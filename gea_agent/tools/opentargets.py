from __future__ import annotations

from typing import Any

import requests


_SEARCH_URL = "https://api.platform.opentargets.org/api/v4/search"
_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _search_entity(query: str, entity_type: str) -> dict[str, Any] | None:
    try:
        response = requests.get(
            _SEARCH_URL,
            params={"q": query, "type": entity_type},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    items = payload.get("data") or payload.get("results") or []
    if not isinstance(items, list) or not items:
        return None

    for item in items:
        if isinstance(item, dict):
            return item
    return None


def _extract_ensembl_id(target_hit: dict[str, Any] | None) -> str:
    if not isinstance(target_hit, dict):
        return ""
    candidates = [
        target_hit.get("id"),
        target_hit.get("ensemblId"),
        target_hit.get("ensgId"),
        target_hit.get("geneId"),
        target_hit.get("targetId"),
    ]
    for candidate in candidates:
        value = _safe_text(candidate)
        if value:
            return value
    return ""


def resolve_gene_to_ensembl_id(gene: str) -> dict[str, Any]:
    gene = _safe_text(gene)
    if not gene:
        return {
            "status": "missing_input",
            "gene": "",
            "ensembl_id": "",
            "message": "Gene is required.",
        }

    gene_hit = _search_entity(gene, "target")
    if not isinstance(gene_hit, dict):
        return {
            "status": "gene_not_found",
            "gene": gene,
            "ensembl_id": "",
            "message": "Gene was not found in OpenTargets search.",
        }

    ensembl_id = _extract_ensembl_id(gene_hit)
    return {
        "status": "ok" if ensembl_id else "lookup_failed",
        "gene": gene,
        "ensembl_id": ensembl_id,
        "gene_result": gene_hit,
        "message": "Resolved gene to Ensembl ID." if ensembl_id else "Could not resolve an Ensembl ID.",
    }


def _graphql_association(target_id: str, disease_id: str) -> dict[str, Any] | None:
    query = """
    query Association($targetId: String!, $diseaseId: String!) {
      target(ensemblId: $targetId) {
        approvedSymbol
        associatedDiseases {
          count
          rows {
            disease {
              id
              name
            }
            score
          }
        }
      }
    }
    """
    try:
        response = requests.post(
            _GRAPHQL_URL,
            json={"query": query, "variables": {"targetId": target_id, "diseaseId": disease_id}},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {"status": "request_failed", "message": str(exc)}

    target = (((payload or {}).get("data") or {}).get("target") or {})
    rows = (((target or {}).get("associatedDiseases") or {}).get("rows") or [])
    if not isinstance(rows, list):
        rows = []

    association = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        disease = row.get("disease") or {}
        if isinstance(disease, dict) and str(disease.get("id") or "") == disease_id:
            association = row
            break

    return {
        "status": "ok",
        "target": {
            "id": target_id,
            "symbol": target.get("approvedSymbol", ""),
        },
        "association": association,
        "all_rows": rows[:10],
    }


def check_gene_disease_association(gene: str, disease: str) -> dict[str, Any]:
    gene = _safe_text(gene)
    disease = _safe_text(disease)
    if not gene or not disease:
        return {
            "status": "missing_input",
            "gene": gene,
            "disease": disease,
            "associated": False,
            "message": "Both gene and disease are required.",
        }

    gene_resolution = resolve_gene_to_ensembl_id(gene)
    disease_hit = _search_entity(disease, "disease")

    if not isinstance(disease_hit, dict):
        return {
            "status": "disease_not_found",
            "gene": gene,
            "disease": disease,
            "associated": False,
            "gene_resolution": gene_resolution,
            "message": "Disease was not found in OpenTargets search.",
        }

    target_id = str(gene_resolution.get("ensembl_id") or "").strip()
    disease_id = str(disease_hit.get("id") or disease_hit.get("efoId") or "").strip()

    if not target_id or not disease_id:
        return {
            "status": "lookup_failed",
            "gene": gene,
            "disease": disease,
            "associated": False,
            "gene_resolution": gene_resolution,
            "disease_result": disease_hit,
            "message": "Could not resolve OpenTargets identifiers.",
        }

    assoc = _graphql_association(target_id, disease_id)
    if not isinstance(assoc, dict):
        return {
            "status": "request_failed",
            "gene": gene,
            "disease": disease,
            "associated": False,
            "gene_resolution": gene_resolution,
            "disease_result": disease_hit,
            "message": "OpenTargets association lookup failed.",
        }

    row = assoc.get("association")
    score = None
    if isinstance(row, dict):
        try:
            score = float(row.get("score"))
        except Exception:
            score = None

    return {
        "status": "ok",
        "gene": gene,
        "disease": disease,
        "gene_resolution": gene_resolution,
        "disease_result": disease_hit,
        "ensembl_id": target_id,
        "associated": bool(row),
        "association_score": score,
        "association": row,
        "candidate_associations": assoc.get("all_rows", []),
        "message": "Association found." if row else "No direct association found in top OpenTargets rows.",
    }
