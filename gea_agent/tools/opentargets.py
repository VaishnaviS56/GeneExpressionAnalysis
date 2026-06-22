from __future__ import annotations

from typing import Any

import requests

try:
    import mygene
except Exception:  # pragma: no cover - keeps the tool importable before deps are installed
    mygene = None


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


def _extract_mygene_ensembl_id(hit: dict[str, Any]) -> str:
    ensembl = hit.get("ensembl")
    candidates: list[Any] = []
    if isinstance(ensembl, list):
        candidates.extend(ensembl)
    elif isinstance(ensembl, dict):
        candidates.append(ensembl)
    elif ensembl:
        candidates.append(ensembl)

    for candidate in candidates:
        if isinstance(candidate, dict):
            value = _safe_text(candidate.get("gene"))
        else:
            value = _safe_text(candidate)
        if value.startswith("ENSG"):
            return value
    return ""


def resolve_genes_to_ensembl_ids(genes: list[str]) -> dict[str, Any]:
    cleaned_genes = []
    for gene in genes:
        text = _safe_text(gene).upper()
        if text and text not in cleaned_genes:
            cleaned_genes.append(text)

    if not cleaned_genes:
        return {
            "status": "missing_input",
            "genes": [],
            "resolved": [],
            "unresolved": [],
            "message": "At least one gene is required.",
        }

    if mygene is None:
        return {
            "status": "dependency_missing",
            "genes": cleaned_genes,
            "resolved": [],
            "unresolved": cleaned_genes,
            "message": "The mygene package is required to resolve genes to Ensembl IDs.",
        }

    try:
        mg = mygene.MyGeneInfo()
        hits = mg.querymany(
            cleaned_genes,
            scopes="symbol,alias,ensembl.gene",
            fields="symbol,name,ensembl.gene",
            species="human",
            as_dataframe=False,
            verbose=False,
        )
    except Exception as exc:
        return {
            "status": "request_failed",
            "genes": cleaned_genes,
            "resolved": [],
            "unresolved": cleaned_genes,
            "message": f"MyGene lookup failed: {exc}",
        }

    best_by_query: dict[str, dict[str, Any]] = {}
    for hit in hits if isinstance(hits, list) else []:
        if not isinstance(hit, dict) or hit.get("notfound"):
            continue
        query = _safe_text(hit.get("query")).upper()
        ensembl_id = _extract_mygene_ensembl_id(hit)
        if not query or not ensembl_id or query in best_by_query:
            continue
        best_by_query[query] = {
            "gene": query,
            "symbol": _safe_text(hit.get("symbol")).upper(),
            "name": _safe_text(hit.get("name")),
            "ensembl_id": ensembl_id,
            "mygene_result": hit,
        }

    resolved = [best_by_query[gene] for gene in cleaned_genes if gene in best_by_query]
    unresolved = [gene for gene in cleaned_genes if gene not in best_by_query]

    if not resolved:
        status = "not_found"
    elif unresolved:
        status = "partial"
    else:
        status = "ok"

    return {
        "status": status,
        "genes": cleaned_genes,
        "resolved": resolved,
        "unresolved": unresolved,
        "message": f"Resolved {len(resolved)} of {len(cleaned_genes)} genes to Ensembl IDs.",
    }


def resolve_gene_to_ensembl_id(gene: str) -> dict[str, Any]:
    gene = _safe_text(gene)
    if not gene:
        return {
            "status": "missing_input",
            "gene": "",
            "ensembl_id": "",
            "message": "Gene is required.",
        }

    lookup = resolve_genes_to_ensembl_ids([gene])
    resolved = lookup.get("resolved", [])
    if not isinstance(resolved, list) or not resolved:
        return {
            "status": lookup.get("status") or "gene_not_found",
            "gene": gene,
            "ensembl_id": "",
            "gene_result": lookup,
            "message": lookup.get("message") or "Gene was not resolved by MyGene.",
        }

    gene_hit = resolved[0]
    ensembl_id = _safe_text(gene_hit.get("ensembl_id"))
    return {
        "status": "ok" if ensembl_id else "lookup_failed",
        "gene": gene,
        "ensembl_id": ensembl_id,
        "gene_result": gene_hit,
        "message": "Resolved gene to Ensembl ID." if ensembl_id else "Could not resolve an Ensembl ID.",
    }


def _graphql_association(target_id: str, disease_id: str = "") -> dict[str, Any] | None:
    query = """
    query Association($targetId: String!) {
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
            json={"query": query, "variables": {"targetId": target_id}},
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
        if disease_id and isinstance(disease, dict) and str(disease.get("id") or "") == disease_id:
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


def find_diseases_for_gene(gene: str) -> dict[str, Any]:
    gene = _safe_text(gene)
    if not gene:
        return {
            "status": "missing_input",
            "gene": "",
            "associated": False,
            "top_diseases": [],
            "message": "Gene is required.",
        }

    gene_resolution = resolve_gene_to_ensembl_id(gene)
    target_id = str(gene_resolution.get("ensembl_id") or "").strip()
    if not target_id:
        return {
            "status": gene_resolution.get("status") or "lookup_failed",
            "gene": gene,
            "associated": False,
            "gene_resolution": gene_resolution,
            "top_diseases": [],
            "message": gene_resolution.get("message") or "Could not resolve an Ensembl ID.",
        }

    assoc = _graphql_association(target_id)
    if not isinstance(assoc, dict) or assoc.get("status") != "ok":
        return {
            "status": "request_failed",
            "gene": gene,
            "associated": False,
            "gene_resolution": gene_resolution,
            "ensembl_id": target_id,
            "top_diseases": [],
            "message": assoc.get("message") if isinstance(assoc, dict) else "OpenTargets association lookup failed.",
        }

    top_diseases: list[dict[str, Any]] = []
    for row in assoc.get("all_rows", []):
        if not isinstance(row, dict):
            continue
        disease = row.get("disease") or {}
        if not isinstance(disease, dict):
            continue
        try:
            score = float(row.get("score"))
        except Exception:
            score = None
        top_diseases.append(
            {
                "id": disease.get("id"),
                "name": disease.get("name"),
                "score": score,
            }
        )

    return {
        "status": "ok",
        "gene": gene,
        "gene_resolution": gene_resolution,
        "ensembl_id": target_id,
        "associated": bool(top_diseases),
        "top_diseases": top_diseases,
        "candidate_associations": assoc.get("all_rows", []),
        "message": "Retrieved top OpenTargets disease associations." if top_diseases else "No associated diseases found in top OpenTargets rows.",
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


def check_gene_list_disease_associations(genes: list[str], disease: str) -> dict[str, Any]:
    disease = _safe_text(disease)
    resolution = resolve_genes_to_ensembl_ids(genes)
    resolved = resolution.get("resolved", [])
    if not disease:
        return {
            "status": "missing_input",
            "genes": resolution.get("genes", []),
            "disease": disease,
            "associated": False,
            "gene_resolution": resolution,
            "results": [],
            "message": "Disease is required.",
        }
    if not isinstance(resolved, list) or not resolved:
        return {
            "status": resolution.get("status") or "gene_not_found",
            "genes": resolution.get("genes", []),
            "disease": disease,
            "associated": False,
            "gene_resolution": resolution,
            "results": [],
            "message": resolution.get("message") or "No genes were resolved to Ensembl IDs.",
        }

    disease_hit = _search_entity(disease, "disease")
    if not isinstance(disease_hit, dict):
        return {
            "status": "disease_not_found",
            "genes": resolution.get("genes", []),
            "disease": disease,
            "associated": False,
            "gene_resolution": resolution,
            "results": [],
            "message": "Disease was not found in OpenTargets search.",
        }

    disease_id = str(disease_hit.get("id") or disease_hit.get("efoId") or "").strip()
    results: list[dict[str, Any]] = []
    for gene_hit in resolved:
        if not isinstance(gene_hit, dict):
            continue
        gene = _safe_text(gene_hit.get("symbol") or gene_hit.get("gene"))
        target_id = _safe_text(gene_hit.get("ensembl_id"))
        assoc = _graphql_association(target_id, disease_id) if target_id and disease_id else None
        row = assoc.get("association") if isinstance(assoc, dict) else None
        score = None
        if isinstance(row, dict):
            try:
                score = float(row.get("score"))
            except Exception:
                score = None
        results.append(
            {
                "gene": gene,
                "ensembl_id": target_id,
                "associated": bool(row),
                "association_score": score,
                "association": row,
                "candidate_associations": assoc.get("all_rows", []) if isinstance(assoc, dict) else [],
            }
        )

    return {
        "status": "ok",
        "genes": resolution.get("genes", []),
        "disease": disease,
        "gene_resolution": resolution,
        "disease_result": disease_hit,
        "associated": any(result.get("associated") for result in results),
        "results": results,
        "message": f"Checked {len(results)} resolved genes against {disease}.",
    }
