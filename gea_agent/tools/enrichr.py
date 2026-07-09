from __future__ import annotations

import json
import math
import re
from typing import Any

try:
    import gget
except Exception:  # pragma: no cover - dependency guard
    gget = None


def _as_records(obj: Any) -> list[dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            recs = obj.to_dict(orient="records")
            if isinstance(recs, list):
                return [x for x in recs if isinstance(x, dict)]
        except Exception:
            return []
    return []


def _normalize_terms(records: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    def pick(row: dict[str, Any], *keys: str):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return None

    def to_float(value: Any) -> float | None:
        try:
            out = float(value) if value is not None else None
        except Exception:
            return None
        if out is None or math.isnan(out) or math.isinf(out):
            return None
        return out

    normalized: list[dict[str, Any]] = []

    for row in records:
        term = pick(row, "path_name", "term", "term_name", "name", "Path", "Term")
        if not isinstance(term, str):
            continue

        p = pick(row, "p_val", "p_value", "pvalue", "P-value", "P_value")
        adj = pick(
            row,
            "adj_p_val",
            "adjusted_p_value",
            "adj_p_value",
            "adjusted_pval",
            "Adjusted P-value",
            "Adjusted P-value ",
            "adjP",
        )
        cs = pick(row, "combined_score", "Combined Score", "combinedScore")
        overlap = pick(row, "overlapping_genes", "overlap_genes", "genes", "Overlap")

        p_f = to_float(p)
        adj_f = to_float(adj)
        cs_f = to_float(cs)

        if isinstance(overlap, str):
            overlap_genes = [g.strip() for g in re.split(r"[;,]", overlap) if g.strip()]
        elif isinstance(overlap, list):
            overlap_genes = [str(g).strip() for g in overlap if str(g).strip()]
        else:
            overlap_genes = []
        normalized.append(
            {
                "term": term,
                "p_value": p_f,
                "adjusted_p_value": adj_f,
                "combined_score": cs_f,
                "overlapping_genes": overlap_genes,
                "n_overlap_genes": len(overlap_genes),
            }
        )

    normalized.sort(
        key=lambda x: (
            x["adjusted_p_value"]
            if x["adjusted_p_value"] is not None
            else float("inf")
        )
    )

    return normalized[:top_n]


def enrichr_pathways(
    genes: list[str],
    *,
    background_genes: list[str] | None = None,
    libraries: list[str] | None = None,
    top_n: int = 10,
    species: str = "human",
) -> dict[str, Any]:
    genes = [g.strip().upper() for g in genes if g and g.strip()]
    genes = list(dict.fromkeys(genes))
    background_genes = background_genes or []
    background_genes = [g.strip().upper() for g in background_genes if g and g.strip()]
    background_genes = list(dict.fromkeys(background_genes + genes))

    if not genes:
        return {"status": "missing_input", "input_genes": [], "background_genes": background_genes, "libraries": {}}

    if gget is None:
        return {
            "status": "dependency_missing",
            "input_genes": genes,
            "background_genes": background_genes,
            "libraries": {},
            "messages": ["gget is not installed, so enrichment could not be run."],
        }

    libraries = libraries or [
        "Reactome_2022",
        "KEGG_2021_Human",
        "GO_Biological_Process_2023",
        "GO_Molecular_Function_2023",
        "GO_Cellular_Component_2023",
    ]

    print("Enrichr genes: ", genes)

    out: dict[str, list[dict[str, Any]]] = {}
    messages: list[str] = []
    for lib in libraries:
        try:
            raw = gget.enrichr(
                genes=genes,
                database=lib,
                species=species,
            )
            records = _as_records(raw)
            out[lib] = _normalize_terms(records, top_n=top_n)
        except Exception as exc:
            out[lib] = []
            messages.append(f"{lib}: {exc}")

    status = "ok" if any(out.values()) else "error"
    return {
        "status": status,
        "input_genes": genes,
        "background_genes": background_genes,
        "libraries": out,
        "top_pathways": {lib: terms[:10] for lib, terms in out.items() if isinstance(terms, list)},
        "messages": messages,
    }
