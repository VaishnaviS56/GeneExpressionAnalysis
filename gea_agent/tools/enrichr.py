from __future__ import annotations

from typing import Any

import gget


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

    out: list[dict[str, Any]] = []
    for row in records[: max(top_n * 3, top_n)]:
        term = pick(row, "path_name", "term", "term_name", "name")
        if not isinstance(term, str):
            continue

        p = pick(row, "p_val", "p_value")
        adj = pick(row, "adj_p_val", "adjusted_p_value", "adj_p_value")
        cs = pick(row, "combined_score")
        overlap = pick(row, "overlapping_genes", "overlap_genes", "genes")

        try:
            p_f = float(p) if p is not None else None
        except Exception:
            p_f = None
        try:
            adj_f = float(adj) if adj is not None else None
        except Exception:
            adj_f = None
        try:
            cs_f = float(cs) if cs is not None else None
        except Exception:
            cs_f = None

        if isinstance(overlap, str):
            overlap_genes = [g.strip() for g in overlap.split(";") if g.strip()]
        elif isinstance(overlap, list):
            overlap_genes = [str(g).strip() for g in overlap if str(g).strip()]
        else:
            overlap_genes = []

        term_obj: dict[str, Any] = {"term": term, "overlapping_genes": overlap_genes}
        if p_f is not None:
            term_obj["p_value"] = p_f
        if adj_f is not None:
            term_obj["adjusted_p_value"] = adj_f
        if cs_f is not None:
            term_obj["combined_score"] = cs_f

        out.append(term_obj)
        if len(out) >= top_n:
            break

    return out


def enrichr_pathways(
    genes: list[str],
    *,
    background_genes: list[str] | None = None,
    libraries: list[str] | None = None,
    top_n: int = 10,
    species: str = "human",
) -> dict[str, Any]:
    """
    Enrichment via `gget.enrichr(..., background_list=...)`.

    Returns simple data structures:
    {
      "input_genes": [...],
      "background_genes": [...],
      "libraries": {
         "Reactome_2022": [ {"term":..., "adjusted_p_value":..., ...}, ...],
      }
    }
    """
    genes = [g.strip().upper() for g in genes if g and g.strip()]
    genes = list(dict.fromkeys(genes))

    background_genes = background_genes or []
    background_genes = [g.strip().upper() for g in background_genes if g and g.strip()]
    background_genes = list(dict.fromkeys(background_genes + genes))

    if not genes:
        return {"input_genes": [], "background_genes": background_genes, "libraries": {}}

    libraries = libraries or [
        "Reactome_2022",
        "KEGG_2021_Human",
        "GO_Biological_Process_2023",
    ]

    out: dict[str, list[dict[str, Any]]] = {}
    for lib in libraries:
        raw = gget.enrichr(
            genes=genes,
            database=lib,
            species=species,
            background_list=background_genes if background_genes else None,
            json=True,
            verbose=False,
        )
        records = _as_records(raw)
        out[lib] = _normalize_terms(records, top_n=top_n)

    return {
        "input_genes": genes,
        "background_genes": background_genes,
        "libraries": out,
    }