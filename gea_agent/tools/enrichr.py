from __future__ import annotations

from typing import Any, Literal, TypedDict

import gget


EnrichrLibrary = Literal[
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
]


class EnrichrTerm(TypedDict, total=False):
    term: str
    p_value: float
    adjusted_p_value: float
    combined_score: float
    overlapping_genes: list[str]


class EnrichrResults(TypedDict):
    input_genes: list[str]
    background_genes: list[str]
    libraries: dict[str, list[EnrichrTerm]]


def _as_records(obj: Any) -> list[dict[str, Any]]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    # pandas DataFrame-like
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            recs = obj.to_dict(orient="records")
            if isinstance(recs, list):
                return [x for x in recs if isinstance(x, dict)]
        except Exception:
            return []
    return []


def _normalize_terms(records: list[dict[str, Any]], *, top_n: int) -> list[EnrichrTerm]:
    out: list[EnrichrTerm] = []

    def pick(row: dict[str, Any], *keys: str):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return None

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

        term_obj: EnrichrTerm = {"term": term, "overlapping_genes": overlap_genes}
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
    libraries: list[EnrichrLibrary] | None = None,
    top_n: int = 10,
    species: str = "human",
) -> EnrichrResults:
    """
    Enrichment via `gget.enrichr(..., background_list=...)` as in the referenced notebook.

    `background_genes` should represent the universe (e.g. all genes in the STRING network,
    or all expressed genes in the experiment).
    """
    genes = [g.strip().upper() for g in genes if g and g.strip()]
    genes = list(dict.fromkeys(genes))

    background_genes = background_genes or []
    background_genes = [g.strip().upper() for g in background_genes if g and g.strip()]
    # Ensure query genes are included in the background list
    background_genes = list(dict.fromkeys(background_genes + genes))

    if not genes:
        return {"input_genes": [], "background_genes": background_genes, "libraries": {}}

    libraries = libraries or [
        "Reactome_2022",
        "KEGG_2021_Human",
        "GO_Biological_Process_2023",
        "ChEA_2022"
    ]

    out: dict[str, list[EnrichrTerm]] = {}
    for lib in libraries:
        raw = gget.enrichr(
            genes=genes,
            database=lib,
            species=species,
            background_list=background_genes if background_genes else None
            # json=True,
            # verbose=False,
        )
        records = _as_records(raw)
        out[lib] = _normalize_terms(records, top_n=top_n)

    return {
        "input_genes": genes,
        "background_genes": background_genes,
        "libraries": out,
    }