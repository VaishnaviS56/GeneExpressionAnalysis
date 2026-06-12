from __future__ import annotations

import json
from typing import Any

import networkx as nx

from gea_agent.tools.llm import get_llm


def _compact_text(value: Any, *, limit: int) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _compact_deg_analysis(deg_analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(deg_analysis, dict):
        return None

    rows = deg_analysis.get("rows")
    genes = deg_analysis.get("genes", [])
    compact_rows: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows[:10]:
            if not isinstance(row, dict):
                continue
            compact_rows.append(
                {
                    "g": row.get("hgnc_symbol") or row.get("external_gene_name") or row.get("Ensembl"),
                    "l2fc": row.get("log2FoldChange"),
                    "p": row.get("pvalue"),
                }
            )

    return {
        "status": deg_analysis.get("status"),
        "n": len(rows) if isinstance(rows, list) else 0,
        "genes": genes[:10] if isinstance(genes, list) else [],
        "rows": compact_rows,
    }


def _compact_rwr(rwr_genes: list[tuple[str, float]]) -> list[dict[str, Any]]:
    return [{"g": g, "s": round(float(score), 4)} for g, score in rwr_genes[:10]]


def _compact_enrichr(enrichr: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(enrichr, dict):
        return None

    libs = enrichr.get("libraries")
    if not isinstance(libs, dict):
        return None

    out: dict[str, Any] = {}
    for lib_name, terms in list(libs.items())[:2]:
        if not isinstance(terms, list):
            continue
        compact_terms: list[dict[str, Any]] = []
        for term in terms[:3]:
            if not isinstance(term, dict):
                continue
            compact_terms.append(
                {
                    "t": term.get("term"),
                    "adj": term.get("adjusted_p_value"),
                }
            )
        out[str(lib_name)] = compact_terms

    return out or None


def synthesize_technical_response(
    *,
    user_query: str,
    seed_genes: list[str],
    deg_analysis: dict[str, Any] | None,
    rwr_genes: list[tuple[str, float]],
    graph: nx.Graph,
    enrichr: dict[str, Any],
) -> str:
    llm = get_llm()

    payload = {
        "q": _compact_text(user_query, limit=400),
        "deg": _compact_deg_analysis(deg_analysis),
        "net": {"n": graph.number_of_nodes(), "e": graph.number_of_edges()},
        "rwr": _compact_rwr(rwr_genes),
        "enr": _compact_enrichr(enrichr),
    }
    resp = llm.invoke(
        [
            ("system", "Write a brief bioinformatics summary using the compact JSON provided."),
            ("user", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        ]
    )
    return getattr(resp, "content", "")
