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


def _compact_seed_list(genes: list[str] | None, *, limit: int = 10) -> list[str]:
    if not isinstance(genes, list):
        return []
    return [str(g) for g in genes[:limit] if str(g).strip()]


def _compact_enrichr(enrichr: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(enrichr, dict):
        return None

    libs = enrichr.get("libraries")
    if not isinstance(libs, dict):
        return None

    out: dict[str, Any] = {}
    for lib_name, terms in list(libs.items())[:4]:
        if not isinstance(terms, list):
            continue
        compact_terms: list[dict[str, Any]] = []
        for term in terms[:10]:
            if not isinstance(term, dict):
                continue
            compact_terms.append(
                {
                    "t": term.get("term"),
                    "p": term.get("p_value"),
                    "adj": term.get("adjusted_p_value"),
                    "cs": term.get("combined_score"),
                    "genes": term.get("overlapping_genes"),
                }
            )
        out[str(lib_name)] = compact_terms

    return out or None


def _compact_opentargets(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None

    compact: dict[str, Any] = {
        "status": result.get("status"),
        "gene": result.get("gene"),
        "disease": result.get("disease"),
        "ensembl_id": result.get("ensembl_id"),
        "associated": result.get("associated"),
        "association_score": result.get("association_score"),
        "message": result.get("message"),
    }

    top_diseases = result.get("top_diseases")
    if isinstance(top_diseases, list):
        compact["top_diseases"] = [
            {
                "name": row.get("name"),
                "score": row.get("score"),
            }
            for row in top_diseases[:10]
            if isinstance(row, dict)
        ]

    results = result.get("results")
    if isinstance(results, list):
        compact["results"] = [
            {
                "gene": row.get("gene"),
                "ensembl_id": row.get("ensembl_id"),
                "associated": row.get("associated"),
                "association_score": row.get("association_score"),
            }
            for row in results[:10]
            if isinstance(row, dict)
        ]

    top_drugs = result.get("top_drugs")
    if isinstance(top_drugs, list):
        compact["top_drugs"] = [
            {
                "name": row.get("name"),
                "phase": row.get("phase"),
                "status": row.get("status"),
                "disease_name": row.get("disease_name"),
            }
            for row in top_drugs[:10]
            if isinstance(row, dict)
        ]

    return {key: value for key, value in compact.items() if value not in (None, "", [])}


def _compact_l1000cds2(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None

    compact: dict[str, Any] = {
        "status": result.get("status"),
        "message": result.get("message"),
        "mode": result.get("mode"),
        "requested_cell_lines": result.get("requested_cell_lines"),
        "cell_line_filter_applied": result.get("cell_line_filter_applied"),
        "up_gene_count": result.get("up_gene_count"),
        "down_gene_count": result.get("down_gene_count"),
        "signature_count": result.get("signature_count"),
    }

    top_drugs = result.get("top_drugs")
    if isinstance(top_drugs, list):
        compact["top_drugs"] = [
            {
                "name": row.get("name"),
                "pert_id": row.get("pert_id"),
                "best_rank": row.get("best_rank"),
                "best_score": row.get("best_score"),
                "cell_lines": row.get("cell_lines"),
                "signature_count": row.get("signature_count"),
            }
            for row in top_drugs[:10]
            if isinstance(row, dict)
        ]

    top_signatures = result.get("top_signatures")
    if isinstance(top_signatures, list):
        compact["top_signatures"] = [
            {
                "rank": row.get("rank"),
                "perturbation": row.get("perturbation"),
                "pert_id": row.get("pert_id"),
                "cell_line": row.get("cell_line"),
                "dose": row.get("dose"),
                "dose_unit": row.get("dose_unit"),
                "time": row.get("time"),
                "time_unit": row.get("time_unit"),
            }
            for row in top_signatures[:5]
            if isinstance(row, dict)
        ]

    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_pubchem(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None

    properties = result.get("properties") if isinstance(result.get("properties"), dict) else {}
    compact = {
        "status": result.get("status"),
        "message": result.get("message"),
        "drug_name": result.get("drug_name"),
        "pert_id": result.get("pert_id"),
        "matched_query": result.get("matched_query"),
        "matched_strategy": result.get("matched_strategy"),
        "title": result.get("title"),
        "cid": result.get("cid"),
        "properties": {
            key: properties.get(key)
            for key in (
                "MolecularFormula",
                "MolecularWeight",
                "CanonicalSMILES",
                "IsomericSMILES",
                "InChI",
                "InChIKey",
                "XLogP",
                "TPSA",
                "HBondDonorCount",
                "HBondAcceptorCount",
                "RotatableBondCount",
                "Complexity",
            )
            if properties.get(key) not in (None, "")
        },
        "synonyms": result.get("synonyms")[:30] if isinstance(result.get("synonyms"), list) else [],
        "descriptions": result.get("descriptions")[:10] if isinstance(result.get("descriptions"), list) else [],
        "annotation_lines": result.get("annotation_lines")[:60] if isinstance(result.get("annotation_lines"), list) else [],
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_primekg(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    raw_rows = result.get("raw_result")
    compact_rows: list[dict[str, Any]] = []
    if isinstance(raw_rows, list):
        for row in raw_rows[:10]:
            if not isinstance(row, dict):
                compact_rows.append({"value": _compact_text(row, limit=160)})
                continue
            compact_row: dict[str, Any] = {}
            for key, value in list(row.items())[:8]:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    compact_row[str(key)] = value
                else:
                    compact_row[str(key)] = _compact_text(value, limit=160)
            if compact_row:
                compact_rows.append(compact_row)

    edges = result.get("edges")
    compact_edges: list[dict[str, Any]] = []
    if isinstance(edges, list):
        for edge in edges[:20]:
            if not isinstance(edge, dict):
                continue
            source = edge.get("source") if isinstance(edge.get("source"), dict) else {}
            target = edge.get("target") if isinstance(edge.get("target"), dict) else {}
            compact_edges.append(
                {
                    "relation": edge.get("display_relation") or edge.get("relation"),
                    "source": source.get("name"),
                    "source_type": source.get("type"),
                    "target": target.get("name"),
                    "target_type": target.get("type"),
                }
            )

    compact = {
        "status": result.get("status"),
        "question": result.get("question") or result.get("query"),
        "answer": result.get("answer"),
        "cypher": result.get("cypher"),
        "candidate_count": result.get("candidate_count"),
        "selected_count": result.get("selected_count"),
        "ranking_method": result.get("ranking_method"),
        "row_count": len(raw_rows) if isinstance(raw_rows, list) else result.get("count"),
        "rows": compact_rows,
        "edges": compact_edges,
        "message": result.get("message"),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_literature(
    papers: list[dict[str, Any]] | None,
    ranked_papers: list[dict[str, Any]] | None,
    key_points: list[dict[str, Any]] | None,
    references: list[dict[str, Any]] | None,
    summary: str | None,
) -> dict[str, Any] | None:
    out: dict[str, Any] = {}

    if summary:
        out["summary"] = summary

    if isinstance(key_points, list):
        out["key_points"] = [
            {
                "point": row.get("point"),
                "paper_ids": row.get("paper_ids"),
            }
            for row in key_points[:6]
            if isinstance(row, dict)
        ]

    if isinstance(ranked_papers, list):
        out["top_papers"] = [
            {
                "paper_id": row.get("id"),
                "source": row.get("source"),
                "title": row.get("title"),
                "year": row.get("year"),
                "doi": row.get("doi"),
                "pmid": row.get("pmid"),
                "reason": row.get("reason"),
                "relevance": row.get("relevance"),
            }
            for row in ranked_papers[:5]
            if isinstance(row, dict)
        ]

    if isinstance(references, list):
        out["references"] = [
            {
                "paper_id": row.get("paper_id"),
                "source": row.get("source"),
                "title": row.get("title"),
                "year": row.get("year"),
                "doi": row.get("doi"),
                "pmid": row.get("pmid"),
                "url": row.get("url"),
            }
            for row in references[:8]
            if isinstance(row, dict)
        ]
    elif isinstance(papers, list):
        out["references"] = [
            {
                "paper_id": index,
                "source": row.get("source"),
                "title": row.get("title"),
                "year": row.get("year"),
                "doi": row.get("doi"),
                "pmid": row.get("pmid"),
                "url": row.get("url"),
            }
            for index, row in enumerate(papers[:8], start=1)
            if isinstance(row, dict)
        ]

    return out or None


def synthesize_technical_response(
    *,
    user_query: str,
    analysis_arm: str,
    seed_genes: list[str],
    srp_ids: list[str] | None,
    disease_name: str | None,
    deg_analysis: dict[str, Any] | None,
    rwr_genes: list[tuple[str, float]],
    graph: nx.Graph,
    enrichr: dict[str, Any],
    literature_papers: list[dict[str, Any]] | None = None,
    ranked_literature_papers: list[dict[str, Any]] | None = None,
    literature_key_points: list[dict[str, Any]] | None = None,
    literature_references: list[dict[str, Any]] | None = None,
    literature_summary: str | None = None,
) -> str:
    llm = get_llm()

    arm = (analysis_arm or "disease").strip().lower()
    payload: dict[str, Any] = {
        "arm": arm,
        "q": _compact_text(user_query, limit=350),
        "seeds": _compact_seed_list(seed_genes),
        "enr": _compact_enrichr(enrichr),
    }

    if arm == "srp":
        payload["srp"] = _compact_seed_list(srp_ids, limit=10)
        payload["deg"] = _compact_deg_analysis(deg_analysis)
    elif arm == "l1000cds2":
        payload["l1000"] = _compact_l1000cds2(deg_analysis)
    elif arm == "pubchem":
        payload["pubchem"] = _compact_pubchem(deg_analysis)
    elif arm == "opentargets":
        payload["ot"] = _compact_opentargets(deg_analysis)
    elif arm == "primekg":
        payload["kg"] = _compact_primekg(deg_analysis)
    elif arm == "memory_rwr":
        payload["rwr"] = _compact_rwr(rwr_genes)
        payload["net"] = {"n": graph.number_of_nodes(), "e": graph.number_of_edges()}
    else:
        payload["disease"] = _compact_text(disease_name, limit=120)
        payload["lit"] = _compact_literature(
            literature_papers,
            ranked_literature_papers,
            literature_key_points,
            literature_references,
            literature_summary,
        )
        payload["rwr"] = _compact_rwr(rwr_genes)
        payload["net"] = {"n": graph.number_of_nodes(), "e": graph.number_of_edges()}

    resp = llm.invoke(
        [
            (
                "system",
                "You are the final synthesis stage for a biomedical analysis agent. "
                "Write the final user-facing answer using only the structured payload you receive. "
                "If a field is absent or empty, omit it instead of guessing. "
                "Do not mention analysis arms, tool names, prompts, routing logic, or hidden intermediate steps. "
                "Return plain text only, not JSON and not markdown code fences. "
                "Answer the user's question directly in clear technical language. "
                "Lead with the highest-signal findings, then add only the most relevant supporting detail. "
                "State uncertainty explicitly whenever evidence is limited, mixed, or indirect. "
                "Respect the active context: "
                "for `srp`, summarize only DEG plus any provided enrichment context; "
                "for `l1000cds2`, summarize only the returned small-molecule matches, requested cell-line filter, and whether the result reflects reversal or mimic mode; "
                "for `pubchem`, identify only genes, pathways, and diseases that are explicitly supported or reasonably inferable from the provided PubChem text and annotations; organize that answer with short `Genes`, `Pathways`, and `Diseases` labels when possible; if PubChem content does not support one of those categories, say so clearly; "
                "for `primekg`, answer only from the provided knowledge-graph relationships; "
                "for `opentargets`, summarize only the association evidence provided; "
                "for `memory_rwr`, summarize only the stored-gene RWR prioritization; "
                "for `disease`, summarize literature findings first, then relevant network or enrichment context if present. "
                "Keep the answer concise but scientifically useful. "
                "When literature references are available, end with a short `References:` section.",
            ),
            ("user", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        ]
    )
    return getattr(resp, "content", "")
