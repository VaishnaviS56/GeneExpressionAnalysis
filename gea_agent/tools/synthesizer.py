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


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item not in (None, ""):
                parts.append(str(item))
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "").strip()
    return str(content or "").strip()


def _compact_deg_analysis(deg_analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(deg_analysis, dict):
        return None

    rows = deg_analysis.get("rows")
    genes = deg_analysis.get("genes", [])
    up_rows = deg_analysis.get("upregulated_rows")
    down_rows = deg_analysis.get("downregulated_rows")

    def compact_rows(source: Any) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        if isinstance(source, list):
            for row in source[:10]:
                if not isinstance(row, dict):
                    continue
                compact.append(
                    {
                        "g": row.get("gene") or row.get("hgnc_symbol") or row.get("external_gene_name") or row.get("Ensembl"),
                        "l2fc": row.get("log2FoldChange"),
                        "p": row.get("pvalue"),
                    }
                )
        return compact

    compact_up_rows = compact_rows(up_rows)
    compact_down_rows = compact_rows(down_rows)

    if not compact_up_rows or not compact_down_rows:
        fallback_up: list[dict[str, Any]] = []
        fallback_down: list[dict[str, Any]] = []
        if isinstance(rows, list):
            sorted_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    log2fc = float(row.get("log2FoldChange"))
                except Exception:
                    continue
                sorted_rows.append((log2fc, row))
            fallback_up = [
                row
                for log2fc, row in sorted(sorted_rows, key=lambda item: item[0], reverse=True)
                if log2fc > 0
            ]
            fallback_down = [
                row
                for log2fc, row in sorted(sorted_rows, key=lambda item: item[0])
                if log2fc < 0
            ]
        compact_up_rows = compact_up_rows or compact_rows(fallback_up)
        compact_down_rows = compact_down_rows or compact_rows(fallback_down)

    compact_mixed_rows: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows[:10]:
            if not isinstance(row, dict):
                continue
            compact_mixed_rows.append(
                {
                    "g": row.get("gene") or row.get("hgnc_symbol") or row.get("external_gene_name") or row.get("Ensembl"),
                    "l2fc": row.get("log2FoldChange"),
                    "p": row.get("pvalue"),
                }
            )

    return {
        "status": deg_analysis.get("status"),
        "n": len(rows) if isinstance(rows, list) else 0,
        "log2fold": deg_analysis.get("log2fold"),
        "padj": deg_analysis.get("padj"),
        "thresholds_applied_post_hoc": deg_analysis.get("thresholds_applied_post_hoc"),
        "genes": genes[:10] if isinstance(genes, list) else [],
        "up_rows": compact_up_rows,
        "down_rows": compact_down_rows,
        "rows": compact_mixed_rows,
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


def _compact_memory_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None

    compact: dict[str, Any] = {
        "status": result.get("status"),
        "field": result.get("field") or result.get("requested_field") or result.get("resolved_field"),
        "field_length": result.get("field_length"),
        "selection_mode": result.get("selection_mode"),
        "top_n": result.get("top_n"),
        "bottom_n": result.get("bottom_n"),
        "answer": result.get("answer"),
        "message": result.get("message"),
    }

    selected_term = result.get("selected_term")
    if isinstance(selected_term, dict):
        compact["selected_term"] = {
            "library": selected_term.get("library"),
            "term": selected_term.get("term"),
            "rank": selected_term.get("rank"),
        }

    selected_values = result.get("selected_values")
    if isinstance(selected_values, list):
        compact["selected_count"] = len(selected_values)
        compact["selected_values"] = selected_values[:30]

    selected_genes = result.get("selected_gene_candidates")
    if isinstance(selected_genes, list):
        compact["selected_gene_candidates"] = selected_genes[:50]

    intersection = result.get("intersection_genes")
    if isinstance(intersection, list):
        compact["intersection_genes"] = intersection[:50]

    pathway_genes = result.get("pathway_genes")
    if isinstance(pathway_genes, list):
        compact["pathway_genes"] = pathway_genes[:50]

    deg_genes = result.get("deg_genes")
    if isinstance(deg_genes, list):
        compact["deg_genes"] = deg_genes[:50]

    inspections = result.get("inspections")
    if isinstance(inspections, list):
        compact["inspections"] = inspections[:20]

    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _fallback_answer(payload: dict[str, Any]) -> str:
    arm = str(payload.get("arm") or "general")
    lines: list[str] = ["**Summary**"]

    memory = payload.get("memory")
    if isinstance(memory, dict) and memory:
        field = memory.get("field")
        selected = memory.get("selected_values")
        intersection = memory.get("intersection_genes")
        if field and selected:
            lines.append(
                f"Selected {len(selected) if isinstance(selected, list) else memory.get('selected_count', 'the requested')} item(s) from `{field}`."
            )
            if isinstance(selected, list):
                lines.append("")
                lines.append("**Selected Values**")
                lines.append(", ".join(str(value) for value in selected[:30]))
            return "\n".join(lines).strip()
        if isinstance(intersection, list) and intersection:
            lines.append(f"Found {len(intersection)} overlapping gene(s).")
            lines.append("")
            lines.append("**Genes**")
            lines.append(", ".join(str(value) for value in intersection[:50]))
            return "\n".join(lines).strip()
        raw_answer = str(memory.get("answer") or memory.get("message") or "").strip()
        if raw_answer:
            lines.append(raw_answer)
            return "\n".join(lines).strip()

    enrichr = payload.get("enr")
    if isinstance(enrichr, dict) and enrichr:
        lines.append("Pathway enrichment results were generated.")
        lines.append("")
        lines.append("**Top Terms**")
        for library, terms in list(enrichr.items())[:4]:
            if not isinstance(terms, list) or not terms:
                continue
            first = terms[0] if isinstance(terms[0], dict) else {}
            term = first.get("t") or first.get("term") or "top returned term"
            genes = first.get("genes")
            suffix = f" ({len(genes)} overlapping genes)" if isinstance(genes, list) else ""
            lines.append(f"- {library}: {term}{suffix}")
        return "\n".join(lines).strip()

    deg = payload.get("deg")
    if isinstance(deg, dict) and deg:
        count = deg.get("n") or len(deg.get("genes") or [])
        lines.append(f"Differential expression analysis completed with {count} retained row(s).")
        for title, key in (("Top Up-Regulated Genes", "up_rows"), ("Top Down-Regulated Genes", "down_rows")):
            rows = deg.get(key)
            if not isinstance(rows, list) or not rows:
                continue
            lines.append("")
            lines.append(f"**{title}**")
            lines.append("| Gene | log2FC | p-value |")
            lines.append("|---|---:|---:|")
            for row in rows[:10]:
                if not isinstance(row, dict):
                    continue
                lines.append(f"| {row.get('g', '')} | {row.get('l2fc', '')} | {row.get('p', '')} |")
        return "\n".join(lines).strip()

    if arm == "general":
        lines.append("The agent completed the requested step, but there was not enough structured output to produce a detailed summary.")
    else:
        lines.append(f"The `{arm}` result was generated, but no additional summary text was returned.")
    return "\n".join(lines).strip()


def _suggested_followup(payload: dict[str, Any]) -> str:
    arm = str(payload.get("arm") or "general").strip().lower()
    if arm == "srp":
        return "I can next run pathway enrichment on the up-regulated and down-regulated DEG sets, or make a volcano plot from these DEG rows."
    if arm == "pathway" or payload.get("enr"):
        return "I can next visualize a top KEGG pathway or prioritize candidate targets from the enriched gene set with RWR."
    if arm in {"memory_rwr", "general"} and payload.get("rwr"):
        return "I can next check the top RWR candidates against OpenTargets or PrimeKG for disease and drug relationships."
    if arm == "l1000cds2":
        return "I can next look up a top compound in PubChem to summarize supported genes, pathways, and disease annotations."
    if arm == "pubchem":
        return "I can next compare the PubChem-supported annotations with your stored DEG or pathway results."
    if arm == "opentargets":
        return "I can next follow the strongest associated genes into PrimeKG or generate hypothesis candidates."
    if arm == "primekg":
        return "I can next validate the returned genes with OpenTargets or turn the related genes into an enrichment analysis."
    if arm == "hypothesis":
        return "I can next run a separate literature or OpenTargets evidence check for any hypothesis you want to investigate."
    if arm in {"literature", "research_literature", "disease"}:
        return "I can next convert the literature-supported genes into RWR target prioritization or pathway enrichment."
    if arm in {"memory_lookup", "state_lookup", "memory_slice"}:
        return "I can next use this stored selection as input for pathway enrichment, RWR, or literature support."
    return "I can next run a supported follow-up using the stored genes, pathways, literature, or DEG results from this session."


def _ensure_suggested_followup(answer: str, payload: dict[str, Any]) -> str:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return cleaned
    if "**suggested follow-up**" in cleaned.lower() or "suggested follow-up" in cleaned.lower():
        return cleaned
    return f"{cleaned}\n\n**Suggested Follow-Up**\n{_suggested_followup(payload)}".strip()


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
    memory_lookup_result: dict[str, Any] | None = None,
    state_lookup_result: dict[str, Any] | None = None,
    memory_slice_result: dict[str, Any] | None = None,
) -> str:
    llm = get_llm()

    arm = (analysis_arm or "disease").strip().lower()
    payload: dict[str, Any] = {
        "arm": arm,
        "q": _compact_text(user_query, limit=350),
        "seeds": _compact_seed_list(seed_genes),
    }

    if arm == "srp":
        payload["srp"] = _compact_seed_list(srp_ids, limit=10)
        payload["deg"] = _compact_deg_analysis(deg_analysis)
    elif arm == "pathway":
        payload["enr"] = _compact_enrichr(enrichr)
    elif arm == "l1000cds2":
        payload["l1000"] = _compact_l1000cds2(deg_analysis)
    elif arm == "pubchem":
        payload["pubchem"] = _compact_pubchem(deg_analysis)
    elif arm == "opentargets":
        payload["ot"] = _compact_opentargets(deg_analysis)
    elif arm == "primekg":
        payload["kg"] = _compact_primekg(deg_analysis)
    elif arm in {"memory_lookup", "state_lookup", "memory_slice"}:
        if arm == "memory_lookup":
            payload["memory"] = _compact_memory_result(memory_lookup_result or deg_analysis)
        elif arm == "state_lookup":
            payload["memory"] = _compact_memory_result(state_lookup_result or deg_analysis)
        else:
            payload["memory"] = _compact_memory_result(memory_slice_result or deg_analysis)
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

    try:
        resp = llm.invoke(
            [
                (
                    "system",
                    "You are the final synthesis stage for a biomedical analysis agent. "
                    "Write the final user-facing answer using only the structured payload you receive. "
                    "If a field is absent or empty, omit it instead of guessing. "
                    "Do not mention analysis arms, tool names, prompts, routing logic, or hidden intermediate steps. "
                    "Return plain text only, not JSON and not markdown code fences. "
                    "Answer the user's question directly in clear, professional technical language. "
                    "Format the response as a polished technical report using Markdown. "
                    "Use bold section headings such as `**Summary**`, `**Key Findings**`, `**Interpretation**`, `**Evidence**`, and `**References**` when relevant and necessary. Do not repeat the same information under different headings; instead, synthesize and integrate the evidence into a coherent narrative. "
                    "Start with a clear summary, then briefly describe the key points with evidence when available. "
                    "When multiple findings are present, use organized bullets so the answer remains readable while still being thorough. "
                    "Include methods/context, key results, notable genes or terms, interpretation, caveats, and practical next steps when those are supported by the payload. "
                    "Do not compress away important evidence solely for brevity. "
                    "Avoid repetitions across sections; instead, synthesize and integrate the evidence into a coherent narrative. "
                    "Lead with the highest-signal findings, then add relevant supporting detail from the payload. "
                    "State uncertainty explicitly whenever evidence is limited, mixed, or indirect. "
                    "Sound like a professional biomedical analyst: precise, neutral, well organized, and conversational enough to guide the user through the next move. "
                    "Avoid filler, repetition, and unsupported enthusiasm. "
                    "Respect the active context: "
                    "for `srp`, summarize only DEG results and include two separate Markdown tables named `Top Up-Regulated Genes` and `Top Down-Regulated Genes` using the provided `up_rows` and `down_rows`; each table should include Gene, log2FC, and p-value columns and at most 10 rows; if log2FC and padj are not given by user say that the default values have been used; "
                    "for `visualize`, if payload contains *_path fields, just say that the visualization was generated successfully; "
                    "for `l1000cds2`, summarize only the returned small-molecule matches, requested cell-line filter, and whether the result reflects reversal or mimic mode; "
                    "for `pubchem`, identify only genes, pathways, and diseases that are explicitly supported or reasonably inferable from the provided PubChem text and annotations; organize that answer with `Genes`, `Pathways`, and `Diseases` labels when possible; if PubChem content does not support one of those categories, say so clearly; "
                    "for `primekg`, answer only from the provided knowledge-graph relationships; "
                    "for `opentargets`, summarize only the association evidence provided; "
                    "for `memory_lookup`, `state_lookup`, and `memory_slice`, convert the stored values into a clean readable answer and avoid raw JSON unless the user asked for literal state; "
                    "for `pathway`, summarize the enrichment results from `enr`, highlighting the top terms, libraries, adjusted p-values when available, and overlapping genes; "
                    "Only mention pathway enrichment, Enrichr, KEGG, Reactome, GO terms, adjusted p-values, or overlapping pathway genes when the active arm is exactly `pathway`; "
                    "for `memory_rwr`, summarize only the RWR/network-prioritization result; do not mention pathway enrichment, Enrichr terms, KEGG/Reactome/GO terms, adjusted p-values, or pathway overlap genes even if those genes were used internally as seeds; "
                    "for `research_literature` and `literature`, always synthesize the literature summary, key points, and references into a polished final answer; do not return raw JSON, raw tool output, or unsynthesized citation lists; "
                    "for `disease`, summarize literature findings first, then relevant network or enrichment context if present. "
                    "For the general, literature, research-literature, or disease-style response, prefer this order when supported by the payload: `**Summary**`, `**Key Findings**`, `**Interpretation**`, and `**References**`. "
                    "The answer may be comprehensive; prioritize completeness, traceability, and scientific usefulness over brevity. "
                    "After completing an analysis, include exactly one short `**Suggested Follow-Up**` section with a concrete next analysis the agent can perform from the available state, such as pathway enrichment, RWR target prioritization, literature support, OpenTargets checks, L1000CDS2 drug matching, PubChem lookup, hypothesis generation, or a supported visualization. "
                    "Do not suggest unavailable capabilities. "
                    "When literature references are available, place `**Suggested Follow-Up**` after the `**References**` section; otherwise end with `**Suggested Follow-Up**`.",
                ),
                ("user", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
            ]
        )
        answer = _message_content_text(getattr(resp, "content", ""))
    except Exception:
        answer = ""
    return _ensure_suggested_followup(answer or _fallback_answer(payload), payload)
