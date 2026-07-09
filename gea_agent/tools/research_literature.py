from __future__ import annotations

import json
from typing import Any

from gea_agent.tools.disease_literature import fetch_openalex_papers_and_genes, identify_disease_from_query
from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.llm import get_llm
from gea_agent.tools.result_utils import sanitize_exception_message, tool_error_result


def _normalize_genes(genes: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in genes or []:
        gene = str(value or "").strip().upper()
        if gene and gene not in normalized:
            normalized.append(gene)
    return normalized


def _fallback_no_evidence_answer(
    *,
    disease_name: str,
    genes: list[str],
    source_status: dict[str, Any],
) -> str:
    scope_parts = [part for part in [disease_name, ", ".join(genes[:8])] if part]
    scope = " / ".join(scope_parts) if scope_parts else "this query"
    source_bits: list[str] = []
    for source_name, status in source_status.items():
        if not isinstance(status, dict):
            continue
        label = str(status.get("status") or "unknown")
        count = status.get("count")
        source_bits.append(f"{source_name}: {label}" + (f" ({count})" if count is not None else ""))
    source_text = "; ".join(source_bits) if source_bits else "No literature sources were available."
    return (
        f"I could not find enough grounded literature evidence to answer confidently for {scope}. "
        f"Source status: {source_text}"
    )


def run_publication_research_assistant(
    user_query: str,
    *,
    disease_name: str = "",
    genes: list[str] | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    query = str(user_query or "").strip()
    if not query:
        return {
            "status": "not_found",
            "analysis_arm": "research_literature",
            "answer": "No literature query was provided.",
            "message": "No literature query was provided.",
            "literature_references": [],
            "literature_key_points": [],
            "literature_source_status": {},
            "literature_summary": "",
            "should_finalize": True,
        }

    normalized_genes = _normalize_genes(genes)
    resolved_disease = str(disease_name or "").strip()
    if not resolved_disease:
        disease_result = identify_disease_from_query(query)
        if isinstance(disease_result, dict):
            resolved_disease = str(disease_result.get("disease") or "").strip()

    if not normalized_genes:
        normalized_genes = _normalize_genes(extract_genes_from_text(query, mode="strict"))

    literature = fetch_openalex_papers_and_genes(
        resolved_disease,
        top_n=top_n,
        user_query=query,
        genes=normalized_genes,
    )
    papers = list(literature.get("papers") or [])
    references = list(literature.get("references") or [])
    key_points = list(literature.get("key_points") or [])
    summary = str(literature.get("literature_summary") or "").strip()
    source_status = literature.get("source_status") if isinstance(literature.get("source_status"), dict) else {}

    if not papers and not references and not key_points and not summary:
        answer = _fallback_no_evidence_answer(
            disease_name=resolved_disease,
            genes=normalized_genes,
            source_status=source_status,
        )
        return {
            "status": "not_found",
            "analysis_arm": "research_literature",
            "answer": answer,
            "message": answer,
            "disease_name": resolved_disease,
            "openalex_genes": list(literature.get("genes") or normalized_genes),
            "openalex_papers": papers,
            "ranked_openalex_papers": list(literature.get("ranked_papers") or []),
            "literature_key_points": key_points,
            "literature_references": references,
            "literature_summary": summary,
            "literature_source_status": source_status,
            "literature_query": str(literature.get("query") or query),
            "should_finalize": True,
        }

    answer = summary
    try:
        response = get_llm().invoke(
            [
                (
                    "system",
                    "You are a biomedical literature synthesis assistant. "
                    "Answer only from the provided grounded literature evidence. "
                    "Do not invent mechanisms, citations, or findings that are not present in the supplied evidence. "
                    "If the evidence is sparse or incomplete, say so clearly. "
                    "Return plain text only. "
                    "End with a `References:` section when references are available.",
                ),
                (
                    "user",
                    json.dumps(
                        {
                            "query": query,
                            "disease_name": resolved_disease,
                            "genes": normalized_genes,
                            "literature_summary": summary,
                            "key_points": key_points[:8],
                            "references": references[:8],
                            "ranked_papers": list(literature.get("ranked_papers") or [])[:6],
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        answer = str(getattr(response, "content", "") or "").strip() or summary
    except Exception:
        answer = summary

    if not answer:
        answer = _fallback_no_evidence_answer(
            disease_name=resolved_disease,
            genes=normalized_genes,
            source_status=source_status,
        )

    return {
        "status": "ok",
        "analysis_arm": "research_literature",
        "answer": answer,
        "message": "Grounded literature answer generated.",
        "disease_name": resolved_disease,
        "openalex_genes": list(literature.get("genes") or normalized_genes),
        "openalex_papers": papers,
        "ranked_openalex_papers": list(literature.get("ranked_papers") or []),
        "literature_key_points": key_points,
        "literature_references": references,
        "literature_summary": summary,
        "literature_source_status": source_status,
        "literature_query": str(literature.get("query") or query),
        "should_finalize": True,
    }


def run_publication_research_assistant_safe(
    user_query: str,
    *,
    disease_name: str = "",
    genes: list[str] | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    try:
        return run_publication_research_assistant(
            user_query,
            disease_name=disease_name,
            genes=genes,
            top_n=top_n,
        )
    except Exception as exc:
        return tool_error_result(
            "research_literature",
            f"Literature analysis failed: {sanitize_exception_message(exc)}",
            analysis_arm="research_literature",
            literature_references=[],
            literature_key_points=[],
            literature_source_status={},
            literature_summary="",
            should_finalize=True,
        )
