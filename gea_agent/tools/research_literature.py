from __future__ import annotations

from typing import Any

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
    if not normalized_genes:
        normalized_genes = _normalize_genes(extract_genes_from_text(query, mode="strict"))

    resolved_disease = str(disease_name or "").strip()
    prompt_context: list[str] = []
    if resolved_disease:
        prompt_context.append(f"Disease context: {resolved_disease}")
    if normalized_genes:
        prompt_context.append("Genes mentioned or inferred: " + ", ".join(normalized_genes[:20]))
    if top_n:
        prompt_context.append(f"Requested depth hint: top_n={int(top_n)}")

    prompt = (
        "You are a biomedical literature-style research assistant. "
        "Answer the user's question directly using your own model knowledge in a ChatGPT-style research tone. "
        "Do not require retrieved evidence and do not claim to have searched or verified external sources unless explicit evidence is provided. "
        "If you provide references, they are model-generated best-effort references and should be presented as unverified. "
        "When uncertain, say so clearly. "
        "Return plain text only.\n\n"
        + ("\n".join(prompt_context) + "\n\n" if prompt_context else "")
        + f"User query: {query}"
    )

    response = get_llm().invoke([("user", prompt)])
    answer = str(getattr(response, "content", "") or "").strip()
    if not answer:
        answer = "I could not generate a research-style answer for that query."

    return {
        "status": "ok",
        "analysis_arm": "research_literature",
        "answer": answer,
        "message": "LLM-only research-style answer generated.",
        "disease_name": resolved_disease,
        "openalex_genes": normalized_genes,
        "openalex_papers": [],
        "ranked_openalex_papers": [],
        "literature_key_points": [],
        "literature_references": [],
        "literature_summary": answer,
        "literature_source_status": {"mode": "llm_only_unverified"},
        "literature_query": query,
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
            literature_source_status={"mode": "llm_only_unverified"},
            literature_summary="",
            should_finalize=True,
        )
