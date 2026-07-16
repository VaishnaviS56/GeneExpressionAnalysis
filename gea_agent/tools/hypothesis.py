from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from gea_agent.tools.disease_literature import fetch_openalex_papers_and_genes, identify_disease_from_query
from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.llm import get_llm, parse_json_object
from gea_agent.tools.result_utils import sanitize_exception_message, tool_error_result


def _normalize_genes(values: list[str] | None) -> list[str]:
    genes: list[str] = []
    for value in values or []:
        gene = str(value or "").strip().upper()
        if gene and gene not in genes:
            genes.append(gene)
    return genes


def _conversation_history(messages: list[BaseMessage] | None) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages or []:
        content = " ".join(str(getattr(message, "content", "") or "").split()).strip()
        if not content:
            continue
        role = "assistant"
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, ToolMessage):
            role = "tool"
        elif isinstance(message, AIMessage):
            role = "assistant"
        history.append({"role": role, "content": content})
    return history


def _compact_history(history: list[dict[str, str]], *, max_turns: int = 24) -> list[dict[str, str]]:
    trimmed = history[-max_turns:]
    return [
        {
            "role": row.get("role", ""),
            "content": str(row.get("content", ""))[:1200],
        }
        for row in trimmed
        if str(row.get("content", "")).strip()
    ]


def _memory_snapshot(memory_state: dict[str, Any] | None) -> dict[str, Any]:
    state = memory_state if isinstance(memory_state, dict) else {}
    return {
        "memory_control_name": str(state.get("memory_control_name") or ""),
        "memory_test_name": str(state.get("memory_test_name") or ""),
        "memory_disease_name": str(state.get("memory_disease_name") or ""),
        "memory_deg_genes": list(state.get("memory_deg_genes") or [])[:200],
        "memory_upregulated_genes": list(state.get("memory_upregulated_genes") or [])[:200],
        "memory_downregulated_genes": list(state.get("memory_downregulated_genes") or [])[:200],
        "memory_rwr_seed_genes": list(state.get("memory_rwr_seed_genes") or [])[:100],
        "memory_rwr_genes": list(state.get("memory_rwr_genes") or [])[:100],
        "memory_openalex_genes": list(state.get("memory_openalex_genes") or [])[:100],
        "memory_enrichr": state.get("memory_enrichr") if isinstance(state.get("memory_enrichr"), dict) else {},
        "memory_lookup_result": state.get("memory_lookup_result") if isinstance(state.get("memory_lookup_result"), dict) else {},
        "memory_slice_result": state.get("memory_slice_result") if isinstance(state.get("memory_slice_result"), dict) else {},
        "memory_l1000cds2_result": state.get("memory_l1000cds2_result") if isinstance(state.get("memory_l1000cds2_result"), dict) else {},
        "memory_pubchem_result": state.get("memory_pubchem_result") if isinstance(state.get("memory_pubchem_result"), dict) else {},
    }


def _candidate_genes(user_query: str, explicit_genes: list[str] | None, memory_state: dict[str, Any] | None) -> list[str]:
    genes = _normalize_genes(explicit_genes)
    state = memory_state if isinstance(memory_state, dict) else {}
    memory_candidates = [
        *(state.get("memory_deg_genes") or []),
        *(state.get("memory_upregulated_genes") or []),
        *(state.get("memory_downregulated_genes") or []),
        *(state.get("memory_openalex_genes") or []),
        *(state.get("memory_rwr_seed_genes") or []),
    ]
    genes.extend(_normalize_genes(memory_candidates))
    genes.extend(_normalize_genes(extract_genes_from_text(user_query, mode="strict")))

    deduped: list[str] = []
    for gene in genes:
        if gene and gene not in deduped:
            deduped.append(gene)
    return deduped[:40]


def _resolve_disease_name(user_query: str, disease_name: str, memory_state: dict[str, Any] | None) -> str:
    resolved = str(disease_name or "").strip()
    if resolved:
        return resolved
    state = memory_state if isinstance(memory_state, dict) else {}
    remembered = str(state.get("memory_disease_name") or "").strip()
    if remembered:
        return remembered
    result = identify_disease_from_query(user_query)
    if isinstance(result, dict):
        return str(result.get("disease") or "").strip()
    return ""


def _format_reference_line(index: int, reference: dict[str, Any]) -> str:
    title = str(reference.get("title") or "Untitled reference").strip()
    year = str(reference.get("year") or "").strip()
    source = str(reference.get("source") or reference.get("journal") or "").strip()
    doi = str(reference.get("doi") or "").strip()
    url = str(reference.get("url") or "").strip()
    parts = [f"[{index}] {title}"]
    tail = ", ".join(part for part in [year, source] if part)
    if tail:
        parts.append(f"({tail})")
    if doi:
        parts.append(f"DOI: {doi}")
    elif url:
        parts.append(url)
    return " ".join(parts)


def _fallback_hypothesis_answer(
    *,
    validation_goal: str,
    disease_name: str,
    genes: list[str],
    references: list[dict[str, Any]],
) -> str:
    scope = validation_goal or "the requested validation"
    context_parts = [part for part in [disease_name, ", ".join(genes[:8])] if part]
    lines = [
        f"Here are LLM-generated experimental hypotheses for {scope}.",
    ]
    if context_parts:
        lines.append(f"Context used: {' | '.join(context_parts)}.")
    lines.extend(
        [
            "",
            "1. Perturb the top candidate gene set in the most disease-relevant model and test whether the user-described phenotype shifts in the expected direction.",
            "Rationale: this is the most direct validation path when prior DEG or disease-context genes are already available.",
            "Suggested readouts: qPCR or RNA-seq, protein-level confirmation, and phenotype-specific functional assays.",
            "",
            "2. Compare loss-of-function and gain-of-function perturbations to distinguish correlation from causal directionality.",
            "Rationale: reciprocal perturbation helps clarify whether the observed signature is mechanistically upstream or downstream.",
            "Suggested readouts: rescue experiments, pathway markers, and quantitative phenotype measurements.",
        ]
    )
    if references:
        lines.append("")
        lines.append("Possible related references:")
        for index, reference in enumerate(references[:5], start=1):
            if isinstance(reference, dict):
                lines.append(_format_reference_line(index, reference))
    return "\n".join(lines).strip()


def _format_hypothesis_answer(
    *,
    validation_goal: str,
    disease_name: str,
    genes: list[str],
    hypotheses: list[dict[str, Any]],
    references: list[dict[str, Any]],
    literature_summary: str,
) -> str:
    lines: list[str] = []
    lines.append("## Experimental Hypotheses")
    if validation_goal:
        lines.append(f"Validation goal: {validation_goal}")
    context_bits = [part for part in [disease_name, ", ".join(genes[:10])] if part]
    if context_bits:
        lines.append(f"Context used: {' | '.join(context_bits)}")
    if literature_summary:
        lines.append("")
        lines.append("### Literature context")
        lines.append(literature_summary)

    for index, hypothesis in enumerate(hypotheses, start=1):
        if not isinstance(hypothesis, dict):
            continue
        title = str(hypothesis.get("title") or f"Hypothesis {index}").strip()
        rationale = str(hypothesis.get("rationale") or "").strip()
        experiment = str(hypothesis.get("experiment_design") or "").strip()
        expected = str(hypothesis.get("expected_observation") or "").strip()
        readouts = hypothesis.get("readouts") if isinstance(hypothesis.get("readouts"), list) else []
        evidence_state = str(hypothesis.get("existing_evidence") or "").strip()
        novelty = str(hypothesis.get("novelty_assessment") or "").strip()
        ref_ids = [int(value) for value in hypothesis.get("supporting_reference_ids", []) if str(value).isdigit()]

        lines.append("")
        lines.append(f"### {index}. {title}")
        if rationale:
            lines.append(f"Rationale: {rationale}")
        if experiment:
            lines.append(f"Experiment: {experiment}")
        if expected:
            lines.append(f"Expected result: {expected}")
        if readouts:
            lines.append("Readouts: " + ", ".join(str(value).strip() for value in readouts if str(value).strip()))
        if evidence_state:
            lines.append(f"Prior evidence: {evidence_state}")
        if novelty:
            lines.append(f"Novelty: {novelty}")
        if ref_ids:
            lines.append("Supporting references: " + ", ".join(f"[{value}]" for value in ref_ids))

    if references:
        lines.append("")
        lines.append("## References")
        for index, reference in enumerate(references, start=1):
            if isinstance(reference, dict):
                lines.append(_format_reference_line(index, reference))

    return "\n".join(lines).strip()


def generate_experimental_hypotheses(
    *,
    user_query: str,
    validation_goal: str = "",
    disease_name: str = "",
    genes: list[str] | None = None,
    conversation_messages: list[BaseMessage] | None = None,
    memory_state: dict[str, Any] | None = None,
    hypothesis_count: int = 3,
    include_references: bool = True,
) -> dict[str, Any]:
    query = str(user_query or "").strip()
    goal = str(validation_goal or "").strip() or query
    if not query:
        return {
            "status": "not_found",
            "analysis_arm": "hypothesis",
            "answer": "No validation request was provided for hypothesis generation.",
            "message": "No validation request was provided for hypothesis generation.",
            "hypotheses": [],
            "literature_references": [],
            "should_finalize": True,
        }

    memory = _memory_snapshot(memory_state)
    history = _compact_history(_conversation_history(conversation_messages))
    resolved_disease = _resolve_disease_name(query, disease_name, memory_state)
    resolved_genes = _candidate_genes(query, genes, memory_state)

    literature: dict[str, Any] = {
        "status": "not_requested",
        "papers": [],
        "ranked_papers": [],
        "references": [],
        "key_points": [],
        "literature_summary": "",
        "source_status": {},
        "genes": resolved_genes,
    }
    if include_references:
        literature = fetch_openalex_papers_and_genes(
            resolved_disease,
            top_n=max(8, min(20, hypothesis_count * 4)),
            user_query=query,
            genes=resolved_genes,
        )

    references = list(literature.get("references") or [])
    ranked_papers = list(literature.get("ranked_papers") or [])
    key_points = list(literature.get("key_points") or [])
    literature_summary = str(literature.get("literature_summary") or "").strip()
    source_status = literature.get("source_status") if isinstance(literature.get("source_status"), dict) else {}

    reference_bundle = []
    for index, reference in enumerate(references[:8], start=1):
        if isinstance(reference, dict):
            reference_bundle.append(
                {
                    "id": index,
                    "title": reference.get("title"),
                    "year": reference.get("year"),
                    "source": reference.get("source") or reference.get("journal"),
                    "doi": reference.get("doi"),
                    "url": reference.get("url"),
                    "note": reference.get("note") or reference.get("summary"),
                }
            )

    llm_payload = {
        "user_query": query,
        "validation_goal": goal,
        "disease_name": resolved_disease,
        "genes": resolved_genes,
        "conversation_history": history,
        "memory_state": memory,
        "literature_summary": literature_summary,
        "literature_key_points": key_points[:8],
        "references": reference_bundle,
        "requested_hypothesis_count": max(1, min(int(hypothesis_count or 3), 6)),
    }

    hypotheses: list[dict[str, Any]] = []
    llm_summary = ""
    try:
        response = get_llm().invoke(
            [
                (
                    "system",
                    "You are a biomedical experimental design assistant inside a gene expression analysis agent. "
                    "Generate plausible scientific validation hypotheses grounded in the provided conversation history, "
                    "stored memory, and any supplied literature evidence. "
                    "Do not invent citations. If prior evidence suggests a similar experiment has already been performed, "
                    "say so explicitly in `existing_evidence` and cite only from the supplied numbered references. "
                    "Return strict JSON with keys `overall_summary` and `hypotheses`. "
                    "Each hypothesis must include: `title`, `rationale`, `experiment_design`, `expected_observation`, "
                    "`readouts` (array), `novelty_assessment`, `existing_evidence`, and `supporting_reference_ids` (array of integers).",
                ),
                ("user", json.dumps(llm_payload, ensure_ascii=False)),
            ]
        )
        parsed = parse_json_object(getattr(response, "content", "") or "")
        if isinstance(parsed.get("hypotheses"), list):
            hypotheses = [row for row in parsed.get("hypotheses", []) if isinstance(row, dict)]
        llm_summary = str(parsed.get("overall_summary") or "").strip()
    except Exception:
        hypotheses = []

    if not hypotheses:
        answer = _fallback_hypothesis_answer(
            validation_goal=goal,
            disease_name=resolved_disease,
            genes=resolved_genes,
            references=references,
        )
        return {
            "status": "ok",
            "analysis_arm": "hypothesis",
            "answer": answer,
            "message": "Fallback hypotheses generated.",
            "validation_goal": goal,
            "disease_name": resolved_disease,
            "genes": resolved_genes,
            "hypotheses": [],
            "hypothesis_summary": llm_summary,
            "openalex_papers": list(literature.get("papers") or []),
            "ranked_openalex_papers": ranked_papers,
            "literature_key_points": key_points,
            "literature_references": references,
            "literature_summary": literature_summary,
            "literature_source_status": source_status,
            "should_finalize": True,
        }

    answer = _format_hypothesis_answer(
        validation_goal=goal,
        disease_name=resolved_disease,
        genes=resolved_genes,
        hypotheses=hypotheses,
        references=references[:8],
        literature_summary=llm_summary or literature_summary,
    )
    return {
        "status": "ok",
        "analysis_arm": "hypothesis",
        "answer": answer,
        "message": "Experimental hypotheses generated.",
        "validation_goal": goal,
        "disease_name": resolved_disease,
        "genes": resolved_genes,
        "hypotheses": hypotheses,
        "hypothesis_summary": llm_summary,
        "openalex_papers": list(literature.get("papers") or []),
        "ranked_openalex_papers": ranked_papers,
        "literature_key_points": key_points,
        "literature_references": references,
        "literature_summary": literature_summary,
        "literature_source_status": source_status,
        "should_finalize": True,
    }


def generate_experimental_hypotheses_safe(**kwargs: Any) -> dict[str, Any]:
    try:
        return generate_experimental_hypotheses(**kwargs)
    except Exception as exc:
        return tool_error_result(
            "hypothesis",
            f"Hypothesis generation failed: {sanitize_exception_message(exc)}",
            analysis_arm="hypothesis",
            hypotheses=[],
            literature_references=[],
            literature_key_points=[],
            literature_source_status={},
            literature_summary="",
            should_finalize=True,
        )
