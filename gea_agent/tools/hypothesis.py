from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from gea_agent.tools.disease_literature import identify_disease_from_query
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


def _fallback_hypothesis_answer(
    *,
    hypothesis_goal: str,
    disease_name: str,
    genes: list[str],
) -> str:
    scope = hypothesis_goal or "the requested question"
    context_parts = [part for part in [disease_name, ", ".join(genes[:8])] if part]
    lines = [
        "**Hypothesis Goal**",
        scope,
    ]
    if context_parts:
        lines.extend(["", "**Context Used**", " | ".join(context_parts)])
    lines.extend(
        [
            "",
            "**Hypothesis 1: Candidate genes form a coordinated disease-associated regulatory module**",
            "- **Rationale:** Repeatedly surfaced genes across the current analysis context may reflect a shared biological process rather than isolated signals.",
            "- **Validation Approach:** Prioritize two to five candidate genes, perturb them in a disease-relevant cell or tissue model, and compare transcriptional and phenotype responses against matched controls.",
            "- **Expected Observation:** Perturbing true driver genes should shift both pathway markers and the disease-relevant phenotype in a coherent direction.",
            "- **Readouts:** Target-gene expression; pathway-marker expression; protein abundance where feasible; phenotype-specific functional endpoint.",
            "- **Controls And Caveats:** Include non-targeting and positive controls, verify perturbation efficiency, and interpret weak effects cautiously if the model does not capture the disease context.",
            "",
            "**Hypothesis 2: Directionally consistent genes reflect an upstream pathway state**",
            "- **Rationale:** Concordant up- or down-regulation can indicate a common regulator, pathway activation state, or cellular composition shift.",
            "- **Validation Approach:** Test whether modulating the suspected upstream pathway changes the candidate-gene signature and downstream phenotype.",
            "- **Expected Observation:** Pathway modulation should move the candidate-gene signature and phenotype together if the pathway is mechanistically relevant.",
            "- **Readouts:** Pathway activity markers; candidate-gene expression; quantitative phenotype measurements.",
            "- **Controls And Caveats:** Separate direct pathway effects from nonspecific stress responses and consider time-course sampling if directionality is unclear.",
        ]
    )
    return "\n".join(lines).strip()


def _format_list(values: list[Any]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return "; ".join(cleaned)


def _format_hypothesis_answer(
    *,
    hypothesis_goal: str,
    disease_name: str,
    genes: list[str],
    hypotheses: list[dict[str, Any]],
    overall_summary: str,
) -> str:
    lines: list[str] = []
    lines.append("**Experimental Hypotheses**")
    if hypothesis_goal:
        lines.extend(["", "**Hypothesis Goal**", hypothesis_goal])
    context_bits = [part for part in [disease_name, ", ".join(genes[:10])] if part]
    if context_bits:
        lines.extend(["", "**Context Used**", " | ".join(context_bits)])
    if overall_summary:
        lines.extend(["", "**Summary**", overall_summary])

    for index, hypothesis in enumerate(hypotheses, start=1):
        if not isinstance(hypothesis, dict):
            continue
        title = str(hypothesis.get("title") or f"Hypothesis {index}").strip()
        rationale = str(hypothesis.get("rationale") or "").strip()
        experiment = str(hypothesis.get("experiment_design") or "").strip()
        expected = str(hypothesis.get("expected_observation") or hypothesis.get("expected_pattern") or "").strip()
        readouts = hypothesis.get("readouts") if isinstance(hypothesis.get("readouts"), list) else []
        controls = hypothesis.get("controls") if isinstance(hypothesis.get("controls"), list) else []
        interpretation = str(hypothesis.get("interpretation") or "").strip()
        caveats = hypothesis.get("caveats") if isinstance(hypothesis.get("caveats"), list) else []
        assumptions = hypothesis.get("key_assumptions") if isinstance(hypothesis.get("key_assumptions"), list) else []

        lines.append("")
        lines.append(f"**Hypothesis {index}: {title}**")
        if rationale:
            lines.append(f"- **Rationale:** {rationale}")
        if experiment:
            lines.append(f"- **Validation Approach:** {experiment}")
        if expected:
            lines.append(f"- **Expected Observation:** {expected}")
        if readouts:
            lines.append(f"- **Readouts:** {_format_list(readouts)}")
        if controls:
            lines.append(f"- **Controls:** {_format_list(controls)}")
        if interpretation:
            lines.append(f"- **Interpretation:** {interpretation}")
        if assumptions:
            lines.append(f"- **Key Assumptions:** {_format_list(assumptions)}")
        if caveats:
            lines.append(f"- **Caveats:** {_format_list(caveats)}")

    return "\n".join(lines).strip()


def generate_experimental_hypotheses(
    *,
    user_query: str,
    validation_goal: str = "",
    hypothesis_goal: str = "",
    disease_name: str = "",
    genes: list[str] | None = None,
    conversation_messages: list[BaseMessage] | None = None,
    memory_state: dict[str, Any] | None = None,
    hypothesis_count: int = 3,
    include_references: bool = True,
) -> dict[str, Any]:
    query = str(user_query or "").strip()
    goal = str(hypothesis_goal or validation_goal or "").strip() or query
    if not query:
        return {
            "status": "not_found",
            "analysis_arm": "hypothesis",
            "answer": "No request was provided for hypothesis generation.",
            "message": "No request was provided for hypothesis generation.",
            "hypotheses": [],
            "literature_references": [],
            "should_finalize": True,
        }

    memory = _memory_snapshot(memory_state)
    history = _compact_history(_conversation_history(conversation_messages))
    resolved_disease = _resolve_disease_name(query, disease_name, memory_state)
    resolved_genes = _candidate_genes(query, genes, memory_state)

    llm_payload = {
        "user_query": query,
        "hypothesis_goal": goal,
        "disease_name": resolved_disease,
        "genes": resolved_genes,
        "conversation_history": history,
        "memory_state": memory,
        "requested_hypothesis_count": max(1, min(int(hypothesis_count or 3), 6)),
    }

    hypotheses: list[dict[str, Any]] = []
    llm_summary = ""
    try:
        response = get_llm().invoke(
            [
                (
                    "system",
                    "You are a biomedical hypothesis-generation assistant inside a gene expression analysis agent. "
                    "Generate plausible scientific hypotheses grounded only in the provided conversation history and stored memory. "
                    "When the user asks to suggest experiments or validation experiments, clearly propose experimental validation ideas. "
                    "Do not validate hypotheses against external sources, search or cite literature, assess novelty, or claim that a hypothesis is proven. "
                    "Keep experiment suggestions conceptual and analysis-oriented rather than a step-by-step wet-lab protocol. "
                    "Make the answer clear, detailed, and useful for planning follow-up biological validation. "
                    "Return strict JSON with keys `overall_summary` and `hypotheses`. "
                    "Each hypothesis must include: `title`, `rationale`, `experiment_design`, `expected_observation`, "
                    "`readouts` (array), `controls` (array), `interpretation`, `key_assumptions` (array), and `caveats` (array).",
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
            hypothesis_goal=goal,
            disease_name=resolved_disease,
            genes=resolved_genes,
        )
        return {
            "status": "ok",
            "analysis_arm": "hypothesis",
            "answer": answer,
            "message": "Fallback hypotheses generated.",
            "hypothesis_goal": goal,
            "validation_goal": goal,
            "disease_name": resolved_disease,
            "genes": resolved_genes,
            "hypotheses": [],
            "hypothesis_summary": llm_summary,
            "openalex_papers": [],
            "ranked_openalex_papers": [],
            "literature_key_points": [],
            "literature_references": [],
            "literature_summary": "",
            "literature_source_status": {"hypothesis": {"status": "not_requested", "reason": "hypothesis tool does not perform validation or literature retrieval"}},
            "should_finalize": True,
        }

    answer = _format_hypothesis_answer(
        hypothesis_goal=goal,
        disease_name=resolved_disease,
        genes=resolved_genes,
        hypotheses=hypotheses,
        overall_summary=llm_summary,
    )
    return {
        "status": "ok",
        "analysis_arm": "hypothesis",
        "answer": answer,
        "message": "Hypotheses generated.",
        "hypothesis_goal": goal,
        "validation_goal": goal,
        "disease_name": resolved_disease,
        "genes": resolved_genes,
        "hypotheses": hypotheses,
        "hypothesis_summary": llm_summary,
        "openalex_papers": [],
        "ranked_openalex_papers": [],
        "literature_key_points": [],
        "literature_references": [],
        "literature_summary": "",
        "literature_source_status": {"hypothesis": {"status": "not_requested", "reason": "hypothesis tool does not perform validation or literature retrieval"}},
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
