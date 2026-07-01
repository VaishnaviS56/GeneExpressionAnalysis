from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import networkx as nx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from gea_agent.agent.state import AgentState
from gea_agent.config import SETTINGS
from gea_agent.tools.disease_literature import fetch_openalex_papers_and_genes, identify_disease_from_query
from gea_agent.tools.deg_analysis import run_deg_r_analysis
from gea_agent.tools.enrichr import enrichr_pathways
from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.llm import get_llm
from gea_agent.tools.opentargets import (
    check_gene_disease_association,
    check_gene_list_disease_associations,
    find_diseases_for_gene,
    find_drugs_for_gene,
)
from gea_agent.tools.random_walk_restart import top_rwr_genes
from gea_agent.tools.string_local_graph import build_weighted_graph_from_string_files
from gea_agent.tools.synthesizer import synthesize_technical_response
from gea_agent.tools.primekg import query_primekg
from gea_agent.tools.srp_ids import extract_srp_ids_from_text
from gea_agent.tools.visualizers import (
    build_kegg_pathway_visualization,
    build_network_visualization,
    build_volcano_plot,
)


MAX_AGENT_STEPS = 10

TOOL_USE_INSTRUCTIONS = '''
Tool selection guide:
1. deg_analysis: Use when the request includes SRP accessions or asks for differential expression between cohorts. Extract and pass `srp_ids`, `control_name`, and `test_name` whenever possible, and reuse stored cohort labels on follow-up turns.
2. pathway: Use as the default enrichment specialist for pathway and GO-term questions. Prefer stored DEG genes when the request refers to up-regulated, down-regulated, or previously identified DEGs.
3. rwr_analysis: Use when the task is to prioritize candidate genes or targets from a gene set. The input genes may come from the user, literature extraction, memory lookup, or stored DEG results.
4. literature: Use when the disease context, disease-linked genes, or literature-backed mechanisms need to be established from papers. This tool is a strong precursor when downstream pathway or RWR analysis needs genes that are not already available.
5. identify_disease_from_query: Use when the next step depends on a disease label and the query implies one but does not state it in a clean structured way.
6. primekg_query: Use for local biomedical graph questions, especially multi-hop relationship discovery across genes, drugs, diseases, phenotypes, and pathways.
7. opentargets_association: Use for gene-to-disease association evidence, disease checks for a gene or gene set, and drug associations tied to a gene.
8. visualize: Use only when the user explicitly asks for a visual output such as a network, KEGG pathway image, or volcano plot.
9. memory_lookup: Use for follow-up questions that can be answered from stored pathway overlap genes, GO terms, DEG sets, or simple intersections/membership checks.

Operational guidance:
- If the user asks a simple non-technical question or casual follow-up, answer directly without tools.
- Call at most one specialist at a time. After each tool result, reassess whether the user is already answered or whether one additional tool is necessary.
- Prefer recovering missing prerequisites with tools instead of guessing. Example: use `literature` to get disease genes before `pathway` or `rwr_analysis`.
- Prefer memory and current state before recomputing the same result.
- Use `pathway` first for enrichment questions unless the user explicitly asks for knowledge-graph pathway relationships.
- Use `primekg_query` first for "what connects", "what links", mediator, multi-hop, and graph-neighborhood questions.
- Use `opentargets_association` when the main task is evidence-backed association rather than broader graph exploration.
- For stored DEG follow-ups, interpret positive `log2FoldChange` as up-regulated and negative `log2FoldChange` as down-regulated.
- If a specialist is used, let the workflow continue through the specialist/final synthesis path rather than answering from partial assumptions.
'''


def _trace_tool_call(name: str) -> None:
    print(f"[tool] {name}")


def _compact_text(value: Any, *, limit: int = 240) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _merge_unique(*groups: list[str] | tuple[str, ...] | None) -> list[str]:
    merged: list[str] = []
    for group in groups:
        if not group:
            continue
        for value in group:
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _genes_from_deg_records(records: Any, *, top_n: int | None = None) -> list[str]:
    if not isinstance(records, list):
        return []

    ranked: list[tuple[str, float]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        gene = str(row.get("gene") or "").strip().upper()
        if not gene:
            continue
        try:
            log2fc = float(row.get("log2FoldChange"))
        except Exception:
            log2fc = 0.0
        ranked.append((gene, log2fc))

    ranked.sort(key=lambda item: item[1], reverse=True)
    genes = [gene for gene, _ in ranked]
    if top_n is not None:
        return genes[: max(0, top_n)]
    return genes


def _genes_from_deg_records_by_direction(
    records: Any,
    *,
    direction: str = "all",
    top_n: int | None = None,
) -> list[str]:
    if not isinstance(records, list):
        return []

    ranked: list[tuple[str, float, float]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        gene = str(row.get("gene") or "").strip().upper()
        if not gene:
            continue
        try:
            log2fc = float(row.get("log2FoldChange"))
        except Exception:
            log2fc = 0.0
        if direction == "up" and log2fc <= 0:
            continue
        if direction == "down" and log2fc >= 0:
            continue
        ranked.append((gene, log2fc, abs(log2fc)))

    if direction == "down":
        ranked.sort(key=lambda item: item[1])
    elif direction in {"all", "both"}:
        ranked.sort(key=lambda item: item[2], reverse=True)
    else:
        ranked.sort(key=lambda item: item[1], reverse=True)

    genes = [gene for gene, _, _ in ranked]
    if top_n is not None:
        return genes[: max(0, top_n)]
    return genes


def _deg_direction_from_query(text: str | None) -> str:
    query = str(text or "").lower()
    has_down = "down-regulated" in query or "down regulated" in query or "downregulated" in query
    has_up = "up-regulated" in query or "up regulated" in query or "upregulated" in query
    if (
        "both up and down" in query
        or "both up-regulated and down-regulated" in query
        or "both upregulated and downregulated" in query
        or "both up regulated and down regulated" in query
        or ("both" in query and has_up and has_down)
    ):
        return "both"
    if has_down:
        return "down"
    if has_up:
        return "up"
    return "all"


def _memory_gene_query_requested(text: str | None) -> bool:
    query = str(text or "").lower()
    return any(
        marker in query
        for marker in (
            "up-regulated genes",
            "up regulated genes",
            "upregulated genes",
            "down-regulated genes",
            "down regulated genes",
            "downregulated genes",
            "deg genes",
            "differentially expressed genes",
        )
    )


def _memory_lookup_gene_candidates(state: AgentState) -> list[str]:
    result = state.get("memory_lookup_result")
    if not isinstance(result, dict):
        return []

    genes = _merge_unique(
        result.get("intersection_genes") if isinstance(result.get("intersection_genes"), list) else [],
        result.get("deg_genes") if isinstance(result.get("deg_genes"), list) else [],
        (
            result.get("selected_term", {}).get("overlapping_genes")
            if isinstance(result.get("selected_term"), dict)
            and isinstance(result.get("selected_term", {}).get("overlapping_genes"), list)
            else []
        ),
        result.get("mentioned_genes") if isinstance(result.get("mentioned_genes"), list) else [],
    )
    return [str(value).strip().upper() for value in genes if str(value).strip()]


def _resolve_rwr_source_genes(
    state: AgentState,
    args: dict[str, Any],
    *,
    prefer_seed_genes: bool = False,
) -> tuple[list[str], str]:
    query = str(args.get("text") or state.get("query") or "")
    direct_key = "seed_genes" if prefer_seed_genes else "genes"
    direct_genes = args.get(direct_key)
    if isinstance(direct_genes, list) and direct_genes:
        return [str(value).strip().upper() for value in direct_genes if str(value).strip()], "tool_args"

    memory_lookup_genes = _memory_lookup_gene_candidates(state)
    if memory_lookup_genes:
        return memory_lookup_genes, "memory_lookup"

    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    deg_records = (
        state.get("memory_deg_gene_records")
        if analysis_arm == "memory_rwr"
        else (state.get("deg_gene_records") or state.get("memory_deg_gene_records"))
    )
    directional_deg_genes = _genes_from_deg_records_by_direction(
        deg_records,
        direction=direction,
        top_n=top_n if isinstance(top_n, int) and top_n > 0 else (20 if prefer_seed_genes else None),
    )
    if directional_deg_genes:
        return directional_deg_genes, "stored_deg_directional"

    if not _memory_gene_query_requested(query):
        query_genes = [
            str(value).strip().upper()
            for value in extract_genes_from_text(query, mode="strict")
            if str(value).strip()
        ]
        if query_genes:
            return list(dict.fromkeys(query_genes)), "query_genes"

    if analysis_arm == "memory_rwr":
        genes = _merge_unique(
            _genes_from_deg_records(state.get("memory_deg_gene_records"), top_n=20 if prefer_seed_genes else None),
            state.get("memory_deg_genes"),
            state.get("memory_rwr_seed_genes") if prefer_seed_genes else [],
            state.get("genes"),
        )
    else:
        genes = _merge_unique(
            _genes_from_deg_records(state.get("deg_gene_records") or state.get("memory_deg_gene_records"), top_n=20 if prefer_seed_genes else None),
            state.get("deg_genes"),
            state.get("memory_deg_genes"),
            state.get("openalex_genes"),
            state.get("memory_openalex_genes"),
            state.get("genes"),
        )
    return [str(value).strip().upper() for value in genes if str(value).strip()], "state_fallback"


def _disease_from_association_query(text: str | None) -> str:
    query = str(text or "").strip()
    patterns = (
        r"\bassociated\s+with\s+(.+?)(?:\?|$)",
        r"\bassociation\s+with\s+(.+?)(?:\?|$)",
        r"\blinked\s+to\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            disease = re.sub(r"\b(the|a|an)\b", "", match.group(1), flags=re.IGNORECASE)
            disease = " ".join(disease.split()).strip(" .,:;")
            if disease:
                return disease
    return ""


def _drug_association_query_requested(text: str | None) -> bool:
    query = str(text or "").lower()
    return any(
        marker in query
        for marker in (
            "drug associated with",
            "drugs associated with",
            "drug association for",
            "drugs for",
            "drug for gene",
            "drugs for gene",
            "what drugs",
            "which drugs",
            "target this gene with drugs",
        )
    )


def _primekg_target_types_from_query(text: str | None) -> list[str]:
    query = str(text or "").lower()
    types: list[str] = []
    explicit_kg = "primekg" in query or "kg" in query or "knowledge graph" in query
    if "drug" in query or "drugs" in query or "compound" in query or "treatment" in query:
        types.append("drug")
    if "disease" in query or "diseases" in query or "diabetes" in query or "cancer" in query:
        types.append("disease")
    if "phenotype" in query or "phenotypes" in query or "symptom" in query or "side effect" in query:
        types.append("effect/phenotype")
    if explicit_kg and ("pathway" in query or "pathways" in query):
        types.append("pathway")
    if "gene" in query or "genes" in query or "protein" in query or "proteins" in query:
        types.append("gene/protein")
    return types


def _parse_top_n_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\btop\s+(\d+)\b", str(text), re.IGNORECASE)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except Exception:
        return None
    return value if value > 0 else None


def _normalize_text_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _find_enrichr_term_from_state(
    state: AgentState,
    pathway_term: str | None,
    *,
    query: str = "",
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    enrichr = state.get("enrichr")
    if not isinstance(enrichr, dict) or not isinstance(enrichr.get("libraries"), dict):
        enrichr = state.get("memory_enrichr")
    if not isinstance(enrichr, dict):
        return None, None, None

    libraries = enrichr.get("libraries")
    if not isinstance(libraries, dict):
        return None, None, None

    desired = _normalize_text_token(pathway_term or "")
    query_norm = _normalize_text_token(query)
    best_match: tuple[dict[str, Any], str, int, int] | None = None

    for library_name, terms in libraries.items():
        if not isinstance(terms, list):
            continue
        for index, term in enumerate(terms, start=1):
            if not isinstance(term, dict):
                continue
            label = str(term.get("term") or "").strip()
            if not label:
                continue
            label_norm = _normalize_text_token(label)
            score = 0
            if desired:
                if label_norm == desired:
                    score = 1000
                elif desired in label_norm:
                    score = 800
            elif label_norm and label_norm in query_norm:
                score = 600 + len(label_norm)
            elif query_norm and all(token in query_norm for token in label_norm.split()[:3] if token):
                score = 300 + len(label_norm)

            if score <= 0:
                continue
            if best_match is None or score > best_match[3]:
                best_match = (term, str(library_name), index, score)

    if not best_match:
        return None, None, None
    return best_match[0], best_match[1], best_match[2]


def _run_memory_lookup(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("text") or state.get("query") or "")
    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    pathway_term = str(args.get("pathway_term") or "").strip()
    selected_term, selected_library, selected_rank = _find_enrichr_term_from_state(
        state,
        pathway_term,
        query=query,
    )
    pathway_genes = []
    if isinstance(selected_term, dict) and isinstance(selected_term.get("overlapping_genes"), list):
        pathway_genes = [
            str(g).strip().upper()
            for g in selected_term.get("overlapping_genes", [])
            if str(g).strip()
        ]

    deg_records = state.get("deg_gene_records") or state.get("memory_deg_gene_records")
    deg_genes = _genes_from_deg_records_by_direction(
        deg_records,
        direction=direction,
        top_n=top_n if isinstance(top_n, int) and top_n > 0 else None,
    )
    if not deg_genes:
        deg_genes = [
            str(g).strip().upper()
            for g in (state.get("deg_genes") or state.get("memory_deg_genes") or [])
            if str(g).strip()
        ]
        if isinstance(top_n, int) and top_n > 0:
            deg_genes = deg_genes[:top_n]

    mentioned_genes = [
        str(g).strip().upper()
        for g in extract_genes_from_text(query, mode="strict")
        if str(g).strip()
    ]
    mentioned_genes = list(dict.fromkeys(mentioned_genes))

    pathway_gene_set = set(pathway_genes)
    deg_gene_set = set(deg_genes)
    intersection = sorted(pathway_gene_set.intersection(deg_gene_set))

    gene_membership = []
    for gene in mentioned_genes:
        gene_membership.append(
            {
                "gene": gene,
                "in_pathway_or_go_term": gene in pathway_gene_set if pathway_gene_set else None,
                "in_selected_deg_set": gene in deg_gene_set if deg_gene_set else None,
            }
        )

    payload = {
        "query": query,
        "direction": direction,
        "top_n": top_n,
        "selected_term": {
            "library": selected_library,
            "term": selected_term.get("term") if isinstance(selected_term, dict) else "",
            "rank": selected_rank,
            "overlapping_genes": pathway_genes[:100],
        } if isinstance(selected_term, dict) else None,
        "pathway_gene_count": len(pathway_genes),
        "deg_gene_count": len(deg_genes),
        "deg_genes": deg_genes[:200],
        "intersection_count": len(intersection),
        "intersection_genes": intersection[:200],
        "mentioned_genes": mentioned_genes,
        "gene_membership": gene_membership,
    }

    llm = get_llm()
    response = llm.invoke(
        [
            (
                "system",
                "You are the stored-state lookup specialist inside a biomedical agent workflow. "
                "Use only the structured payload provided. "
                "If a pathway or GO term is present, treat its overlapping genes as the active term gene set. "
                "If the user asks for the overlapping genes of a stored pathway or GO term, return that overlapping gene list directly and name the matched term. "
                "If the user asks for an intersection, report the intersecting genes clearly. "
                "If the user asks whether a named gene is present, answer yes or no and state which set it belongs to. "
                "If required data is missing, say that clearly instead of guessing. "
                "Keep the answer concise and direct.",
            ),
            ("user", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        ]
    )
    return {
        "status": "ok",
        "analysis_arm": "memory_lookup",
        "direction": direction,
        "top_n": top_n,
        "selected_term": payload.get("selected_term"),
        "pathway_genes": pathway_genes,
        "deg_genes": deg_genes,
        "intersection_genes": intersection,
        "mentioned_genes": mentioned_genes,
        "gene_membership": gene_membership,
        "answer": getattr(response, "content", ""),
    }


def _graph_summary(graph: nx.Graph | None) -> dict[str, Any]:
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "top_degree": []}
    degrees = sorted(graph.degree(), key=lambda item: item[1], reverse=True)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "top_degree": [{"gene": gene, "degree": int(degree)} for gene, degree in degrees[:10]],
    }


def _latest_ai_message(messages: list[BaseMessage] | None) -> AIMessage | None:
    if not messages:
        return None
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _latest_tool_call(state: AgentState, tool_name: str) -> dict[str, Any] | None:
    ai_message = _latest_ai_message(list(state.get("messages") or []))
    if not ai_message or not getattr(ai_message, "tool_calls", None):
        return None

    for call in ai_message.tool_calls:
        if str(call.get("name", "")).strip() == tool_name:
            return call

    if len(ai_message.tool_calls) == 1:
        return ai_message.tool_calls[0]
    return None


def _serialize_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    for key in ("status", "message", "disease", "disease_name", "gene", "answer"):
        value = result.get(key)
        if value not in (None, ""):
            payload[key] = value

    if isinstance(result.get("genes"), list):
        payload["genes"] = result["genes"][:50]
    if isinstance(result.get("srp_ids"), list):
        payload["srp_ids"] = result["srp_ids"][:20]
    for key in ("control_name", "test_name"):
        if result.get(key):
            payload[key] = result.get(key)
    if isinstance(result.get("deg_genes"), list):
        payload["deg_genes"] = result["deg_genes"][:50]
    if isinstance(result.get("openalex_genes"), list):
        payload["openalex_genes"] = result["openalex_genes"][:50]
    if isinstance(result.get("rwr_genes"), list):
        payload["rwr_genes"] = [
            {"gene": gene, "score": round(float(score), 4)} for gene, score in result["rwr_genes"][:20]
        ]
    if isinstance(result.get("graph"), nx.Graph):
        payload["graph"] = _graph_summary(result["graph"])
    if isinstance(result.get("openalex_papers"), list):
        payload["openalex_papers"] = [
            {"title": paper.get("title"), "year": paper.get("year")}
            for paper in result["openalex_papers"][:5]
            if isinstance(paper, dict)
        ]
    if isinstance(result.get("ranked_openalex_papers"), list):
        payload["ranked_openalex_papers"] = [
            {
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "relevance": paper.get("relevance"),
                "reason": paper.get("reason"),
            }
            for paper in result["ranked_openalex_papers"][:5]
            if isinstance(paper, dict)
        ]
    if isinstance(result.get("literature_key_points"), list):
        payload["literature_key_points"] = [
            {
                "point": row.get("point"),
                "paper_ids": row.get("paper_ids"),
            }
            for row in result["literature_key_points"][:5]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("literature_references"), list):
        payload["literature_references"] = [
            {
                "paper_id": row.get("paper_id"),
                "title": row.get("title"),
                "year": row.get("year"),
                "doi": row.get("doi"),
            }
            for row in result["literature_references"][:8]
            if isinstance(row, dict)
        ]
    if result.get("literature_summary"):
        payload["literature_summary"] = _compact_text(result.get("literature_summary"), limit=1000)
    if result.get("associated") is not None:
        payload["associated"] = bool(result.get("associated"))
    if result.get("association_score") is not None:
        payload["association_score"] = result.get("association_score")
    for key in ("gene_set_source", "direction", "top_n"):
        if result.get(key) not in (None, ""):
            payload[key] = result.get(key)
    if isinstance(result.get("top_diseases"), list):
        payload["top_diseases"] = [
            {"name": disease.get("name"), "score": disease.get("score")}
            for disease in result["top_diseases"][:10]
            if isinstance(disease, dict)
        ]
    if isinstance(result.get("top_drugs"), list):
        payload["top_drugs"] = [
            {
                "name": row.get("name"),
                "phase": row.get("phase"),
                "status": row.get("status"),
                "disease_name": row.get("disease_name"),
            }
            for row in result["top_drugs"][:10]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("results"), list):
        payload["results"] = [
            {
                "gene": row.get("gene"),
                "ensembl_id": row.get("ensembl_id"),
                "associated": row.get("associated"),
                "association_score": row.get("association_score"),
            }
            for row in result["results"][:20]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("intersection_genes"), list):
        payload["intersection_genes"] = result["intersection_genes"][:100]
    if isinstance(result.get("mentioned_genes"), list):
        payload["mentioned_genes"] = result["mentioned_genes"][:20]
    if isinstance(result.get("gene_membership"), list):
        payload["gene_membership"] = result["gene_membership"][:20]
    if isinstance(result.get("selected_term"), dict):
        payload["selected_term"] = {
            "library": result["selected_term"].get("library"),
            "term": result["selected_term"].get("term"),
            "rank": result["selected_term"].get("rank"),
        }
    if isinstance(result.get("edges"), list):
        payload["edges"] = [
            {
                "relation": edge.get("display_relation") or edge.get("relation"),
                "source": (edge.get("source") or {}).get("name") if isinstance(edge.get("source"), dict) else "",
                "source_type": (edge.get("source") or {}).get("type") if isinstance(edge.get("source"), dict) else "",
                "target": (edge.get("target") or {}).get("name") if isinstance(edge.get("target"), dict) else "",
                "target_type": (edge.get("target") or {}).get("type") if isinstance(edge.get("target"), dict) else "",
            }
            for edge in result["edges"][:25]
            if isinstance(edge, dict)
        ]
        payload["count"] = result.get("count", len(result.get("edges") or []))
    if isinstance(result.get("deg_analysis"), dict):
        deg_analysis = result["deg_analysis"]
        payload["deg_analysis"] = {
            "status": deg_analysis.get("status"),
            "genes": deg_analysis.get("genes", [])[:20] if isinstance(deg_analysis.get("genes"), list) else [],
            "rows": len(deg_analysis.get("rows", [])) if isinstance(deg_analysis.get("rows"), list) else 0,
        }
    if isinstance(result.get("deg_gene_records"), list):
        payload["deg_gene_records"] = [
            {
                "gene": row.get("gene"),
                "pvalue": row.get("pvalue"),
                "pdj": row.get("pdj"),
                "log2FoldChange": row.get("log2FoldChange"),
            }
            for row in result["deg_gene_records"][:20]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("enrichr"), dict):
        libs = result["enrichr"].get("libraries")
        if isinstance(libs, dict):
            payload["enrichr"] = {
                lib: [
                    {
                        "term": term.get("term"),
                        "p_value": term.get("p_value"),
                        "adjusted_p_value": term.get("adjusted_p_value"),
                        "combined_score": term.get("combined_score"),
                        "overlapping_genes": term.get("overlapping_genes"),
                    }
                    for term in terms[:3]
                    if isinstance(term, dict)
                ]
                for lib, terms in libs.items()
                if isinstance(terms, list)
            }
    if result.get("cypher"):
        payload["cypher"] = result["cypher"]

    if result.get("answer"):
        payload["answer"] = result["answer"]
    if result.get("pyvis_html_path"):
        payload["pyvis_html_path"] = result["pyvis_html_path"]
    if result.get("kegg_pathway_path"):
        payload["kegg_pathway_path"] = result["kegg_pathway_path"]
    if result.get("volcano_plot_path"):
        payload["volcano_plot_path"] = result["volcano_plot_path"]
    if isinstance(result.get("kegg_enrichr_results"), list):
        payload["kegg_enrichr_results"] = result["kegg_enrichr_results"][:5]
    if isinstance(result.get("selected_pathway"), dict):
        payload["selected_pathway"] = {
            "library": result["selected_pathway"].get("library"),
            "term": result["selected_pathway"].get("term"),
            "rank": result["selected_pathway"].get("rank"),
            "overlapping_genes": result["selected_pathway"].get("overlapping_genes"),
        }

    return payload or {"keys": sorted(result.keys())}


def _infer_analysis_arm(state: AgentState) -> str:
    arm = str(state.get("analysis_arm") or "").strip().lower()
    if arm in {"general", "srp", "disease", "memory_rwr", "primekg", "opentargets", "memory_lookup"}:
        return arm
    if state.get("memory_lookup_result"):
        return "memory_lookup"
    if state.get("primekg_result"):
        return "primekg"
    if state.get("opentargets_result"):
        return "opentargets"
    if state.get("deg_analysis"):
        return "srp"
    if state.get("openalex_papers") or state.get("openalex_genes") or state.get("rwr_genes") or state.get("disease_name"):
        return "disease"
    if state.get("memory_deg_genes") and state.get("rwr_genes"):
        return "memory_rwr"
    return "general"


def _build_tool_list_text() -> str:
    lines = []
    for tool_obj in TOOL_SCHEMAS:
        lines.append(f"- {tool_obj.name}: {tool_obj.description}")
    return "\n".join(lines)


def _build_system_prompt(state: AgentState) -> str:
    query = _compact_text(state.get("query"), limit=400)
    memory_summary = _compact_text(state.get("memory_summary"), limit=500) or "No prior memory."
    state_snapshot = {
        "analysis_arm": _infer_analysis_arm(state),
        "step_count": int(state.get("step_count") or 0),
        "genes": state.get("genes") or [],
        "srp_ids": state.get("srp_ids") or [],
        "control_name": state.get("control_name") or state.get("memory_control_name") or "",
        "test_name": state.get("test_name") or state.get("memory_test_name") or "",
        "disease_name": state.get("disease_name") or "",
        "memory_disease_name": state.get("memory_disease_name") or "",
        "deg_gene_count": len(state.get("deg_genes") or []),
        "openalex_gene_count": len(state.get("openalex_genes") or []),
        "memory_deg_gene_count": len(state.get("memory_deg_genes") or []),
        "memory_enrichr_libraries": sorted(list((_ensure_dict(state.get("enrichr") or state.get("memory_enrichr")).get("libraries") or {}).keys())),
        "rwr_gene_count": len(state.get("rwr_genes") or []),
        "has_graph": bool(isinstance(state.get("graph"), nx.Graph) and state["graph"].number_of_nodes() > 0),
        "graph_summary": _graph_summary(state.get("graph") if isinstance(state.get("graph"), nx.Graph) else None),
        "recent_tools": (state.get("tool_history") or [])[-5:],
    }

    return (
        "You are the orchestration layer for a gene expression analysis agent.\n"
        "Your job is to choose the single best next action for the current turn.\n"
        "Think like an agent supervisor: inspect the live state, decide whether to answer or call exactly one tool, then reassess after the tool returns.\n"
        "Do not plan a long sequence in text. Make the next grounded move.\n"
        "If the query is simple general knowledge, conversational, or already fully answered by state, answer directly without tools.\n"
        "If the query needs technical analysis, prefer specialist tools over unsupported free-form reasoning.\n\n"
        "Specialist guidance:\n"
        f"{TOOL_USE_INSTRUCTIONS}\n\n"
        "Available tools and what they do:\n"
        f"{_build_tool_list_text()}\n\n"
        f"Current user query: {query}\n"
        f"Memory summary: {memory_summary}\n"
        f"Current state snapshot: {json.dumps(state_snapshot, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Decision rules:\n"
        "- Choose only from the listed specialist tools when a tool is needed.\n"
        "- Make the smallest correct next decision rather than narrating a full workflow.\n"
        "- Prefer using state and memory before asking tools to rediscover the same information.\n"
        "- Reuse stored genes, DEG results, disease names, pathway results, and graph state whenever they already satisfy the request.\n"
        "- If a required input is missing, choose the tool that can recover it instead of asking the user unless the gap cannot be inferred or recovered.\n"
        "- For pathway enrichment, prefer stored DEG genes and respect up/down regulation cues.\n"
        "- For DEG analysis, extract control, test, and SRP identifiers from the query or memory before running the tool.\n"
        "- For visualization, use stored pathway overlaps, RWR targets, DEG rows, or graphs whenever available.\n"
        "- Do not invent unavailable data, hidden evidence, or tool outputs.\n"
        "- Stop using tools once the user’s question is sufficiently answered.\n"
        "- Keep any direct answer concise, technically accurate, and aligned to the user's exact ask.\n"
    )


def _get_bound_llm():
    return get_llm().bind_tools(TOOL_SCHEMAS)


def _prepare_context(state: AgentState) -> AgentState:
    messages = list(state.get("messages") or [])
    if not messages:
        query = str(state.get("query") or "")
        messages = [HumanMessage(content=query)]

    update: AgentState = {
        "messages": messages,
        "step_count": int(state.get("step_count") or 0),
        "tool_history": list(state.get("tool_history") or []),
        "memory_summary": str(state.get("memory_summary") or ""),
    }
    if state.get("query"):
        update["query"] = str(state.get("query") or "")
    if state.get("memory_deg_genes") is not None:
        update["memory_deg_genes"] = list(state.get("memory_deg_genes") or [])
    if state.get("memory_deg_analysis") is not None:
        update["memory_deg_analysis"] = _ensure_dict(state.get("memory_deg_analysis"))
    if state.get("memory_deg_gene_records") is not None:
        update["memory_deg_gene_records"] = list(state.get("memory_deg_gene_records") or [])
    if state.get("memory_control_name") is not None:
        update["memory_control_name"] = str(state.get("memory_control_name") or "")
    if state.get("memory_test_name") is not None:
        update["memory_test_name"] = str(state.get("memory_test_name") or "")
    if state.get("memory_enrichr") is not None:
        update["memory_enrichr"] = _ensure_dict(state.get("memory_enrichr"))
    if state.get("memory_rwr_seed_genes") is not None:
        update["memory_rwr_seed_genes"] = list(state.get("memory_rwr_seed_genes") or [])
    if state.get("memory_rwr_genes") is not None:
        update["memory_rwr_genes"] = list(state.get("memory_rwr_genes") or [])
    if state.get("memory_disease_name") is not None:
        update["memory_disease_name"] = str(state.get("memory_disease_name") or "")
    if state.get("memory_openalex_genes") is not None:
        update["memory_openalex_genes"] = list(state.get("memory_openalex_genes") or [])
    if state.get("memory_opentargets_results") is not None:
        update["memory_opentargets_results"] = list(state.get("memory_opentargets_results") or [])
    if state.get("memory_lookup_result") is not None:
        update["memory_lookup_result"] = _ensure_dict(state.get("memory_lookup_result"))
    return update


def _agent(state: AgentState) -> AgentState:
    _trace_tool_call("llm_agent")
    llm = _get_bound_llm()
    messages = [SystemMessage(content=_build_system_prompt(state)), *list(state.get("messages") or [])]
    response = llm.invoke(messages)

    update: AgentState = {
        "messages": [response],
        "step_count": int(state.get("step_count") or 0) + 1,
    }

    if not getattr(response, "tool_calls", None):
        update["answer"] = _compact_text(getattr(response, "content", ""), limit=4000)
        update["should_finalize"] = True
    return update


def _route_after_agent(state: AgentState) -> str:
    if int(state.get("step_count") or 0) >= MAX_AGENT_STEPS:
        return "finalize"
    ai_message = _latest_ai_message(list(state.get("messages") or []))
    if not ai_message or not getattr(ai_message, "tool_calls", None):
        return "finalize"

    tool_name = str(ai_message.tool_calls[0].get("name", "")).strip()
    if tool_name in TOOL_EXECUTORS:
        return tool_name
    return "finalize"


def _route_after_tool(state: AgentState) -> str:
    return "finalize" if state.get("should_finalize") else "agent"


def _run_extract_genes(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text") or state.get("query") or "")
    mode = str(args.get("mode") or "strict")
    whitelist = args.get("whitelist")
    whitelist_set = None
    if isinstance(whitelist, list):
        whitelist_set = {str(value).strip().upper() for value in whitelist if str(value).strip()}
    genes = extract_genes_from_text(text, whitelist=whitelist_set, mode=mode)
    return {"genes": genes}


def _run_extract_srp_ids(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text") or state.get("query") or "")
    from gea_agent.tools.srp_ids import extract_srp_ids_from_text

    return {"srp_ids": extract_srp_ids_from_text(text)}


def _run_extract_deg_groups(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    control_name = " ".join(str(args.get("control_name") or state.get("control_name") or state.get("memory_control_name") or "").split()).strip()
    test_name = " ".join(str(args.get("test_name") or state.get("test_name") or state.get("memory_test_name") or "").split()).strip()
    if control_name and test_name:
        return {"control_name": control_name, "test_name": test_name}

    text = str(args.get("text") or state.get("query") or "")
    llm = get_llm()
    response = llm.invoke(
        [
            (
                "system",
                "You are a normalization step inside a biomedical agent workflow. "
                "Extract DEG comparison groups from the user request. "
                "Return strict JSON only with keys `control_name` and `test_name`. "
                "Interpret synonyms such as control, baseline, healthy, normal, untreated, disease, case, treated, or condition when the comparison is implied. "
                "Preserve the original human-readable cohort labels exactly when possible. "
                "If either group is missing or ambiguous, return an empty string for that field. "
                "Do not add explanations, reasoning, or extra keys.",
            ),
            ("user", text),
        ]
    )
    try:
        parsed = _ensure_dict(json.loads(getattr(response, "content", "") or "{}"))
    except Exception:
        parsed = {}
    control_name = " ".join(str(parsed.get("control_name") or "").split()).strip()
    test_name = " ".join(str(parsed.get("test_name") or "").split()).strip()
    return {"control_name": control_name, "test_name": test_name}


def _run_identify_disease(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or state.get("query") or "")
    disease_result = identify_disease_from_query(query)
    disease_name = disease_result.get("disease", "") or state.get("memory_disease_name") or ""
    return {"disease_name": disease_name}


def _run_opentargets_association(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(state.get("query") or "")
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    genes_arg = args.get("genes")
    genes: list[str] = []
    if isinstance(genes_arg, list):
        genes = [str(value).strip().upper() for value in genes_arg if str(value).strip()]
    if not genes and _memory_gene_query_requested(query):
        direction = _deg_direction_from_query(query)
        deg_records = state.get("deg_gene_records") or state.get("memory_deg_gene_records")
        genes = _genes_from_deg_records_by_direction(
            deg_records,
            direction=direction,
            top_n=top_n if isinstance(top_n, int) and top_n > 0 else 500,
        )
        if not genes:
            genes = [
                str(value).strip().upper()
                for value in (state.get("deg_genes") or state.get("memory_deg_genes") or state.get("genes") or [])
                if str(value).strip()
            ]
            if isinstance(top_n, int) and top_n > 0:
                genes = genes[:top_n]

    gene = str(
        args.get("gene")
        or state.get("disease_gene")
        or (state.get("genes") or state.get("memory_deg_genes") or [""])[0]
        or ""
    )
    if not gene:
        extracted = extract_genes_from_text(str(state.get("query") or ""), mode="strict")
        gene = str((extracted or [""])[0] or "")
    disease = str(
        args.get("disease")
        or args.get("disease_name")
        or state.get("disease_name")
        or state.get("memory_disease_name")
        or ""
    )
    if not disease and _memory_gene_query_requested(query):
        disease = _disease_from_association_query(query)
        if not disease:
            disease_result = identify_disease_from_query(query)
            disease = str(disease_result.get("disease") or "").strip()
    if _drug_association_query_requested(query) and not genes:
        result = find_drugs_for_gene(gene)
        result["association_kind"] = "gene_drug"
    elif genes and not disease:
        if len(genes) == 1:
            result = find_diseases_for_gene(genes[0])
        else:
            results = [find_diseases_for_gene(value) for value in genes]
            result = {
                "status": "ok",
                "genes": genes,
                "associated": any(item.get("associated") for item in results if isinstance(item, dict)),
                "results": results,
                "message": f"Retrieved top OpenTargets disease associations for {len(results)} genes.",
            }
    elif genes:
        result = check_gene_list_disease_associations(genes, disease)
        result["gene_set_source"] = "stored_deg_genes" if _memory_gene_query_requested(query) else "tool_args"
        if isinstance(top_n, int) and top_n > 0:
            result["top_n"] = top_n
        result["direction"] = _deg_direction_from_query(query)
    elif not disease:
        result = find_diseases_for_gene(gene)
    else:
        result = check_gene_disease_association(gene, disease)
    result["analysis_arm"] = "opentargets"
    return result


# def _run_primekg_query(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
#     query = str(state.get("query") or "")
#     top_n = args.get("top_n")
#     if top_n is None:
#         top_n = _parse_top_n_from_text(query)
#     if isinstance(top_n, str) and top_n.isdigit():
#         top_n = int(top_n)

#     source_terms = args.get("source_terms") or args.get("genes") or args.get("entities")
#     if isinstance(source_terms, str):
#         source_terms = [source_terms]
#     if not isinstance(source_terms, list):
#         source_terms = []

#     if not source_terms and args.get("gene"):
#         source_terms = [str(args.get("gene"))]

#     if not source_terms and _memory_gene_query_requested(query):
#         source_terms = _genes_from_deg_records_by_direction(
#             state.get("deg_gene_records") or state.get("memory_deg_gene_records"),
#             direction=_deg_direction_from_query(query),
#             top_n=top_n if isinstance(top_n, int) and top_n > 0 else 50,
#         )
#         if not source_terms:
#             source_terms = list(state.get("deg_genes") or state.get("memory_deg_genes") or state.get("genes") or [])
#             if isinstance(top_n, int) and top_n > 0:
#                 source_terms = source_terms[:top_n]

#     if not source_terms:
#         source_terms = extract_genes_from_text(query, mode="strict")

#     target_terms = args.get("target_terms")
#     if isinstance(target_terms, str):
#         target_terms = [target_terms]
#     if not isinstance(target_terms, list):
#         target_terms = []

#     disease = str(args.get("disease") or args.get("disease_name") or "").strip()
#     if not disease and ("associated with" in query.lower() or "linked to" in query.lower()):
#         disease = _disease_from_association_query(query)
#     if disease and source_terms and not target_terms:
#         target_terms = [disease]
#     elif disease and not source_terms:
#         source_terms = [disease]

#     source_types = args.get("source_types")
#     target_types = args.get("target_types")
#     if not isinstance(source_types, list) or not source_types:
#         source_types = ["disease"] if disease and source_terms == [disease] else (["gene/protein"] if source_terms else [])
#     if not isinstance(target_types, list) or not target_types:
#         target_types = _primekg_target_types_from_query(query)
#     relation_terms = args.get("relation_terms")
#     result = query_primekg(
#         source_terms=[str(value).strip().upper() for value in source_terms if str(value).strip()],
#         target_terms=[str(value).strip() for value in target_terms if str(value).strip()],
#         source_types=source_types if isinstance(source_types, list) else [],
#         target_types=target_types if isinstance(target_types, list) else [],
#         relation_terms=relation_terms if isinstance(relation_terms, list) else [],
#         limit=int(args.get("limit") or 50),
#     )
#     result["analysis_arm"] = "primekg"
#     if _memory_gene_query_requested(query):
#         result["gene_set_source"] = "stored_deg_genes"
#         result["direction"] = _deg_direction_from_query(query)
#     return result

def _run_primekg_query(
    state: AgentState,
    args: dict[str, Any]
):
    question = (
        args.get("question")
        or state.get("query")
        or ""
    )

    return query_primekg(question)

def _run_fetch_openalex(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    disease_name = str(
        args.get("disease_name")
        or args.get("disease")
        or state.get("disease_name")
        or state.get("memory_disease_name")
        or ""
    )
    top_n = int(args.get("top_n") or 20)
    openalex_result = fetch_openalex_papers_and_genes(
        disease_name,
        top_n=top_n,
        user_query=str(state.get("query") or args.get("question") or disease_name),
    )
    genes = openalex_result.get("genes", [])
    return {
        "analysis_arm": "disease",
        "disease_name": openalex_result.get("disease", disease_name),
        "openalex_papers": openalex_result.get("papers", []),
        "ranked_openalex_papers": openalex_result.get("ranked_papers", []),
        "openalex_genes": genes,
        "literature_key_points": openalex_result.get("key_points", []),
        "literature_references": openalex_result.get("references", []),
        "literature_summary": openalex_result.get("literature_summary", ""),
        "genes": _merge_unique(state.get("genes"), genes),
    }

def _run_deg_analysis(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    srp_ids = args.get("srp_ids")
    if isinstance(srp_ids, list):
        srp_ids = [str(value).strip().upper() for value in srp_ids if str(value).strip()]
    else:
        srp_ids = list(state.get("srp_ids") or [])
    if not srp_ids:
        srp_ids = _run_extract_srp_ids(state, args).get("srp_ids", [])
    group_result = _run_extract_deg_groups(state, args)
    control_name = str(group_result.get("control_name") or "").strip()
    test_name = str(group_result.get("test_name") or "").strip()
    deg_result = run_deg_r_analysis(
        srp_ids=srp_ids,
        control_name=control_name,
        test_name=test_name,
    )
    deg_genes = deg_result.get("genes", [])
    deg_rows = deg_result.get("rows", [])
    deg_gene_records: list[dict[str, Any]] = []
    if isinstance(deg_rows, list):
        for row in deg_rows:
            if not isinstance(row, dict):
                continue
            gene = row.get("hgnc_symbol") or row.get("external_gene_name") or row.get("Ensembl") or row.get("entrezgene_accession") or ""
            gene = str(gene).strip()
            if not gene:
                continue
            deg_gene_records.append(
                {
                    "gene": gene,
                    "pvalue": row.get("pvalue"),
                    "pdj": row.get("pdj"),
                    "log2FoldChange": row.get("log2FoldChange"),
                    "description": row.get("description"),
                }
            )
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(str(args.get("text") or state.get("query") or ""))
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)
    if isinstance(top_n, int) and top_n > 0:
        deg_genes = deg_genes[:top_n]
        deg_gene_records = deg_gene_records[:top_n]
    return {
        "analysis_arm": "srp",
        "srp_ids": srp_ids,
        "control_name": control_name,
        "test_name": test_name,
        "deg_analysis": deg_result,
        "deg_genes": deg_genes,
        "deg_gene_records": deg_gene_records,
        "genes": _merge_unique(state.get("genes"), deg_genes),
    }


def _run_build_string_graph(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    genes = args.get("genes")
    if not isinstance(genes, list) or not genes:
        genes, gene_set_source = _resolve_rwr_source_genes(state, args, prefer_seed_genes=False)
    else:
        gene_set_source = "tool_args"
    genes = [str(value).strip().upper() for value in genes if str(value).strip()]
    graph = build_weighted_graph_from_string_files(
        genes=genes,
        info_path=str(args.get("info_path") or SETTINGS.string_info_path),
        links_path=str(args.get("links_path") or SETTINGS.string_links_path),
        required_score=int(args.get("required_score") or SETTINGS.string_required_score),
        mode=str(args.get("mode") or SETTINGS.string_local_mode),
    )
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    update: dict[str, Any] = {"graph": graph, "genes": genes, "rwr_seed_genes": genes, "gene_set_source": gene_set_source}
    if analysis_arm:
        update["analysis_arm"] = analysis_arm
    return update


def _run_top_rwr(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return {"rwr_genes": [], "rwr_seed_genes": list(state.get("genes") or [])}

    query = str(args.get("text") or state.get("query") or "")
    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    seed_genes = args.get("seed_genes")
    if not isinstance(seed_genes, list) or not seed_genes:
        seed_genes, gene_set_source = _resolve_rwr_source_genes(state, args, prefer_seed_genes=True)
    else:
        gene_set_source = "tool_args"
    seed_genes = [str(value).strip().upper() for value in seed_genes if str(value).strip()]
    if isinstance(top_n, int) and top_n > 0:
        seed_genes = seed_genes[:top_n]

    rwr = top_rwr_genes(
        graph,
        seed_genes,
        top_k=int(top_n or args.get("top_k") or 30),
        restart_prob=float(args.get("restart_prob") or 0.5),
        exclude=args.get("exclude"),
        exclude_hubs=bool(args.get("exclude_hubs", True)),
    )
    update: dict[str, Any] = {
        "rwr_genes": rwr,
        "rwr_seed_genes": seed_genes,
        "direction": direction,
        "gene_set_source": gene_set_source,
    }
    if isinstance(top_n, int) and top_n > 0:
        update["top_n"] = top_n
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    if analysis_arm in {"disease", "memory_rwr"}:
        update["analysis_arm"] = analysis_arm
    return update


def _run_enrichr(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    query = str(args.get("text") or state.get("query") or "")
    direction = _deg_direction_from_query(query)
    genes = args.get("genes")
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)
    if isinstance(genes, list) and isinstance(top_n, int) and top_n > 0:
        genes = genes[:top_n]
    if not isinstance(genes, list) or not genes:
        if analysis_arm == "srp":
            genes = _genes_from_deg_records_by_direction(
                state.get("deg_gene_records") or state.get("memory_deg_gene_records"),
                direction=direction,
                top_n=top_n,
            ) or list(state.get("deg_genes") or [])
        else:
            genes = _merge_unique(
                _genes_from_deg_records_by_direction(
                    state.get("memory_deg_gene_records"),
                    direction=direction,
                    top_n=top_n,
                ),
                state.get("genes"),
                [gene for gene, _ in (state.get("rwr_genes") or [])],
            )

    background = list((state.get("graph") or nx.Graph()).nodes()) if isinstance(state.get("graph"), nx.Graph) else []
    if analysis_arm == "srp":
        background = list(state.get("deg_genes") or genes)

    return {
        "direction": direction,
        "gene_set_source": "stored_deg_genes" if _memory_gene_query_requested(query) or state.get("deg_gene_records") or state.get("memory_deg_gene_records") else "tool_args",
        "enrichr": enrichr_pathways(
            genes,
            top_n=int(top_n or args.get("top_n") or 10),
            background_genes=background,
        )
    }


def _visualization_gene_set(
    state: AgentState,
    args: dict[str, Any],
    *,
    query: str,
) -> tuple[list[str], str, str, int | None]:
    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    explicit_genes = [str(g).strip().upper() for g in _tool_arg_list(args.get("genes")) if str(g).strip()]
    if explicit_genes:
        if isinstance(top_n, int) and top_n > 0:
            explicit_genes = explicit_genes[:top_n]
        return explicit_genes, "tool_args", direction, top_n

    deg_records = state.get("deg_gene_records") or state.get("memory_deg_gene_records")
    deg_genes = _genes_from_deg_records_by_direction(
        deg_records,
        direction=direction,
        top_n=top_n,
    )
    if deg_genes:
        return deg_genes, "stored_deg_genes", direction, top_n

    fallback = _merge_unique(
        state.get("genes"),
        [gene for gene, _ in (state.get("rwr_genes") or [])],
        state.get("openalex_genes"),
        state.get("memory_openalex_genes"),
    )
    if isinstance(top_n, int) and top_n > 0:
        fallback = fallback[:top_n]
    return fallback, "state_fallback", direction, top_n


def _memory_rwr_targets(state: AgentState, *, top_k: int = 20) -> list[str]:
    targets = state.get("rwr_genes") or state.get("memory_rwr_genes") or []
    out: list[str] = []
    for row in targets[:top_k]:
        if isinstance(row, (list, tuple)) and row:
            gene = str(row[0]).strip().upper()
        elif isinstance(row, dict):
            gene = str(row.get("gene") or "").strip().upper()
        else:
            gene = ""
        if gene:
            out.append(gene)
    return out


def _run_visualize(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("text") or state.get("query") or "")
    visualization_type = str(args.get("visualization_type") or args.get("kind") or "").strip().lower()
    if not visualization_type:
        lowered = query.lower()
        if "volcano" in lowered:
            visualization_type = "volcano"
        elif "kegg" in lowered:
            visualization_type = "kegg"
        else:
            visualization_type = "network"

    if visualization_type == "network":
        graph = state.get("graph")
        seed_genes = [
            str(g).strip().upper()
            for g in (
                args.get("genes")
                or state.get("rwr_seed_genes")
                or state.get("memory_rwr_seed_genes")
                or state.get("genes")
                or []
            )
            if str(g).strip()
        ]
        top_targets = _memory_rwr_targets(state, top_k=int(args.get("top_k") or 20))
        allowed_nodes = _merge_unique(seed_genes, top_targets)
        if (not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0) and allowed_nodes:
            graph = build_weighted_graph_from_string_files(
                genes=allowed_nodes,
                info_path=str(args.get("info_path") or SETTINGS.string_info_path),
                links_path=str(args.get("links_path") or SETTINGS.string_links_path),
                required_score=int(args.get("required_score") or SETTINGS.string_required_score),
                mode=str(args.get("mode") or SETTINGS.string_local_mode),
            )
        result = build_network_visualization(
            graph,
            output_path=str(args.get("output_path") or "pyvis_network.html"),
            select_top_degree=int(args.get("select_top_degree") or max(len(allowed_nodes), 20) or 20),
            allowed_nodes=allowed_nodes,
        )
        result["visualization_type"] = "network"
        result["seed_genes"] = seed_genes
        result["top_targets"] = top_targets
        return result

    if visualization_type == "kegg":
        pathway_term = str(args.get("pathway_term") or "").strip()
        selected_term, selected_library, selected_rank = _find_enrichr_term_from_state(
            state,
            pathway_term,
            query=query,
        )
        if selected_term and isinstance(selected_term.get("overlapping_genes"), list) and selected_term.get("overlapping_genes"):
            genes = [
                str(g).strip().upper()
                for g in selected_term.get("overlapping_genes", [])
                if str(g).strip()
            ]
            gene_set_source = "stored_pathway_overlapping_genes"
            direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
            top_n = len(genes)
        else:
            genes, gene_set_source, direction, top_n = _visualization_gene_set(state, args, query=query)
        result = build_kegg_pathway_visualization(
            genes,
            output_path=str(args.get("output_path") or "kegg_pathway.png"),
            kegg_rank=int(args.get("kegg_rank") or selected_rank or 1),
            species=str(args.get("species") or "human"),
        )
        result["visualization_type"] = "kegg"
        result["gene_set_source"] = gene_set_source
        result["direction"] = direction
        result["top_n"] = top_n
        if selected_term:
            result["selected_pathway"] = {
                "library": selected_library,
                "term": selected_term.get("term"),
                "overlapping_genes": selected_term.get("overlapping_genes"),
                "rank": selected_rank,
            }
        return result

    if visualization_type == "volcano":
        deg_rows = state.get("deg_gene_records") or state.get("memory_deg_gene_records") or []
        result = build_volcano_plot(
            deg_rows,
            output_path=str(args.get("output_path") or "deg_volcano.png"),
            pvalue_threshold=float(args.get("pvalue_threshold") or 0.05),
            log2fc_threshold=float(args.get("log2fc_threshold") or 1.0),
        )
        result["visualization_type"] = "volcano"
        return result

    return {
        "status": "unsupported_visualization",
        "message": f"Unsupported visualization type: {visualization_type}",
        "visualization_type": visualization_type,
    }


def _run_synthesize(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    answer = synthesize_technical_response(
        user_query=str(args.get("user_query") or state.get("query") or ""),
        analysis_arm=str(args.get("analysis_arm") or state.get("analysis_arm") or _infer_analysis_arm(state)).strip().lower(),
        seed_genes=list(args.get("seed_genes") or state.get("genes") or []),
        srp_ids=list(args.get("srp_ids") or state.get("srp_ids") or []),
        disease_name=str(args.get("disease_name") or state.get("disease_name") or ""),
        deg_analysis=_ensure_dict(state.get("opentargets_result") or state.get("primekg_result") or state.get("deg_analysis")),
        rwr_genes=list(state.get("rwr_genes") or []),
        graph=graph if isinstance(graph, nx.Graph) else nx.Graph(),
        enrichr=_ensure_dict(state.get("enrichr")),
        literature_papers=list(state.get("openalex_papers") or []),
        ranked_literature_papers=list(state.get("ranked_openalex_papers") or []),
        literature_key_points=list(state.get("literature_key_points") or []),
        literature_references=list(state.get("literature_references") or []),
        literature_summary=str(state.get("literature_summary") or ""),
    )
    return {
        "answer": answer,
        "analysis_arm": str(args.get("analysis_arm") or state.get("analysis_arm") or "disease").strip().lower(),
        "should_finalize": True,
    }


def _specialist_history_update(state: AgentState, tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    history = list(state.get("tool_history") or [])
    history.append({"tool": tool_name, "args": args, "result": _serialize_tool_result(result)})
    return {"tool_history": history}


def _tool_observations(state: AgentState, call: dict[str, Any] | None, tool_name: str, result: dict[str, Any]) -> list[ToolMessage]:
    ai_message = _latest_ai_message(list(state.get("messages") or []))
    tool_calls = list(getattr(ai_message, "tool_calls", []) or [])
    if not tool_calls and call:
        tool_calls = [call]

    selected_id = str(call.get("id")) if call and call.get("id") else ""
    messages: list[ToolMessage] = []
    for tool_call in tool_calls:
        call_id = str(tool_call.get("id") or "")
        if not call_id:
            continue
        name = str(tool_call.get("name") or tool_name)
        if selected_id and call_id == selected_id:
            payload = _serialize_tool_result(result)
        else:
            payload = {
                "status": "deferred",
                "message": (
                    "This tool call was not executed in this step because the agent executes "
                    "one specialist at a time. Call it again if it is still needed."
                ),
            }
        messages.append(
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str),
                name=name,
                tool_call_id=call_id,
            )
        )
    return messages


def _specialist_node(tool_name: str) -> Callable[[AgentState], AgentState]:
    def node(state: AgentState) -> AgentState:
        call = _latest_tool_call(state, tool_name)
        args = dict(call.get("args") or {}) if call else {}
        _trace_tool_call(tool_name)

        if tool_name == "deg_analysis":
            result = _run_extract_srp_ids(state, args)
            update = _specialist_history_update(state, "extract_srp_ids_from_text", args, result)
            update = {**update, **result}
            state = {**state, **update}

            result = _run_deg_analysis(state, args)
            update = _specialist_history_update(state, "run_deg_r_analysis", args, result)
            update = {**update, **result}
            state = {**state, **update}
            return {**state, **result, "analysis_arm": "srp", "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "pathway":
            result = _run_enrichr(state, args)
            update = _specialist_history_update(state, "enrichr_pathways", args, result)
            update = {**update, **result}
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "rwr_analysis":
            build_result = _run_build_string_graph(state, args)
            update = _specialist_history_update(state, "build_weighted_graph_from_string_files", args, build_result)
            update = {**update, **build_result}
            state = {**state, **update}

            rwr_result = _run_top_rwr(state, args)
            update = _specialist_history_update(state, "top_rwr_genes", args, rwr_result)
            update = {**update, **rwr_result}
            state = {**state, **update}
            return {**state, **rwr_result, "messages": _tool_observations(state, call, tool_name, rwr_result)}

        if tool_name == "visualize":
            result = _run_visualize(state, args)
            update = _specialist_history_update(state, "visualize", args, result)
            update = {**update, **result}
            update["visualization_result"] = result
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "memory_lookup":
            result = _run_memory_lookup(state, args)
            update = _specialist_history_update(state, "memory_lookup", args, result)
            update = {**update, **result}
            update["memory_lookup_result"] = result
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "literature":
            disease_result = _run_identify_disease(state, args)
            update = _specialist_history_update(state, "identify_disease_from_query", args, disease_result)
            update = {**update, **disease_result}
            state = {**state, **update}

            openalex_result = _run_fetch_openalex(state, args)
            update = _specialist_history_update(state, "fetch_openalex_papers_and_genes", args, openalex_result)
            update = {**update, **openalex_result}
            state = {**state, **update}

            gene_result = _run_extract_genes(state, args)
            update = _specialist_history_update(state, "extract_genes_from_text", args, gene_result)
            update = {**update, **gene_result}
            state = {**state, **update}
            return {**state, **openalex_result, "messages": _tool_observations(state, call, tool_name, openalex_result)}

        if tool_name == "identify_disease_from_query":
            result = _run_identify_disease(state, args)
            update = _specialist_history_update(state, "identify_disease_from_query", args, result)
            update = {**update, **result}
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "primekg_query":
            result = _run_primekg_query(state, args)
            update = _specialist_history_update(state, "primekg_query", args, result)
            update = {**update, **result}
            update["primekg_result"] = result
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "opentargets_association":
            result = _run_opentargets_association(state, args)
            update = _specialist_history_update(state, "opentargets_association", args, result)
            update = {**update, **result}
            update["opentargets_result"] = result
            if result.get("gene"):
                update["disease_gene"] = str(result.get("gene") or "").strip().upper()
            if result.get("status") == "ok":
                history = list(state.get("memory_opentargets_results") or [])
                history.append(result)
                update["memory_opentargets_results"] = history
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        return state

    return node


def _finalize(state: AgentState) -> AgentState:
    answer = str(state.get("answer") or "").strip()
    if not answer:
        ai_message = _latest_ai_message(list(state.get("messages") or []))
        if ai_message and not getattr(ai_message, "tool_calls", None):
            answer = _compact_text(getattr(ai_message, "content", ""), limit=4000)

    if state.get("tool_history"):
        analysis_arm = _infer_analysis_arm(state)
        if analysis_arm == "memory_lookup" and isinstance(state.get("memory_lookup_result"), dict):
            answer = str(state["memory_lookup_result"].get("answer") or answer or "").strip()
            graph = state.get("graph")
            meta = {
                "analysis_arm": analysis_arm,
                "is_followup": bool(state.get("is_followup", False)),
                "route_rationale": state.get("route_rationale", ""),
                "srp_ids": list(state.get("srp_ids") or []),
                "control_name": str(state.get("control_name") or ""),
                "test_name": str(state.get("test_name") or ""),
                "memory_control_name": str(state.get("memory_control_name") or ""),
                "memory_test_name": str(state.get("memory_test_name") or ""),
                "memory_deg_genes": list(state.get("memory_deg_genes") or []),
                "memory_deg_analysis": _ensure_dict(state.get("memory_deg_analysis")),
                "memory_deg_gene_records": list(state.get("memory_deg_gene_records") or []),
                "memory_disease_name": str(state.get("memory_disease_name") or ""),
                "memory_openalex_genes": list(state.get("memory_openalex_genes") or []),
                "memory_opentargets_results": list(state.get("memory_opentargets_results") or []),
                "disease_name": str(state.get("disease_name") or ""),
                "disease_gene": str(state.get("disease_gene") or ""),
                "memory_lookup_result": _ensure_dict(state.get("memory_lookup_result")),
                "openalex_papers": list(state.get("openalex_papers") or []),
                "ranked_openalex_papers": list(state.get("ranked_openalex_papers") or []),
                "openalex_genes": list(state.get("openalex_genes") or []),
                "literature_key_points": list(state.get("literature_key_points") or []),
                "literature_references": list(state.get("literature_references") or []),
                "literature_summary": str(state.get("literature_summary") or ""),
                "primekg_result": _ensure_dict(state.get("primekg_result")),
                "opentargets_result": _ensure_dict(state.get("opentargets_result")),
                "deg_analysis": _ensure_dict(state.get("deg_analysis")),
                "deg_genes": list(state.get("deg_genes") or []),
                "deg_gene_records": list(state.get("deg_gene_records") or []),
                "genes": list(state.get("genes") or []),
                "rwr_seed_genes": list(state.get("rwr_seed_genes") or []),
                "network": _graph_summary(graph if isinstance(graph, nx.Graph) else None),
                "rwr_genes": list(state.get("rwr_genes") or []),
                "enrichr": _ensure_dict(state.get("enrichr")),
                "pyvis_html_path": str(state.get("pyvis_html_path") or ""),
                "kegg_pathway_path": str(state.get("kegg_pathway_path") or ""),
                "volcano_plot_path": str(state.get("volcano_plot_path") or ""),
                "visualization_result": _ensure_dict(state.get("visualization_result")),
                "tool_history": list(state.get("tool_history") or [])[-10:],
            }
            return {
                "answer": answer,
                "meta": meta,
                "analysis_arm": analysis_arm,
                "graph": graph if isinstance(graph, nx.Graph) else None,
            }
        specialist_payload = _ensure_dict(state.get("deg_analysis"))
        if analysis_arm == "primekg":
            specialist_payload = _ensure_dict(state.get("primekg_result"))
        elif analysis_arm == "opentargets":
            specialist_payload = _ensure_dict(state.get("opentargets_result"))
        answer = synthesize_technical_response(
            user_query=str(state.get("query") or ""),
            analysis_arm=analysis_arm,
            seed_genes=list(state.get("genes") or []),
            srp_ids=list(state.get("srp_ids") or []),
            disease_name=str(state.get("disease_name") or ""),
            deg_analysis=specialist_payload,
            rwr_genes=list(state.get("rwr_genes") or []),
            graph=state.get("graph") if isinstance(state.get("graph"), nx.Graph) else nx.Graph(),
            enrichr=_ensure_dict(state.get("enrichr")),
            literature_papers=list(state.get("openalex_papers") or []),
            ranked_literature_papers=list(state.get("ranked_openalex_papers") or []),
            literature_key_points=list(state.get("literature_key_points") or []),
            literature_references=list(state.get("literature_references") or []),
            literature_summary=str(state.get("literature_summary") or ""),
        )

    analysis_arm = _infer_analysis_arm(state)
    graph = state.get("graph")
    meta = {
        "analysis_arm": analysis_arm,
        "is_followup": bool(state.get("is_followup", False)),
        "route_rationale": state.get("route_rationale", ""),
        "srp_ids": list(state.get("srp_ids") or []),
        "control_name": str(state.get("control_name") or ""),
        "test_name": str(state.get("test_name") or ""),
        "memory_control_name": str(state.get("memory_control_name") or ""),
        "memory_test_name": str(state.get("memory_test_name") or ""),
        "memory_deg_genes": list(state.get("memory_deg_genes") or []),
        "memory_deg_analysis": _ensure_dict(state.get("memory_deg_analysis")),
        "memory_deg_gene_records": list(state.get("memory_deg_gene_records") or []),
        "memory_disease_name": str(state.get("memory_disease_name") or ""),
        "memory_openalex_genes": list(state.get("memory_openalex_genes") or []),
        "memory_opentargets_results": list(state.get("memory_opentargets_results") or []),
        "disease_name": str(state.get("disease_name") or ""),
        "disease_gene": str(state.get("disease_gene") or ""),
        "openalex_papers": list(state.get("openalex_papers") or []),
        "ranked_openalex_papers": list(state.get("ranked_openalex_papers") or []),
        "openalex_genes": list(state.get("openalex_genes") or []),
        "literature_key_points": list(state.get("literature_key_points") or []),
        "literature_references": list(state.get("literature_references") or []),
        "literature_summary": str(state.get("literature_summary") or ""),
        "primekg_result": _ensure_dict(state.get("primekg_result")),
        "opentargets_result": _ensure_dict(state.get("opentargets_result")),
        "memory_lookup_result": _ensure_dict(state.get("memory_lookup_result")),
        "deg_analysis": _ensure_dict(state.get("deg_analysis")),
        "deg_genes": list(state.get("deg_genes") or []),
        "deg_gene_records": list(state.get("deg_gene_records") or []),
        "genes": list(state.get("genes") or []),
        "rwr_seed_genes": list(state.get("rwr_seed_genes") or []),
        "network": _graph_summary(graph if isinstance(graph, nx.Graph) else None),
        "rwr_genes": list(state.get("rwr_genes") or []),
        "enrichr": _ensure_dict(state.get("enrichr")),
        "pyvis_html_path": str(state.get("pyvis_html_path") or ""),
        "kegg_pathway_path": str(state.get("kegg_pathway_path") or ""),
        "volcano_plot_path": str(state.get("volcano_plot_path") or ""),
        "visualization_result": _ensure_dict(state.get("visualization_result")),
        "tool_history": list(state.get("tool_history") or [])[-10:],
    }
    return {
        "answer": answer,
        "meta": meta,
        "analysis_arm": analysis_arm,
        "graph": graph if isinstance(graph, nx.Graph) else None,
    }


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("prepare_context", _prepare_context)
    graph.add_node("agent", _agent)
    graph.add_node("finalize", _finalize)

    for tool_name in TOOL_EXECUTORS:
        graph.add_node(tool_name, _specialist_node(tool_name))

    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {tool_name: tool_name for tool_name in TOOL_EXECUTORS} | {"finalize": "finalize"},
    )
    for tool_name in TOOL_EXECUTORS:
        graph.add_conditional_edges(
            tool_name,
            _route_after_tool,
            {"agent": "agent", "finalize": "finalize"},
        )
    graph.add_edge("finalize", END)

    return graph.compile()


def _tool_arg_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# New specialist checklist:
# 1. Add a tool schema here so the orchestrator can call it.
# 2. Add its runtime handler in `_specialist_node(...)` and include any state updates.
# 3. Register the tool name in `TOOL_EXECUTORS` so routing can reach it.
# 4. Update `TOOL_USE_INSTRUCTIONS`, prompt guidance, and `_finalize(...)` metadata if the tool introduces new memory/results.
# 5. Extend `AgentState` when the tool stores new fields across turns.
TOOL_SCHEMAS = [
    tool(
        "extract_genes_from_text",
        description="Extract candidate gene symbols from free text.",
        return_direct=False,
    )(lambda text, mode="strict", whitelist=None: {"genes": extract_genes_from_text(text, whitelist={str(value).strip().upper() for value in whitelist} if isinstance(whitelist, list) else None, mode=mode)}),
    tool(
        "extract_srp_ids_from_text",
        description="Extract SRP accession identifiers from text.",
        return_direct=False,
    )(lambda text: {"srp_ids": extract_srp_ids_from_text(text)}),
    tool(
        "identify_disease_from_query",
        description="Infer the disease name from the user query.",
        return_direct=False,
    )(lambda query: identify_disease_from_query(query)),
    tool(
        "literature",
        description="Search OpenAlex for disease literature and extract genes from the returned abstracts.",
        return_direct=False,
    )(lambda disease_name, top_n=20: fetch_openalex_papers_and_genes(disease_name, top_n=int(top_n))),
    tool(
        "deg_analysis",
        description="""
        Run differential expression analysis.

        Args:
            control_name: Control cohort label such as
                    "Healthy lung tissue".
            test_name: Test cohort label such as
                    "COPD lung tissue".
            srp_ids: List of SRP accessions such as
                    ["SRP277202"].
            text: Optional text containing control, test, and SRP IDs.

        Returns:
            Differentially expressed genes and statistics.
        """,
        return_direct=False,
    )(
        lambda srp_ids=None, control_name=None, test_name=None, text=None: {
            "srp_ids": srp_ids,
            "control_name": control_name,
            "test_name": test_name,
            "text": text,
        }
    ),
    tool(
        "rwr_analysis",
        description="Build a STRING protein interaction graph from the local STRING downloads.",
        return_direct=False,
    )(lambda genes, info_path=SETTINGS.string_info_path, links_path=SETTINGS.string_links_path, required_score=SETTINGS.string_required_score, mode=SETTINGS.string_local_mode: {
        "graph": build_weighted_graph_from_string_files(
            genes=list(genes or []),
            info_path=info_path,
            links_path=links_path,
            required_score=int(required_score),
            mode=mode,
        )
    }),
    tool(
        "top_rwr_genes",
        description="Rank genes using random walk with restart on the current STRING graph.",
        return_direct=False,
    )(lambda seed_genes, top_k=30, restart_prob=0.5: {"rwr_genes": seed_genes, "top_k": top_k, "restart_prob": restart_prob}),
    tool(
        "pathway",
        description="Primary enrichment tool using Enrichr. Use for pathway or GO-term questions, including stored up-regulated genes with positive log2FoldChange, down-regulated genes with negative log2FoldChange, or both together when requested.",
        return_direct=False,
    )(lambda genes, top_n=500, background_genes=None: enrichr_pathways(list(genes or []), top_n=int(top_n), background_genes=list(background_genes or []))),
    tool(
        "visualize",
        description="Create technical visualizations independently. Supported visualization_type values are `network` for STRING/PyVis network rendering, `kegg` for KEGG pathway visualization from genes using gget, and `volcano` for DEG volcano plots. For KEGG follow-ups, you can pass `pathway_term` to reuse overlapping genes from stored Enrichr pathways.",
        return_direct=False,
    )(
        lambda visualization_type, genes=None, top_n=None, direction=None, output_path=None, kegg_rank=1, species="human", select_top_degree=300, pvalue_threshold=0.05, log2fc_threshold=1.0, pathway_term=None, text=None: {
            "visualization_type": visualization_type,
            "genes": list(genes or []),
            "top_n": top_n,
            "direction": direction,
            "output_path": output_path,
            "kegg_rank": kegg_rank,
            "species": species,
            "select_top_degree": select_top_degree,
            "pvalue_threshold": pvalue_threshold,
            "log2fc_threshold": log2fc_threshold,
            "pathway_term": pathway_term,
            "text": text,
        }
    ),
    tool(
        "memory_lookup",
        description="Answer stored list lookup and matching questions from chat memory. Use for intersections between stored pathway or GO overlap genes and stored up-regulated, down-regulated, or combined DEG genes, or for checking whether a named gene is present in a stored pathway, GO term, or DEG set.",
        return_direct=False,
    )(
        lambda pathway_term=None, direction=None, top_n=None, text=None: {
            "pathway_term": pathway_term,
            "direction": direction,
            "top_n": top_n,
            "text": text,
        }
    ),
    tool(
        "primekg_query",
        description="""
        Query PrimeKG using natural language.

        Use for:
        - disease to gene relationships
        - gene to pathway relationships
        - drug to target relationships
        - disease to drug relationships
        - phenotype relationships
        - indirect graph questions that require 2-hop or 3-hop joins
        - "what connects X and Y" style graph exploration

        Input:
            question: Natural language question.

        Returns:
            Answer plus generated Cypher query and raw graph results.
        """,
        return_direct=False,
    )(
        lambda question: {"question": question}
    ),
    tool(
        "opentargets_association",
        description="Query OpenTargets for gene associations. Use for single-gene disease lookup, gene-versus-disease checks, checking stored DEG/up-regulated/down-regulated genes against a disease, or retrieving drugs associated with a gene. Genes are standardized to Ensembl IDs with MyGene before querying OpenTargets.",
        return_direct=False,
    )(lambda gene=None, genes=None, disease=None, disease_name=None: {"gene": gene or "", "genes": list(genes or []), "disease": disease or disease_name or ""}),
    tool(
        "synthesize_technical_response",
        description="Write the final technical summary from the available analysis state.",
        return_direct=False,
    )(lambda user_query, analysis_arm="disease", seed_genes=None, srp_ids=None, disease_name="", deg_analysis=None, rwr_genes=None, graph=None, enrichr=None: synthesize_technical_response(
        user_query=user_query,
        analysis_arm=analysis_arm,
        seed_genes=list(seed_genes or []),
        srp_ids=list(srp_ids or []),
        disease_name=disease_name,
        deg_analysis=_ensure_dict(deg_analysis),
        rwr_genes=list(rwr_genes or []),
        graph=graph if isinstance(graph, nx.Graph) else nx.Graph(),
        enrichr=_ensure_dict(enrichr),
    )),
]


TOOL_EXECUTORS: dict[str, Callable[[AgentState, dict[str, Any]], dict[str, Any]]] = {
    "deg_analysis": lambda state, args: {},
    "pathway": lambda state, args: {},
    "rwr_analysis": lambda state, args: {},
    "literature": lambda state, args: {},
    "identify_disease_from_query": lambda state, args: {},
    "visualize": lambda state, args: {},
    "memory_lookup": lambda state, args: {},
    "primekg_query": lambda state, args: {},
    "opentargets_association": lambda state, args: {},
}
