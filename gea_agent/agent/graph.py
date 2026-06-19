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
from gea_agent.tools.opentargets import check_gene_disease_association
from gea_agent.tools.pyvis_visualizer import build_pyvis_html
from gea_agent.tools.random_walk_restart import top_rwr_genes
from gea_agent.tools.string_local_graph import build_weighted_graph_from_string_files
from gea_agent.tools.synthesizer import synthesize_technical_response


MAX_AGENT_STEPS = 10

TOOL_USE_INSTRUCTIONS = '''
deg_analysis: When the user gives SRP ids, run this tool to perform diferentially expressed genes. These genes can then be used downstream. This tool returns the DEGs and their adjusted p-values.
pathway: Call this tool either directly or in chain when we need to identify pathways for genes. This tool returns the pathways and their adjusted p-values.
rwr_analysis: Call this tool when user want to identify potential targets from gene set, the genes can be provided by user or can be taken from state of graph depending on the query. This tool will build the STRING graph, run RWR, render PyVis, then synthesize the technical result.
literature: Call this tool when user wants to identify the disease context, fetch OpenAlex papers, extract genes, then synthesize the technical result. The extracted genes can be saved in state for further downstream analysis for both RWR and pathways.
identify_disease_from_query: Call this tool when user wants to identify the disease context from the query. This tool will return the disease name which can be used for downstream analysis.
opentargets_association: Call this tool when the user wants to know what disease a specific gene is associated with. This tool queries OpenTargets and returns the association details.

Consider:
- If the query is general and you can answer directly, do not call any specialist.
- If any specialist tool is called, the specialist will finish by synthesizing the technical response.
- You are allowed to chain tools, if you don't have genes to run RWR or pathway analysis, you can first call the literature tool to extract genes from the disease context and then use those genes for downstream analysis.
- Use opentargets_association for gene-disease association questions.
- If the user asks for pathways of top upregulated genes, use pathway and prefer stored DEG genes or DEG gene records.
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
    if result.get("associated") is not None:
        payload["associated"] = bool(result.get("associated"))
    if result.get("association_score") is not None:
        payload["association_score"] = result.get("association_score")
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
                    {"term": term.get("term"), "adjusted_p_value": term.get("adjusted_p_value")}
                    for term in terms[:3]
                    if isinstance(term, dict)
                ]
                for lib, terms in libs.items()
                if isinstance(terms, list)
            }
    if result.get("pyvis_html_path"):
        payload["pyvis_html_path"] = result["pyvis_html_path"]

    return payload or {"keys": sorted(result.keys())}


def _infer_analysis_arm(state: AgentState) -> str:
    arm = str(state.get("analysis_arm") or "").strip().lower()
    if arm in {"general", "srp", "disease", "memory_rwr"}:
        return arm
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
        "disease_name": state.get("disease_name") or "",
        "memory_disease_name": state.get("memory_disease_name") or "",
        "deg_gene_count": len(state.get("deg_genes") or []),
        "openalex_gene_count": len(state.get("openalex_genes") or []),
        "memory_deg_gene_count": len(state.get("memory_deg_genes") or []),
        "rwr_gene_count": len(state.get("rwr_genes") or []),
        "has_graph": bool(isinstance(state.get("graph"), nx.Graph) and state["graph"].number_of_nodes() > 0),
        "graph_summary": _graph_summary(state.get("graph") if isinstance(state.get("graph"), nx.Graph) else None),
        "recent_tools": (state.get("tool_history") or [])[-5:],
    }

    return (
        "You are a Gene Expression Analysis orchestrator.\n"
        "Use the bound specialist tools independently and flexibly.\n"
        "Call one specialist at a time, observe the result, and decide whether another specialist is needed.\n"
        "If the query is general and you can answer directly, do not call any specialist.\n"
        "If a specialist is called, it should contribute to a chain that may continue with later specialists.\n\n"
        "Specialist guidance:\n"
        f"{TOOL_USE_INSTRUCTIONS}\n\n"
        "Available tools and what they do:\n"
        f"{_build_tool_list_text()}\n\n"
        f"Current user query: {query}\n"
        f"Memory summary: {memory_summary}\n"
        f"Current state snapshot: {json.dumps(state_snapshot, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Rules:\n"
        "- Choose only from the specialist tools.\n"
        "- Do not call a specialist if the answer is clearly general chat.\n"
        "- Keep reasoning concise and choose only the next best specialist.\n"
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
    if state.get("memory_disease_name") is not None:
        update["memory_disease_name"] = str(state.get("memory_disease_name") or "")
    if state.get("memory_openalex_genes") is not None:
        update["memory_openalex_genes"] = list(state.get("memory_openalex_genes") or [])
    if state.get("memory_opentargets_results") is not None:
        update["memory_opentargets_results"] = list(state.get("memory_opentargets_results") or [])
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


def _run_identify_disease(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or state.get("query") or "")
    disease_result = identify_disease_from_query(query)
    disease_name = disease_result.get("disease", "") or state.get("memory_disease_name") or ""
    return {"disease_name": disease_name}


def _run_opentargets_association(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    gene = str(
        args.get("gene")
        or state.get("disease_gene")
        or (state.get("genes") or [""])[0]
        or ""
    )
    disease = str(
        args.get("disease")
        or args.get("disease_name")
        or state.get("disease_name")
        or state.get("memory_disease_name")
        or ""
    )
    result = check_gene_disease_association(gene, disease)
    result["analysis_arm"] = "opentargets"
    return result


def _run_fetch_openalex(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    disease_name = str(
        args.get("disease_name")
        or args.get("disease")
        or state.get("disease_name")
        or state.get("memory_disease_name")
        or ""
    )
    top_n = int(args.get("top_n") or 20)
    openalex_result = fetch_openalex_papers_and_genes(disease_name, top_n=top_n)
    genes = openalex_result.get("genes", [])
    return {
        "analysis_arm": "disease",
        "disease_name": openalex_result.get("disease", disease_name),
        "openalex_papers": openalex_result.get("papers", []),
        "openalex_genes": genes,
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
    deg_result = run_deg_r_analysis(srp_ids=srp_ids)
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
    return {
        "analysis_arm": "srp",
        "srp_ids": srp_ids,
        "deg_analysis": deg_result,
        "deg_genes": deg_genes,
        "deg_gene_records": deg_gene_records,
        "genes": _merge_unique(state.get("genes"), deg_genes),
    }


def _run_build_string_graph(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    genes = args.get("genes")
    if not isinstance(genes, list) or not genes:
        deg_records = state.get("deg_gene_records") or state.get("memory_deg_gene_records")
        ranked_deg_genes = _genes_from_deg_records(deg_records)
        genes = _merge_unique(state.get("genes"), state.get("openalex_genes"), ranked_deg_genes, state.get("deg_genes"))
        if str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower() == "memory_rwr":
            genes = _merge_unique(_genes_from_deg_records(state.get("memory_deg_gene_records")), state.get("memory_deg_genes"), genes)
    genes = [str(value).strip().upper() for value in genes if str(value).strip()]
    graph = build_weighted_graph_from_string_files(
        genes=genes,
        info_path=str(args.get("info_path") or SETTINGS.string_info_path),
        links_path=str(args.get("links_path") or SETTINGS.string_links_path),
        required_score=int(args.get("required_score") or SETTINGS.string_required_score),
        mode=str(args.get("mode") or SETTINGS.string_local_mode),
    )
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    update: dict[str, Any] = {"graph": graph, "genes": genes, "rwr_seed_genes": genes}
    if analysis_arm:
        update["analysis_arm"] = analysis_arm
    return update


def _run_top_rwr(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return {"rwr_genes": [], "rwr_seed_genes": list(state.get("genes") or [])}

    seed_genes = args.get("seed_genes")
    if not isinstance(seed_genes, list) or not seed_genes:
        analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
        if analysis_arm == "memory_rwr":
            seed_genes = _genes_from_deg_records(state.get("memory_deg_gene_records"), top_n=20) or list(state.get("memory_deg_genes") or state.get("genes") or [])
        else:
            seed_genes = _genes_from_deg_records(state.get("deg_gene_records") or state.get("memory_deg_gene_records"), top_n=20) or list(state.get("genes") or [])
    seed_genes = [str(value).strip().upper() for value in seed_genes if str(value).strip()]

    rwr = top_rwr_genes(
        graph,
        seed_genes,
        top_k=int(args.get("top_k") or 30),
        restart_prob=float(args.get("restart_prob") or 0.5),
        exclude=args.get("exclude"),
        exclude_hubs=bool(args.get("exclude_hubs", True)),
    )
    update: dict[str, Any] = {"rwr_genes": rwr, "rwr_seed_genes": seed_genes}
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    if analysis_arm in {"disease", "memory_rwr"}:
        update["analysis_arm"] = analysis_arm
    return update


def _run_enrichr(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    genes = args.get("genes")
    if not isinstance(genes, list) or not genes:
        if analysis_arm == "srp":
            genes = _genes_from_deg_records(state.get("deg_gene_records") or state.get("memory_deg_gene_records"), top_n=50) or list(state.get("deg_genes") or [])
        else:
            genes = _merge_unique(
                _genes_from_deg_records(state.get("memory_deg_gene_records"), top_n=50),
                state.get("genes"),
                [gene for gene, _ in (state.get("rwr_genes") or [])],
            )

    background = list((state.get("graph") or nx.Graph()).nodes()) if isinstance(state.get("graph"), nx.Graph) else []
    if analysis_arm == "srp":
        background = list(state.get("deg_genes") or genes)

    return {
        "enrichr": enrichr_pathways(
            genes,
            top_n=int(args.get("top_n") or 10),
            background_genes=background,
        )
    }


def _run_build_pyvis(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return {}
    return {
        "pyvis_html_path": build_pyvis_html(
            graph,
            output_path=str(args.get("output_path") or "pyvis_network.html"),
            select_top_degree=int(args.get("select_top_degree") or 300),
        )
    }


def _run_synthesize(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    answer = synthesize_technical_response(
        user_query=str(args.get("user_query") or state.get("query") or ""),
        analysis_arm=str(args.get("analysis_arm") or state.get("analysis_arm") or _infer_analysis_arm(state)).strip().lower(),
        seed_genes=list(args.get("seed_genes") or state.get("genes") or []),
        srp_ids=list(args.get("srp_ids") or state.get("srp_ids") or []),
        disease_name=str(args.get("disease_name") or state.get("disease_name") or ""),
        deg_analysis=_ensure_dict(state.get("deg_analysis")),
        rwr_genes=list(state.get("rwr_genes") or []),
        graph=graph if isinstance(graph, nx.Graph) else nx.Graph(),
        enrichr=_ensure_dict(state.get("enrichr")),
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
            return {**state, **result, "analysis_arm": "srp"}

        if tool_name == "pathway":
            result = _run_enrichr(state, args)
            update = _specialist_history_update(state, "enrichr_pathways", args, result)
            update = {**update, **result}
            return {**state, **update}

        if tool_name == "rwr_analysis":
            build_result = _run_build_string_graph(state, args)
            update = _specialist_history_update(state, "build_weighted_graph_from_string_files", args, build_result)
            update = {**update, **build_result}
            state = {**state, **update}

            rwr_result = _run_top_rwr(state, args)
            update = _specialist_history_update(state, "top_rwr_genes", args, rwr_result)
            update = {**update, **rwr_result}
            state = {**state, **update}

            pyvis_result = _run_build_pyvis(state, args)
            update = _specialist_history_update(state, "build_pyvis_html", args, pyvis_result)
            update = {**update, **pyvis_result}
            state = {**state, **update}
            return {**state, **rwr_result}

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
            return {**state, **openalex_result}

        if tool_name == "identify_disease_from_query":
            result = _run_identify_disease(state, args)
            update = _specialist_history_update(state, "identify_disease_from_query", args, result)
            update = {**update, **result}
            return {**state, **update}

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
            return {**state, **update}

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
        if state.get("opentargets_result"):
            analysis_arm = "opentargets"
        answer = synthesize_technical_response(
            user_query=str(state.get("query") or ""),
            analysis_arm=analysis_arm,
            seed_genes=list(state.get("genes") or []),
            srp_ids=list(state.get("srp_ids") or []),
            disease_name=str(state.get("disease_name") or ""),
            deg_analysis=_ensure_dict(state.get("deg_analysis")),
            rwr_genes=list(state.get("rwr_genes") or []),
            graph=state.get("graph") if isinstance(state.get("graph"), nx.Graph) else nx.Graph(),
            enrichr=_ensure_dict(state.get("enrichr")),
        )

    analysis_arm = _infer_analysis_arm(state)
    graph = state.get("graph")
    meta = {
        "analysis_arm": analysis_arm,
        "is_followup": bool(state.get("is_followup", False)),
        "route_rationale": state.get("route_rationale", ""),
        "srp_ids": list(state.get("srp_ids") or []),
        "memory_deg_genes": list(state.get("memory_deg_genes") or []),
        "memory_deg_analysis": _ensure_dict(state.get("memory_deg_analysis")),
        "memory_deg_gene_records": list(state.get("memory_deg_gene_records") or []),
        "memory_disease_name": str(state.get("memory_disease_name") or ""),
        "memory_openalex_genes": list(state.get("memory_openalex_genes") or []),
        "memory_opentargets_results": list(state.get("memory_opentargets_results") or []),
        "disease_name": str(state.get("disease_name") or ""),
        "disease_gene": str(state.get("disease_gene") or ""),
        "openalex_papers": list(state.get("openalex_papers") or []),
        "openalex_genes": list(state.get("openalex_genes") or []),
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
        description="Run the DEG R workflow for one or more SRP identifiers.",
        return_direct=False,
    )(lambda srp_ids=None, text=None: run_deg_r_analysis(srp_ids=list(srp_ids or []))),
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
        description="Run pathway enrichment on the current gene set.",
        return_direct=False,
    )(lambda genes, top_n=10, background_genes=None: enrichr_pathways(list(genes or []), top_n=int(top_n), background_genes=list(background_genes or []))),
    tool(
        "build_pyvis_html",
        description="Render the current STRING graph to an interactive PyVis HTML file.",
        return_direct=False,
    )(lambda select_top_degree=300, output_path="pyvis_network.html": {"select_top_degree": select_top_degree, "output_path": output_path}),
    tool(
        "opentargets_association",
        description="Check whether a gene is associated with a disease using the OpenTargets platform.",
        return_direct=False,
    )(lambda gene=None, disease=None, disease_name=None: {"gene": gene or "", "disease": disease or disease_name or ""}),
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
    "opentargets_association": lambda state, args: {},
}
