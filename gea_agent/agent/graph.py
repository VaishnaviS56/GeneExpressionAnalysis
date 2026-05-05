from __future__ import annotations

from typing import cast

import networkx as nx
from langgraph.graph import END, START, StateGraph

from gea_agent.agent.state import AgentState, Route
from gea_agent.config import SETTINGS
from gea_agent.tools.classify_query import classify_query
from gea_agent.tools.enrichr import enrichr_pathways
from gea_agent.tools.llm import get_llm
from gea_agent.tools.random_walk_restart import top_rwr_genes
from gea_agent.tools.string_local_graph import build_weighted_graph_from_string_files
from gea_agent.tools.synthesizer import synthesize_technical_response


def _route(state: AgentState) -> Route:
    classification = state.get("classification")
    if not classification:
        return "general"
    return cast(Route, "technical" if classification["kind"] == "technical" else "general")


def node_classify(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    classification = classify_query(query)
    return {"classification": classification, "genes": classification.get("genes", [])}


def node_general_answer(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    llm = get_llm()
    resp = llm.invoke(
        [
            ("system", "You are a helpful assistant. Answer the user clearly and concisely."),
            ("user", query),
        ]
    )
    return {"answer": getattr(resp, "content", "") or ""}


def node_fetch_string(state: AgentState) -> AgentState:
    """Build STRING graph from downloaded local files."""
    genes = state.get("genes") or []
    graph = build_weighted_graph_from_string_files(
        genes=genes,
        info_path=SETTINGS.string_info_path,
        links_path=SETTINGS.string_links_path,
        required_score=SETTINGS.string_required_score,
        mode=SETTINGS.string_local_mode,
    )
    return {"graph": graph}


def _graph_summary(graph: nx.Graph) -> dict[str, object]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    top = [{"gene": g, "degree": int(d)} for g, d in degrees[:10]]
    return {"nodes": n, "edges": m, "top_degree": top}


def node_rwr(state: AgentState) -> AgentState:
    graph = state.get("graph") or nx.Graph()
    genes = state.get("genes") or []
    rwr = top_rwr_genes(graph, genes, top_k=20, restart_prob=0.5)
    return {"rwr_genes": rwr}


def node_enrichr(state: AgentState) -> AgentState:
    genes = state.get("genes") or []
    rwr = state.get("rwr_genes") or []
    expanded = genes + [g for g, _ in rwr]
    results = enrichr_pathways(
        expanded,
        top_n=10,
        background_genes=list((state.get("graph") or nx.Graph()).nodes()),
    )
    return {"enrichr": results}


def node_synthesize(state: AgentState) -> AgentState:
    query = state.get("query") or ""
    genes = state.get("genes") or []
    graph = state.get("graph") or nx.Graph()
    rwr = state.get("rwr_genes") or []
    enrichr = state.get("enrichr") or {}

    summary = _graph_summary(graph)
    answer = synthesize_technical_response(
        user_query=query,
        seed_genes=genes,
        rwr_genes=rwr,
        graph=graph,
        enrichr=enrichr,
    )

    meta = {
        "network": summary,
        "rwr_genes": rwr,
        "enrichr": enrichr,
    }
    return {"answer": answer, "meta": meta}


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("classify", node_classify)
    graph.add_node("general_answer", node_general_answer)

    graph.add_node("fetch_string", node_fetch_string)
    graph.add_node("rwr", node_rwr)
    graph.add_node("enrichr", node_enrichr)
    graph.add_node("synthesize", node_synthesize)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _route,
        {
            "general": "general_answer",
            "technical": "fetch_string",
        },
    )

    graph.add_edge("general_answer", END)

    graph.add_edge("fetch_string", "rwr")
    graph.add_edge("rwr", "enrichr")
    graph.add_edge("enrichr", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()