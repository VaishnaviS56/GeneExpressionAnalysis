from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

import networkx as nx
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from gea_agent.tools.types import QueryClassification


class AgentState(TypedDict, total=False):
    # messages: list[dict[str, str]]  # {"role": "user"|"assistant", "content": "..."}
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    classification: QueryClassification
    memory_summary: str
    is_followup: bool
    route_rationale: str
    step_count: int
    planner_action: dict[str, Any]
    tool_name: str
    tool_args: dict[str, Any]
    tool_history: list[dict[str, Any]]
    should_finalize: bool

    # technical routing
    analysis_arm: str
    srp_ids: list[str]
    control_name: str
    test_name: str
    memory_deg_genes: list[str]
    memory_upregulated_genes: list[str]
    memory_downregulated_genes: list[str]
    memory_deg_analysis: dict[str, Any]
    memory_deg_gene_records: list[dict[str, Any]]
    memory_srp_metadata_result: dict[str, Any]
    memory_control_name: str
    memory_test_name: str
    memory_enrichr: dict[str, Any]
    memory_rwr_seed_genes: list[str]
    memory_rwr_genes: list[tuple[str, float]]
    memory_disease_name: str
    memory_openalex_genes: list[str]
    memory_opentargets_results: list[dict[str, Any]]
    memory_l1000cds2_result: dict[str, Any]
    memory_pubchem_result: dict[str, Any]
    memory_hypothesis_result: dict[str, Any]
    memory_druggability_result: dict[str, Any]
    memory_pdb_visualization_result: dict[str, Any]

    # disease literature branch
    disease_name: str
    disease_gene: str
    primekg_result: dict[str, Any]
    opentargets_result: dict[str, Any]
    memory_lookup_result: dict[str, Any]
    state_lookup_result: dict[str, Any]
    memory_slice_result: dict[str, Any]
    openalex_papers: list[dict[str, Any]]
    openalex_genes: list[str]
    ranked_openalex_papers: list[dict[str, Any]]
    literature_key_points: list[dict[str, Any]]
    literature_references: list[dict[str, Any]]
    literature_summary: str
    literature_source_status: dict[str, Any]
    literature_query: str

    # DEG branch
    deg_analysis: dict[str, Any]
    deg_genes: list[str]
    upregulated_genes: list[str]
    downregulated_genes: list[str]
    deg_gene_records: list[dict[str, Any]]
    srp_metadata_result: dict[str, Any]

    # technical branch
    genes: list[str]
    rwr_seed_genes: list[str]
    graph: nx.Graph
    rwr_genes: list[tuple[str, float]]
    enrichr: dict[str, Any]
    pyvis_html_path: str
    kegg_pathway_path: str
    volcano_plot_path: str
    visualization_result: dict[str, Any]
    l1000cds2_result: dict[str, Any]
    pubchem_result: dict[str, Any]
    hypothesis_result: dict[str, Any]
    druggability_result: dict[str, Any]
    pdb_visualization_result: dict[str, Any]

    # final
    answer: str
    meta: dict[str, Any]


Route = Literal["general", "srp", "disease", "memory_rwr"]
