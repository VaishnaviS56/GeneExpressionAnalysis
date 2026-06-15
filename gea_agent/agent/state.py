from __future__ import annotations

from typing import Any, Literal, TypedDict

import networkx as nx

from gea_agent.tools.types import QueryClassification


class AgentState(TypedDict, total=False):
    # messages: list[dict[str, str]]  # {"role": "user"|"assistant", "content": "..."}
    query: str
    classification: QueryClassification

    # disease literature branch
    disease_name: str
    openalex_papers: list[dict[str, Any]]
    openalex_genes: list[str]

    # DEG branch
    deg_analysis: dict[str, Any]
    deg_genes: list[str]

    # technical branch
    genes: list[str]
    rwr_seed_genes: list[str]
    graph: nx.Graph
    rwr_genes: list[tuple[str, float]]
    enrichr: dict[str, Any]
    pyvis_html_path: str

    # final
    answer: str
    meta: dict[str, Any]


Route = Literal["general", "technical"]
