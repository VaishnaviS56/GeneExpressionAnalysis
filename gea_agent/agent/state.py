from __future__ import annotations

from typing import Any, Literal, TypedDict

import networkx as nx

from gea_agent.tools.types import QueryClassification, StringEdge


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]  # {"role": "user"|"assistant", "content": "..."}
    query: str
    classification: QueryClassification

    # technical branch
    genes: list[str]
    string_edges: list[StringEdge]
    graph: nx.Graph
    rwr_genes: list[tuple[str, float]]
    enrichr: dict[str, Any]

    # final
    answer: str
    meta: dict[str, Any]


Route = Literal["general", "technical"]