from __future__ import annotations

import networkx as nx

from gea_agent.tools.types import StringEdge


def build_weighted_graph_from_string_edges(edges: list[StringEdge]) -> nx.Graph:
    """
    Build an undirected weighted graph from STRING edges.
    Edge weight = STRING 'score' (0..1).
    """
    graph = nx.Graph()
    for edge in edges:
        a = edge["preferredName_A"]
        b = edge["preferredName_B"]
        w = float(edge["score"])
        if a == b:
            continue
        if graph.has_edge(a, b):
            # keep max weight if duplicates appear
            graph[a][b]["weight"] = max(graph[a][b].get("weight", 0.0), w)
        else:
            graph.add_edge(a, b, weight=w)
    return graph

