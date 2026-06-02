from __future__ import annotations

import os
import pickle
from functools import lru_cache

import networkx as nx

from gea_agent.tools.string_local_graph import load_string_id_to_gene


def build_full_string_graph_from_files(
    *,
    info_path: str,
    links_path: str,
    required_score: int = 700,
) -> nx.Graph:
    """Build the full weighted STRING graph from local downloads.

    - `info_path` is a TSV with `#string_protein_id` and `preferred_name`
    - `links_path` is space-separated with `protein1 protein2 combined_score`

    Edge weight = combined_score / 1000.0
    """
    id_to_gene = load_string_id_to_gene(info_path)

    graph = nx.Graph()
    with open(links_path, "r", encoding="utf-8", newline="") as f:
        _ = f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            p1, p2, s = parts[0], parts[1], parts[2]
            try:
                score = int(s)
            except Exception:
                continue
            if score < required_score:
                continue

            a = id_to_gene.get(p1, p1)
            b = id_to_gene.get(p2, p2)
            if a == b:
                continue
            w = score / 1000.0
            if graph.has_edge(a, b):
                graph[a][b]["weight"] = max(graph[a][b].get("weight", 0.0), w)
            else:
                graph.add_edge(a, b, weight=w)

    return graph


def save_graph_pickle(graph: nx.Graph, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_pickle(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, nx.Graph):
        raise TypeError("Cached object is not a networkx.Graph")
    return obj


@lru_cache(maxsize=4)
def load_or_build_full_string_graph(
    *,
    info_path: str,
    links_path: str,
    required_score: int,
    cache_path: str,
    force_rebuild: bool = False,
) -> nx.Graph:
    """Load cached full graph from disk, or build and save it."""
    if (not force_rebuild) and cache_path and os.path.exists(cache_path):
        return load_graph_pickle(cache_path)

    graph = build_full_string_graph_from_files(
        info_path=info_path,
        links_path=links_path,
        required_score=required_score,
    )

    if cache_path:
        save_graph_pickle(graph, cache_path)

    return graph