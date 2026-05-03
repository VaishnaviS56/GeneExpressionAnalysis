from __future__ import annotations

from collections.abc import Iterable

import networkx as nx


def random_walk_with_restart(
    graph: nx.Graph,
    seed_genes: list[str],
    *,
    restart_prob: float = 0.12,
    max_iter: int = 100,
    tol: float = 1e-10,
    weight: str = "weight",
) -> dict[str, float]:
    """
    Random Walk with Restart (RWR) on an undirected weighted graph.

    Returns stationary probabilities for each node.
    """
    if graph.number_of_nodes() == 0 or not seed_genes:
        return {}

    nodes = list(graph.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    seeds = [g for g in seed_genes if g in node_index]
    if not seeds:
        return {}

    n = len(nodes)
    p0 = [0.0] * n
    for g in seeds:
        p0[node_index[g]] = 1.0 / len(seeds)

    # Precompute row-normalized transition probabilities
    nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u in nodes:
        ui = node_index[u]
        total = 0.0
        edges: list[tuple[int, float]] = []
        for v, attrs in graph[u].items():
            vi = node_index[v]
            w = float(attrs.get(weight, 1.0))
            if w <= 0:
                continue
            edges.append((vi, w))
            total += w
        if total > 0:
            nbrs[ui] = [(vi, w / total) for vi, w in edges]

    p = p0[:]
    for _ in range(max_iter):
        p_next = [restart_prob * p0[i] for i in range(n)]
        for i in range(n):
            if not nbrs[i]:
                continue
            spread = (1.0 - restart_prob) * p[i]
            if spread == 0:
                continue
            for j, prob in nbrs[i]:
                p_next[j] += spread * prob

        delta = sum(abs(p_next[i] - p[i]) for i in range(n))
        p = p_next
        if delta < tol:
            break

    return {nodes[i]: float(p[i]) for i in range(n)}


def top_rwr_genes(
    graph: nx.Graph,
    seed_genes: list[str],
    *,
    top_k: int = 5,
    restart_prob: float = 0.5,
    exclude: Iterable[str] | None = None,
) -> list[tuple[str, float]]:
    """Get top-k genes by RWR score, excluding seed genes by default."""
    exclude_set = set(exclude) if exclude is not None else set(seed_genes)
    scores = random_walk_with_restart(graph, seed_genes, restart_prob=restart_prob)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    filtered = [(g, s) for g, s in ranked if g not in exclude_set]
    return filtered[:top_k]