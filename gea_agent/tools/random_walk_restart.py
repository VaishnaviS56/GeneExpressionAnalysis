from __future__ import annotations

import math
import random
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
    print("count")
    """Random Walk with Restart (RWR) on an undirected weighted graph."""
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


def identify_hub_genes(graph: nx.Graph, *, percentile: float = 0.99) -> set[str]:
    """Identify hub genes by degree percentile (default: top 1% by degree)."""
    if graph.number_of_nodes() == 0:
        return set()

    degrees = [(str(n), int(d)) for n, d in graph.degree()]
    deg_values = sorted(d for _, d in degrees)
    if not deg_values:
        return set()

    p = float(percentile)
    if p <= 0:
        cutoff = deg_values[0]
    elif p >= 1:
        cutoff = deg_values[-1]
    else:
        idx = max(0, min(len(deg_values) - 1, math.ceil(p * len(deg_values)) - 1))
        cutoff = deg_values[idx]

    return {n for n, d in degrees if d >= cutoff and cutoff > 0}


def _aggregate_scores(score_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not score_dicts:
        return {}
    acc: dict[str, float] = {}
    for scores in score_dicts:
        for gene, score in scores.items():
            acc[gene] = acc.get(gene, 0.0) + float(score)
    k = float(len(score_dicts))
    return {g: s / k for g, s in acc.items()}


def top_rwr_genes(
    graph: nx.Graph,
    seed_genes: list[str],
    *,
    top_k: int = 20,
    restart_prob: float = 0.5,
    exclude: Iterable[str] | None = None,
    exclude_hubs: bool = True,
    hub_percentile: float = 0.99,
    runs: int = 25,
    seed_subset_frac: float = 0.8,
    random_state: int | None = 42,
) -> list[tuple[str, float]]:
    """
    Get top-k genes by RWR score.

    - Excludes seed genes by default.
    - Optionally excludes hub genes (top `hub_percentile` by degree).
    - If `runs` > 1, runs RWR multiple times on random subsets of the seed genes and
      returns genes ranked by mean stationary probability across runs.
    """
    print(f"Running RWR with restart_prob={restart_prob}, top_k={top_k}, runs={runs}")
    exclude_set = set(exclude) if exclude is not None else set(seed_genes)
    if exclude_hubs:
        exclude_set |= identify_hub_genes(graph, percentile=hub_percentile)

    seed_genes = [g for g in seed_genes if g in graph]
    if not seed_genes:
        return []
    
    r = max(1, int(runs))
    if r == 1:
        scores = random_walk_with_restart(graph, seed_genes, restart_prob=restart_prob)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(g, s) for g, s in ranked if g not in exclude_set][:top_k]

    rng = random.Random(random_state)
    score_runs: list[dict[str, float]] = []


    for _ in range(r):
        x=_*0.001
        if len(seed_genes) == 1:
            chosen = seed_genes
        else:
            k = max(1, int(math.ceil(seed_subset_frac * len(seed_genes))))
            chosen = rng.sample(seed_genes, k=min(k, len(seed_genes)))
        score_runs.append(random_walk_with_restart(graph, chosen, restart_prob=restart_prob-x))

    avg_scores = _aggregate_scores(score_runs)
    ranked = sorted(avg_scores.items(), key=lambda kv: kv[1], reverse=True)
    print("Excluded genes:", exclude_set)
    filtered = [(g, s) for g, s in ranked if g not in exclude_set]
    return filtered[:top_k]