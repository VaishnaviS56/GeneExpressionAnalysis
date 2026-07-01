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
    """Random Walk with Restart (RWR) on an undirected weighted graph."""
    if graph.number_of_nodes() == 0 or not seed_genes:
        return {}
    restart_prob = min(max(float(restart_prob), 0.0), 1.0)
    max_iter = max(1, int(max_iter))
    tol = max(float(tol), 0.0)

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


def permutation_pvalues(
    graph: nx.Graph,
    *,
    seed_genes: list[str],
    candidate_genes: list[str],
    restart_prob: float,
    permutations: int = 1000,
    random_state: int | None = 42,
    exclude_hubs_from_sampling: bool = True,
    hub_percentile: float = 0.99,
    compare_seed_stat: str = "mean",
) -> dict[str, float]:
    """
    Permutation test described as:
      p-value(prot) = Y / permutations
    where Y is the number of RWR runs (with random seed sets) in which
    the probability of the candidate protein is higher than that of the actual seed nodes.

    Implementation detail:
    - Compute baseline from the *actual* RWR run using the real `seed_genes`.
      baseline_seed_score is the mean (or max) of the stationary probabilities of seed genes.
    - For each permutation: run RWR with random seeds (same count as seed_genes),
      and count if score_perm(candidate) > baseline_seed_score.
    """
    seed_genes = [g for g in seed_genes if g in graph]
    candidate_genes = [g for g in candidate_genes if g in graph]
    candidate_genes = list(dict.fromkeys(candidate_genes))

    if not seed_genes or not candidate_genes:
        return {g: 1.0 for g in candidate_genes}

    actual_scores = random_walk_with_restart(graph, seed_genes, restart_prob=restart_prob)
    seed_vals = [actual_scores.get(g, 0.0) for g in seed_genes]

    if compare_seed_stat == "max":
        baseline = max(seed_vals) if seed_vals else 0.0
    else:
        baseline = (sum(seed_vals) / len(seed_vals)) if seed_vals else 0.0

    hubs = identify_hub_genes(graph, percentile=hub_percentile) if exclude_hubs_from_sampling else set()
    population = [n for n in graph.nodes() if n not in hubs and n not in seed_genes]

    if len(population) < len(seed_genes):
        population = [n for n in graph.nodes() if n not in seed_genes]

    rng = random.Random(random_state)

    Y: dict[str, int] = {g: 0 for g in candidate_genes}
    k = len(seed_genes)

    for _ in range(int(permutations)):
        perm_seeds = rng.sample(population, k=min(k, len(population)))
        perm_scores = random_walk_with_restart(graph, perm_seeds, restart_prob=restart_prob)

        for g in candidate_genes:
            if perm_scores.get(g, 0.0) > baseline:
                Y[g] += 1

    return {g: (Y[g] / float(permutations)) for g in candidate_genes}


def top_rwr_genes(
    graph: nx.Graph,
    seed_genes: list[str],
    *,
    top_k: int = 20,
    restart_prob: float = 0.3,
    exclude: Iterable[str] | None = None,
    exclude_hubs: bool = True,
    hub_percentile: float = 0.99,
    runs: int = 1,
    seed_subset_frac: float = 0.8,
    random_state: int | None = 42,
    permutation_test: bool = True,
    permutations: int = 100,
    alpha: float = 0.05,
) -> list[tuple[str, float]]:
    """
    Get top-k genes by RWR score.

    - Excludes seed genes by default.
    - Optionally excludes hub genes.
    - If `runs` > 1, runs RWR multiple times on random subsets of the seed genes and
      ranks genes by mean stationary probability across runs.
    - If `permutation_test` is True, filters candidates to those with p-value < alpha
      using the permutation test described in the prompt.
    """
    top_k = max(0, int(top_k))
    if top_k == 0:
        return []
    restart_prob = min(max(float(restart_prob), 0.0), 1.0)
    alpha = min(max(float(alpha), 0.0), 1.0)
    permutations = max(1, int(permutations))
    exclude_set = set(exclude) if exclude is not None else set(seed_genes)
    if exclude_hubs:
        exclude_set |= identify_hub_genes(graph, percentile=hub_percentile)

    seed_genes = [g for g in seed_genes if g in graph]
    if not seed_genes:
        return []
    print(seed_genes)
    r = max(1, int(runs))
    if r == 1:
        scores = random_walk_with_restart(graph, seed_genes, restart_prob=restart_prob)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [(g, s) for g, s in ranked if g not in exclude_set]
    else:
        rng = random.Random(random_state)
        score_runs: list[dict[str, float]] = []

        for _ in range(r):
            if len(seed_genes) == 1:
                chosen = seed_genes
            else:
                k = max(1, int(math.ceil(seed_subset_frac * len(seed_genes))))
                chosen = rng.sample(seed_genes, k=min(k, len(seed_genes)))
            score_runs.append(random_walk_with_restart(graph, chosen, restart_prob=restart_prob))

        avg_scores = _aggregate_scores(score_runs)
        ranked = sorted(avg_scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [(g, s) for g, s in ranked if g not in exclude_set]

    candidates = candidates[: max(top_k * 5, top_k)]

    if not permutation_test:
        return candidates[:top_k]

    candidate_genes = [g for g, _ in candidates]
    pvals = permutation_pvalues(
        graph,
        seed_genes=seed_genes,
        candidate_genes=candidate_genes,
        restart_prob=restart_prob,
        permutations=permutations,
        random_state=random_state,
        exclude_hubs_from_sampling=exclude_hubs,
        hub_percentile=hub_percentile,
    )

    kept = [(g, s) for g, s in candidates if pvals.get(g, 1.0) < alpha]
    print(kept)
    return kept[:top_k]
