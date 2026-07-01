from __future__ import annotations

import csv
import pickle
from functools import lru_cache
from pathlib import Path

import networkx as nx

from gea_agent.config import SETTINGS


def _resolve_path(raw_path: str) -> Path:
    return Path(SETTINGS.resolve_path(raw_path)).resolve()


@lru_cache(maxsize=4)
def load_string_id_to_gene(info_path: str) -> dict[str, str]:
    """Load mapping: #string_protein_id -> preferred_name from STRING protein.info TSV."""
    mapping: dict[str, str] = {}
    resolved = _resolve_path(info_path)
    with resolved.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = (row.get("#string_protein_id") or "").strip()
            gene = (row.get("preferred_name") or "").strip().upper()
            if pid and gene:
                mapping[pid] = gene
    return mapping


@lru_cache(maxsize=4)
def load_gene_to_string_id(info_path: str) -> dict[str, str]:
    """Reverse mapping: preferred_name -> #string_protein_id (first occurrence wins)."""
    out: dict[str, str] = {}
    resolved = _resolve_path(info_path)
    with resolved.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = (row.get("#string_protein_id") or "").strip()
            gene = (row.get("preferred_name") or "").strip().upper()
            if pid and gene and gene not in out:
                out[gene] = pid
    return out


def _load_cached_full_graph(cache_path: Path, *, required_score: int) -> nx.Graph | None:
    if SETTINGS.string_force_rebuild or not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if int(payload.get("required_score", -1)) != int(required_score):
        return None

    graph = payload.get("graph")
    return graph if isinstance(graph, nx.Graph) else None


def _write_cached_full_graph(cache_path: Path, graph: nx.Graph, *, required_score: int) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump({"required_score": int(required_score), "graph": graph}, handle)
    except Exception:
        return


def build_weighted_graph_from_string_files(
    *,
    genes: list[str],
    info_path: str,
    links_path: str,
    required_score: int = 700,
    mode: str = "seed_1hop_closed",
) -> nx.Graph:
    """
    Build a weighted NetworkX graph from local STRING downloads.

    Files:
    - protein.info: #string_protein_id, preferred_name, ... (TSV)
    - protein.links: protein1, protein2, combined_score (space-separated)

    Weight is combined_score / 1000.0.

    mode:
    - seed_1hop: include edges where either endpoint is a seed protein
    - seed_1hop_closed: seed_1hop plus add edges among discovered nodes (2nd pass)
    - full: include all edges and then return the seed-induced subgraph
    """
    mode = str(mode or "seed_1hop_closed").strip().lower()
    genes = [g.strip().upper() for g in genes if g and g.strip()]
    genes = list(dict.fromkeys(genes))
    graph = nx.Graph()
    if not genes:
        return graph

    info_resolved = _resolve_path(info_path)
    links_resolved = _resolve_path(links_path)
    if not info_resolved.exists() or not links_resolved.exists():
        return graph

    id_to_gene = load_string_id_to_gene(str(info_resolved))
    gene_to_id = load_gene_to_string_id(str(info_resolved))

    seed_ids = {gene_to_id.get(g) for g in genes}
    seed_ids.discard(None)
    seed_ids = {x for x in seed_ids if isinstance(x, str) and x}
    if not seed_ids:
        return graph

    discovered: set[str] = set(seed_ids)

    def add_edge(target_graph: nx.Graph, p1: str, p2: str, score: int) -> None:
        a = id_to_gene.get(p1, p1)
        b = id_to_gene.get(p2, p2)
        if a == b:
            return
        w = score / 1000.0
        if target_graph.has_edge(a, b):
            target_graph[a][b]["weight"] = max(target_graph[a][b].get("weight", 0.0), w)
        else:
            target_graph.add_edge(a, b, weight=w)

    if mode == "full":
        cache_path = Path(SETTINGS.string_graph_cache_path)
        cached = _load_cached_full_graph(cache_path, required_score=required_score)
        if cached is None:
            cached = nx.Graph()
            with links_resolved.open("r", encoding="utf-8", newline="") as f:
                _ = f.readline()
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
                    add_edge(cached, p1, p2, score)
            _write_cached_full_graph(cache_path, cached, required_score=required_score)

        seed_nodes = {id_to_gene.get(pid, pid) for pid in seed_ids}
        keep = set(seed_nodes)
        for node in list(seed_nodes):
            if node in cached:
                keep.update(cached.neighbors(node))
        return cached.subgraph(keep).copy() if keep else nx.Graph()

    with links_resolved.open("r", encoding="utf-8", newline="") as f:
        _ = f.readline()
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

            if p1 in seed_ids or p2 in seed_ids:
                add_edge(graph, p1, p2, score)
                discovered.add(p1)
                discovered.add(p2)

    if mode == "seed_1hop_closed" and len(discovered) > 1:
        with links_resolved.open("r", encoding="utf-8", newline="") as f:
            _ = f.readline()
            for line in f:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                p1, p2, s = parts[0], parts[1], parts[2]
                if p1 not in discovered or p2 not in discovered:
                    continue
                try:
                    score = int(s)
                except Exception:
                    continue
                if score < required_score:
                    continue
                add_edge(graph, p1, p2, score)

    return graph
