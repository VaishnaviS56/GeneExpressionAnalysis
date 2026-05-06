from __future__ import annotations

import csv

import networkx as nx


def load_string_id_to_gene(info_path: str) -> dict[str, str]:
    """Load mapping: #string_protein_id -> preferred_name from STRING protein.info TSV."""
    mapping = {}
    with open(info_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = (row.get("#string_protein_id") or "").strip()
            gene = (row.get("preferred_name") or "").strip()
            if pid and gene:
                mapping[pid] = gene
    return mapping


def load_gene_to_string_id(info_path: str) -> dict[str, str]:
    """Reverse mapping: preferred_name -> #string_protein_id (first occurrence wins)."""
    out = {}
    with open(info_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = (row.get("#string_protein_id") or "").strip()
            gene = (row.get("preferred_name") or "").strip()
            if pid and gene and gene not in out:
                out[gene] = pid
    return out


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
    - full: include all edges (can be huge; not recommended)
    """
    genes = [g.strip().upper() for g in genes if g and g.strip()]
    genes = list(dict.fromkeys(genes))
    graph = nx.Graph()
    if not genes:
        return graph

    id_to_gene = load_string_id_to_gene(info_path)
    gene_to_id = load_gene_to_string_id(info_path)

    seed_ids = {gene_to_id.get(g) for g in genes}
    seed_ids.discard(None)
    seed_ids = {x for x in seed_ids if isinstance(x, str) and x}
    if not seed_ids:
        return graph

    discovered: set[str] = set(seed_ids)

    def add_edge(p1: str, p2: str, score: int):
        a = id_to_gene.get(p1, p1)
        b = id_to_gene.get(p2, p2)
        if a == b:
            return
        w = score / 1000.0
        if graph.has_edge(a, b):
            graph[a][b]["weight"] = max(graph[a][b].get("weight", 0.0), w)
        else:
            graph.add_edge(a, b, weight=w)

    # Pass 1
    with open(links_path, "r", encoding="utf-8", newline="") as f:
        header = f.readline()  # consume header
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

            if mode == "full":
                add_edge(p1, p2, score)
                continue

            if p1 in seed_ids or p2 in seed_ids:
                add_edge(p1, p2, score)
                discovered.add(p1)
                discovered.add(p2)

    # Pass 2 (close over discovered nodes)
    if mode == "seed_1hop_closed" and len(discovered) > 1:
        with open(links_path, "r", encoding="utf-8", newline="") as f:
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
                add_edge(p1, p2, score)

    return graph