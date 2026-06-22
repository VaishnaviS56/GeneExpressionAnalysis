from __future__ import annotations

import csv
import os
import re
from typing import Any

from gea_agent.config import SETTINGS


FOCUSED_TYPES = {"drug", "disease", "effect/phenotype", "gene/protein", "pathway"}
TYPE_ALIASES = {
    "gene": "gene/protein",
    "genes": "gene/protein",
    "protein": "gene/protein",
    "proteins": "gene/protein",
    "phenotype": "effect/phenotype",
    "phenotypes": "effect/phenotype",
    "effect": "effect/phenotype",
    "effects": "effect/phenotype",
    "drug": "drug",
    "drugs": "drug",
    "disease": "disease",
    "diseases": "disease",
    "pathway": "pathway",
    "pathways": "pathway",
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _clean_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _clean_types(values: Any) -> set[str]:
    types = set()
    for value in _clean_list(values):
        mapped = TYPE_ALIASES.get(_norm(value), value)
        if mapped in FOCUSED_TYPES:
            types.add(mapped)
    return types


def _node(row: dict[str, str], prefix: str) -> dict[str, str]:
    return {
        "id": row.get(f"{prefix}_id", ""),
        "type": row.get(f"{prefix}_type", ""),
        "name": row.get(f"{prefix}_name", ""),
        "source": row.get(f"{prefix}_source", ""),
    }


def _edge(row: dict[str, str]) -> dict[str, Any]:
    return {
        "relation": row.get("relation", ""),
        "display_relation": row.get("display_relation", ""),
        "source": _node(row, "x"),
        "target": _node(row, "y"),
    }


def _edge_oriented(row: dict[str, str], *, reverse: bool = False) -> dict[str, Any]:
    if not reverse:
        return _edge(row)
    return {
        "relation": row.get("relation", ""),
        "display_relation": row.get("display_relation", ""),
        "source": _node(row, "y"),
        "target": _node(row, "x"),
    }


def _matches_term(name: str, node_id: str, terms: list[str], normalized_terms: set[str]) -> bool:
    if not terms:
        return True
    name_norm = _norm(name)
    id_norm = _norm(node_id)
    return name_norm in normalized_terms or id_norm in normalized_terms


def _matches_text(value: str, filters: list[str]) -> bool:
    if not filters:
        return True
    normalized = _norm(value)
    return any(_norm(item) in normalized for item in filters)


def query_primekg(
    *,
    source_terms: list[str] | None = None,
    target_terms: list[str] | None = None,
    source_types: list[str] | None = None,
    target_types: list[str] | None = None,
    relation_terms: list[str] | None = None,
    limit: int = 50,
    kg_path: str | None = None,
) -> dict[str, Any]:
    path = kg_path or SETTINGS.primekg_csv_path
    if not os.path.exists(path):
        return {
            "status": "missing_file",
            "edges": [],
            "message": f"PrimeKG CSV was not found at {path}.",
        }

    source_terms_clean = _clean_list(source_terms)
    target_terms_clean = _clean_list(target_terms)
    source_norm = {_norm(value) for value in source_terms_clean}
    target_norm = {_norm(value) for value in target_terms_clean}
    source_type_set = _clean_types(source_types)
    target_type_set = _clean_types(target_types)
    relation_filters = _clean_list(relation_terms)
    limit = max(1, min(int(limit or 50), 200))

    edges: list[dict[str, Any]] = []
    scanned = 0
    focused_seen = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned += 1
            x_type = row.get("x_type", "")
            y_type = row.get("y_type", "")
            if x_type not in FOCUSED_TYPES or y_type not in FOCUSED_TYPES:
                continue
            focused_seen += 1
            if not _matches_text(row.get("display_relation", "") or row.get("relation", ""), relation_filters):
                continue

            forward = (
                (not source_type_set or x_type in source_type_set)
                and (not target_type_set or y_type in target_type_set)
                and _matches_term(row.get("x_name", ""), row.get("x_id", ""), source_terms_clean, source_norm)
                and _matches_term(row.get("y_name", ""), row.get("y_id", ""), target_terms_clean, target_norm)
            )
            reverse = (
                (not source_type_set or y_type in source_type_set)
                and (not target_type_set or x_type in target_type_set)
                and _matches_term(row.get("y_name", ""), row.get("y_id", ""), source_terms_clean, source_norm)
                and _matches_term(row.get("x_name", ""), row.get("x_id", ""), target_terms_clean, target_norm)
            )
            if not forward and not reverse:
                continue

            edges.append(_edge_oriented(row, reverse=bool(reverse and not forward)))
            if len(edges) >= limit:
                break

    return {
        "status": "ok",
        "edges": edges,
        "count": len(edges),
        "scanned_rows": scanned,
        "focused_rows_seen": focused_seen,
        "query": {
            "source_terms": source_terms_clean,
            "target_terms": target_terms_clean,
            "source_types": sorted(source_type_set),
            "target_types": sorted(target_type_set),
            "relation_terms": relation_filters,
            "limit": limit,
        },
        "message": f"Found {len(edges)} PrimeKG relationships.",
    }
