from __future__ import annotations

from typing import Any

import requests

from gea_agent.config import SETTINGS
from gea_agent.tools.types import StringEdge


def fetch_string_network_edges(
    genes: list[str],
    *,
    species: int | None = None,
    required_score: int | None = None,
    limit: int | None = None,
    timeout_s: float = 30.0,
) -> list[StringEdge]:
    """
    Fetch interaction edges from STRING for Homo sapiens with high confidence.

    - species: NCBI taxonomy id, default 9606 (Homo sapiens)
    - required_score: STRING score threshold, default 700 (== 0.700)
    """
    if not genes:
        return []

    species = species if species is not None else SETTINGS.string_species
    required_score = (
        required_score if required_score is not None else SETTINGS.string_required_score
    )
    limit = limit if limit is not None else SETTINGS.string_limit

    url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "%0d".join(genes),  # STRING expects CR-delimited
        "species": species,
        "required_score": required_score,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    payload: list[dict[str, Any]] = resp.json()

    edges: list[StringEdge] = []
    for item in payload:
        a = item.get("preferredName_A")
        b = item.get("preferredName_B")
        score = item.get("score")
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if not isinstance(score, (int, float)):
            continue
        edges.append(
            {
                "preferredName_A": a,
                "preferredName_B": b,
                "score": float(score),
            }
        )
    return edges

