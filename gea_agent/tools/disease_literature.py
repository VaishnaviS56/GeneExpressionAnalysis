from __future__ import annotations

import json
from typing import Any

import requests

from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.llm import get_llm


def _safe_parse_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def identify_disease_from_query(query: str) -> dict[str, Any]:
    llm = get_llm()
    resp = llm.invoke(
        [
            ("system", "Extract the disease name from the user query. Return only JSON: {\"disease\":\"...\"}."),
            ("user", query),
        ]
    )
    data = _safe_parse_json(getattr(resp, "content", "") or "") or {}
    disease = str(data.get("disease", "")).strip()
    return {
        "status": "ok" if disease else "not_found",
        "disease": disease,
    }


def _abstract_from_inverted_index(inverted_index: dict[str, list[int]] | None) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""

    positions: dict[int, str] = {}
    for word, indexes in inverted_index.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            try:
                positions[int(index)] = str(word)
            except Exception:
                continue

    return " ".join(positions[index] for index in sorted(positions))


def fetch_openalex_papers_and_genes(
    disease: str,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    disease = " ".join(str(disease).split()).strip()
    if not disease:
        return {
            "status": "no_disease",
            "disease": "",
            "papers": [],
            "genes": [],
        }

    query = f'{disease} homo sapiens gene'
    params = {
        "search": query,
        "per-page": top_n,
    }

    try:
        response = requests.get("https://api.openalex.org/works", params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "status": "request_failed",
            "disease": disease,
            "papers": [],
            "genes": [],
            "message": str(exc),
        }

    results = payload.get("results", [])
    papers: list[dict[str, Any]] = []
    genes: list[str] = []

    for work in results[:top_n]:
        if not isinstance(work, dict):
            continue

        title = str(work.get("display_name", "")).strip()
        abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
        text = f"{title} {abstract}".strip()
        paper_genes = extract_genes_from_text(text)

        for gene in paper_genes:
            if gene not in genes:
                genes.append(gene)

        papers.append(
            {
                "title": title,
                "year": work.get("publication_year"),
                "doi": work.get("doi"),
                "genes": paper_genes[:20],
            }
        )

    return {
        "status": "ok",
        "disease": disease,
        "papers": papers,
        "genes": genes,
    }
