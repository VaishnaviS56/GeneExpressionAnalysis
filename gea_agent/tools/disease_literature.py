from __future__ import annotations

import json
from typing import Any

from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.config import SETTINGS
from gea_agent.tools.http_utils import get_retrying_session
from gea_agent.tools.llm import get_llm


def _safe_parse_json(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def identify_disease_from_query(query: str) -> dict[str, Any]:
    try:
        llm = get_llm()
        resp = llm.invoke(
            [
                (
                    "system",
                    "You are a normalization step inside a biomedical agent workflow. "
                    "Extract the single main disease or condition that should drive downstream analysis. "
                    "Return strict JSON only: {\"disease\":\"...\"}. "
                    "If no disease or condition is clearly present, return {\"disease\":\"\"}. "
                    "Do not include explanations, reasoning, or extra keys.",
                ),
                ("user", query),
            ]
        )
        data = _safe_parse_json(getattr(resp, "content", "") or "") or {}
        disease = str(data.get("disease", "")).strip()
    except Exception as exc:
        return {
            "status": "error",
            "disease": "",
            "message": f"Disease extraction failed: {exc}",
        }
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


def _extract_literature_evidence(
    *,
    user_query: str,
    disease: str,
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    if not papers:
        return {"key_points": [], "references": []}

    compact_papers: list[dict[str, Any]] = []
    for index, paper in enumerate(papers[:8], start=1):
        if not isinstance(paper, dict):
            continue
        compact_papers.append(
            {
                "id": index,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "abstract": paper.get("abstract"),
                "genes": paper.get("genes", []),
            }
        )

    if not compact_papers:
        return {"key_points": [], "references": []}

    llm = get_llm()
    try:
        response = llm.invoke(
            [
                (
                    "system",
                    "You are an evidence-extraction specialist inside a biomedical agent workflow. "
                    "Use only the provided titles and abstracts. "
                    "Extract only findings that directly help answer the user's query and can be tied to specific papers. "
                    "Prefer disease mechanisms, gene associations, biomarkers, pathways, perturbation effects, and clinically relevant observations when present. "
                    "Return strict JSON with keys `key_points` and `references`. "
                    "`key_points` must be a list of objects with `point` and `paper_ids`. "
                    "`references` must be a list of objects with `paper_id`, `title`, `year`, and `doi`. "
                    "Each point must be concise, factual, non-duplicative, and traceable to the cited paper ids. "
                    "Do not speculate or include unsupported claims.",
                ),
                (
                    "user",
                    json.dumps(
                        {
                            "query": user_query,
                            "disease": disease,
                            "papers": compact_papers,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
    except Exception:
        return {"key_points": [], "references": []}
    parsed = _safe_parse_json(getattr(response, "content", "") or "") or {}
    key_points = parsed.get("key_points", [])
    references = parsed.get("references", [])

    if not isinstance(key_points, list):
        key_points = []
    if not isinstance(references, list):
        references = []

    return {
        "key_points": [row for row in key_points[:8] if isinstance(row, dict)],
        "references": [row for row in references[:8] if isinstance(row, dict)],
    }


def _rank_literature_papers(
    *,
    user_query: str,
    disease: str,
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not papers:
        return []

    compact_papers: list[dict[str, Any]] = []
    for index, paper in enumerate(papers[:12], start=1):
        if not isinstance(paper, dict):
            continue
        compact_papers.append(
            {
                "id": index,
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "abstract": paper.get("abstract"),
                "genes": paper.get("genes", []),
            }
        )

    if not compact_papers:
        return []

    llm = get_llm()
    try:
        response = llm.invoke(
            [
                (
                    "system",
                    "You are a retrieval-ranking specialist inside a biomedical agent workflow. "
                    "Rank the papers by how useful they are for answering the user's question next. "
                    "Use only the provided titles and abstracts. "
                    "Prefer direct relevance to the disease, genes, pathways, phenotype, comparison, mechanism, or treatment context mentioned in the query. "
                    "Return strict JSON with key `ranked_papers`, where each item has `paper_id`, `relevance`, and `reason`. "
                    "Use a relevance score from 0 to 100. "
                    "Keep each reason short, concrete, and comparative.",
                ),
                (
                    "user",
                    json.dumps(
                        {
                            "query": user_query,
                            "disease": disease,
                            "papers": compact_papers,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
    except Exception:
        return compact_papers[:8]
    parsed = _safe_parse_json(getattr(response, "content", "") or "") or {}
    ranked_rows = parsed.get("ranked_papers", [])
    if not isinstance(ranked_rows, list):
        return compact_papers[:8]

    by_id = {paper["id"]: paper for paper in compact_papers}
    ranked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in ranked_rows:
        if not isinstance(row, dict):
            continue
        try:
            paper_id = int(row.get("paper_id"))
        except Exception:
            continue
        paper = by_id.get(paper_id)
        if not paper or paper_id in seen:
            continue
        seen.add(paper_id)
        ranked.append(
            {
                **paper,
                "relevance": row.get("relevance"),
                "reason": row.get("reason"),
            }
        )

    for paper in compact_papers:
        paper_id = int(paper.get("id", 0) or 0)
        if paper_id and paper_id not in seen:
            ranked.append(paper)

    return ranked[:8]


def _summarize_literature_answer(
    *,
    user_query: str,
    disease: str,
    ranked_papers: list[dict[str, Any]],
    key_points: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> str:
    if not ranked_papers:
        return ""

    llm = get_llm()
    try:
        response = llm.invoke(
            [
                (
                    "system",
                    "You are the literature-synthesis specialist inside a biomedical agent workflow. "
                    "Read the provided paper titles and abstracts and answer the user's question directly. "
                    "Use only paper-supported claims from the provided papers, key points, and references. "
                    "Lead with the most decision-relevant literature findings, then add concise supporting context. "
                    "If evidence is mixed, weak, or sparse, say so clearly. "
                    "Do not mention retrieval steps, ranking steps, or internal reasoning. "
                    "End with a `References:` section listing the cited papers by title, year, and DOI when available.",
                ),
                (
                    "user",
                    json.dumps(
                        {
                            "query": user_query,
                            "disease": disease,
                            "ranked_papers": ranked_papers[:6],
                            "key_points": key_points[:8],
                            "references": references[:8],
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        return str(getattr(response, "content", "") or "").strip()
    except Exception:
        if key_points:
            lines = [str(row.get("point") or "").strip() for row in key_points if isinstance(row, dict)]
            lines = [line for line in lines if line]
            return " ".join(lines[:3])
        return ""


def fetch_openalex_papers_and_genes(
    disease: str,
    *,
    top_n: int = 20,
    user_query: str = "",
) -> dict[str, Any]:
    disease = " ".join(str(disease).split()).strip()
    if not disease:
        return {
            "status": "no_disease",
            "disease": "",
            "papers": [],
            "genes": [],
            "key_points": [],
            "references": [],
        }

    query = f'{disease} homo sapiens gene'
    params = {
        "search": query,
        "per-page": top_n,
    }

    try:
        response = get_retrying_session().get(
            "https://api.openalex.org/works",
            params=params,
            timeout=SETTINGS.http_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "status": "request_failed",
            "disease": disease,
            "papers": [],
            "genes": [],
            "key_points": [],
            "references": [],
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
                "abstract": abstract,
                "genes": paper_genes[:20],
            }
        )

    ranked_papers = _rank_literature_papers(
        user_query=user_query or disease,
        disease=disease,
        papers=papers,
    )
    evidence = _extract_literature_evidence(
        user_query=user_query or disease,
        disease=disease,
        papers=ranked_papers or papers,
    )
    literature_summary = _summarize_literature_answer(
        user_query=user_query or disease,
        disease=disease,
        ranked_papers=ranked_papers or papers,
        key_points=evidence.get("key_points", []),
        references=evidence.get("references", []),
    )

    return {
        "status": "ok",
        "disease": disease,
        "papers": papers,
        "ranked_papers": ranked_papers,
        "genes": genes,
        "key_points": evidence.get("key_points", []),
        "references": evidence.get("references", []),
        "literature_summary": literature_summary,
    }
