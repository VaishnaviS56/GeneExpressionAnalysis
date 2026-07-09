from __future__ import annotations

import json
import re
from html import unescape
from typing import Any
from xml.etree import ElementTree

from gea_agent.config import SETTINGS
from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.http_utils import get_retrying_session
from gea_agent.tools.llm import get_llm, parse_json_object


def _safe_parse_json(text: str) -> dict[str, Any] | None:
    data = parse_json_object(text)
    return data or None


def identify_disease_from_query(query: str) -> dict[str, Any]:
    try:
        llm = get_llm()
        resp = llm.invoke(
            [
                (
                    "system",
                    "You are a normalization step inside a biomedical agent workflow. "
                    "Extract the single main disease or condition that should drive downstream analysis. "
                    "Return exactly one JSON object with this schema: {\"disease\":\"...\"}. "
                    "Do not use markdown, code fences, commentary, or extra keys. "
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


def _clean_whitespace(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    return _clean_whitespace(unescape(cleaned))


def _extract_year(text: Any) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(text or ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _extract_doi(text: Any) -> str:
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", str(text or ""), flags=re.IGNORECASE)
    return match.group(0).rstrip(" .);,]") if match else ""


def _normalize_genes(genes: list[str] | None, *, limit: int = 8) -> list[str]:
    normalized: list[str] = []
    for gene in genes or []:
        value = str(gene or "").strip().upper()
        if value and value not in normalized:
            normalized.append(value)
        if len(normalized) >= limit:
            break
    return normalized


_QUERY_STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "among",
    "and",
    "are",
    "asthma",
    "because",
    "between",
    "could",
    "disease",
    "does",
    "following",
    "from",
    "genes",
    "glucocorticoid",
    "glucocorticoids",
    "have",
    "into",
    "literature",
    "many",
    "more",
    "not",
    "papers",
    "related",
    "results",
    "show",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "these",
    "those",
    "using",
    "what",
    "which",
    "with",
}


def _chunk_list(values: list[str], *, chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        return [values]
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def _query_keywords(user_query: str, genes: list[str], disease: str) -> list[str]:
    disease_tokens = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", disease)}
    gene_tokens = {token.upper() for token in genes}
    keywords: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", user_query):
        lower = token.lower()
        if lower in _QUERY_STOPWORDS or lower in disease_tokens or token.upper() in gene_tokens:
            continue
        if lower not in keywords:
            keywords.append(lower)
    return keywords[:8]


def _unique_queries(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = _clean_whitespace(value)
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        unique.append(query)
    return unique


def _paper_identity_key(paper: dict[str, Any]) -> str:
    doi = _clean_whitespace(paper.get("doi")).lower()
    if doi:
        return f"doi:{doi}"
    pmid = _clean_whitespace(paper.get("pmid"))
    if pmid:
        return f"pmid:{pmid}"
    title = re.sub(r"[^a-z0-9]+", " ", _clean_whitespace(paper.get("title")).lower()).strip()
    year = _clean_whitespace(paper.get("year"))
    return f"title:{title}|year:{year}"


def _annotate_paper(paper: dict[str, Any]) -> dict[str, Any]:
    title = _clean_whitespace(paper.get("title"))
    abstract = _clean_whitespace(paper.get("abstract"))
    combined = f"{title} {abstract}".strip()
    genes = extract_genes_from_text(combined) if combined else []
    return {
        "source": _clean_whitespace(paper.get("source")).lower(),
        "source_id": _clean_whitespace(paper.get("source_id")),
        "title": title,
        "year": paper.get("year"),
        "doi": _clean_whitespace(paper.get("doi")),
        "pmid": _clean_whitespace(paper.get("pmid")),
        "abstract": abstract,
        "url": _clean_whitespace(paper.get("url")),
        "journal": _clean_whitespace(paper.get("journal")),
        "authors": paper.get("authors", []) if isinstance(paper.get("authors"), list) else [],
        "genes": genes[:20],
    }


def _dedupe_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for raw_paper in papers:
        if not isinstance(raw_paper, dict):
            continue
        paper = _annotate_paper(raw_paper)
        if not paper.get("title"):
            continue
        key = _paper_identity_key(paper)
        existing = merged.get(key)
        if existing is None:
            merged[key] = paper
            continue

        if len(paper.get("abstract", "")) > len(existing.get("abstract", "")):
            existing["abstract"] = paper.get("abstract", "")
        if not existing.get("doi") and paper.get("doi"):
            existing["doi"] = paper.get("doi")
        if not existing.get("pmid") and paper.get("pmid"):
            existing["pmid"] = paper.get("pmid")
        if not existing.get("url") and paper.get("url"):
            existing["url"] = paper.get("url")
        if not existing.get("journal") and paper.get("journal"):
            existing["journal"] = paper.get("journal")
        if not existing.get("authors") and paper.get("authors"):
            existing["authors"] = paper.get("authors")
        existing_genes = list(existing.get("genes") or [])
        for gene in paper.get("genes", []):
            if gene not in existing_genes:
                existing_genes.append(gene)
        existing["genes"] = existing_genes[:20]
        if paper.get("source") and paper.get("source") not in str(existing.get("source") or ""):
            existing["source"] = ",".join(
                sorted(
                    {
                        part.strip()
                        for part in (str(existing.get("source") or "") + "," + str(paper.get("source") or "")).split(",")
                        if part.strip()
                    }
                )
            )
    return list(merged.values())


def _build_search_queries(
    *,
    user_query: str,
    disease: str,
    genes: list[str],
) -> dict[str, list[str]]:
    genes = _normalize_genes(genes, limit=24)
    disease = _clean_whitespace(disease)
    user_query = _clean_whitespace(user_query)
    keywords = _query_keywords(user_query, genes, disease)
    keyword_phrase = " ".join(keywords[:4]).strip()
    disease_context = " ".join(part for part in (disease, keyword_phrase) if part).strip()
    gene_chunks = _chunk_list(genes[:24], chunk_size=5)

    if user_query:
        plain_query = user_query
    elif disease and genes:
        plain_query = f"{disease} {' '.join(genes[:5])} mechanism biomarker gene expression"
    elif disease:
        plain_query = f"{disease} mechanism biomarker gene expression"
    elif genes:
        plain_query = f"{' '.join(genes[:5])} gene function mechanism disease association"
    else:
        plain_query = "gene disease mechanism literature"

    openalex_queries: list[str] = [plain_query]
    scholar_queries: list[str] = [plain_query]
    pubmed_queries: list[str] = []

    if disease_context:
        openalex_queries.extend(
            [
                f"{disease_context} mechanism",
                f"{disease_context} biomarker",
                f"{disease_context} gene expression",
            ]
        )
        scholar_queries.extend(
            [
                f"{disease_context} review",
                f"{disease_context} mechanism",
                f"{disease_context} biomarker",
            ]
        )
        pubmed_queries.extend(
            [
                f"({disease_context}[Title/Abstract])",
                f"({disease}[Title/Abstract]) AND ({keyword_phrase}[Title/Abstract])" if disease and keyword_phrase else "",
                f"({disease}[Title/Abstract]) AND (mechanism[Title/Abstract] OR biomarker[Title/Abstract] OR gene expression[Title/Abstract])" if disease else "",
            ]
        )

    for chunk in gene_chunks[:5]:
        gene_or = " OR ".join(f"{gene}[Title/Abstract]" for gene in chunk)
        chunk_text = " ".join(chunk)
        if disease_context:
            openalex_queries.append(f"{disease_context} {chunk_text}")
            scholar_queries.append(f"{disease_context} {chunk_text}")
            pubmed_queries.append(f"({disease_context}[Title/Abstract]) AND ({gene_or})")
        elif disease:
            openalex_queries.append(f"{disease} {chunk_text}")
            scholar_queries.append(f"{disease} {chunk_text}")
            pubmed_queries.append(f"({disease}[Title/Abstract]) AND ({gene_or})")
        elif keyword_phrase:
            openalex_queries.append(f"{keyword_phrase} {chunk_text}")
            scholar_queries.append(f"{keyword_phrase} {chunk_text}")
            pubmed_queries.append(f"({keyword_phrase}[Title/Abstract]) AND ({gene_or})")
        pubmed_queries.append(f"({gene_or})")

    if genes and not disease and not keyword_phrase:
        for chunk in gene_chunks[:3]:
            chunk_text = " ".join(chunk)
            openalex_queries.append(f"{chunk_text} gene function")
            scholar_queries.append(f"{chunk_text} gene disease association")

    return {
        "plain": _unique_queries(openalex_queries),
        "pubmed": _unique_queries(pubmed_queries or [plain_query]),
        "scholar": _unique_queries(scholar_queries),
    }


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
                "source": paper.get("source"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "pmid": paper.get("pmid"),
                "url": paper.get("url"),
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
                    "Return exactly one JSON object with keys `key_points` and `references`. "
                    "Do not use markdown, code fences, commentary, or extra keys. "
                    "`key_points` must be a list of objects with `point` and `paper_ids`. "
                    "`references` must be a list of objects with `paper_id`, `source`, `title`, `year`, `doi`, `pmid`, and `url`. "
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
                "source": paper.get("source"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "pmid": paper.get("pmid"),
                "url": paper.get("url"),
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
                    "Return exactly one JSON object with key `ranked_papers`. "
                    "Do not use markdown, code fences, commentary, or extra keys. "
                    "Each item in `ranked_papers` must have `paper_id`, `relevance`, and `reason`. "
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

    def _format_reference_block(rows: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for row in rows[:8]:
            if not isinstance(row, dict):
                continue
            source = _clean_whitespace(row.get("source")) or "literature"
            title = _clean_whitespace(row.get("title")) or "Untitled"
            year = _clean_whitespace(row.get("year"))
            doi = _clean_whitespace(row.get("doi"))
            pmid = _clean_whitespace(row.get("pmid"))
            locator = doi or (f"PMID: {pmid}" if pmid else "")
            entry = f"- {source}: {title}"
            if year:
                entry += f" ({year})"
            if locator:
                entry += f" [{locator}]"
            lines.append(entry)
        return "\n".join(lines)

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
                    "Return plain text only, not JSON or markdown code fences. "
                    "End with a `References:` section listing the cited papers by source, title, year, and DOI or PMID when available.",
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
        summary = str(getattr(response, "content", "") or "").strip()
        reference_block = _format_reference_block(references or ranked_papers)
        if reference_block and "references:" not in summary.lower():
            summary = (summary + "\n\nReferences:\n" + reference_block).strip()
        return summary
    except Exception:
        if key_points:
            lines = [str(row.get("point") or "").strip() for row in key_points if isinstance(row, dict)]
            lines = [line for line in lines if line]
            summary = " ".join(lines[:3])
        else:
            summary = ""
        reference_block = _format_reference_block(references or ranked_papers)
        if reference_block:
            summary = (summary + "\n\nReferences:\n" + reference_block).strip()
        return summary


def _search_openalex(query: str, *, top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = {"search": query, "per-page": top_n}
    try:
        response = get_retrying_session().get(
            "https://api.openalex.org/works",
            params=params,
            timeout=SETTINGS.http_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return [], {"status": "request_failed", "message": str(exc)}

    papers: list[dict[str, Any]] = []
    for work in payload.get("results", [])[:top_n]:
        if not isinstance(work, dict):
            continue
        primary_location = work.get("primary_location")
        if not isinstance(primary_location, dict):
            primary_location = {}
        source_info = primary_location.get("source")
        if not isinstance(source_info, dict):
            source_info = {}
        title = _clean_whitespace(work.get("display_name"))
        abstract = _clean_whitespace(_abstract_from_inverted_index(work.get("abstract_inverted_index")))
        url = _clean_whitespace(primary_location.get("landing_page_url"))
        papers.append(
            {
                "source": "openalex",
                "source_id": _clean_whitespace(work.get("id")),
                "title": title,
                "year": work.get("publication_year"),
                "doi": _clean_whitespace(work.get("doi")),
                "abstract": abstract,
                "url": url,
                "journal": _clean_whitespace(source_info.get("display_name")),
            }
        )
    return papers, {"status": "ok", "query": query, "count": len(papers)}


def _search_openalex_many(queries: list[str], *, top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for query in queries[:6]:
        rows, status = _search_openalex(query, top_n=top_n)
        papers.extend(rows)
        attempts.append(status)
    return papers, {"status": "ok" if papers else "no_results", "queries": attempts, "count": len(papers)}


def _fetch_pubmed_abstracts(pmids: list[str]) -> dict[str, str]:
    if not pmids:
        return {}
    try:
        response = get_retrying_session().get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            timeout=SETTINGS.http_timeout_seconds,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
    except Exception:
        return {}

    abstracts: dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = _clean_whitespace(article.findtext(".//PMID"))
        segments = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = _clean_whitespace(node.attrib.get("Label"))
            text = _clean_whitespace("".join(node.itertext()))
            if label and text:
                segments.append(f"{label}: {text}")
            elif text:
                segments.append(text)
        if pmid and segments:
            abstracts[pmid] = " ".join(segments)
    return abstracts


def _search_pubmed(query: str, *, top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        search_response = get_retrying_session().get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "retmode": "json", "sort": "relevance", "retmax": top_n, "term": query},
            timeout=SETTINGS.http_timeout_seconds,
        )
        search_response.raise_for_status()
        ids = (search_response.json().get("esearchresult") or {}).get("idlist") or []
    except Exception as exc:
        return [], {"status": "request_failed", "message": str(exc)}

    pmids = [str(value).strip() for value in ids if str(value).strip()]
    if not pmids:
        return [], {"status": "ok", "query": query, "count": 0}

    try:
        summary_response = get_retrying_session().get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "retmode": "json", "id": ",".join(pmids)},
            timeout=SETTINGS.http_timeout_seconds,
        )
        summary_response.raise_for_status()
        summary_payload = summary_response.json().get("result") or {}
    except Exception as exc:
        return [], {"status": "request_failed", "message": str(exc)}

    abstracts = _fetch_pubmed_abstracts(pmids)
    papers: list[dict[str, Any]] = []
    for pmid in pmids:
        row = summary_payload.get(pmid)
        if not isinstance(row, dict):
            continue
        articleids = row.get("articleids") if isinstance(row.get("articleids"), list) else []
        doi = ""
        for item in articleids:
            if isinstance(item, dict) and str(item.get("idtype") or "").lower() == "doi":
                doi = _clean_whitespace(item.get("value"))
                break
        papers.append(
            {
                "source": "pubmed",
                "source_id": pmid,
                "pmid": pmid,
                "title": _clean_whitespace(row.get("title")),
                "year": _extract_year(row.get("pubdate")),
                "doi": doi,
                "abstract": abstracts.get(pmid, ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "journal": _clean_whitespace(row.get("fulljournalname") or row.get("source")),
            }
        )
    return papers, {"status": "ok", "query": query, "count": len(papers)}


def _search_pubmed_many(queries: list[str], *, top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for query in queries[:10]:
        rows, status = _search_pubmed(query, top_n=top_n)
        papers.extend(rows)
        attempts.append(status)
        if len(papers) >= top_n * 3:
            break
    return papers, {"status": "ok" if papers else "no_results", "queries": attempts, "count": len(papers)}


def _search_google_scholar(query: str, *, top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = {"hl": "en", "q": query, "num": max(1, min(int(top_n), 20))}
    try:
        response = get_retrying_session().get(
            "https://scholar.google.com/scholar",
            params=params,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                )
            },
            timeout=SETTINGS.http_timeout_seconds,
        )
        response.raise_for_status()
        html = response.text
    except Exception as exc:
        return [], {"status": "request_failed", "message": str(exc)}

    blocks = re.findall(r'(?s)<div class="gs_ri".*?</div>\s*</div>', html)
    papers: list[dict[str, Any]] = []
    for block in blocks[:top_n]:
        title_match = re.search(r'(?s)<h3 class="gs_rt".*?>(.*?)</h3>', block)
        if not title_match:
            continue
        title_html = title_match.group(1)
        link_match = re.search(r'href="([^"]+)"', title_html)
        title = _strip_html(title_html)
        meta_match = re.search(r'(?s)<div class="gs_a">(.*?)</div>', block)
        abstract_match = re.search(r'(?s)<div class="gs_rs">(.*?)</div>', block)
        meta_text = _strip_html(meta_match.group(1)) if meta_match else ""
        abstract = _strip_html(abstract_match.group(1)) if abstract_match else ""
        url = _clean_whitespace(link_match.group(1)) if link_match else ""
        papers.append(
            {
                "source": "google_scholar",
                "source_id": url or title,
                "title": title,
                "year": _extract_year(meta_text),
                "doi": _extract_doi(f"{meta_text} {abstract} {url}"),
                "abstract": abstract,
                "url": url,
                "journal": meta_text,
            }
        )
    return papers, {"status": "ok", "query": query, "count": len(papers)}


def _search_google_scholar_many(queries: list[str], *, top_n: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for query in queries[:5]:
        rows, status = _search_google_scholar(query, top_n=top_n)
        papers.extend(rows)
        attempts.append(status)
        if len(papers) >= top_n * 2:
            break
    return papers, {"status": "ok" if papers else "no_results", "queries": attempts, "count": len(papers)}


def fetch_openalex_papers_and_genes(
    disease: str,
    *,
    top_n: int = 20,
    user_query: str = "",
    genes: list[str] | None = None,
) -> dict[str, Any]:
    disease = _clean_whitespace(disease)
    user_query = _clean_whitespace(user_query)
    genes = _normalize_genes(genes)
    if not disease and not genes and not user_query:
        return {
            "status": "no_query",
            "disease": "",
            "papers": [],
            "genes": [],
            "key_points": [],
            "references": [],
            "source_status": {},
        }

    queries = _build_search_queries(user_query=user_query, disease=disease, genes=genes)
    per_source = max(5, min(int(top_n or 20), 20))

    openalex_papers, openalex_status = _search_openalex_many(queries["plain"], top_n=per_source)
    pubmed_papers, pubmed_status = _search_pubmed_many(queries["pubmed"], top_n=per_source)
    scholar_papers, scholar_status = _search_google_scholar_many(queries["scholar"], top_n=max(5, min(per_source, 10)))

    merged_papers = _dedupe_papers(openalex_papers + pubmed_papers + scholar_papers)
    collected_genes = list(genes)
    for paper in merged_papers:
        for gene in paper.get("genes", []):
            if gene not in collected_genes:
                collected_genes.append(gene)

    ranked_papers = _rank_literature_papers(
        user_query=user_query or disease or " ".join(genes),
        disease=disease,
        papers=merged_papers,
    )
    evidence = _extract_literature_evidence(
        user_query=user_query or disease or " ".join(genes),
        disease=disease,
        papers=ranked_papers or merged_papers,
    )
    literature_summary = _summarize_literature_answer(
        user_query=user_query or disease or " ".join(genes),
        disease=disease,
        ranked_papers=ranked_papers or merged_papers,
        key_points=evidence.get("key_points", []),
        references=evidence.get("references", []),
    )

    return {
        "status": "ok" if merged_papers else "no_results",
        "disease": disease,
        "query": queries["plain"][0] if queries["plain"] else "",
        "queries": queries,
        "papers": merged_papers[: max(per_source, 12)],
        "ranked_papers": ranked_papers,
        "genes": collected_genes,
        "key_points": evidence.get("key_points", []),
        "references": evidence.get("references", []),
        "literature_summary": literature_summary,
        "source_status": {
            "openalex": openalex_status,
            "pubmed": pubmed_status,
            "google_scholar": scholar_status,
        },
    }
