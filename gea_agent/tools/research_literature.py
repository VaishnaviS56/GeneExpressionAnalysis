from __future__ import annotations

import json
from typing import Any

from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.disease_literature import fetch_openalex_papers_and_genes
from gea_agent.tools.llm import get_llm, parse_json_object
from gea_agent.tools.result_utils import sanitize_exception_message, tool_error_result


def _normalize_genes(genes: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in genes or []:
        gene = str(value or "").strip().upper()
        if gene and gene not in normalized:
            normalized.append(gene)
    return normalized


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item not in (None, ""):
                parts.append(str(item))
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "").strip()
    return str(content or "").strip()


def _coerce_references(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(value, start=1):
        if isinstance(row, str):
            ref = {"title": _clean_text(row)}
        elif isinstance(row, dict):
            ref = {
                "paper_id": row.get("paper_id") or row.get("id") or index,
                "source": _clean_text(row.get("source") or row.get("database") or "Model-generated reference"),
                "title": _clean_text(row.get("title") or row.get("citation") or row.get("reference")),
                "authors": _clean_text(row.get("authors") or row.get("author")),
                "journal": _clean_text(row.get("journal") or row.get("venue")),
                "year": row.get("year"),
                "doi": _clean_text(row.get("doi")),
                "pmid": _clean_text(row.get("pmid") or row.get("pubmed_id")),
                "url": _clean_text(row.get("url")),
                "note": _clean_text(row.get("note") or row.get("relevance")),
            }
        else:
            continue

        title = _clean_text(ref.get("title"))
        if not title:
            continue
        key = "|".join(
            str(ref.get(part) or "").strip().lower()
            for part in ("title", "year", "doi", "pmid")
        )
        if key in seen:
            continue
        seen.add(key)
        ref["paper_id"] = ref.get("paper_id") or len(references) + 1
        ref["title"] = title
        references.append({key: val for key, val in ref.items() if val not in (None, "", [])})

    return references


def _format_reference(reference: dict[str, Any], index: int) -> str:
    authors = _clean_text(reference.get("authors"))
    year = _clean_text(reference.get("year"))
    title = _clean_text(reference.get("title"))
    journal = _clean_text(reference.get("journal"))
    doi = _clean_text(reference.get("doi"))
    pmid = _clean_text(reference.get("pmid"))
    url = _clean_text(reference.get("url"))
    note = _clean_text(reference.get("note"))

    pieces: list[str] = []
    if authors:
        pieces.append(authors)
    if year:
        pieces.append(f"({year})")
    if title:
        pieces.append(title)
    if journal:
        pieces.append(journal)
    suffixes = []
    if doi:
        suffixes.append(f"DOI: {doi}")
    if pmid:
        suffixes.append(f"PMID: {pmid}")
    if url:
        suffixes.append(url)
    if note:
        suffixes.append(note)
    body = ". ".join(piece.rstrip(".") for piece in pieces if piece).strip()
    if suffixes:
        body = f"{body}. " + "; ".join(suffixes) if body else "; ".join(suffixes)
    return f"{index}. {body}".strip()


def _references_markdown(references: list[dict[str, Any]]) -> str:
    if not references:
        return ""
    lines = ["**References**"]
    lines.extend(_format_reference(reference, index) for index, reference in enumerate(references, start=1))
    return "\n".join(lines).strip()


def _ensure_references_in_answer(answer: str, references: list[dict[str, Any]]) -> str:
    cleaned = str(answer or "").strip()
    refs = _references_markdown(references)
    if not refs:
        return cleaned
    if "**references**" in cleaned.lower() or "\nreferences" in cleaned.lower():
        return cleaned
    return f"{cleaned}\n\n{refs}".strip()


def _extract_references_from_text(raw: str) -> list[dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return []

    lowered = text.lower()
    marker_index = lowered.rfind("references")
    if marker_index == -1:
        return []

    refs_text = text[marker_index:].splitlines()[1:]
    references: list[dict[str, Any]] = []
    for line in refs_text:
        cleaned = line.strip().lstrip("-*").strip()
        if not cleaned:
            continue
        while cleaned and cleaned[0].isdigit():
            cleaned = cleaned[1:].strip()
        cleaned = cleaned.lstrip(".]").strip()
        if len(cleaned) < 12:
            continue
        references.append(
            {
                "paper_id": len(references) + 1,
                "source": "Model-generated reference",
                "title": cleaned,
                "note": "Parsed from the LLM text response; bibliographic details should be verified.",
            }
        )
        if len(references) >= 10:
            break
    return references


def _fallback_reference_resources(query: str) -> list[dict[str, Any]]:
    return [
        {
            "paper_id": 1,
            "source": "Reference note",
            "title": f"No specific bibliographic references were generated for: {query}",
            "note": "The research_literature tool is LLM-only and did not perform live PubMed, OpenAlex, or Google Scholar retrieval.",
        },
    ]


def _references_from_retrieved_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for index, paper in enumerate(papers[:10], start=1):
        if not isinstance(paper, dict):
            continue
        title = _clean_text(paper.get("title"))
        if not title:
            continue
        references.append(
            {
                "paper_id": paper.get("id") or index,
                "source": _clean_text(paper.get("source") or "literature database"),
                "title": title,
                "authors": ", ".join(paper.get("authors", [])[:8]) if isinstance(paper.get("authors"), list) else _clean_text(paper.get("authors")),
                "journal": _clean_text(paper.get("journal")),
                "year": paper.get("year"),
                "doi": _clean_text(paper.get("doi")),
                "pmid": _clean_text(paper.get("pmid")),
                "url": _clean_text(paper.get("url")),
                "note": _clean_text(paper.get("reason") or "Retrieved from an external literature source for this query."),
            }
        )
    return _coerce_references(references)


def _retrieved_literature_answer(
    *,
    query: str,
    disease_name: str,
    genes: list[str],
    top_n: int,
    first_pass_only: bool = False,
) -> dict[str, Any] | None:
    result = fetch_openalex_papers_and_genes(
        disease_name,
        top_n=max(5, min(int(top_n or 20), 25)),
        user_query=query,
        genes=genes,
        first_pass_only=first_pass_only,
        source_query_limit=2 if first_pass_only else None,
        source_timeout_seconds=8 if first_pass_only else None,
        source_use_retries=not first_pass_only,
    )
    if not isinstance(result, dict):
        return None

    papers = result.get("papers") if isinstance(result.get("papers"), list) else []
    ranked_papers = result.get("ranked_papers") if isinstance(result.get("ranked_papers"), list) else []
    dataset_accessions = result.get("dataset_accessions") if isinstance(result.get("dataset_accessions"), list) else []
    if str(result.get("status") or "").lower() != "ok" and not dataset_accessions:
        return None
    if not (papers or ranked_papers or dataset_accessions):
        return None

    references = _coerce_references(result.get("references")) or _references_from_retrieved_papers(ranked_papers or papers)
    summary = _clean_text(result.get("literature_summary"))
    if not summary:
        key_points = result.get("key_points") if isinstance(result.get("key_points"), list) else []
        point_lines = [
            _clean_text(row.get("point") if isinstance(row, dict) else row)
            for row in key_points[:5]
        ]
        point_lines = [line for line in point_lines if line]
        summary = "\n".join(f"- {line}" for line in point_lines)
    if not summary:
        summary = "Retrieved relevant literature records, but no concise synthesis could be generated from the abstracts."
    if not (papers or ranked_papers) and dataset_accessions:
        summary = "Retrieved dataset accession evidence for the query from the evidence-based literature retrieval layer."

    answer = _ensure_references_in_answer(summary, references)
    key_points = result.get("key_points") if isinstance(result.get("key_points"), list) else []
    source_status = result.get("source_status") if isinstance(result.get("source_status"), dict) else {}

    return {
        "status": "ok",
        "analysis_arm": "research_literature",
        "answer": answer,
        "message": "Evidence-grounded literature answer generated from retrieved literature records.",
        "disease_name": disease_name,
        "openalex_genes": result.get("genes") if isinstance(result.get("genes"), list) else genes,
        "openalex_papers": papers,
        "ranked_openalex_papers": ranked_papers,
        "literature_key_points": [row for row in key_points[:10] if isinstance(row, dict)],
        "candidate_gene_evidence": [],
        "literature_dataset_accessions": dataset_accessions,
        "literature_references": references,
        "literature_summary": answer,
        "literature_source_status": {
            "mode": "retrieved_evidence",
            "retrieval_status": source_status,
            "paper_count": len(papers),
            "ranked_paper_count": len(ranked_papers),
            "reference_count": len(references),
            "dataset_accession_count": len(dataset_accessions),
            "candidate_gene_count": len(genes),
            "reference_notice": "References were retrieved from external literature sources and claims are synthesized from retrieved titles/abstracts. Verify critical claims against the linked papers before high-stakes use.",
        },
        "literature_query": query,
        "should_finalize": True,
    }


def _compact_retrieved_context_for_llm(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}

    def compact_paper(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "paper_id": row.get("paper_id") or row.get("id"),
                "title": _clean_text(row.get("title")),
                "year": row.get("year"),
                "journal": _clean_text(row.get("journal") or row.get("source")),
                "doi": _clean_text(row.get("doi")),
                "pmid": _clean_text(row.get("pmid")),
                "url": _clean_text(row.get("url")),
                "relevance": row.get("relevance"),
                "reason": _clean_text(row.get("reason") or row.get("note")),
                "abstract": _clean_text(row.get("abstract"))[:900],
            }.items()
            if value not in (None, "", [])
        }

    ranked_papers = result.get("ranked_openalex_papers") if isinstance(result.get("ranked_openalex_papers"), list) else []
    papers = result.get("openalex_papers") if isinstance(result.get("openalex_papers"), list) else []
    key_points = result.get("literature_key_points") if isinstance(result.get("literature_key_points"), list) else []
    references = result.get("literature_references") if isinstance(result.get("literature_references"), list) else []
    dataset_accessions = result.get("literature_dataset_accessions") if isinstance(result.get("literature_dataset_accessions"), list) else []

    context = {
        "retrieval_message": result.get("message"),
        "retrieval_summary": _clean_text(result.get("literature_summary"))[:2000],
        "ranked_papers": [
            compact_paper(row)
            for row in ranked_papers[:8]
            if isinstance(row, dict)
        ],
        "additional_papers": [
            compact_paper(row)
            for row in papers[:8]
            if isinstance(row, dict)
        ],
        "key_points": [
            {
                "point": _clean_text(row.get("point") if isinstance(row, dict) else row),
                "paper_ids": row.get("paper_ids") if isinstance(row, dict) and isinstance(row.get("paper_ids"), list) else [],
            }
            for row in key_points[:10]
        ],
        "dataset_accessions": dataset_accessions[:20],
        "references": references[:10],
    }
    return {key: value for key, value in context.items() if value not in (None, "", [], {})}


def _coerce_candidate_gene_evidence(value: Any, candidate_genes: list[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    allowed = set(candidate_genes)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in value:
        if not isinstance(row, dict):
            continue
        gene = _clean_text(row.get("gene")).upper()
        if not gene or gene in seen:
            continue
        if allowed and gene not in allowed:
            continue
        status = _clean_text(row.get("status") or row.get("evidence_status")).lower()
        if status not in {"supported", "plausible_indirect", "uncertain", "no_known_support"}:
            status = "uncertain"
        rows.append(
            {
                "gene": gene,
                "status": status,
                "phenotypes": [
                    _clean_text(value)
                    for value in row.get("phenotypes", [])
                    if _clean_text(value)
                ]
                if isinstance(row.get("phenotypes"), list)
                else [],
                "evidence": _clean_text(row.get("evidence") or row.get("rationale")),
                "paper_ids": row.get("paper_ids") if isinstance(row.get("paper_ids"), list) else [],
            }
        )
        seen.add(gene)

    if allowed:
        order = {gene: index for index, gene in enumerate(candidate_genes)}
        rows.sort(key=lambda item: order.get(str(item.get("gene") or ""), len(order)))
    return rows


def _parse_literature_response(
    raw: str,
    candidate_genes: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    data = parse_json_object(raw)
    if not data:
        return (
            str(raw or "").strip(),
            _extract_references_from_text(raw),
            [],
            [],
            "text_fallback",
        )

    answer = str(data.get("answer") or data.get("summary") or "").strip()
    references = _coerce_references(data.get("references") or data.get("literature_references"))
    candidate_gene_evidence = _coerce_candidate_gene_evidence(
        data.get("candidate_gene_evidence") or data.get("gene_evidence"),
        candidate_genes or [],
    )
    key_points: list[dict[str, Any]] = []
    raw_points = data.get("key_points") or data.get("literature_key_points")
    if isinstance(raw_points, list):
        for row in raw_points[:10]:
            if isinstance(row, str):
                point = _clean_text(row)
                if point:
                    key_points.append({"point": point, "paper_ids": []})
            elif isinstance(row, dict):
                point = _clean_text(row.get("point") or row.get("text") or row.get("finding"))
                if point:
                    paper_ids = row.get("paper_ids") if isinstance(row.get("paper_ids"), list) else []
                    key_points.append({"point": point, "paper_ids": paper_ids})

    if not answer:
        answer = str(raw or "").strip()
    return answer, references, key_points, candidate_gene_evidence, "json"


def run_publication_research_assistant(
    user_query: str,
    *,
    disease_name: str = "",
    genes: list[str] | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    query = str(user_query or "").strip()
    if not query:
        return {
            "status": "not_found",
            "analysis_arm": "research_literature",
            "answer": "No literature query was provided.",
            "message": "No literature query was provided.",
            "literature_references": [],
            "literature_key_points": [],
            "literature_source_status": {},
            "literature_summary": "",
            "should_finalize": True,
        }

    caller_provided_genes = isinstance(genes, list) and bool(_normalize_genes(genes))
    normalized_genes = _normalize_genes(genes)
    if not normalized_genes:
        normalized_genes = _normalize_genes(extract_genes_from_text(query, mode="strict"))

    resolved_disease = str(disease_name or "").strip()
    print("[research_literature] query:")
    print(query)
    print("[research_literature] genes:")
    print(", ".join(normalized_genes) if normalized_genes else "None provided")
    if caller_provided_genes:
        print("[research_literature] gene list provided; skipping retrieved literature branch")
        retrieved_result = None
    else:
        try:
            print("[research_literature] trying retrieved literature branch")
            retrieved_result = _retrieved_literature_answer(
                query=query,
                disease_name=resolved_disease,
                genes=normalized_genes,
                top_n=top_n,
                first_pass_only=True,
            )
            print(
                "[research_literature] retrieved literature branch complete "
                f"status={retrieved_result.get('status') if isinstance(retrieved_result, dict) else 'not_used'}",
                flush=True,
            )
        except Exception as exc:
            print(f"[research_literature] retrieved literature branch failed: {sanitize_exception_message(exc)}")
            retrieved_result = None

    genes_text = ", ".join(normalized_genes) if normalized_genes else "None provided"
    retrieved_context = _compact_retrieved_context_for_llm(retrieved_result)
    retrieved_context_text = (
        json.dumps(retrieved_context, ensure_ascii=False, indent=2)
        if retrieved_context
        else "No retrieved literature records were available; use cautious model knowledge and mark references as unverified."
    )
    prompt = (
        "Answer the user query using the supplied genes and include references. "
        "Conduct a deep reasearch-style search of the literature and make sure to go though all the genes in the list"
        "For the query do an extensive research and provide a detailed answer with references. "
        "Return one JSON object only with this schema: "
        '{"answer":"Markdown answer ending before references","key_points":[{"point":"concise finding","paper_ids":[1]}],'
        '"references":[{"paper_id":1,"title":"paper or review title","authors":"authors if known","journal":"journal if known","year":"year if known",'
        '"doi":"doi if known","pmid":"pmid if known","url":"url if known","source":"source type","note":"short relevance note"}]}. '
        f"User query: {query}\n"
        f"Genes: {genes_text}"
    )

    print("[research_literature] LLM prompt:")
    print(prompt)

    print("[research_literature] starting LLM synthesis", flush=True)
    response = get_llm().invoke([("user", prompt)])
    print("[research_literature] LLM synthesis complete", flush=True)
    raw_answer = _message_content_text(getattr(response, "content", ""))
    answer, references, key_points, candidate_gene_evidence, response_format = _parse_literature_response(raw_answer, normalized_genes)
    if not answer:
        answer = "I could not generate a research-style answer for that query."
    used_fallback_resources = False
    retrieved_references = (
        retrieved_result.get("literature_references")
        if isinstance(retrieved_result, dict) and isinstance(retrieved_result.get("literature_references"), list)
        else []
    )
    retrieved_key_points = (
        retrieved_result.get("literature_key_points")
        if isinstance(retrieved_result, dict) and isinstance(retrieved_result.get("literature_key_points"), list)
        else []
    )
    retrieved_dataset_accessions = (
        retrieved_result.get("literature_dataset_accessions")
        if isinstance(retrieved_result, dict) and isinstance(retrieved_result.get("literature_dataset_accessions"), list)
        else []
    )
    if not references:
        references = _coerce_references(retrieved_references)
    if not references:
        references = _fallback_reference_resources(query)
        response_format = f"{response_format}_without_specific_references"
        used_fallback_resources = True
    if not key_points and retrieved_key_points:
        key_points = [row for row in retrieved_key_points[:10] if isinstance(row, dict)]
    dataset_accessions = retrieved_dataset_accessions
    answer = _ensure_references_in_answer(answer, references)

    retrieved_papers = (
        retrieved_result.get("openalex_papers")
        if isinstance(retrieved_result, dict) and isinstance(retrieved_result.get("openalex_papers"), list)
        else []
    )
    retrieved_ranked_papers = (
        retrieved_result.get("ranked_openalex_papers")
        if isinstance(retrieved_result, dict) and isinstance(retrieved_result.get("ranked_openalex_papers"), list)
        else []
    )
    retrieved_genes = (
        retrieved_result.get("openalex_genes")
        if isinstance(retrieved_result, dict) and isinstance(retrieved_result.get("openalex_genes"), list)
        else normalized_genes
    )
    retrieved_source_status = (
        retrieved_result.get("literature_source_status")
        if isinstance(retrieved_result, dict) and isinstance(retrieved_result.get("literature_source_status"), dict)
        else {}
    )
    used_retrieved_evidence = bool(retrieved_context)

    return {
        "status": "ok",
        "analysis_arm": "research_literature",
        "answer": answer,
        "message": (
            "LLM research-style answer generated from retrieved literature evidence and model synthesis."
            if used_retrieved_evidence
            else "LLM-only research-style answer generated with model-generated references."
        ),
        "disease_name": resolved_disease,
        "openalex_genes": retrieved_genes,
        "openalex_papers": retrieved_papers,
        "ranked_openalex_papers": retrieved_ranked_papers,
        "literature_key_points": key_points,
        "candidate_gene_evidence": candidate_gene_evidence,
        "literature_dataset_accessions": dataset_accessions,
        "literature_references": references,
        "literature_summary": answer,
        "literature_source_status": {
            "mode": "retrieved_evidence_plus_llm" if used_retrieved_evidence else "llm_only_unverified",
            "response_format": response_format,
            "retrieval_status": retrieved_source_status,
            "paper_count": len(retrieved_papers),
            "ranked_paper_count": len(retrieved_ranked_papers),
            "reference_count": len(references),
            "dataset_accession_count": len(dataset_accessions),
            "candidate_gene_count": len(normalized_genes),
            "candidate_gene_evidence_count": len(candidate_gene_evidence),
            "used_fallback_reference_resources": used_fallback_resources,
            "llm_synthesis_used": True,
            "reference_notice": (
                "References include retrieved literature records when available plus any LLM-provided references; verify critical claims against primary databases."
                if used_retrieved_evidence
                else "References are model-generated from LLM knowledge and should be verified against primary databases."
            ),
        },
        "literature_query": query,
        "should_finalize": True,
    }


def run_publication_research_assistant_safe(
    user_query: str,
    *,
    disease_name: str = "",
    genes: list[str] | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    try:
        return run_publication_research_assistant(
            user_query,
            disease_name=disease_name,
            genes=genes,
            top_n=top_n,
        )
    except Exception as exc:
        return tool_error_result(
            "research_literature",
            f"Literature analysis failed: {sanitize_exception_message(exc)}",
            analysis_arm="research_literature",
            literature_references=[],
            literature_key_points=[],
            literature_source_status={"mode": "llm_only_unverified"},
            literature_summary="",
            should_finalize=True,
        )
