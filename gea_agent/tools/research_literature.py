from __future__ import annotations

from typing import Any

from gea_agent.tools.extract_genes import extract_genes_from_text
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
        return str(raw or "").strip(), _extract_references_from_text(raw), [], [], "text_fallback"

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

    normalized_genes = _normalize_genes(genes)
    if not normalized_genes:
        normalized_genes = _normalize_genes(extract_genes_from_text(query, mode="strict"))

    resolved_disease = str(disease_name or "").strip()
    genes_text = ", ".join(normalized_genes) if normalized_genes else "None provided"
    prompt = (
        "Answer the user query using the supplied genes and include references. "
        "You are a research assitant that provides a research-style answer to the user query. "
        "Return one JSON object only with this schema: "
        '{"answer":"Markdown answer ending before references","key_points":[{"point":"concise finding","paper_ids":[1]}],'
        '"references":[{"paper_id":1,"title":"paper or review title","authors":"authors if known","journal":"journal if known","year":"year if known",'
        '"doi":"doi if known","pmid":"pmid if known","url":"url if known","source":"source type","note":"short relevance note"}]}. '
        f"User query: {query}\n"
        f"Genes: {genes_text}"
    )

    print("[research_literature] LLM prompt:")
    print(prompt)

    response = get_llm().invoke([("user", prompt)])
    raw_answer = _message_content_text(getattr(response, "content", ""))
    answer, references, key_points, candidate_gene_evidence, response_format = _parse_literature_response(raw_answer, normalized_genes)
    if not answer:
        answer = "I could not generate a research-style answer for that query."
    used_fallback_resources = False
    if not references:
        references = _fallback_reference_resources(query)
        response_format = f"{response_format}_without_specific_references"
        used_fallback_resources = True
    answer = _ensure_references_in_answer(answer, references)

    return {
        "status": "ok",
        "analysis_arm": "research_literature",
        "answer": answer,
        "message": "LLM-only research-style answer generated with model-generated references.",
        "disease_name": resolved_disease,
        "openalex_genes": normalized_genes,
        "openalex_papers": [],
        "ranked_openalex_papers": [],
        "literature_key_points": key_points,
        "candidate_gene_evidence": candidate_gene_evidence,
        "literature_references": references,
        "literature_summary": answer,
        "literature_source_status": {
            "mode": "llm_only_unverified",
            "response_format": response_format,
            "reference_count": len(references),
            "candidate_gene_count": len(normalized_genes),
            "candidate_gene_evidence_count": len(candidate_gene_evidence),
            "used_fallback_reference_resources": used_fallback_resources,
            "reference_notice": "References are model-generated from LLM knowledge and should be verified against primary databases.",
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
