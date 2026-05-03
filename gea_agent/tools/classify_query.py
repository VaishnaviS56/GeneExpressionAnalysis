from __future__ import annotations

import json
import re
from typing import Any

from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.llm import get_llm
from gea_agent.tools.types import QueryClassification


_SYSTEM = (
    "You are a strict classifier for a biomedical chat assistant. "
    "Your job is to decide whether the user needs gene-network / pathway analysis.\n\n"
    "Return ONLY valid JSON: {\"kind\": \"general\"|\"technical\", \"rationale\": \"...\"}.\n\n"
    "Guidelines:\n"
    "- general: definitions, explanations, or a single-gene question (no analysis requested)\n"
    "- technical: the user provides multiple genes OR asks for pathways/enrichment/interaction/network analysis\n"
)


def safe_parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
    return None


def heuristic_classification(query: str, genes: list[str]) -> QueryClassification:
    q = (query or "").lower()
    keywords = ["pathway", "interact", "interaction", "network", "enrichment", "enrichr", "string"]

    if len(genes) >= 2:
        return {"kind": "technical", "rationale": "Multiple genes detected.", "genes": genes}

    if any(k in q for k in keywords) and len(genes) >= 1:
        return {
            "kind": "technical",
            "rationale": "Analysis keyword detected with at least one gene present.",
            "genes": genes,
        }

    return {"kind": "general", "rationale": "No analysis trigger detected.", "genes": genes}


def classify_query(query: str) -> QueryClassification:
    llm = get_llm()

    genes = extract_genes_from_text(query)

    response = llm.invoke(
        [
            ("system", _SYSTEM),
            ("user", query),
        ]
    )

    data = safe_parse_json(getattr(response, "content", "") or "")
    if not isinstance(data, dict):
        return heuristic_classification(query, genes)

    kind = str(data.get("kind", "general")).strip().lower()
    if kind not in {"general", "technical"}:
        kind = "general"

    # Guardrails: technical only when it makes sense for this app
    q = (query or "").lower()
    keywords = ["pathway", "interact", "interaction", "network", "enrichment", "enrichr", "string"]
    if kind == "technical" and not (len(genes) >= 2 or (len(genes) >= 1 and any(k in q for k in keywords))):
        kind = "general"

    if len(genes) >= 2:
        kind = "technical"
    if any(k in q for k in keywords) and len(genes) >= 1:
        kind = "technical"

    return {
        "kind": kind,
        "rationale": str(data.get("rationale", "")),
        "genes": genes,
    }