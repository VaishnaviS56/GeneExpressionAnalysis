from __future__ import annotations

from typing import Any

from gea_agent.tools.research_literature import run_publication_research_assistant_safe


LITERATURE_SYSTEM_PROMPT = (
    "You are a biomedical literature agent. "
    "Answer the user's query directly using your literature knowledge. "
    "Preserve the user's query as-is for intent; do not rewrite it into a different task. "
    "Support the answer with references. "
    "Keep the response concise but informative. "
    "Return plain text only, not JSON and not markdown code fences. "
    "End with a `References:` section. "
    "For each reference, include as much bibliographic detail as you can confidently provide, such as authors, title, year, journal, DOI, or PMID. "
    "If you are not confident about a citation detail, say that the reference should be verified."
)


def run_literature_agent(user_query: str) -> dict[str, Any]:
    print("running literature agent...")
    query = str(user_query or "").strip()
    result = run_publication_research_assistant_safe(query)
    answer = str(result.get("answer") or "").strip()
    return {
        "answer": answer,
        "status": result.get("status", "ok"),
        "message": result.get("message", ""),
        "literature_references": result.get("literature_references", []),
        "literature_key_points": result.get("literature_key_points", []),
        "literature_source_status": result.get("literature_source_status", {}),
        "literature_summary": result.get("literature_summary", ""),
        "meta": {
            "analysis_arm": "literature",
            "agent_type": "literature",
            "route_rationale": "Handled by the dedicated literature agent.",
        },
    }
