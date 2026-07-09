from __future__ import annotations

from typing import Any

from gea_agent.tools.llm import get_llm


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
    llm = get_llm()
    response = llm.invoke(
        [
            ("system", LITERATURE_SYSTEM_PROMPT),
            (
                "user",
                (
                    "User query:\n"
                    f"{query}\n\n"
                    "Please answer the query and provide a `References:` section for the answer."
                ),
            ),
        ]
    )
    answer = str(getattr(response, "content", "") or "").strip()
    return {
        "answer": answer,
        "meta": {
            "analysis_arm": "literature",
            "agent_type": "literature",
            "route_rationale": "Handled by the dedicated literature agent.",
        },
    }
