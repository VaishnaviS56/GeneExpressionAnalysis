from functools import lru_cache
import json
import os
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


JUDGE_MODEL = os.getenv("HALLUCINATION_JUDGE_MODEL", "gpt-5")


response_groundedness_prompt = """
You are evaluating a biomedical research support agent.

You will receive:

1. User query
2. Tool outputs used by the agent
3. Final response

Determine whether the final response is supported by the
tool outputs.

Do NOT penalize:
- hypothesis
- clearly marked speculation
- suggestions for future investigation

Penalize:
- invented genes
- invented proteins
- invented pathways
- invented diseases
- invented compounds
- invented scores
- invented literature findings
- invented differential expression results

Scoring:

5 = Fully grounded in tool evidence
4 = Mostly grounded with minor unsupported details
3 = Some unsupported claims
2 = Multiple unsupported claims
1 = Significant hallucination

Return ONLY valid JSON:

{
    "score": 1,
    "reason": "short explanation"
}
"""


evidence_validity_prompt = """
You are evaluating biomedical tool outputs.

You will receive:

1. User query
2. Tool outputs

Determine whether the tool outputs actually answer
the user's question.

Penalize:

- wrong disease
- wrong cohort
- wrong comparison
- wrong evidence
- irrelevant entities
- contradictory evidence
- outputs that fail to answer the question

Scoring:

5 = Fully valid evidence
4 = Mostly valid evidence
3 = Some issues
2 = Significant issues
1 = Invalid or contradictory evidence

Return ONLY valid JSON:

{
    "score": 1,
    "reason": "short explanation"
}
"""


def extract_tool_evidence(run) -> str:
    """
    Extract all tool outputs from a trace.
    """

    evidence = []

    def add_from_output(output: Any) -> None:
        if not isinstance(output, dict):
            return
        for row in output.get("tool_history") or []:
            if isinstance(row, dict):
                evidence.append(
                    f"""
Tool: {row.get("tool") or row.get("tool_name") or row.get("name")}

Output:
{row.get("result")}
"""
                )
        meta = output.get("meta")
        if isinstance(meta, dict):
            add_from_output(meta)

    def visit(node):
        if getattr(node, "run_type", None) == "tool":

            evidence.append(
                f"""
Tool: {node.name}

Output:
{node.outputs}
"""
            )

        add_from_output(getattr(node, "outputs", None))

        for child in getattr(node, "child_runs", []) or []:
            visit(child)

    visit(run)

    return "\n\n".join(evidence)


def extract_final_response(run) -> str:
    """
    Extract final agent response.
    """

    outputs = getattr(run, "outputs", None)
    if not outputs:
        return ""

    if isinstance(outputs, dict):
        if outputs.get("answer"):
            return str(outputs["answer"])
        nested_outputs = outputs.get("outputs")
        if isinstance(nested_outputs, dict) and nested_outputs.get("answer"):
            return str(nested_outputs["answer"])

    return str(outputs)


def example_query(example) -> str:
    inputs = getattr(example, "inputs", {}) or {}
    if inputs.get("query"):
        return str(inputs["query"])
    messages = inputs.get("messages")
    if isinstance(messages, list):
        return "\n".join(str(message) for message in messages)
    return str(messages or "")


@lru_cache(maxsize=1)
def get_judge_llm():
    return ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0,
    )


def run_judge(prompt: str) -> dict[str, Any]:
    """
    Call judge LLM and parse JSON response.
    """

    response = get_judge_llm().invoke(
        [HumanMessage(content=prompt)]
    )

    content = response.content

    try:
        return json.loads(content)

    except Exception:
        return {
            "score": 1,
            "reason": f"Unable to parse judge response: {content}",
        }


def evaluate_response_groundedness(run, example):

    user_query = example_query(example)

    tool_outputs = extract_tool_evidence(run)

    final_response = extract_final_response(run)

    prompt = f"""
{response_groundedness_prompt}

USER QUERY

{user_query}

TOOL OUTPUTS

{tool_outputs}

FINAL RESPONSE

{final_response}
"""

    result = run_judge(prompt)

    return {
        "key": "response_groundedness",
        "score": result["score"],
        "comment": result.get("reason", ""),
    }


def evaluate_evidence_validity(run, example):

    user_query = example_query(example)

    tool_outputs = extract_tool_evidence(run)

    prompt = f"""
{evidence_validity_prompt}

USER QUERY

{user_query}

TOOL OUTPUTS

{tool_outputs}
"""

    result = run_judge(prompt)

    return {
        "key": "evidence_validity",
        "score": result["score"],
        "comment": result.get("reason", ""),
    }
