from __future__ import annotations

from typing import Any

from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain

from gea_agent.tools.llm import get_llm
from gea_agent.config import SETTINGS


from langchain_core.prompts import PromptTemplate


PRIMEKG_CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
You are an expert biomedical graph database engineer.

You are generating Cypher queries for PrimeKG.

PrimeKG contains biomedical entities such as:

Entity Types:
- Gene
- Drug
- Disease
- Pathway
- Phenotype
- Anatomy
- BiologicalProcess
- MolecularFunction
- CellularComponent
- Exposure

The graph schema is:

{schema}

Rules:

1. Generate ONLY Cypher.
2. Do NOT explain your reasoning.
3. Do NOT return markdown.
4. Use case-insensitive matching when searching names.
5. Use CONTAINS instead of exact equality whenever possible.
6. Return meaningful names, not internal IDs.
7. Limit results to 25 unless the user explicitly asks for more.
8. Prefer shortest graph traversals.
9. Never invent labels or relationship types that are not present in the schema.
10. If the user asks for genes, return gene/protein entities.
11. If the user asks for pathways, return pathway entities.
12. If the user asks for drugs, return drug entities.
13. If the user asks for diseases, return disease entities.

Examples

Question:
What genes are associated with type 2 diabetes?

Cypher:
MATCH (d:Entity)-[r]-(g:Entity)
WHERE d.type = "disease"
AND g.type = "gene/protein"
WHERE toLower(d.name) CONTAINS "diabetes"
RETURN DISTINCT g.name
LIMIT 25

Question:
What drugs target JAK2?

Cypher:
MATCH (d:Entity)-[r]-(g:Entity)
WHERE d.type = "disease"
AND g.type = "gene/protein"
WHERE toLower(g.name) CONTAINS "jak2"
RETURN DISTINCT drug.name
LIMIT 25

Question:
What pathways involve TP53?

Cypher:
MATCH (d:Entity)-[r]-(g:Entity)
WHERE d.type = "disease"
AND g.type = "gene/protein"
WHERE toLower(g.name) CONTAINS "tp53"
RETURN DISTINCT p.name
LIMIT 25

Question:
{question}

Cypher:
"""
)

PRIMEKG_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a biomedical research assistant.

You are given query results retrieved from PrimeKG.

Use only the provided data.

Context:
{context}

Question:
{question}

Instructions:

- Answer clearly and concisely.
- Mention genes, drugs, diseases and pathways explicitly.
- Do not invent information.
- If no results were found, state that.
- If multiple entities are returned, summarize the major findings.

Answer:
"""
)

_graph = None
_chain = None


def _get_graph() -> Neo4jGraph:
    global _graph

    if _graph is None:
        _graph = Neo4jGraph(
            url=SETTINGS.neo4j_uri,
            username=SETTINGS.neo4j_username,
            password=SETTINGS.neo4j_password,
        )

    return _graph


def _get_chain() -> GraphCypherQAChain:
    global _chain

    if _chain is None:
        llm = get_llm()

        _chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=_get_graph(),
            cypher_prompt=PRIMEKG_CYPHER_PROMPT,
            qa_prompt=PRIMEKG_QA_PROMPT,
            validate_cypher=True,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )

    return _chain

def refresh_primekg_schema() -> str:
    global _graph

    if _graph is None:
        _graph = _get_graph()

    _graph.refresh_schema()
    return _graph.schema

def get_primekg_schema() -> dict[str, Any]:
    graph = _get_graph()
    graph.refresh_schema()

    return {
        "status": "ok",
        "schema": graph.schema
    }

def test_primekg_connection() -> dict[str, Any]:
    try:
        graph = _get_graph()

        result = graph.query(
            "RETURN 'connected' AS status"
        )

        return {
            "status": "ok",
            "result": result,
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

def query_primekg(question: str) -> dict[str, Any]:
    """
    Description:
        Query PrimeKG using natural language.

        The LLM receives the Neo4j schema,
        generates a Cypher query,
        executes it against PrimeKG,
        and returns both the result and generated Cypher.

    Args:
        question:
            Natural language biomedical question.

    Output:
        {
            "status": "ok",
            "question": "...",
            "answer": "...",
            "cypher": "...",
            "raw_result": [...]
        }
    """

    try:

        result = _get_chain().invoke(
            {"query": question}
        )

        cypher = ""

        steps = result.get(
            "intermediate_steps",
            []
        )

        for step in steps:
            if isinstance(step, dict):
                cypher = (
                    step.get("query")
                    or step.get("cypher")
                    or cypher
                )
        print("Cypher: ", cypher)
        print("Raw Result: ", result.get("result"))
        return {
            "status": "ok",
            "question": question,
            "answer": result.get("result", ""),
            "cypher": cypher,
            "raw_result": result,
        }

    except Exception as exc:

        return {
            "status": "error",
            "question": question,
            "answer": "",
            "cypher": "",
            "message": str(exc),
        }