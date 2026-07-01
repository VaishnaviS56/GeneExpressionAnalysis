from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph

from gea_agent.config import SETTINGS
from gea_agent.tools.llm import get_llm


PRIMEKG_CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
You are the Cypher-generation specialist for a biomedical agent workflow.
Return one valid read-only Cypher query and nothing else.

Schema constraints:
- The only node label is `:Entity`.
- The only relationship type is `:RELATED_TO`.
- Never invent labels such as `:Gene`, `:Disease`, `:Drug`, or `:Pathway`.
- Never invent relationship types such as `TREATS`, `TARGETS`, or `INTERACTS_WITH`.
- Each entity may have `id`, `name`, `type`, and `source`.
- Relationships may include `relation` and `display_relation`.

Entity typing:
- Always filter biological categories with `Entity.type`.
- Valid types include `gene/protein`, `disease`, `drug`, `pathway`, `anatomy`,
  `biological_process`, `molecular_function`, `cellular_component`,
  `effect/phenotype`, and `exposure`.

Query style:
- Use only read-only Cypher.
- Use `MATCH`.
- Use `Entity.type` instead of labels for biological categories.
- Use `toLower(...)` for matching.
- Prefer `CONTAINS` unless the user explicitly requests exact matching.
- Return names instead of ids whenever possible.
- Use `DISTINCT` when returning entities.
- Do not use `LIMIT` unless requested or necessary to prevent an unbounded query.
- Use relationship metadata only when it materially improves specificity.

Intent mapping:
- disease -> genes: one-hop disease to gene/protein query
- gene -> diseases: one-hop gene/protein to disease query
- gene -> pathways: one-hop gene/protein to pathway query
- disease -> drugs: one-hop disease to drug query
- drug -> genes: one-hop drug to gene/protein query
- "connected", "path between", "what links": shortest path or bounded multi-hop path
- "neighbors", "related to", "associated with": one-hop neighborhood query
- "shared": common-neighbor query
- "side effects": use `r.relation="drug_effect"`
- "contraindications": use `r.relation="contraindication"`

Preferred templates:
```cypher
MATCH (d:Entity)-[:RELATED_TO]-(g:Entity)
WHERE d.type="disease"
AND g.type="gene/protein"
AND toLower(d.name) CONTAINS "<disease>"
RETURN DISTINCT g.name
ORDER BY g.name
```

```cypher
MATCH (g:Entity)-[:RELATED_TO]-(p:Entity)
WHERE g.type="gene/protein"
AND p.type="pathway"
AND toLower(g.name) CONTAINS "<gene>"
RETURN DISTINCT p.name
ORDER BY p.name
```

```cypher
MATCH (n:Entity)-[r:RELATED_TO]-(m:Entity)
WHERE toLower(n.name)=toLower("<entity>")
RETURN DISTINCT m.name, m.type, r.relation, r.display_relation
```

```cypher
MATCH (a:Entity),(b:Entity)
WHERE toLower(a.name)=toLower("<entity1>")
AND toLower(b.name)=toLower("<entity2>")
MATCH p=shortestPath((a)-[:RELATED_TO*]-(b))
RETURN p
```

Validation checklist:
- Uses only `:Entity`
- Uses only `:RELATED_TO`
- Uses `Entity.type` when filtering categories
- Uses `toLower(...)`
- Uses `CONTAINS` unless exact matching is explicitly requested
- Is read-only

User question:
{question}

Cypher:
"""
)

PRIMEKG_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the answer-synthesis specialist for PrimeKG inside a biomedical agent workflow.
Use only the provided query results.

Context:
{context}

Question:
{question}

Instructions:
- Answer the question directly from the returned graph results.
- Mention genes, drugs, diseases, phenotypes, and pathways explicitly when present.
- If the results imply a path or relationship, describe it plainly.
- Do not invent facts, mechanisms, or missing entities.
- If no relevant results were found, state that clearly.
- If multiple entities are returned, summarize the dominant findings rather than listing everything.

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
            verbose=False,
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
        "schema": graph.schema,
    }


def test_primekg_connection() -> dict[str, Any]:
    try:
        graph = _get_graph()

        result = graph.query("RETURN 'connected' AS status")

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
        result = _get_chain().invoke({"query": question})

        cypher = ""
        steps = result.get("intermediate_steps", [])

        for step in steps:
            if isinstance(step, dict):
                cypher = step.get("query") or step.get("cypher") or cypher

        cypher = str(cypher or "").strip()
        if not cypher:
            return {
                "status": "error",
                "question": question,
                "answer": "",
                "cypher": "",
                "raw_result": [],
                "message": "PrimeKG did not generate a Cypher query.",
            }

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
