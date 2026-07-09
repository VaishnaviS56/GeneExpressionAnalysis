from __future__ import annotations

import re
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

READ_ONLY_CYPHER_PREFIXES = ("match", "with", "return", "call", "unwind")
ENTITY_ALIASES = {
    "type 2 diabetes": ["type 2 diabetes", "t2d", "diabetes mellitus noninsulin dependent", "noninsulin-dependent"],
    "type ii diabetes": ["type ii diabetes", "type 2 diabetes", "t2d", "diabetes mellitus noninsulin dependent"],
    "t2d": ["t2d", "type 2 diabetes", "diabetes mellitus noninsulin dependent", "noninsulin-dependent"],
    "type 1 diabetes": ["type 1 diabetes", "t1d", "diabetes mellitus insulin dependent", "insulin-dependent"],
    "t1d": ["t1d", "type 1 diabetes", "diabetes mellitus insulin dependent", "insulin-dependent"],
}
ENTITY_NAME_STOPWORDS = {
    "what",
    "which",
    "who",
    "whom",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "in",
    "on",
    "and",
    "or",
    "between",
    "related",
    "associated",
    "connected",
    "genes",
    "gene",
    "diseases",
    "disease",
    "drugs",
    "drug",
    "pathways",
    "pathway",
}

_graph = None
_chain = None


def _get_graph() -> Neo4jGraph:
    global _graph

    if _graph is None:
        _graph = Neo4jGraph(
            url=SETTINGS.neo4j_uri,
            username=SETTINGS.neo4j_username,
            password=SETTINGS.neo4j_password,
            database=SETTINGS.neo4j_database,
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


def _graph_counts() -> dict[str, int]:
    graph = _get_graph()
    node_rows = graph.query("MATCH (n) RETURN count(n) AS count")
    rel_rows = graph.query("MATCH ()-[r]->() RETURN count(r) AS count")
    node_count = int(node_rows[0].get("count", 0)) if node_rows else 0
    relationship_count = int(rel_rows[0].get("count", 0)) if rel_rows else 0
    return {
        "nodes": node_count,
        "relationships": relationship_count,
    }


def _has_expected_primekg_shape() -> bool:
    graph = _get_graph()
    rows = graph.query(
        """
        MATCH (n)
        RETURN
            count(CASE WHEN "Entity" IN labels(n) THEN 1 END) AS entity_nodes,
            count(n) AS total_nodes
        """
    )
    if not rows:
        return False

    entity_nodes = int(rows[0].get("entity_nodes", 0) or 0)
    total_nodes = int(rows[0].get("total_nodes", 0) or 0)
    if total_nodes == 0:
        return False
    if entity_nodes == 0:
        return False

    rel_rows = graph.query(
        """
        MATCH ()-[r]->()
        RETURN count(CASE WHEN type(r) = "RELATED_TO" THEN 1 END) AS related_to_edges
        """
    )
    related_to_edges = int(rel_rows[0].get("related_to_edges", 0) or 0) if rel_rows else 0
    return related_to_edges > 0


def _primekg_unavailable_message(counts: dict[str, int]) -> str:
    return (
        "PrimeKG is reachable in Neo4j, but the expected graph is not loaded in the configured "
        f"database `{SETTINGS.neo4j_database}`. Found {counts['nodes']} nodes and "
        f"{counts['relationships']} relationships, but no usable `:Entity` / `:RELATED_TO` "
        "PrimeKG structure. Load PrimeKG into Neo4j or point `NEO4J_DATABASE` at the database "
        "that already contains it."
    )


def _extract_generated_cypher(result: dict[str, Any]) -> str:
    steps = result.get("intermediate_steps", [])
    for step in steps:
        if isinstance(step, dict):
            cypher = step.get("query") or step.get("cypher")
            if cypher:
                return str(cypher).strip()
    return ""


def _is_read_only_cypher(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    if not normalized.startswith(READ_ONLY_CYPHER_PREFIXES):
        return False
    blocked_tokens = (" create ", " merge ", " delete ", " detach ", " set ", " remove ", " drop ")
    compact = f" {normalized} "
    return not any(token in compact for token in blocked_tokens)


def _normalize_entity_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s\-/]", " ", str(text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _entity_search_terms(text: str) -> list[str]:
    normalized = _normalize_entity_text(text)
    if not normalized:
        return []

    terms: list[str] = [normalized]
    for key, aliases in ENTITY_ALIASES.items():
        if key in normalized:
            terms.extend(_normalize_entity_text(alias) for alias in aliases)

    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        compact = term.strip()
        if not compact or compact in seen:
            continue
        seen.add(compact)
        deduped.append(compact)
    return deduped


def _contains_condition(field: str, raw_text: str) -> str:
    terms = _entity_search_terms(raw_text)
    if not terms:
        return ""
    clauses = [f'toLower({field}) CONTAINS "{term}"' for term in terms]
    return "(" + " OR ".join(clauses) + ")"


def _extract_between_entities(question: str) -> list[str]:
    match = re.search(
        r"(?:between|connects?|links?)\s+(.+?)\s+(?:and|to)\s+(.+?)(?:\?|$)",
        str(question or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    return [match.group(1).strip(" .?"), match.group(2).strip(" .?")]


def _extract_focus_entity(question: str) -> str:
    raw = str(question or "").strip()
    for pattern in (
        r"(?:related to|associated with|connected to|for|about)\s+(.+?)(?:\?|$)",
        r"(?:what genes are related to|what drugs are related to|what pathways are related to)\s+(.+?)(?:\?|$)",
    ):
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .?")

    tokens = [token for token in re.split(r"\s+", raw) if token]
    filtered = [token for token in tokens if _normalize_entity_text(token) not in ENTITY_NAME_STOPWORDS]
    if not filtered:
        return ""
    return " ".join(filtered[-4:]).strip(" .?")


def _build_rule_based_cypher(question: str) -> str:
    lowered = str(question or "").lower()

    between_entities = _extract_between_entities(question)
    if len(between_entities) == 2:
        source, target = (_normalize_entity_text(item) for item in between_entities)
        if source and target:
            return f"""
MATCH (a:Entity),(b:Entity)
WHERE toLower(a.name) CONTAINS "{source}"
AND toLower(b.name) CONTAINS "{target}"
MATCH p = shortestPath((a)-[:RELATED_TO*..4]-(b))
RETURN p
LIMIT 1
""".strip()

    entity = _extract_focus_entity(question)
    entity_terms = _entity_search_terms(entity)
    if not entity_terms:
        return ""
    entity_condition = _contains_condition("d.name", entity)
    gene_condition = _contains_condition("g.name", entity)
    drug_condition = _contains_condition("drug.name", entity)
    generic_condition = _contains_condition("n.name", entity)

    asks_for_genes = bool(re.search(r"\bwhat genes\b|\bwhich genes\b|\bgene[s]?\s+related to\b", lowered))
    asks_for_diseases = bool(re.search(r"\bwhat diseases\b|\bwhich diseases\b|\bdiseases related to\b", lowered))

    if asks_for_diseases or ("gene" in lowered and "disease" in lowered):
        if asks_for_diseases:
            return f"""
MATCH (g:Entity)-[r:RELATED_TO]-(d:Entity)
WHERE g.type = "gene/protein"
AND d.type = "disease"
AND {gene_condition}
RETURN DISTINCT d.name AS disease, r.relation AS relation, r.display_relation AS display_relation
ORDER BY disease
LIMIT 25
""".strip()

    if asks_for_genes or ("gene" in lowered and "disease" in lowered):
        return f"""
MATCH (d:Entity)-[r:RELATED_TO]-(g:Entity)
WHERE d.type = "disease"
AND g.type = "gene/protein"
AND {entity_condition}
RETURN DISTINCT g.name AS gene, r.relation AS relation, r.display_relation AS display_relation
ORDER BY gene
LIMIT 25
""".strip()

    if "pathway" in lowered:
        return f"""
MATCH (g:Entity)-[r:RELATED_TO]-(p:Entity)
WHERE g.type = "gene/protein"
AND p.type = "pathway"
AND {gene_condition}
RETURN DISTINCT p.name AS pathway, r.relation AS relation, r.display_relation AS display_relation
ORDER BY pathway
LIMIT 25
""".strip()

    if "drug" in lowered and "disease" in lowered:
        return f"""
MATCH (d:Entity)-[r:RELATED_TO]-(drug:Entity)
WHERE d.type = "disease"
AND drug.type = "drug"
AND {entity_condition}
RETURN DISTINCT drug.name AS drug, r.relation AS relation, r.display_relation AS display_relation
ORDER BY drug
LIMIT 25
""".strip()

    if "drug" in lowered and "gene" in lowered:
        return f"""
MATCH (drug:Entity)-[r:RELATED_TO]-(g:Entity)
WHERE drug.type = "drug"
AND g.type = "gene/protein"
AND {drug_condition}
RETURN DISTINCT g.name AS gene, r.relation AS relation, r.display_relation AS display_relation
ORDER BY gene
LIMIT 25
""".strip()

    return f"""
MATCH (n:Entity)-[r:RELATED_TO]-(m:Entity)
WHERE {generic_condition}
RETURN DISTINCT
    m.name AS related_entity,
    m.type AS related_type,
    r.relation AS relation,
    r.display_relation AS display_relation
ORDER BY related_type, related_entity
LIMIT 25
""".strip()


def _run_cypher_query(question: str, cypher: str, message: str | None = None) -> dict[str, Any]:
    graph = _get_graph()
    raw_rows = graph.query(cypher)
    if message is None:
        if raw_rows:
            message = f"PrimeKG returned {len(raw_rows)} row(s)."
        else:
            message = "PrimeKG returned no matching rows."

    return {
        "status": "ok",
        "question": question,
        "answer": message,
        "cypher": cypher,
        "raw_result": raw_rows,
    }


def refresh_primekg_schema() -> str:
    global _graph

    if _graph is None:
        _graph = _get_graph()

    _graph.refresh_schema()
    return _graph.schema


def get_primekg_schema() -> dict[str, Any]:
    graph = _get_graph()
    graph.refresh_schema()
    counts = _graph_counts()

    return {
        "status": "ok",
        "schema": graph.schema,
        "counts": counts,
        "database": SETTINGS.neo4j_database,
    }


def test_primekg_connection() -> dict[str, Any]:
    try:
        graph = _get_graph()
        result = graph.query("RETURN 'connected' AS status")
        counts = _graph_counts()
        has_primekg_shape = _has_expected_primekg_shape()

        payload: dict[str, Any] = {
            "status": "ok",
            "result": result,
            "counts": counts,
            "database": SETTINGS.neo4j_database,
            "has_expected_primekg_shape": has_primekg_shape,
        }
        if not has_primekg_shape:
            payload["warning"] = _primekg_unavailable_message(counts)
        return payload

    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "database": SETTINGS.neo4j_database,
        }


def query_primekg(question: str) -> dict[str, Any]:
    """
    Description:
        Query PrimeKG using natural language or direct read-only Cypher.

        The preferred path uses an LLM to translate natural language into Cypher.
        If that is unavailable, the tool falls back to a local rule-based query
        builder for common PrimeKG question shapes.
    """

    question = str(question or "").strip()
    if not question:
        return {
            "status": "error",
            "question": "",
            "answer": "",
            "cypher": "",
            "message": "PrimeKG query cannot be empty.",
        }

    try:
        counts = _graph_counts()
        if not _has_expected_primekg_shape():
            return {
                "status": "error",
                "question": question,
                "answer": "",
                "cypher": "",
                "raw_result": [],
                "message": _primekg_unavailable_message(counts),
            }

        if _is_read_only_cypher(question):
            return _run_cypher_query(question, question, "Executed direct read-only Cypher against PrimeKG.")

        llm_error = ""
        try:
            result = _get_chain().invoke({"query": question})
            cypher = _extract_generated_cypher(result)
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
            llm_error = str(exc).strip()

        fallback_cypher = _build_rule_based_cypher(question)
        if fallback_cypher:
            fallback = _run_cypher_query(
                question,
                fallback_cypher,
                "PrimeKG answered using the local fallback query path because the LLM-backed path was unavailable.",
            )
            fallback["fallback_reason"] = llm_error or "LLM-backed path unavailable."
            return fallback

        return {
            "status": "error",
            "question": question,
            "answer": "",
            "cypher": "",
            "raw_result": [],
            "message": (
                "PrimeKG could not answer the question. The LLM-backed query path failed"
                + (f": {llm_error}" if llm_error else ".")
            ),
        }

    except Exception as exc:
        return {
            "status": "error",
            "question": question,
            "answer": "",
            "cypher": "",
            "message": str(exc),
        }
