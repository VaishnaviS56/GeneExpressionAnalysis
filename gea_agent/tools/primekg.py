from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph

from gea_agent.config import SETTINGS
from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.llm import get_llm
from gea_agent.tools.llm import parse_json_object


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
- Treat row fields such as `related_type`, `relation`, and `display_relation` as key graph semantics when they are present.
- Use `display_relation` as the preferred human-readable relation label; fall back to `relation` if `display_relation` is missing.
- Use `related_type` to disambiguate whether a result is a gene/protein, disease, drug, pathway, or phenotype.
- Mention genes, drugs, diseases, phenotypes, and pathways explicitly when present.
- If the results imply a path or relationship, describe it plainly.
- Do not invent facts, mechanisms, or missing entities.
- If no relevant results were found, state that clearly.
- If multiple entities are returned, summarize the dominant findings rather than listing everything.

Answer:
"""
)

READ_ONLY_CYPHER_PREFIXES = ("match", "with", "return", "call", "unwind")
DEFAULT_PRIMEKG_RESULT_LIMIT = 500
DEFAULT_PRIMEKG_CANDIDATE_LIMIT = 500
MAX_RERANK_CANDIDATES = 100
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


def _query_keywords(question: str) -> list[str]:
    normalized = _normalize_entity_text(question)
    if not normalized:
        return []

    seen: set[str] = set()
    keywords: list[str] = []
    for token in normalized.split():
        if len(token) < 3 or token in ENTITY_NAME_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _dedupe_gene_symbols(values: list[str] | None) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        symbol = str(value or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def _resolve_focus_genes(question: str, focus_genes: list[str] | None = None) -> list[str]:
    extracted = extract_genes_from_text(question, mode="strict")
    return _dedupe_gene_symbols(list(focus_genes or []) + list(extracted or []))


def _should_use_gene_focused_query(question: str, focus_genes: list[str]) -> bool:
    if not focus_genes:
        return False
    if len(_extract_between_entities(question)) == 2:
        return False

    lowered = str(question or "").lower()
    specific_target_markers = (
        "biological process",
        "biological processes",
        "biological_process",
        "molecular function",
        "molecular functions",
        "molecular_function",
        "cellular component",
        "cellular components",
        "cellular_component",
        "pathway",
        "pathways",
        "disease",
        "diseases",
        "drug",
        "drugs",
        "phenotype",
        "phenotypes",
        "exposure",
        "anatomy",
    )
    if any(marker in lowered for marker in specific_target_markers):
        return False

    gene_focus_markers = (
        "relationship",
        "relationships",
        "related",
        "associated",
        "connected",
        "connection",
        "connections",
        "neighbor",
        "neighborhood",
        "interact",
        "links",
        "what is linked",
        "what links",
        "primekg",
        "knowledge graph",
    )
    return any(marker in lowered for marker in gene_focus_markers)


def _build_gene_focused_cypher(focus_genes: list[str]) -> str:
    gene_list = json.dumps(_dedupe_gene_symbols(focus_genes), ensure_ascii=True)
    return f"""
MATCH (n:Entity)-[r:RELATED_TO]-(m:Entity)
WHERE n.type = "gene/protein"
AND toUpper(n.name) IN {gene_list}
RETURN DISTINCT
    n.name AS focus_gene,
    m.name AS related_entity,
    m.type AS related_type,
    r.relation AS relation,
    r.display_relation AS display_relation
ORDER BY focus_gene, related_type, related_entity
LIMIT {DEFAULT_PRIMEKG_CANDIDATE_LIMIT}
""".strip()


def _query_target_types(question: str) -> set[str]:
    lowered = str(question or "").lower()
    target_types: set[str] = set()
    if any(token in lowered for token in ("gene", "genes", "protein", "proteins", "target", "targets")):
        target_types.add("gene/protein")
    if any(token in lowered for token in ("disease", "diseases", "diabetes", "cancer", "syndrome")):
        target_types.add("disease")
    if any(token in lowered for token in ("drug", "drugs", "compound", "treatment", "therapy")):
        target_types.add("drug")
    if any(token in lowered for token in ("pathway", "pathways")):
        target_types.add("pathway")
    if any(token in lowered for token in ("phenotype", "phenotypes", "symptom", "side effect")):
        target_types.add("effect/phenotype")
    return target_types


def _row_to_search_text(row: Any) -> str:
    if isinstance(row, dict):
        fragments: list[str] = []
        for key, value in row.items():
            fragments.append(str(key))
            if isinstance(value, (str, int, float, bool)) or value is None:
                fragments.append("" if value is None else str(value))
            else:
                fragments.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return _normalize_entity_text(" ".join(fragments))
    return _normalize_entity_text(str(row))


def _score_primekg_row(question: str, row: Any, index: int) -> dict[str, Any]:
    text = _row_to_search_text(row)
    keywords = _query_keywords(question)
    target_types = _query_target_types(question)
    score = 0.0
    matched_keywords: list[str] = []

    for keyword in keywords:
        if keyword in text:
            score += 2.0
            matched_keywords.append(keyword)

    normalized_question = _normalize_entity_text(question)
    if normalized_question and normalized_question in text:
        score += 4.0

    if isinstance(row, dict):
        relation = _normalize_entity_text(row.get("display_relation") or row.get("relation") or "")
        if relation:
            for keyword in keywords:
                if keyword in relation:
                    score += 1.0
                    break

        if target_types:
            for key, value in row.items():
                if "type" not in str(key).lower():
                    continue
                normalized_value = _normalize_entity_text(value)
                if normalized_value in target_types:
                    score += 2.5

    score += max(0.0, 0.25 - (index * 0.001))
    return {
        "row": row,
        "score": round(score, 4),
        "matched_keywords": matched_keywords[:8],
        "index": index,
    }


def _ensure_candidate_limit(cypher: str, candidate_limit: int = DEFAULT_PRIMEKG_CANDIDATE_LIMIT) -> str:
    text = str(cypher or "").strip()
    if not text:
        return text

    limit_pattern = re.compile(r"(?is)\bLIMIT\s+(\d+)\s*$")
    match = limit_pattern.search(text)
    if match:
        current_limit = int(match.group(1))
        if current_limit >= candidate_limit:
            return text
        return limit_pattern.sub(f"LIMIT {candidate_limit}", text)

    if re.search(r"(?is)\bRETURN\b", text):
        return text + f"\nLIMIT {candidate_limit}"
    return text


def _llm_rerank_primekg_rows(question: str, rows: list[Any], top_k: int) -> dict[str, Any] | None:
    if not rows:
        return None

    candidate_rows = rows[:MAX_RERANK_CANDIDATES]
    serialized_rows = []
    for index, row in enumerate(candidate_rows):
        if isinstance(row, dict):
            compact_row = {str(key): row[key] for key in list(row.keys())[:8]}
        else:
            compact_row = {"value": str(row)}
        serialized_rows.append({"index": index, "row": compact_row})

    prompt = (
        "You are ranking PrimeKG query results for biomedical relevance.\n"
        "Select the rows that best answer the user's question.\n"
        f"Return JSON only with keys `selected_indices` and `reason`.\n"
        f"Choose at most {top_k} indices, ordered best to worst.\n"
        "Prefer rows that directly mention the queried entity, requested entity type, and relevant relation.\n"
        "Pay special attention to `related_type`, `relation`, and `display_relation` when they are present.\n"
        "Use `display_relation` as the most human-readable relationship label and `related_type` to match the requested entity category.\n\n"
        f"Question:\n{question}\n\n"
        f"Rows:\n{json.dumps(serialized_rows, ensure_ascii=False)}"
    )
    response = get_llm().invoke([("user", prompt)])
    parsed = parse_json_object(getattr(response, "content", "") or "")
    indices = parsed.get("selected_indices")
    if not isinstance(indices, list):
        return None

    selected_indices: list[int] = []
    seen: set[int] = set()
    for value in indices:
        try:
            idx = int(value)
        except Exception:
            continue
        if idx < 0 or idx >= len(candidate_rows) or idx in seen:
            continue
        seen.add(idx)
        selected_indices.append(idx)
        if len(selected_indices) >= top_k:
            break

    if not selected_indices:
        return None

    return {
        "rows": [candidate_rows[idx] for idx in selected_indices],
        "ranking_method": "llm",
        "ranking_reason": str(parsed.get("reason") or "").strip(),
        "selected_indices": selected_indices,
    }


def _rerank_primekg_rows(question: str, rows: list[Any], top_k: int = DEFAULT_PRIMEKG_RESULT_LIMIT) -> dict[str, Any]:
    if not rows:
        return {
            "rows": [],
            "candidate_count": 0,
            "selected_count": 0,
            "ranking_method": "none",
            "ranking_reason": "",
            "selected_indices": [],
        }

    try:
        llm_ranked = _llm_rerank_primekg_rows(question, rows, top_k)
        if llm_ranked:
            return {
                **llm_ranked,
                "candidate_count": len(rows),
                "selected_count": len(llm_ranked["rows"]),
            }
    except Exception:
        pass

    scored_rows = [_score_primekg_row(question, row, index) for index, row in enumerate(rows)]
    scored_rows.sort(key=lambda item: (-item["score"], item["index"]))
    selected = scored_rows[:top_k]
    return {
        "rows": [item["row"] for item in selected],
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "ranking_method": "heuristic",
        "ranking_reason": "Selected rows by keyword, relation, and entity-type overlap with the user query.",
        "selected_indices": [item["index"] for item in selected],
    }


def _synthesize_primekg_answer(question: str, rows: list[Any]) -> str:
    if not rows:
        return "PrimeKG returned no relevant rows for this question."

    context = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    response = get_llm().invoke(
        [
            ("system", PRIMEKG_QA_PROMPT.format(context=context, question=question)),
        ]
    )
    answer = str(getattr(response, "content", "") or "").strip()
    return answer or f"PrimeKG returned {len(rows)} relevant row(s)."


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


def _build_rule_based_cypher(question: str, focus_genes: list[str] | None = None) -> str:
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

    resolved_focus_genes = _dedupe_gene_symbols(focus_genes)
    if resolved_focus_genes and _should_use_gene_focused_query(question, resolved_focus_genes):
        return _build_gene_focused_cypher(resolved_focus_genes)

    entity = resolved_focus_genes[0] if resolved_focus_genes else _extract_focus_entity(question)
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
LIMIT 100
""".strip()

    if asks_for_genes or ("gene" in lowered and "disease" in lowered):
        return f"""
MATCH (d:Entity)-[r:RELATED_TO]-(g:Entity)
WHERE d.type = "disease"
AND g.type = "gene/protein"
AND {entity_condition}
RETURN DISTINCT g.name AS gene, r.relation AS relation, r.display_relation AS display_relation
ORDER BY gene
LIMIT 100
""".strip()

    if "pathway" in lowered:
        return f"""
MATCH (g:Entity)-[r:RELATED_TO]-(p:Entity)
WHERE g.type = "gene/protein"
AND p.type = "pathway"
AND {gene_condition}
RETURN DISTINCT p.name AS pathway, r.relation AS relation, r.display_relation AS display_relation
ORDER BY pathway
LIMIT 100
""".strip()

    if "drug" in lowered and "disease" in lowered:
        return f"""
MATCH (d:Entity)-[r:RELATED_TO]-(drug:Entity)
WHERE d.type = "disease"
AND drug.type = "drug"
AND {entity_condition}
RETURN DISTINCT drug.name AS drug, r.relation AS relation, r.display_relation AS display_relation
ORDER BY drug
LIMIT 100
""".strip()

    if "drug" in lowered and "gene" in lowered:
        return f"""
MATCH (drug:Entity)-[r:RELATED_TO]-(g:Entity)
WHERE drug.type = "drug"
AND g.type = "gene/protein"
AND {drug_condition}
RETURN DISTINCT g.name AS gene, r.relation AS relation, r.display_relation AS display_relation
ORDER BY gene
LIMIT 100
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
LIMIT 500
""".strip()


def _run_cypher_query(question: str, cypher: str, message: str | None = None) -> dict[str, Any]:
    graph = _get_graph()
    candidate_cypher = _ensure_candidate_limit(cypher)
    raw_rows = graph.query(candidate_cypher)
    reranked = _rerank_primekg_rows(question, raw_rows, DEFAULT_PRIMEKG_RESULT_LIMIT)
    selected_rows = reranked["rows"]
    if message is None:
        if selected_rows:
            message = f"PrimeKG returned {len(selected_rows)} relevant row(s) from {len(raw_rows)} candidates."
        else:
            message = "PrimeKG returned no matching rows."

    answer = message
    try:
        answer = _synthesize_primekg_answer(question, selected_rows)
    except Exception:
        pass

    return {
        "status": "ok",
        "question": question,
        "answer": answer,
        "cypher": cypher,
        "candidate_cypher": candidate_cypher,
        "raw_result": selected_rows,
        "all_candidates": raw_rows[:MAX_RERANK_CANDIDATES],
        "candidate_count": reranked["candidate_count"],
        "selected_count": reranked["selected_count"],
        "ranking_method": reranked["ranking_method"],
        "ranking_reason": reranked["ranking_reason"],
        "selected_indices": reranked["selected_indices"],
        "message": message,
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


def query_primekg(question: str, focus_genes: list[str] | None = None) -> dict[str, Any]:
    """
    Description:
        Query PrimeKG using natural language or direct read-only Cypher.

        The preferred path uses an LLM to translate natural language into Cypher.
        If that is unavailable, the tool falls back to a local rule-based query
        builder for common PrimeKG question shapes.
    """

    question = str(question or "").strip()
    resolved_focus_genes = _resolve_focus_genes(question, focus_genes)
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
            direct = _run_cypher_query(question, question, "Executed direct read-only Cypher against PrimeKG.")
            if resolved_focus_genes:
                direct["focus_genes"] = resolved_focus_genes
            return direct

        if _should_use_gene_focused_query(question, resolved_focus_genes):
            focused = _run_cypher_query(
                question,
                _build_gene_focused_cypher(resolved_focus_genes),
                "PrimeKG answered using extracted gene focus terms.",
            )
            focused["focus_genes"] = resolved_focus_genes
            return focused

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

            llm_result = _run_cypher_query(question, cypher)
            if resolved_focus_genes:
                llm_result["focus_genes"] = resolved_focus_genes
            return llm_result
        except Exception as exc:
            llm_error = str(exc).strip()

        fallback_cypher = _build_rule_based_cypher(question, focus_genes=resolved_focus_genes)
        if fallback_cypher:
            fallback = _run_cypher_query(
                question,
                fallback_cypher,
                "PrimeKG answered using the local fallback query path because the LLM-backed path was unavailable.",
            )
            if resolved_focus_genes:
                fallback["focus_genes"] = resolved_focus_genes
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
