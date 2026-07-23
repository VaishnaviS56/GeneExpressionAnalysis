from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import networkx as nx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from gea_agent.agent.state import AgentState
from gea_agent.config import SETTINGS
from gea_agent.tools.disease_literature import fetch_openalex_papers_and_genes, identify_disease_from_query
from gea_agent.tools.druggability import run_druggability_analysis_safe, run_pdb_visualization_safe
from gea_agent.tools.deg_analysis import run_deg_r_analysis
from gea_agent.tools.enrichr import enrichr_pathways
from gea_agent.tools.extract_genes import extract_genes_from_text
from gea_agent.tools.hypothesis import generate_experimental_hypotheses_safe
from gea_agent.tools.l1000cds2 import query_l1000cds2
from gea_agent.tools.llm import get_llm, is_gemini_family_provider, parse_json_object
from gea_agent.tools.opentargets import (
    check_gene_disease_association,
    check_gene_list_disease_associations,
    find_diseases_for_gene,
    find_drugs_for_gene,
)
from gea_agent.tools.pubchem import query_pubchem_drug
from gea_agent.tools.research_literature import run_publication_research_assistant_safe
from gea_agent.tools.random_walk_restart import top_rwr_genes
from gea_agent.tools.result_utils import normalize_tool_result, sanitize_exception_message, tool_error_result
from gea_agent.tools.srp_metadata import fetch_srp_metadata_summary_safe
from gea_agent.tools.string_local_graph import build_weighted_graph_from_string_files, load_gene_to_string_id
from gea_agent.tools.synthesizer import synthesize_technical_response
from gea_agent.tools.primekg import query_primekg
from gea_agent.tools.srp_ids import extract_srp_ids_from_text
from gea_agent.tools.visualizers import (
    build_kegg_pathway_visualization,
    build_network_visualization,
    build_volcano_plot,
)


MAX_AGENT_STEPS = 10
MAX_LITERATURE_CALLS_PER_QUERY = 2

TOOL_USE_INSTRUCTIONS = '''
Tool selection guide and capability boundaries:
1. deg_analysis: Runs one supported DEG workflow only: DEE2 SRA project data identified by SRP accessions, analyzed with the configured R script that uses getDEE2/DEE2 inputs and DESeq2. Required parameters are `srp_ids`, `control_name`, and `test_name`; optional thresholds are `log2fold` and `padj`. Do not use for uploaded count matrices, GEO-only IDs without SRP accessions, single-cell data, proteomics, EdgeR, limma/Lemma, voom, NOISeq, custom design formulas, batch correction, or arbitrary local files. If the user requests an unsupported DEG source or method, say it is not available and name the supported option.
1b. srp_metadata: Retrieves DEE2/SRA metadata for SRP accessions when a DEG request has SRP IDs but lacks exact control/test cohort labels. It returns DEE2 descriptions and SRA/BioSample-derived treatment, sample_name, and disease values to help the user choose labels. It does not run DEG analysis or infer a contrast by itself.
2. pathway: Runs over-representation enrichment through Enrichr/gget only. Supported libraries are the configured Reactome, KEGG Human, and GO Biological Process/Molecular Function/Cellular Component sets. Inputs must be explicit genes or stored DEG/RWR/literature genes; use `direction` as `up`, `down`, or `all`, `gene_limit` for top-N gene selection, and `term_limit` for returned terms. Do not promise GSEA, ORA background customization beyond the tool's `background_genes`, WikiPathways, MSigDB, ReactomePA, clusterProfiler, or custom GMT support.
3. rwr_analysis: Performs Random Walk with Restart over the configured local STRING network only. Seed priority is strict: explicit genes named by the user first, then overlapping genes from stored pathway/enrichment results, then stored DEG genes. Supported parameters include `genes`, `top_k`/`top_n`, `restart_prob`, and STRING file/score settings. Do not claim it uses BioGRID, IntAct, tissue-specific networks, directed causal edges, or a newly downloaded network.
4. literature: Retrieves and ranks literature evidence through the agent's OpenAlex/PubMed/Google Scholar pipeline. It can use `disease_name`, `genes`, `top_n`, and `text`. Use it when the user asks to find/search/check evidence for a statement, claim, mechanism, biomarker, gene-disease link, or explicitly asks for PubMed/OpenAlex/Google Scholar/paper searches. Do not use it for full-text paywalled article extraction, systematic-review guarantees, clinical advice, or live database curation beyond the configured sources.
4b. research_literature: Produces a literature-style answer with references from model knowledge, optionally grounded by provided disease/genes. It does not perform live retrieval and its references are best-effort/model-generated unless later verified. Use for broad research, investigate, review, explain, summarize, overview, or "what is known about" requests that do not specifically ask to find/search/check evidence or run a PubMed/OpenAlex/Google Scholar/paper search; do not present its citations as newly searched or verified.
5. identify_disease_from_query: Extracts a disease label from text only. It does not validate diagnosis, map ontology IDs, or prove disease-gene associations.
6. primekg_query: Queries the configured local PrimeKG/Neo4j graph using read-only Cypher. It can answer graph relationships among PrimeKG entities such as genes/proteins, diseases, drugs, pathways, phenotypes, anatomy, biological processes, molecular functions, cellular components, and exposures. Do not use it for live web knowledge, non-PrimeKG facts, graph writes, schema changes, or unsupported labels/relationship types.
7. opentargets_association: Queries OpenTargets for gene-disease associations and gene-linked drugs, with gene normalization to Ensembl IDs via MyGene. Use for association evidence and top disease/drug lookups. Do not use it for pathway enrichment, DEG computation, PubChem chemistry, PrimeKG paths, or full clinical trial interpretation.
8. l1000cds2_query: Queries L1000CDS2 for small-molecule signatures from separate up-regulated and down-regulated gene lists. It supports reverse mode by default and mimic/aggravate mode when requested, optional cell-line filters, combination/share flags, and result limits. Do not use it for CMap APIs outside L1000CDS2, arbitrary drug mechanism lookup, or when only one undirected gene list is available unless a split is clearly specified or stored DEG directions exist.
9. pubchem_drug_lookup: Queries PubChem for a named compound, `pert_desc`, or BRD-like `pert_id`, then summarizes compound properties, synonyms, descriptions, and annotations. It can infer genes/pathways/diseases only when supported by returned PubChem text. Do not use it for drug perturbation signatures, L1000 ranking, target validation, clinical efficacy, or non-compound biomedical graph questions.
10. hypothesis: Generates plausible biomedical hypotheses and conceptual validation experiment ideas from the conversation and stored analysis memory only. It can suggest experiment designs, readouts, controls, expected observations, interpretation, caveats, rationale, and key assumptions. It does not validate hypotheses against external sources, search literature, cite references, assess novelty, retrieve new evidence, or provide step-by-step wet-lab protocols.
11. visualize: Creates only supported visual artifacts: `network` for STRING/PyVis HTML, `kegg` for KEGG/gget pathway image, and `volcano` for DEG volcano HTML. It needs stored graph/genes/pathway/DEG rows or explicit genes. Do not use it for heatmaps, PCA, UMAP, boxplots, survival plots, circos plots, dashboards, or custom figures unless implemented as a supported visualization type.
12. memory_lookup: Answers intersections, overlap-gene lookups, membership checks, and stored pathway/GO/DEG matching from current chat memory only. Do not use it for new analysis or external data.
13. state_lookup: Inspects literal stored state fields, values, and counts. Do not use it for biological interpretation or to invent missing outputs.
14. memory_slice: Selects top N and/or bottom N items from stored list-like state fields and makes that slice reusable. Do not use it for statistical ranking unless the stored list already has meaningful order.
15. druggability: Runs structure-backed binding-pocket/druggability analysis for one gene. It resolves the gene to UniProt, prefers an RCSB PDB when available, falls back to an AlphaFold PDB model, runs PDBFixer/OpenMM sanitization when installed, uploads the sanitized PDB to ProteinsPlus, submits a DoGSiteScorer job, waits for completion, downloads the result table plus top pocket residue/map files, and returns local file paths. Use only when the user asks for druggability, binding pockets, DoGSite, protein pocket prediction, or structure-based target assessment for a specific gene. It requires internet access and the remote ProteinsPlus service; if PDBFixer is missing, the tool reports that and uses the original PDB as a fallback.
16. pdb_visualizer: Fetches and visualizes a protein structure without running DoGSite. It accepts a gene symbol, UniProt accession, protein label, or direct PDB ID, prefers RCSB where available, falls back to AlphaFold via the latest AlphaFold API PDB URL, saves the PDB file, and creates an interactive 3Dmol HTML viewer. Use when the user asks to view, visualize, fetch, or show a PDB/protein structure for a gene/protein and does not ask for druggability or pocket scoring.

Operational guidance:
- Be conversational in direct replies: acknowledge what the user asked, state what can be done, and after completed analyses offer one concrete follow-up the agent can run next.
- If the user asks a simple non-technical question or casual follow-up, answer directly without tools.
- If the user requests a tool, source, method, input type, or output that is outside the capability boundaries above, do not call a substitute tool silently. Clearly say the requested capability is not available and offer the closest supported alternative.
- Call at most one specialist at a time. After each tool result, reassess whether the user is answered or whether another specialist would materially improve the result.
- Credible multi-specialist chains are encouraged when the user asks for a workflow, for example DEG -> pathway -> visualization, DEG -> RWR -> PrimeKG/OpenTargets, literature -> RWR, or L1000CDS2 -> PubChem.
- Keep chains bounded and purposeful. Do not call a specialist just because it is available; each call must add evidence or an artifact the user requested.
- Never call the `literature` specialist more than two times for a single user query. Reuse retrieved papers, references, and summaries after that limit.
- Prefer recovering missing prerequisites with tools instead of guessing. Example: use `literature` to get disease genes, gene-function evidence, or paper support before `pathway` or `rwr_analysis`.
- Prefer memory and current state before recomputing the same result.
- Use `pathway` first for enrichment questions unless the user explicitly asks for knowledge-graph pathway relationships.
- Use `primekg_query` first for "what connects", "what links", mediator, multi-hop, and graph-neighborhood questions.
- Use `opentargets_association` when the main task is evidence-backed association rather than broader graph exploration.
- For stored DEG follow-ups, interpret positive `log2FoldChange` as up-regulated and negative `log2FoldChange` as down-regulated.
- For DEG requests with SRP IDs but missing or ambiguous control/test labels, call `srp_metadata` first so the user can select exact cohort labels from DEE2/SRA metadata.
- If a specialist is used, let the workflow continue through the specialist/final synthesis path rather than answering from partial assumptions.
- Gemini/Gemma tool-calling rule for DEG requests: when a query contains SRP accessions and no unsupported DEG method/source is requested, call `deg_analysis` and place those accessions in the structured `srp_ids` list instead of leaving them only in prose.
- Preferred `deg_analysis` argument shape: `{"srp_ids":["SRP123456"],"control_name":"control cohort","test_name":"test cohort","log2fold":1.0,"padj":0.05,"text":"full user request"}`.
'''


def _trace_tool_call(name: str) -> None:
    print(f"[tool] {name}")


def _compact_text(value: Any, *, limit: int = 240) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


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


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _merge_unique(*groups: list[str] | tuple[str, ...] | None) -> list[str]:
    merged: list[str] = []
    for group in groups:
        if not group:
            continue
        for value in group:
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _genes_from_deg_records(records: Any, *, top_n: int | None = None) -> list[str]:
    if not isinstance(records, list):
        return []

    ranked: list[tuple[str, float]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        gene = str(row.get("gene") or "").strip().upper()
        if not gene:
            continue
        try:
            log2fc = float(row.get("log2FoldChange"))
        except Exception:
            log2fc = 0.0
        ranked.append((gene, log2fc))

    ranked.sort(key=lambda item: item[1], reverse=True)
    genes = [gene for gene, _ in ranked]
    if top_n is not None:
        return genes[: max(0, top_n)]
    return genes


def _genes_from_deg_records_by_direction(
    records: Any,
    *,
    direction: str = "all",
    top_n: int | None = None,
) -> list[str]:
    if not isinstance(records, list):
        return []

    ranked: list[tuple[str, float, float]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        gene = str(row.get("gene") or "").strip().upper()
        if not gene:
            continue
        try:
            log2fc = float(row.get("log2FoldChange"))
        except Exception:
            log2fc = 0.0
        if direction == "up" and log2fc <= 0:
            continue
        if direction == "down" and log2fc >= 0:
            continue
        ranked.append((gene, log2fc, abs(log2fc)))

    if direction == "down":
        ranked.sort(key=lambda item: item[1])
    elif direction in {"all", "both"}:
        ranked.sort(key=lambda item: item[2], reverse=True)
    else:
        ranked.sort(key=lambda item: item[1], reverse=True)

    genes = [gene for gene, _, _ in ranked]
    if top_n is not None:
        return genes[: max(0, top_n)]
    return genes


def _stored_deg_genes_by_direction(
    state: AgentState,
    *,
    direction: str = "all",
    top_n: int | None = None,
) -> list[str]:
    normalized_direction = str(direction or "all").strip().lower()
    if normalized_direction == "both":
        normalized_direction = "all"

    if normalized_direction == "up":
        genes = _merge_unique(
            state.get("upregulated_genes"),
            state.get("memory_upregulated_genes"),
        )
    elif normalized_direction == "down":
        genes = _merge_unique(
            state.get("downregulated_genes"),
            state.get("memory_downregulated_genes"),
        )
    else:
        genes = _merge_unique(
            state.get("deg_genes"),
            state.get("memory_deg_genes"),
        )

    if genes:
        return genes[: max(0, top_n)] if top_n is not None else genes

    return _genes_from_deg_records_by_direction(
        state.get("deg_gene_records") or state.get("memory_deg_gene_records"),
        direction=normalized_direction,
        top_n=top_n,
    )


def _deg_direction_from_query(text: str | None) -> str:
    query = str(text or "").lower()
    has_down = "down-regulated" in query or "down regulated" in query or "downregulated" in query
    has_up = "up-regulated" in query or "up regulated" in query or "upregulated" in query
    if (
        "both up and down" in query
        or "both up-regulated and down-regulated" in query
        or "both upregulated and downregulated" in query
        or "both up regulated and down regulated" in query
        or ("both" in query and has_up and has_down)
    ):
        return "both"
    if has_down:
        return "down"
    if has_up:
        return "up"
    return "all"


def _memory_gene_query_requested(text: str | None) -> bool:
    query = str(text or "").lower()
    return any(
        marker in query
        for marker in (
            "up-regulated genes",
            "up regulated genes",
            "upregulated genes",
            "down-regulated genes",
            "down regulated genes",
            "downregulated genes",
            "deg genes",
            "differentially expressed genes",
        )
    )


def _literature_followup_requested(text: str | None) -> bool:
    query = str(text or "").lower()
    return any(
        marker in query
        for marker in (
            "literature",
            "paper",
            "papers",
            "publication",
            "publications",
            "reference",
            "references",
            "pubmed",
            "google scholar",
            "openalex",
            "evidence",
            "studies",
            "study",
        )
    )


def _hypothesis_requested(text: str | None) -> bool:
    query = str(text or "").lower()
    return any(
        marker in query
        for marker in (
            "hypothesis",
            "hypotheses",
            "validate this",
            "validation experiment",
            "experimental validation",
            "follow-up experiment",
            "follow up experiment",
            "generate a hypothesis",
            "generate hypotheses",
            "propose a hypothesis",
            "propose hypotheses",
            "how can i validate",
            "what experiment should",
            "what experiments should",
            "suggest experiments",
        )
    )


def _literature_memory_gene_requested(text: str | None) -> bool:
    query = str(text or "").lower()
    return any(
        marker in query
        for marker in (
            "these genes",
            "those genes",
            "stored genes",
            "gene list",
            "genes from memory",
            "previous genes",
            "remembered genes",
        )
    )


def _literature_state_gene_candidates(state: AgentState, *, limit: int = 12) -> list[str]:
    candidates = _merge_unique(
        _memory_slice_gene_candidates(state),
        state.get("genes"),
        state.get("deg_genes"),
        state.get("memory_deg_genes"),
        state.get("openalex_genes"),
        state.get("memory_openalex_genes"),
        state.get("rwr_seed_genes"),
        state.get("memory_rwr_seed_genes"),
        [gene for gene, _ in (state.get("rwr_genes") or [])],
        [gene for gene, _ in (state.get("memory_rwr_genes") or [])],
    )
    genes: list[str] = []
    for value in candidates:
        gene = str(value or "").strip().upper()
        if gene and gene not in genes:
            genes.append(gene)
        if len(genes) >= limit:
            break
    return genes


def _memory_slice_field_from_query(text: str | None) -> str | None:
    query = str(text or "").lower()
    if any(marker in query for marker in ("upregulated genes", "up-regulated genes", "up regulated genes")):
        return "memory_upregulated_genes"
    if any(marker in query for marker in ("downregulated genes", "down-regulated genes", "down regulated genes")):
        return "memory_downregulated_genes"
    if any(marker in query for marker in ("deg genes", "differentially expressed genes")):
        return "memory_deg_genes"
    return None


def _slice_request_satisfied_by_memory(state: AgentState, query: str) -> bool:
    result = state.get("memory_slice_result")
    if not isinstance(result, dict):
        return False
    requested_field = _memory_slice_field_from_query(query)
    if requested_field:
        resolved = str(result.get("field") or result.get("requested_field") or "")
        if resolved != requested_field:
            return False
    requested_top_n = _parse_top_n_from_text(query)
    selected_values = result.get("selected_values")
    if not isinstance(selected_values, list) or not selected_values:
        return False
    if isinstance(requested_top_n, int) and requested_top_n > 0 and len(selected_values) < requested_top_n:
        return False
    return True


def _should_use_memory_slice_for_current_query(
    state: AgentState,
    query: str,
    *,
    requested_limit: int | None = None,
) -> bool:
    result = state.get("memory_slice_result")
    if not isinstance(result, dict):
        return False

    selected_values = result.get("selected_values")
    if not isinstance(selected_values, list) or not selected_values:
        return False

    requested_field = _memory_slice_field_from_query(query)
    if requested_field:
        resolved = str(result.get("field") or result.get("requested_field") or "")
        if resolved != requested_field:
            return False

    effective_limit = requested_limit
    if not isinstance(effective_limit, int) or effective_limit <= 0:
        parsed_limit = _parse_top_n_from_text(query)
        effective_limit = parsed_limit if isinstance(parsed_limit, int) and parsed_limit > 0 else None

    if isinstance(effective_limit, int) and effective_limit > 0 and len(selected_values) < effective_limit:
        return False

    return True


def _should_force_memory_slice_for_research_query(state: AgentState) -> bool:
    query = str(state.get("query") or "")
    lowered = query.lower()
    if not query.strip():
        return False
    if not (_should_force_research_literature_tool(state) or _should_force_literature_tool(state)):
        return False
    if not any(marker in lowered for marker in ("top ", "bottom ")):
        return False
    if not _memory_slice_field_from_query(query):
        return False
    if _slice_request_satisfied_by_memory(state, query):
        return False
    return True


def _should_chain_research_after_memory_slice(state: AgentState, result: dict[str, Any]) -> bool:
    query = str(state.get("query") or "")
    if not (_should_force_research_literature_tool(state) or _should_force_literature_tool(state)):
        return False
    if not isinstance(result, dict) or str(result.get("status") or "") != "ok":
        return False
    requested_field = _memory_slice_field_from_query(query)
    resolved = str(result.get("field") or result.get("requested_field") or "")
    if requested_field and resolved != requested_field:
        return False
    return True


def _literature_call_count(state: AgentState) -> int:
    count = 0
    for row in list(state.get("tool_history") or []):
        if not isinstance(row, dict):
            continue
        tool_name = str(row.get("tool") or "").strip()
        if tool_name in {"literature", "fetch_openalex_papers_and_genes"}:
            count += 1
    return count


def _looks_like_literature_query(text: str | None) -> bool:
    query = str(text or "")
    lowered = query.lower()
    literature_markers = (
        "literature",
        "paper",
        "papers",
        "reference",
        "references",
        "pubmed",
        "google scholar",
        "openalex",
        "evidence",
        "study",
        "studies",
    )
    if not any(marker in lowered for marker in literature_markers):
        return False
    return True


def _looks_like_research_literature_query(text: str | None) -> bool:
    query = str(text or "").lower()
    if _looks_like_literature_query(query):
        return True
    research_markers = (
        "research",
        "investigate",
        "look into",
        "review",
        "overview",
        "background on",
        "what is known about",
        "tell me about",
        "explain",
        "summarize",
    )
    return any(marker in query for marker in research_markers)


def _evidence_statement_search_requested(text: str | None) -> bool:
    query = " ".join(str(text or "").lower().split())
    if not query:
        return False
    evidence_markers = (
        "find evidence",
        "find the evidence",
        "look for evidence",
        "search for evidence",
        "show evidence",
        "give evidence",
        "evidence for",
        "evidence that",
        "supporting evidence",
        "literature evidence",
        "papers supporting",
        "studies supporting",
        "is there evidence",
        "any evidence",
    )
    return any(marker in query for marker in evidence_markers)


def _explicit_live_literature_search_requested(text: str | None) -> bool:
    query = " ".join(str(text or "").lower().split())
    if not query:
        return False
    live_markers = (
        "pubmed",
        "openalex",
        "google scholar",
        "search literature",
        "search the literature",
        "search papers",
        "find papers",
        "find publications",
        "find studies",
        "lookup papers",
        "look up papers",
        "paper search",
        "publication search",
        "study search",
    )
    return any(marker in query for marker in live_markers)


def _should_force_literature_tool(state: AgentState) -> bool:
    if _literature_call_count(state) >= MAX_LITERATURE_CALLS_PER_QUERY:
        return False

    query = str(state.get("query") or "")
    if not _looks_like_literature_query(query):
        return False

    lowered = query.lower()
    explicit_non_lit_markers = (
        "visualize",
        "plot",
        "volcano",
        "network",
        "kegg",
        "enrichr",
        "pathway enrichment",
        "l1000",
        "l1000cds2",
        "pubchem",
        "primekg",
    )
    if any(marker in lowered for marker in explicit_non_lit_markers):
        return False

    return bool(_evidence_statement_search_requested(query) or _explicit_live_literature_search_requested(query))


def _should_force_research_literature_tool(state: AgentState) -> bool:
    query = str(state.get("query") or "")
    lowered = query.lower()
    if not _looks_like_research_literature_query(query):
        return False
    if _evidence_statement_search_requested(query):
        return False
    if _explicit_live_literature_search_requested(query):
        return False

    explicit_non_lit_markers = (
        "visualize",
        "plot",
        "volcano",
        "network",
        "kegg",
        "enrichr",
        "pathway enrichment",
        "l1000",
        "l1000cds2",
        "pubchem",
        "primekg",
    )
    if any(marker in lowered for marker in explicit_non_lit_markers):
        return False

    direct_answer_markers = (
        "with references",
        "with citations",
        "give references",
        "give citations",
        "include references",
        "include citations",
        "answer with references",
        "answer with citations",
        "using literature",
        "use literature",
        "research",
        "investigate",
        "look into",
        "review",
        "overview",
        "background on",
        "what is known about",
        "tell me about",
        "explain",
        "summarize",
    )
    if not any(marker in lowered for marker in direct_answer_markers):
        return False

    return True


def _is_simple_conversational_query(text: str | None) -> bool:
    query = " ".join(str(text or "").lower().split())
    if not query:
        return False
    simple_exact = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "great",
        "cool",
        "bye",
    }
    if query in simple_exact:
        return True
    simple_prefixes = (
        "hi ",
        "hello ",
        "hey ",
        "thanks ",
        "thank you ",
    )
    return any(query.startswith(prefix) for prefix in simple_prefixes)


def _unsupported_deg_request_message(text: str | None) -> str:
    query = " ".join(str(text or "").lower().split())
    if not query:
        return ""

    deg_requested = any(
        marker in query
        for marker in (
            "deg",
            "diferential expression",
            "diferentially expressed",
            "differential expression",
            "differentially expressed",
            "deseq",
            "deseq2",
            "edger",
            "edge r",
            "edge-r",
            "limma",
            "lemma",
            "voom",
            "noiseq",
        )
    )
    if not deg_requested:
        return ""

    unsupported_methods = {
        "edger": "edgeR",
        "edge r": "edgeR",
        "edge-r": "edgeR",
        "limma": "limma",
        "lemma": "limma/Lemma",
        "voom": "voom",
        "noiseq": "NOISeq",
    }
    requested_methods = [
        label
        for marker, label in unsupported_methods.items()
        if marker in query
    ]
    if requested_methods:
        methods = ", ".join(dict.fromkeys(requested_methods))
        return (
            f"{methods} is not available in this agent's DEG workflow. "
            "The supported DEG path uses DEE2 datasets identified by SRP accession IDs and runs DESeq2 through the configured R script. "
            "If you provide SRP IDs plus control and test cohort labels, I can run that DESeq2-based DEE2 analysis."
        )

    unsupported_sources = {
        "geo": "GEO-only accessions",
        "gse": "GSE/GEO-only accessions",
        "uploaded": "uploaded files",
        "upload": "uploaded files",
        "count matrix": "custom count matrices",
        "counts matrix": "custom count matrices",
        "single cell": "single-cell data",
        "scrna": "single-cell data",
        "proteomics": "proteomics data",
    }
    requested_sources = [
        label
        for marker, label in unsupported_sources.items()
        if marker in query
    ]
    if requested_sources and not extract_srp_ids_from_text(str(text or "")):
        sources = ", ".join(dict.fromkeys(requested_sources))
        return (
            f"{sources} are not available for DEG analysis in this agent unless they resolve to DEE2 SRP accessions. "
            "The supported DEG path is DEE2 SRP data analyzed with DESeq2. "
            "Send SRP IDs plus control and test cohort labels to run it."
        )

    return ""


def _looks_like_deg_analysis_query(text: str | None) -> bool:
    query = " ".join(str(text or "").lower().split())
    if not query:
        return False
    return any(
        marker in query
        for marker in (
            "deg",
            "differential expression",
            "differentially expressed",
            "diferential expression",
            "diferentially expressed",
            "differentially expressed genes",
            "diferentially expressed genes",
            "identify differential",
            "identify diferential",
        )
    )


def _looks_like_srp_metadata_query(text: str | None) -> bool:
    query = " ".join(str(text or "").lower().split())
    if not query:
        return False
    return any(
        marker in query
        for marker in (
            "srp metadata",
            "sra metadata",
            "dee2 metadata",
            "metadata",
            "sample labels",
            "sample names",
            "treatment values",
            "cohort labels",
            "control and test",
            "control/test",
            "what are the groups",
            "which groups",
        )
    )


def _deg_group_labels_available(state: AgentState, text: str | None, args: dict[str, Any] | None = None) -> bool:
    args = args or {}
    control_name = " ".join(
        str(args.get("control_name") or state.get("control_name") or state.get("memory_control_name") or "").split()
    ).strip()
    test_name = " ".join(
        str(args.get("test_name") or state.get("test_name") or state.get("memory_test_name") or "").split()
    ).strip()
    if control_name and test_name:
        return True

    parsed_groups = _extract_deg_group_labels_from_text(text)
    return bool(
        (control_name or str(parsed_groups.get("control_name") or "").strip())
        and (test_name or str(parsed_groups.get("test_name") or "").strip())
    )


def _should_force_srp_metadata(state: AgentState) -> bool:
    query = str(state.get("query") or "")
    srp_ids = _normalize_srp_ids(state.get("srp_ids")) or _normalize_srp_ids(query)
    if not srp_ids:
        return False
    if _ensure_dict(state.get("srp_metadata_result")):
        return False
    if _looks_like_srp_metadata_query(query):
        return True
    if not _looks_like_deg_analysis_query(query):
        return False
    return not _deg_group_labels_available(state, query)


def _looks_like_druggability_query(text: str | None) -> bool:
    query = " ".join(str(text or "").lower().split())
    if not query:
        return False
    return any(
        marker in query
        for marker in (
            "druggability",
            "druggable",
            "binding pocket",
            "binding pockets",
            "protein pocket",
            "protein pockets",
            "pocket prediction",
            "dogsite",
            "dogsite scorer",
            "dogsitescorer",
            "proteins.plus",
            "proteins plus",
            "structure-based target",
            "structure based target",
        )
    )


def _looks_like_pdb_visualizer_query(text: str | None) -> bool:
    query = " ".join(str(text or "").lower().split())
    if not query:
        return False
    if _looks_like_druggability_query(query):
        return False
    return any(
        marker in query
        for marker in (
            "pdb viewer",
            "pdb visualizer",
            "visualize pdb",
            "visualise pdb",
            "show pdb",
            "fetch pdb",
            "download pdb",
            "protein structure",
            "visualize protein",
            "visualise protein",
            "show protein",
            "view protein",
            "3d structure",
            "3d protein",
            "alphafold structure",
            "alphafold pdb",
        )
    )


def _should_force_pdb_visualizer(state: AgentState) -> bool:
    if not _looks_like_pdb_visualizer_query(state.get("query")):
        return False
    query = str(state.get("query") or "")
    query_genes = extract_genes_from_text(query, mode="strict")
    pdb_ids = re.findall(r"\b[0-9][A-Za-z0-9]{3}\b", query)
    existing = _ensure_dict(state.get("pdb_visualization_result"))
    existing_gene = str(existing.get("gene") or existing.get("uniprot_id") or existing.get("pdb_id") or "").strip().upper()
    current_candidates = [str(value).strip().upper() for value in (query_genes + pdb_ids) if str(value).strip()]
    if existing_gene and current_candidates and all(value == existing_gene for value in current_candidates):
        return False
    return bool(current_candidates or state.get("genes") or state.get("memory_deg_genes"))


def _should_force_druggability(state: AgentState) -> bool:
    if not _looks_like_druggability_query(state.get("query")):
        return False
    query_genes = extract_genes_from_text(str(state.get("query") or ""), mode="strict")
    existing_gene = str(_ensure_dict(state.get("druggability_result")).get("gene") or "").strip().upper()
    if existing_gene and not any(str(gene).strip().upper() != existing_gene for gene in query_genes):
        return False
    return bool(query_genes or state.get("genes") or state.get("memory_deg_genes"))


def _should_force_deg_analysis(state: AgentState) -> bool:
    query = str(state.get("query") or "")
    if _unsupported_deg_request_message(query):
        return False
    srp_ids = _normalize_srp_ids(state.get("srp_ids")) or _normalize_srp_ids(query)
    if not srp_ids:
        return False
    if isinstance(state.get("deg_analysis"), dict) and state.get("deg_analysis"):
        return False
    if not _deg_group_labels_available(state, query):
        return False
    return _looks_like_deg_analysis_query(query)


def _extract_deg_group_labels_from_text(text: str | None) -> dict[str, str]:
    query = " ".join(str(text or "").split()).strip()
    if not query:
        return {"control_name": "", "test_name": ""}

    def find_label(patterns: tuple[str, ...]) -> str:
        for pattern in patterns:
            match = re.search(pattern, query, flags=re.IGNORECASE)
            if not match:
                continue
            value = str(match.group(1) or "").strip().strip("\"'` .,:;")
            value = " ".join(value.split())
            if value:
                return value
        return ""

    control_name = find_label(
        (
            r"\bcontrol\s+(?:as|is|=|:)\s+[\"']([^\"']+)[\"']",
            r"\bcontrol\s+(?:as|is|=|:)\s+(.+?)(?:\s+\band\b\s+test\b|\s+\bvs\b|\s+\bversus\b|$)",
        )
    )
    test_name = find_label(
        (
            r"\btest\s+(?:as|is|=|:)\s+[\"']([^\"']+)[\"']",
            r"\btest\s+(?:as|is|=|:)\s+(.+?)(?:\s+\band\b\s+control\b|\s+\bvs\b|\s+\bversus\b|$)",
        )
    )
    return {"control_name": control_name, "test_name": test_name}


def _looks_like_pathway_enrichment_query(text: str | None) -> bool:
    query = str(text or "").lower()
    enrichment_markers = (
        "pathway enrichment",
        "enrichment",
        "enrichr",
        "go term",
        "go enrichment",
        "pathways",
        "pathway",
        "reactome",
        "kegg",
        "biological process",
    )
    if not any(marker in query for marker in enrichment_markers):
        return False
    blocked_markers = (
        "primekg",
        "knowledge graph",
        "what connects",
        "network",
        "visualize",
        "volcano",
        "paper",
        "papers",
        "literature",
        "pubmed",
    )
    return not any(marker in query for marker in blocked_markers)


def _should_force_pathway_tool(state: AgentState) -> bool:
    query = str(state.get("query") or "")
    if not _looks_like_pathway_enrichment_query(query):
        return False
    if _ensure_dict(state.get("enrichr")).get("libraries") or _ensure_dict(state.get("memory_enrichr")).get("libraries"):
        return False
    genes_in_query = [
        str(value).strip().upper()
        for value in extract_genes_from_text(query, mode="strict")
        if str(value).strip()
    ]
    remembered_genes = _merge_unique(
        _memory_slice_gene_candidates(state),
        state.get("genes"),
        state.get("deg_genes"),
        state.get("memory_deg_genes"),
        [gene for gene, _ in (state.get("rwr_genes") or [])],
        state.get("openalex_genes"),
        state.get("memory_openalex_genes"),
    )
    return bool(genes_in_query or remembered_genes or state.get("deg_gene_records") or state.get("memory_deg_gene_records"))


_KNOWN_CELL_LINES = {
    "A375",
    "A549",
    "ASC",
    "BT20",
    "HA1E",
    "HCC515",
    "HEPG2",
    "HT29",
    "MCF10A",
    "MCF7",
    "NOMO1",
    "NPC",
    "PC3",
    "PHH",
    "SKB",
    "THP1",
    "U2OS",
    "VCAP",
    "YAPC",
}


def _extract_cell_lines_from_text(text: str | None) -> list[str]:
    query = str(text or "")
    matches: list[str] = []
    patterns = (
        r"\b([A-Za-z0-9-]{2,15})\s+cell\s+line\b",
        r"\bcell\s+line\s+([A-Za-z0-9-]{2,15})\b",
        r"\b([A-Za-z0-9-]{2,15})\s+cells\b",
        r"\bcell\s+lines?\s*[:\-]?\s*([A-Za-z0-9,\s-]{2,80})",
    )
    for pattern in patterns:
        for group in re.findall(pattern, query, flags=re.IGNORECASE):
            if not isinstance(group, str):
                continue
            for token in re.split(r"[,/]| and ", group):
                value = str(token or "").strip().upper()
                if 2 <= len(value) <= 15 and any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value):
                    matches.append(value)

    for token in re.findall(r"\b[A-Za-z0-9-]{2,15}\b", query):
        upper = token.upper()
        if upper in _KNOWN_CELL_LINES:
            matches.append(upper)

    return list(dict.fromkeys(matches))


def _l1000_mode_from_query(text: str | None) -> bool:
    query = str(text or "").lower()
    if "mimic" in query or "aggravate" in query:
        return True
    if "reverse" in query or "reversal" in query:
        return False
    return False


def _memory_lookup_gene_candidates(state: AgentState) -> list[str]:
    result = state.get("memory_lookup_result")
    if not isinstance(result, dict):
        return []

    genes = _merge_unique(
        result.get("intersection_genes") if isinstance(result.get("intersection_genes"), list) else [],
        result.get("deg_genes") if isinstance(result.get("deg_genes"), list) else [],
        (
            result.get("selected_term", {}).get("overlapping_genes")
            if isinstance(result.get("selected_term"), dict)
            and isinstance(result.get("selected_term", {}).get("overlapping_genes"), list)
            else []
        ),
        result.get("mentioned_genes") if isinstance(result.get("mentioned_genes"), list) else [],
    )
    return [str(value).strip().upper() for value in genes if str(value).strip()]


def _query_gene_candidates(text: str | None) -> list[str]:
    query = str(text or "")
    strict_genes = [
        str(value).strip().upper()
        for value in extract_genes_from_text(query, mode="strict")
        if str(value).strip()
    ]
    uppercase_symbols: list[str] = []
    blocked = {
        "DEG",
        "DEGS",
        "GO",
        "KEGG",
        "RWR",
        "SRP",
        "STRING",
    }
    for token in re.findall(r"\b[A-Z0-9]{2,12}\b", query):
        symbol = token.strip().upper()
        if symbol in blocked or symbol.startswith("SRP") or symbol.isdigit():
            continue
        if not re.search(r"[A-Z]", symbol):
            continue
        uppercase_symbols.append(symbol)
    return list(
        dict.fromkeys(
            strict_genes + uppercase_symbols
        )
    )


def _enrichr_overlap_gene_candidates(
    state: AgentState,
    *,
    query: str = "",
    limit: int | None = None,
) -> list[str]:
    effective_limit = limit if isinstance(limit, int) and limit > 0 else 50
    pathway_term = _extract_requested_pathway_name(query)
    if pathway_term.lower() in {"this", "it", "that", "these", "the result", "the results", "the pathway", "the pathways"}:
        pathway_term = ""

    if pathway_term:
        selected_term, _selected_library, _selected_rank = _find_enrichr_term_from_state(
            state,
            pathway_term,
            query=query,
        )
        genes = _term_overlapping_genes(selected_term)
        if genes:
            return genes[:effective_limit]

    genes: list[str] = []
    libraries = _enrichr_libraries_from_state(state)
    for terms in libraries.values():
        if not isinstance(terms, list):
            continue
        for term in terms[:3]:
            for gene in _term_overlapping_genes(term if isinstance(term, dict) else None):
                if gene not in genes:
                    genes.append(gene)
                if len(genes) >= effective_limit:
                    return genes
    return genes


def _resolve_rwr_source_genes(
    state: AgentState,
    args: dict[str, Any],
    *,
    prefer_seed_genes: bool = False,
) -> tuple[list[str], str]:
    query = str(args.get("text") or state.get("query") or "")
    direct_keys = ("seed_genes", "genes") if prefer_seed_genes else ("genes", "seed_genes")
    for direct_key in direct_keys:
        direct_genes = args.get(direct_key)
        if isinstance(direct_genes, list) and direct_genes:
            return [str(value).strip().upper() for value in direct_genes if str(value).strip()], "tool_args"

    query_genes = _query_gene_candidates(query)
    if query_genes:
        return query_genes, "query_genes"

    pathway_genes = _enrichr_overlap_gene_candidates(
        state,
        query=query,
        limit=_parse_top_n_from_text(query),
    )
    if pathway_genes:
        return pathway_genes, "stored_enrichr_overlap_genes"

    memory_lookup_genes = _memory_lookup_gene_candidates(state)
    if memory_lookup_genes:
        return memory_lookup_genes, "memory_lookup"

    sliced_genes = _memory_slice_gene_candidates(state)
    if sliced_genes:
        return sliced_genes, "memory_slice"

    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    deg_records = (
        state.get("memory_deg_gene_records")
        if analysis_arm == "memory_rwr"
        else (state.get("deg_gene_records") or state.get("memory_deg_gene_records"))
    )
    directional_deg_genes = _genes_from_deg_records_by_direction(
        deg_records,
        direction=direction,
        top_n=top_n if isinstance(top_n, int) and top_n > 0 else (20 if prefer_seed_genes else None),
    )
    if directional_deg_genes:
        return directional_deg_genes, "stored_deg_directional"

    if analysis_arm == "memory_rwr":
        genes = _merge_unique(
            _genes_from_deg_records(state.get("memory_deg_gene_records"), top_n=20 if prefer_seed_genes else None),
            state.get("memory_deg_genes"),
            state.get("memory_rwr_seed_genes") if prefer_seed_genes else [],
            state.get("genes"),
        )
    else:
        genes = _merge_unique(
            _genes_from_deg_records(state.get("deg_gene_records") or state.get("memory_deg_gene_records"), top_n=20 if prefer_seed_genes else None),
            state.get("deg_genes"),
            state.get("memory_deg_genes"),
            state.get("openalex_genes"),
            state.get("memory_openalex_genes"),
            state.get("genes"),
        )
    return [str(value).strip().upper() for value in genes if str(value).strip()], "state_fallback"


def _disease_from_association_query(text: str | None) -> str:
    query = str(text or "").strip()
    patterns = (
        r"\bassociated\s+with\s+(.+?)(?:\?|$)",
        r"\bassociation\s+with\s+(.+?)(?:\?|$)",
        r"\blinked\s+to\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            disease = re.sub(r"\b(the|a|an)\b", "", match.group(1), flags=re.IGNORECASE)
            disease = " ".join(disease.split()).strip(" .,:;")
            if disease:
                return disease
    return ""


def _drug_association_query_requested(text: str | None) -> bool:
    query = str(text or "").lower()
    return any(
        marker in query
        for marker in (
            "drug associated with",
            "drugs associated with",
            "drug association for",
            "drugs for",
            "drug for gene",
            "drugs for gene",
            "what drugs",
            "which drugs",
            "target this gene with drugs",
        )
    )


def _pubchem_query_requested(text: str | None) -> bool:
    query = str(text or "").lower()
    return "pubchem" in query or (
        ("drug" in query or "compound" in query)
        and any(
            marker in query
            for marker in (
                "what pathways",
                "which pathways",
                "what genes",
                "which genes",
                "what diseases",
                "which diseases",
                "from pubchem",
            )
        )
    )


def _primekg_target_types_from_query(text: str | None) -> list[str]:
    query = str(text or "").lower()
    types: list[str] = []
    explicit_kg = "primekg" in query or "kg" in query or "knowledge graph" in query
    if "drug" in query or "drugs" in query or "compound" in query or "treatment" in query:
        types.append("drug")
    if "disease" in query or "diseases" in query or "diabetes" in query or "cancer" in query:
        types.append("disease")
    if "phenotype" in query or "phenotypes" in query or "symptom" in query or "side effect" in query:
        types.append("effect/phenotype")
    if explicit_kg and ("pathway" in query or "pathways" in query):
        types.append("pathway")
    if "gene" in query or "genes" in query or "protein" in query or "proteins" in query:
        types.append("gene/protein")
    return types


def _parse_top_n_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\btop\s+(\d+)\b", str(text), re.IGNORECASE)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except Exception:
        return None
    return value if value > 0 else None


def _parse_deg_thresholds(text: str | None, args: dict[str, Any] | None = None) -> tuple[float, float]:
    default_log2fold = 1.0
    default_padj = 0.05
    if isinstance(args, dict):
        raw_log2fold = args.get("log2fold")
        raw_padj = args.get("padj")
        try:
            if raw_log2fold not in (None, ""):
                default_log2fold = abs(float(raw_log2fold))
        except Exception:
            pass
        try:
            if raw_padj not in (None, ""):
                default_padj = max(0.0, min(1.0, float(raw_padj)))
        except Exception:
            pass

    query = str(text or "")
    patterns_log2fold = (
        r"\blog2\s*fold(?:change)?\s*[=:<>]?\s*([0-9]*\.?[0-9]+)\b",
        r"\blog2fc\s*[=:<>]?\s*([0-9]*\.?[0-9]+)\b",
        r"\blfc\s*[=:<>]?\s*([0-9]*\.?[0-9]+)\b",
    )
    patterns_padj = (
        r"\bpadj\s*[=:<>]?\s*([0-9]*\.?[0-9]+)\b",
        r"\badjusted\s+p(?:[- ]?value)?\s*[=:<>]?\s*([0-9]*\.?[0-9]+)\b",
        r"\bfdr\s*[=:<>]?\s*([0-9]*\.?[0-9]+)\b",
    )

    for pattern in patterns_log2fold:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            try:
                default_log2fold = abs(float(match.group(1)))
                break
            except Exception:
                pass

    for pattern in patterns_padj:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            try:
                default_padj = max(0.0, min(1.0, float(match.group(1))))
                break
            except Exception:
                pass

    return float(default_log2fold), float(default_padj)


def _normalize_text_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _normalize_pathway_query_token(value: Any) -> str:
    normalized = _normalize_text_token(value)
    stopwords = {
        "the",
        "a",
        "an",
        "please",
        "can",
        "you",
        "visualize",
        "visualise",
        "show",
        "plot",
        "render",
        "pathway",
        "pathways",
        "for",
        "of",
        "using",
        "stored",
        "memory",
    }
    return " ".join(token for token in normalized.split() if token and token not in stopwords)


def _pathway_match_score(label: str, desired: str, query: str) -> int:
    label_norm = _normalize_pathway_query_token(label)
    desired_norm = _normalize_pathway_query_token(desired)
    query_norm = _normalize_pathway_query_token(query)
    if not label_norm:
        return 0

    candidates = [value for value in (desired_norm, query_norm) if value]
    best = 0
    for candidate in candidates:
        if label_norm == candidate:
            best = max(best, 1200)
        elif candidate in label_norm or label_norm in candidate:
            best = max(best, 900 + min(len(candidate), len(label_norm)))
        else:
            label_tokens = set(label_norm.split())
            candidate_tokens = set(candidate.split())
            if not candidate_tokens:
                continue
            overlap = label_tokens.intersection(candidate_tokens)
            if overlap and len(overlap) >= min(2, len(candidate_tokens)):
                best = max(best, 500 + (100 * len(overlap)))
    return best


def _enrichr_libraries_from_state(state: AgentState) -> dict[str, Any]:
    for candidate in (state.get("enrichr"), state.get("memory_enrichr")):
        if not isinstance(candidate, dict):
            continue
        libraries = candidate.get("libraries")
        if isinstance(libraries, dict) and libraries:
            return libraries
        top_pathways = candidate.get("top_pathways")
        if isinstance(top_pathways, dict) and top_pathways:
            return top_pathways
    return {}


def _term_overlapping_genes(term: dict[str, Any] | None) -> list[str]:
    if not isinstance(term, dict):
        return []
    raw = (
        term.get("overlapping_genes")
        or term.get("overlap_genes")
        or term.get("genes")
        or term.get("Overlap")
        or []
    )
    if isinstance(raw, str):
        values = re.split(r"[;,]", raw)
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    return list(
        dict.fromkeys(
            str(value).strip().upper()
            for value in values
            if str(value).strip()
        )
    )


def _extract_requested_pathway_name(text: str | None) -> str:
    query = " ".join(str(text or "").split()).strip()
    if not query:
        return ""

    patterns = (
        r"\b(?:overlap|overlapping|shared|common)\s+genes\s+for\s+(.+)$",
        r"\bgenes\s+for\s+(.+)$",
        r"\bfor\s+(.+)$",
        r"(?:visuali[sz]e|show|plot|render)\s+(?:the\s+)?(.+?)\s+pathway\b",
        r"(?:visuali[sz]e|show|plot|render)\s+(?:the\s+)?(.+)$",
        r"\bpathway\s+(?:called|named)\s+(.+)$",
        r"\b(.+?)\s+pathway\b",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = " ".join(str(match.group(1) or "").split()).strip(" .,:;")
        if candidate:
            return candidate
    return ""


def _should_force_stored_pathway_visualization(state: AgentState) -> bool:
    query = str(state.get("query") or "")
    query_norm = _normalize_text_token(query)
    if not query_norm:
        return False
    if not any(token in query_norm for token in ("visualize", "visualise", "show", "plot", "render")):
        return False
    if any(token in query_norm for token in ("connection", "connections", "connected", "interaction", "interactions", "rwr", "random walk", "network propagation")):
        return False

    libraries = _enrichr_libraries_from_state(state)
    if not libraries:
        return False

    pathway_term = _extract_requested_pathway_name(query)
    selected_term, _selected_library, _selected_rank = _find_enrichr_term_from_state(
        state,
        pathway_term,
        query=query,
    )
    return bool(
        isinstance(selected_term, dict)
        and _term_overlapping_genes(selected_term)
    )


def _should_force_rwr_visualization(state: AgentState) -> bool:
    query_norm = _normalize_text_token(state.get("query"))
    if not query_norm:
        return False
    wants_visual = any(token in query_norm for token in ("visualize", "visualise", "show", "plot", "render"))
    wants_rwr_graph = "rwr" in query_norm and (
        wants_visual
        or any(token in query_norm for token in ("graph", "network", "visualization"))
    )
    has_rwr_memory = bool(
        state.get("rwr_seed_genes")
        or state.get("memory_rwr_seed_genes")
        or state.get("rwr_genes")
        or state.get("memory_rwr_genes")
    )
    return bool(wants_visual and wants_rwr_graph and has_rwr_memory)


def _should_force_volcano_visualization(state: AgentState) -> bool:
    query_norm = _normalize_text_token(state.get("query"))
    if not query_norm or "volcano" not in query_norm:
        return False
    wants_visual = any(token in query_norm for token in ("visualize", "visualise", "show", "plot", "render", "make", "create", "generate"))
    has_deg_rows = bool(state.get("deg_gene_records") or state.get("memory_deg_gene_records"))
    return bool(wants_visual and has_deg_rows)


def _should_force_pathway_rwr(state: AgentState) -> bool:
    query_norm = _normalize_text_token(state.get("query"))
    if not query_norm:
        return False
    if state.get("rwr_genes"):
        return False

    wants_connections = any(
        marker in query_norm
        for marker in (
            "rwr",
            "random walk",
            "network propagation",
            "prioritize",
            "prioritise",
            "candidate targets",
            "target prioritization",
            "target prioritisation",
            "connections",
            "connection",
            "connected",
            "interaction",
            "interactions",
            "around this",
            "around these",
            "around the pathway",
            "around the pathways",
        )
    )
    if not wants_connections:
        return False
    if any(marker in query_norm for marker in ("primekg", "knowledge graph", "opentargets", "pubchem", "l1000", "l1000cds2")):
        return False
    return bool(_enrichr_overlap_gene_candidates(state, query=str(state.get("query") or ""), limit=_parse_top_n_from_text(state.get("query"))))


def _should_force_memory_lookup(state: AgentState) -> bool:
    query = str(state.get("query") or "")
    query_norm = _normalize_text_token(query)
    if not query_norm:
        return False

    wants_memory_overlap = any(
        phrase in query_norm
        for phrase in (
            "overlap genes",
            "overlapping genes",
            "shared genes",
            "common genes",
            "genes in pathway",
            "pathway genes",
            "genes in go term",
            "genes in term",
        )
    )
    if not wants_memory_overlap:
        wants_deg_memory = _memory_gene_query_requested(query)
        asks_for_list_or_count = any(
            phrase in query_norm
            for phrase in (
                "list",
                "show",
                "display",
                "print",
                "which genes",
                "what genes",
                "how many",
                "count",
                "number of",
                "total",
            )
        )
        has_stored_deg_memory = bool(
            state.get("deg_gene_records")
            or state.get("memory_deg_gene_records")
            or state.get("upregulated_genes")
            or state.get("downregulated_genes")
            or state.get("memory_upregulated_genes")
            or state.get("memory_downregulated_genes")
            or state.get("deg_genes")
            or state.get("memory_deg_genes")
        )
        return bool(wants_deg_memory and asks_for_list_or_count and has_stored_deg_memory)

    return bool(_enrichr_libraries_from_state(state))


def _state_field_names() -> list[str]:
    return [str(name) for name in AgentState.__annotations__.keys()]


def _state_field_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {
        "upregulated genes": "upregulated_genes",
        "up regulated genes": "upregulated_genes",
        "downregulated genes": "downregulated_genes",
        "down regulated genes": "downregulated_genes",
        "deg genes": "deg_genes",
        "differentially expressed genes": "deg_genes",
        "rwr seed genes": "rwr_seed_genes",
        "rwr seeds": "rwr_seed_genes",
        "seed genes for rwr": "rwr_seed_genes",
        "seed genes used for rwr": "rwr_seed_genes",
        "seed genes after rwr": "rwr_seed_genes",
        "seeds after rwr": "rwr_seed_genes",
        "stored rwr seed genes": "memory_rwr_seed_genes",
        "stored rwr seeds": "memory_rwr_seed_genes",
        "memory rwr seed genes": "memory_rwr_seed_genes",
        "memory rwr seeds": "memory_rwr_seed_genes",
        "rwr result genes": "rwr_genes",
        "rwr results": "rwr_genes",
        "rwr genes": "rwr_genes",
        "rwr ranked genes": "rwr_genes",
        "rwr output genes": "rwr_genes",
        "rwr target genes": "rwr_genes",
        "rwr hits": "rwr_genes",
        "stored rwr result genes": "memory_rwr_genes",
        "stored rwr results": "memory_rwr_genes",
        "stored rwr genes": "memory_rwr_genes",
        "stored rwr targets": "memory_rwr_genes",
        "memory rwr result genes": "memory_rwr_genes",
        "memory rwr results": "memory_rwr_genes",
        "memory rwr genes": "memory_rwr_genes",
    }
    return {_normalize_text_token(alias): field for alias, field in aliases.items()}


def _match_state_fields(query: str, explicit_fields: Any = None) -> list[str]:
    requested: list[str] = []
    if isinstance(explicit_fields, list):
        requested.extend(str(value).strip() for value in explicit_fields if str(value).strip())
    elif explicit_fields not in (None, ""):
        requested.append(str(explicit_fields).strip())

    text = str(query or "")
    normalized_query = _normalize_text_token(text)
    alias_map: dict[str, str] = {}
    for field in _state_field_names():
        alias_map[field] = field
        alias_map[_normalize_text_token(field)] = field
        alias_map[_normalize_text_token(field.replace("_", " "))] = field
        if field.startswith("memory_"):
            alias_map[_normalize_text_token(field[len("memory_"):])] = field
    alias_map.update(_state_field_aliases())

    matched_aliases: list[tuple[int, str, str]] = []
    for alias, field in alias_map.items():
        if alias and alias in normalized_query:
            matched_aliases.append((len(alias), alias, field))
    matched_aliases.sort(key=lambda item: (-item[0], item[1]))
    for _alias_len, _alias, field in matched_aliases:
        if field not in requested:
            requested.append(field)

    resolved: list[str] = []
    seen: set[str] = set()
    for value in requested:
        normalized = _normalize_text_token(value)
        field = alias_map.get(value) or alias_map.get(normalized)
        if field and field not in seen:
            resolved.append(field)
            seen.add(field)
    return resolved


def _state_value_summary(value: Any, *, max_items: int = 25) -> Any:
    if isinstance(value, nx.Graph):
        return _graph_summary(value)
    if isinstance(value, BaseMessage):
        return {
            "type": type(value).__name__,
            "content": _compact_text(_message_content_text(getattr(value, "content", "")), limit=300),
        }
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in list(value.keys())[:max_items]:
            out[str(key)] = _state_value_summary(value[key], max_items=max_items)
        return out
    if isinstance(value, list):
        return [_state_value_summary(item, max_items=max_items) for item in value[:max_items]]
    if isinstance(value, tuple):
        return [_state_value_summary(item, max_items=max_items) for item in list(value)[:max_items]]
    if isinstance(value, set):
        items = sorted(str(item) for item in value)
        return items[:max_items]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _compact_text(value, limit=300)


def _state_value_length(value: Any) -> int | None:
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value)
    return None


def _as_listlike_state_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(list(value), key=lambda item: str(item))
    return []


def _resolve_listlike_state_field(state: AgentState, field: str) -> tuple[str, list[Any]]:
    resolved_field = str(field or "").strip()
    values = _as_listlike_state_value(state.get(resolved_field))
    if values or not resolved_field or resolved_field.startswith("memory_"):
        return resolved_field, values

    memory_field = f"memory_{resolved_field}"
    memory_values = _as_listlike_state_value(state.get(memory_field))
    if memory_values:
        return memory_field, memory_values
    return resolved_field, values


def _resolve_state_field_value(state: AgentState, field: str) -> tuple[str, bool, Any]:
    resolved_field = str(field or "").strip()
    if resolved_field in state:
        value = state.get(resolved_field)
        if value not in (None, "", [], {}):
            return resolved_field, True, value

    if resolved_field and not resolved_field.startswith("memory_"):
        memory_field = f"memory_{resolved_field}"
        if memory_field in state:
            return memory_field, True, state.get(memory_field)

    return resolved_field, resolved_field in state, state.get(resolved_field)


def _selected_values_to_gene_candidates(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    genes: list[str] = []
    for item in values:
        if isinstance(item, str):
            text = item.strip().upper()
            if text:
                genes.append(text)
            continue
        if isinstance(item, tuple) and item:
            text = str(item[0]).strip().upper()
            if text:
                genes.append(text)
            continue
        if isinstance(item, dict):
            for key in ("gene", "name", "symbol"):
                text = str(item.get(key) or "").strip().upper()
                if text:
                    genes.append(text)
                    break
    return list(dict.fromkeys(genes))


def _memory_slice_gene_candidates(state: AgentState) -> list[str]:
    result = state.get("memory_slice_result")
    if not isinstance(result, dict):
        return []
    return _selected_values_to_gene_candidates(result.get("selected_values"))


def _memory_slice_deg_records(state: AgentState) -> list[dict[str, Any]]:
    result = state.get("memory_slice_result")
    if not isinstance(result, dict):
        return []
    values = result.get("selected_values")
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, dict)]


def _find_enrichr_term_from_state(
    state: AgentState,
    pathway_term: str | None,
    *,
    query: str = "",
    library_filter: set[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    libraries = _enrichr_libraries_from_state(state)
    if not libraries:
        return None, None, None

    best_match: tuple[dict[str, Any], str, int, int] | None = None

    normalized_filter = {str(value).strip().lower() for value in (library_filter or set()) if str(value).strip()}

    for library_name, terms in libraries.items():
        library_text = str(library_name)
        if normalized_filter and library_text.strip().lower() not in normalized_filter:
            continue
        if not isinstance(terms, list):
            continue
        for index, term in enumerate(terms, start=1):
            if not isinstance(term, dict):
                continue
            label = str(
                term.get("term")
                or term.get("path_name")
                or term.get("term_name")
                or term.get("name")
                or term.get("Path")
                or term.get("Term")
                or ""
            ).strip()
            if not label:
                continue
            score = _pathway_match_score(label, pathway_term or "", query)

            if score <= 0:
                continue
            if best_match is None or score > best_match[3]:
                best_match = (term, str(library_name), index, score)

    if not best_match:
        return None, None, None
    return best_match[0], best_match[1], best_match[2]


def _run_memory_lookup(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("text") or state.get("query") or "")
    query_norm = _normalize_text_token(query)
    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    pathway_term = str(args.get("pathway_term") or "").strip()
    if not pathway_term:
        pathway_term = _extract_requested_pathway_name(query)
    selected_term, selected_library, selected_rank = _find_enrichr_term_from_state(
        state,
        pathway_term,
        query=query,
    )
    pathway_genes = []
    if isinstance(selected_term, dict):
        pathway_genes = _term_overlapping_genes(selected_term)

    deg_records = state.get("deg_gene_records") or state.get("memory_deg_gene_records")
    deg_genes = _genes_from_deg_records_by_direction(
        deg_records,
        direction=direction,
        top_n=top_n if isinstance(top_n, int) and top_n > 0 else None,
    )
    if not deg_genes:
        deg_genes = [
            str(g).strip().upper()
            for g in (state.get("deg_genes") or state.get("memory_deg_genes") or [])
            if str(g).strip()
        ]
        if isinstance(top_n, int) and top_n > 0:
            deg_genes = deg_genes[:top_n]

    mentioned_genes = [
        str(g).strip().upper()
        for g in extract_genes_from_text(query, mode="strict")
        if str(g).strip()
    ]
    mentioned_genes = list(dict.fromkeys(mentioned_genes))

    pathway_gene_set = set(pathway_genes)
    deg_gene_set = set(deg_genes)
    intersection = sorted(pathway_gene_set.intersection(deg_gene_set))

    gene_membership = []
    for gene in mentioned_genes:
        gene_membership.append(
            {
                "gene": gene,
                "in_pathway_or_go_term": gene in pathway_gene_set if pathway_gene_set else None,
                "in_selected_deg_set": gene in deg_gene_set if deg_gene_set else None,
            }
        )

    payload = {
        "query": query,
        "direction": direction,
        "top_n": top_n,
        "selected_term": {
            "library": selected_library,
            "term": (
                selected_term.get("term")
                or selected_term.get("path_name")
                or selected_term.get("term_name")
                or selected_term.get("name")
                or selected_term.get("Path")
                or selected_term.get("Term")
            ) if isinstance(selected_term, dict) else "",
            "rank": selected_rank,
            "overlapping_genes": pathway_genes[:100],
        } if isinstance(selected_term, dict) else None,
        "pathway_gene_count": len(pathway_genes),
        "deg_gene_count": len(deg_genes),
        "deg_genes": deg_genes[:200],
        "intersection_count": len(intersection),
        "intersection_genes": intersection[:200],
        "mentioned_genes": mentioned_genes,
        "gene_membership": gene_membership,
    }
    answer_parts: list[str] = []
    selected_term_name = ""
    if isinstance(payload.get("selected_term"), dict):
        selected_term_name = str((payload.get("selected_term") or {}).get("term") or "").strip()

    asks_for_count = any(
        phrase in query_norm
        for phrase in (
            "how many",
            "count",
            "number of",
            "total",
        )
    )
    asks_for_list = any(
        phrase in query_norm
        for phrase in (
            "list",
            "show",
            "display",
            "print",
            "all ",
            "whole list",
            "full list",
            "which genes",
            "what genes",
        )
    )
    asks_for_deg_genes = any(
        phrase in query_norm
        for phrase in (
            "upregulated genes",
            "up regulated genes",
            "up regulated deg genes",
            "upregulated deg genes",
            "downregulated genes",
            "down regulated genes",
            "down regulated deg genes",
            "downregulated deg genes",
            "deg genes",
            "differentially expressed genes",
        )
    )

    if asks_for_deg_genes and asks_for_count:
        direction_label = {
            "up": "up-regulated",
            "down": "down-regulated",
            "both": "combined up- and down-regulated",
            "all": "stored DEG",
        }.get(direction, "stored DEG")
        answer_parts.append(f"{direction_label} gene count: {len(deg_genes)}.")

    if asks_for_deg_genes and asks_for_list:
        direction_label = {
            "up": "Up-regulated genes",
            "down": "Down-regulated genes",
            "both": "Combined DEG genes",
            "all": "Stored DEG genes",
        }.get(direction, "Stored DEG genes")
        if deg_genes:
            answer_parts.append(f"{direction_label} ({len(deg_genes)}): {', '.join(deg_genes)}.")
        else:
            answer_parts.append(f"No {direction_label.lower()} are currently stored.")

    if selected_term_name and any(
        phrase in query_norm
        for phrase in (
            "overlap genes",
            "overlapping genes",
            "genes in pathway",
            "genes in go term",
            "genes in term",
            "pathway genes",
            "go term genes",
        )
    ):
        if pathway_genes:
            answer_parts.append(
                f"Matched term: {selected_term_name}. Overlapping genes ({len(pathway_genes)}): {', '.join(pathway_genes)}."
            )
        else:
            answer_parts.append(f"Matched term: {selected_term_name}, but no overlapping genes are stored for it.")

    if any(
        phrase in query_norm
        for phrase in (
            "intersection",
            "common genes",
            "overlap between",
            "shared genes",
        )
    ):
        if intersection:
            answer_parts.append(f"Intersection genes ({len(intersection)}): {', '.join(intersection)}.")
        else:
            answer_parts.append("No intersecting genes were found between the stored term genes and the selected DEG set.")

    membership_query = any(
        phrase in query_norm
        for phrase in (
            "is present",
            "is in",
            "present in",
            "belongs to",
            "contains",
            "include gene",
            "does it include",
            "is gene",
        )
    )

    if membership_query and mentioned_genes:
        for row in gene_membership:
            gene = str(row.get("gene") or "").strip()
            in_pathway = row.get("in_pathway_or_go_term")
            in_deg = row.get("in_selected_deg_set")
            labels: list[str] = []
            if in_pathway is True:
                labels.append("the stored pathway/GO term")
            if in_deg is True:
                labels.append("the selected DEG set")
            if labels:
                answer_parts.append(f"{gene} is present in {', '.join(labels)}.")
            elif in_pathway is False or in_deg is False:
                answer_parts.append(f"{gene} is not present in the matched stored sets.")

    if not answer_parts:
        if selected_term_name and pathway_genes:
            answer_parts.append(
                f"Matched term: {selected_term_name}. Overlapping genes ({len(pathway_genes)}): {', '.join(pathway_genes[:50])}."
            )
        elif intersection:
            answer_parts.append(f"Intersection genes ({len(intersection)}): {', '.join(intersection[:50])}.")
        elif deg_genes:
            answer_parts.append(f"Selected DEG genes ({len(deg_genes)}): {', '.join(deg_genes[:50])}.")
        else:
            answer_parts.append("No matching stored pathway, GO term, or DEG gene set could be resolved from the current state.")

    status = "ok"
    if not selected_term_name and not pathway_genes and not deg_genes and not intersection and not mentioned_genes:
        status = "not_found"

    answer_text = " ".join(answer_parts).strip()
    return {
        "status": status,
        "analysis_arm": "memory_lookup",
        "direction": direction,
        "top_n": top_n,
        "selected_term": payload.get("selected_term"),
        "pathway_genes": pathway_genes,
        "deg_genes": deg_genes,
        "intersection_genes": intersection,
        "mentioned_genes": mentioned_genes,
        "gene_membership": gene_membership,
        "requested_pathway_term": pathway_term,
        "answer": answer_text,
        "message": answer_text,
        "should_finalize": True,
    }


def _run_state_lookup(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("text") or state.get("query") or "")
    mode = str(args.get("mode") or "").strip().lower()
    query_norm = _normalize_text_token(query)
    if mode not in {"length", "value", "both"}:
        if "length" in query_norm or "how many" in query_norm or "count" in query_norm:
            mode = "length"
        elif "value" in query_norm or "print" in query_norm or "show" in query_norm:
            mode = "value"
        else:
            mode = "both"
    max_items = args.get("max_items")
    if isinstance(max_items, str) and max_items.isdigit():
        max_items = int(max_items)
    if not isinstance(max_items, int) or max_items <= 0:
        max_items = 25

    fields = _match_state_fields(query, args.get("fields"))
    normalized_query = _normalize_text_token(query)
    if not fields and any(
        phrase in normalized_query
        for phrase in (
            "all state",
            "entire state",
            "all fields",
            "what is in memory",
            "what is stored",
            "show memory",
            "show state",
            "list state fields",
        )
    ):
        fields = _state_field_names()

    if not fields:
        fields = sorted(set(_state_field_names()) & set(state.keys()))

    inspections: list[dict[str, Any]] = []
    for field in fields:
        resolved_field, exists, value = _resolve_state_field_value(state, field)
        entry: dict[str, Any] = {
            "field": resolved_field,
            "requested_field": field,
            "exists": exists,
            "type": type(value).__name__ if exists else "missing",
        }
        length = _state_value_length(value)
        if length is not None and mode in {"length", "both"}:
            entry["length"] = length
        if mode in {"value", "both"} and exists:
            entry["value"] = _state_value_summary(value, max_items=max_items)
        inspections.append(entry)

    answer_lines: list[str] = []
    for entry in inspections:
        field = str(entry.get("field") or "")
        if not entry.get("exists"):
            answer_lines.append(f"{field}: missing")
            continue
        parts = [f"{field}: type={entry.get('type')}"]
        if "length" in entry:
            parts.append(f"length={entry.get('length')}")
        if "value" in entry:
            value_text = json.dumps(entry.get("value"), ensure_ascii=False, default=str)
            parts.append(f"value={value_text}")
        answer_lines.append(", ".join(parts))

    answer_text = "\n".join(answer_lines) if answer_lines else "No matching state fields were found."
    return {
        "status": "ok" if inspections else "not_found",
        "analysis_arm": "state_lookup",
        "mode": mode,
        "fields": fields,
        "inspections": inspections,
        "answer": answer_text,
        "message": answer_text,
        "should_finalize": True,
    }


def _state_lookup_has_useful_result(result: dict[str, Any]) -> bool:
    inspections = result.get("inspections")
    if not isinstance(inspections, list) or not inspections:
        return False
    for entry in inspections:
        if not isinstance(entry, dict) or not entry.get("exists"):
            continue
        if "length" in entry:
            if entry.get("length") not in (None, 0):
                return True
            continue
        value = entry.get("value")
        if value not in (None, "", [], {}):
            return True
        entry_type = str(entry.get("type") or "")
        if entry_type not in {"NoneType", "missing"}:
            return True
    return False


def _run_state_lookup_with_memory_fallback(
    state: AgentState,
    args: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_result = _run_state_lookup(state, args)
    update: dict[str, Any] = {"state_lookup_result": state_result}
    if _state_lookup_has_useful_result(state_result):
        return state_result, update

    slice_args = {
        **args,
        "fields": args.get("fields") or state_result.get("fields") or [],
        "text": args.get("text") or state.get("query") or "",
    }
    slice_result = _run_memory_slice(state, slice_args)
    update["memory_slice_result"] = slice_result
    if str(slice_result.get("status") or "") == "ok" and slice_result.get("selected_values"):
        result = {
            **slice_result,
            "analysis_arm": "memory_slice",
            "state_lookup_result": state_result,
            "fallback_from": "state_lookup",
            "message": str(slice_result.get("message") or slice_result.get("answer") or ""),
        }
        return result, update

    lookup_result = _run_memory_lookup(state, args)
    update["memory_lookup_result"] = lookup_result
    if str(lookup_result.get("status") or "") == "ok":
        result = {
            **lookup_result,
            "analysis_arm": "memory_lookup",
            "state_lookup_result": state_result,
            "memory_slice_result": slice_result,
            "fallback_from": "state_lookup",
            "message": str(lookup_result.get("message") or lookup_result.get("answer") or ""),
        }
        return result, update

    return state_result, update


def _run_memory_slice(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("text") or state.get("query") or "")
    fields = _match_state_fields(query, args.get("fields"))
    requested_field = str((fields or [""])[0] or "").strip()
    field = requested_field
    if not field:
        return {
            "status": "missing_field",
            "analysis_arm": "memory_slice",
            "answer": "No matching list-like state field was found.",
            "message": "No matching list-like state field was found.",
            "selected_values": [],
            "should_finalize": True,
        }

    field, values = _resolve_listlike_state_field(state, field)
    if not values:
        return {
            "status": "not_list_like",
            "analysis_arm": "memory_slice",
            "field": requested_field,
            "resolved_field": field,
            "answer": f"{requested_field} is not a non-empty list-like state field.",
            "message": f"{requested_field} is not a non-empty list-like state field.",
            "selected_values": [],
            "should_finalize": True,
        }

    top_n = args.get("top_n")
    bottom_n = args.get("bottom_n")
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)
    if isinstance(bottom_n, str) and bottom_n.isdigit():
        bottom_n = int(bottom_n)

    query_norm = _normalize_text_token(query)
    if not isinstance(top_n, int) and not isinstance(bottom_n, int):
        top_match = re.search(r"\btop\s+(\d+)\b", query, flags=re.IGNORECASE)
        bottom_match = re.search(r"\bbottom\s+(\d+)\b", query, flags=re.IGNORECASE)
        if top_match:
            top_n = int(top_match.group(1))
        if bottom_match:
            bottom_n = int(bottom_match.group(1))
        if not isinstance(top_n, int) and not isinstance(bottom_n, int):
            inferred = _parse_top_n_from_text(query)
            if inferred:
                if "bottom" in query_norm:
                    bottom_n = inferred
                else:
                    top_n = inferred

    if not isinstance(top_n, int) and not isinstance(bottom_n, int):
        top_n = min(10, len(values))

    top_slice = values[: max(0, top_n)] if isinstance(top_n, int) and top_n > 0 else []
    bottom_slice = values[-max(0, bottom_n):] if isinstance(bottom_n, int) and bottom_n > 0 else []
    if top_slice and bottom_slice:
        selected_values = top_slice + [item for item in bottom_slice if item not in top_slice]
        selection_mode = "top_and_bottom"
    elif top_slice:
        selected_values = top_slice
        selection_mode = "top"
    else:
        selected_values = bottom_slice
        selection_mode = "bottom"

    answer_parts = [f"{field}: total_length={len(values)}"]
    if isinstance(top_n, int) and top_n > 0:
        answer_parts.append(f"top_n={top_n}")
    if isinstance(bottom_n, int) and bottom_n > 0:
        answer_parts.append(f"bottom_n={bottom_n}")
    answer_parts.append(f"selected_count={len(selected_values)}")
    summary_limit = len(selected_values) if selected_values else 50
    answer_parts.append(
        "selected_values=" + json.dumps(_state_value_summary(selected_values, max_items=summary_limit), ensure_ascii=False, default=str)
    )

    answer_text = ", ".join(answer_parts)
    return {
        "status": "ok",
        "analysis_arm": "memory_slice",
        "field": field,
        "requested_field": requested_field,
        "field_length": len(values),
        "top_n": top_n if isinstance(top_n, int) and top_n > 0 else None,
        "bottom_n": bottom_n if isinstance(bottom_n, int) and bottom_n > 0 else None,
        "selection_mode": selection_mode,
        "selected_values": selected_values,
        "selected_gene_candidates": _selected_values_to_gene_candidates(selected_values),
        "answer": answer_text,
        "message": answer_text,
        "should_finalize": True,
    }


def _graph_summary(graph: nx.Graph | None) -> dict[str, Any]:
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "top_degree": []}
    degrees = sorted(graph.degree(), key=lambda item: item[1], reverse=True)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "top_degree": [{"gene": gene, "degree": int(degree)} for gene, degree in degrees[:10]],
    }


def _latest_ai_message(messages: list[BaseMessage] | None) -> AIMessage | None:
    if not messages:
        return None
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _latest_tool_call(state: AgentState, tool_name: str) -> dict[str, Any] | None:
    ai_message = _latest_ai_message(list(state.get("messages") or []))
    if not ai_message or not getattr(ai_message, "tool_calls", None):
        return None

    for call in ai_message.tool_calls:
        if str(call.get("name", "")).strip() == tool_name:
            return call

    if len(ai_message.tool_calls) == 1:
        return ai_message.tool_calls[0]
    return None


def _serialize_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    for key in ("status", "message", "disease", "disease_name", "gene", "answer"):
        value = result.get(key)
        if value not in (None, ""):
            payload[key] = value
    for key in ("mode", "signature_count", "result_limit", "log2fold", "padj"):
        value = result.get(key)
        if value not in (None, ""):
            payload[key] = value

    if isinstance(result.get("genes"), list):
        payload["genes"] = result["genes"][:50]
    if isinstance(result.get("mapped_seed_genes"), list):
        payload["mapped_seed_genes"] = result["mapped_seed_genes"][:50]
    if isinstance(result.get("unmapped_seed_genes"), list):
        payload["unmapped_seed_genes"] = result["unmapped_seed_genes"][:50]
    if isinstance(result.get("srp_ids"), list):
        payload["srp_ids"] = result["srp_ids"][:20]
    for key in ("control_name", "test_name"):
        if result.get(key):
            payload[key] = result.get(key)
    if isinstance(result.get("deg_genes"), list):
        payload["deg_genes"] = result["deg_genes"][:50]
    if isinstance(result.get("openalex_genes"), list):
        payload["openalex_genes"] = result["openalex_genes"][:50]
    if isinstance(result.get("rwr_genes"), list):
        payload["rwr_genes"] = [
            {"gene": gene, "score": round(float(score), 4)} for gene, score in result["rwr_genes"][:20]
        ]
    if isinstance(result.get("graph"), nx.Graph):
        payload["graph"] = _graph_summary(result["graph"])
    if isinstance(result.get("openalex_papers"), list):
        payload["openalex_papers"] = [
            {"source": paper.get("source"), "title": paper.get("title"), "year": paper.get("year")}
            for paper in result["openalex_papers"][:5]
            if isinstance(paper, dict)
        ]
    if isinstance(result.get("ranked_openalex_papers"), list):
        payload["ranked_openalex_papers"] = [
            {
                "source": paper.get("source"),
                "title": paper.get("title"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "pmid": paper.get("pmid"),
                "relevance": paper.get("relevance"),
                "reason": paper.get("reason"),
            }
            for paper in result["ranked_openalex_papers"][:5]
            if isinstance(paper, dict)
        ]
    if isinstance(result.get("literature_key_points"), list):
        payload["literature_key_points"] = [
            {
                "point": row.get("point"),
                "paper_ids": row.get("paper_ids"),
            }
            for row in result["literature_key_points"][:5]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("literature_references"), list):
        payload["literature_references"] = [
            {
                "paper_id": row.get("paper_id"),
                "source": row.get("source"),
                "title": row.get("title"),
                "year": row.get("year"),
                "doi": row.get("doi"),
                "pmid": row.get("pmid"),
            }
            for row in result["literature_references"][:8]
            if isinstance(row, dict)
        ]
    if result.get("literature_summary"):
        payload["literature_summary"] = _compact_text(result.get("literature_summary"), limit=1000)
    if isinstance(result.get("literature_source_status"), dict):
        payload["literature_source_status"] = result.get("literature_source_status")
    if result.get("literature_query"):
        payload["literature_query"] = result.get("literature_query")
    if result.get("associated") is not None:
        payload["associated"] = bool(result.get("associated"))
    if result.get("association_score") is not None:
        payload["association_score"] = result.get("association_score")
    for key in ("gene_set_source", "direction", "top_n", "gene_limit", "term_limit", "input_gene_count"):
        if result.get(key) not in (None, ""):
            payload[key] = result.get(key)
    if isinstance(result.get("top_diseases"), list):
        payload["top_diseases"] = [
            {"name": disease.get("name"), "score": disease.get("score")}
            for disease in result["top_diseases"][:10]
            if isinstance(disease, dict)
        ]
    if isinstance(result.get("top_drugs"), list):
        payload["top_drugs"] = [
            {
                "name": row.get("name"),
                "pert_id": row.get("pert_id"),
                "best_rank": row.get("best_rank"),
                "best_score": row.get("best_score"),
                "cell_lines": row.get("cell_lines"),
                "signature_count": row.get("signature_count"),
                "phase": row.get("phase"),
                "status": row.get("status"),
                "disease_name": row.get("disease_name"),
            }
            for row in result["top_drugs"][:10]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("requested_cell_lines"), list):
        payload["requested_cell_lines"] = result["requested_cell_lines"][:20]
    if isinstance(result.get("top_signatures"), list):
        payload["top_signatures"] = [
            {
                "rank": row.get("rank"),
                "perturbation": row.get("perturbation"),
                "pert_id": row.get("pert_id"),
                "cell_line": row.get("cell_line"),
                "dose": row.get("dose"),
                "dose_unit": row.get("dose_unit"),
                "time": row.get("time"),
                "time_unit": row.get("time_unit"),
            }
            for row in result["top_signatures"][:10]
            if isinstance(row, dict)
        ]
    if result.get("cid") not in (None, ""):
        payload["cid"] = result.get("cid")
    for key in ("drug_name", "pert_id", "title", "matched_query", "matched_strategy"):
        if result.get(key):
            payload[key] = result.get(key)
    if isinstance(result.get("properties"), dict):
        payload["properties"] = {
            key: result["properties"].get(key)
            for key in (
                "MolecularFormula",
                "MolecularWeight",
                "CanonicalSMILES",
                "InChIKey",
                "XLogP",
                "TPSA",
            )
            if result["properties"].get(key) not in (None, "")
        }
    if isinstance(result.get("synonyms"), list):
        payload["synonyms"] = result["synonyms"][:20]
    if isinstance(result.get("descriptions"), list):
        payload["descriptions"] = result["descriptions"][:5]
    if isinstance(result.get("annotation_lines"), list):
        payload["annotation_lines"] = result["annotation_lines"][:20]
    if isinstance(result.get("fields"), list):
        payload["fields"] = result["fields"][:50]
    if isinstance(result.get("inspections"), list):
        payload["inspections"] = result["inspections"][:20]
    if result.get("field"):
        payload["field"] = result.get("field")
    for key in ("field_length", "selection_mode", "top_n", "bottom_n"):
        if result.get(key) not in (None, ""):
            payload[key] = result.get(key)
    if isinstance(result.get("selected_values"), list):
        payload["selected_values"] = _state_value_summary(result.get("selected_values"), max_items=20)
    if isinstance(result.get("selected_gene_candidates"), list):
        payload["selected_gene_candidates"] = result["selected_gene_candidates"][:50]
    if isinstance(result.get("results"), list):
        payload["results"] = [
            {
                "gene": row.get("gene"),
                "ensembl_id": row.get("ensembl_id"),
                "associated": row.get("associated"),
                "association_score": row.get("association_score"),
            }
            for row in result["results"][:20]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("intersection_genes"), list):
        payload["intersection_genes"] = result["intersection_genes"][:100]
    if isinstance(result.get("mentioned_genes"), list):
        payload["mentioned_genes"] = result["mentioned_genes"][:20]
    if isinstance(result.get("gene_membership"), list):
        payload["gene_membership"] = result["gene_membership"][:20]
    if isinstance(result.get("selected_term"), dict):
        payload["selected_term"] = {
            "library": result["selected_term"].get("library"),
            "term": result["selected_term"].get("term"),
            "rank": result["selected_term"].get("rank"),
        }
    if isinstance(result.get("edges"), list):
        payload["edges"] = [
            {
                "relation": edge.get("display_relation") or edge.get("relation"),
                "source": (edge.get("source") or {}).get("name") if isinstance(edge.get("source"), dict) else "",
                "source_type": (edge.get("source") or {}).get("type") if isinstance(edge.get("source"), dict) else "",
                "target": (edge.get("target") or {}).get("name") if isinstance(edge.get("target"), dict) else "",
                "target_type": (edge.get("target") or {}).get("type") if isinstance(edge.get("target"), dict) else "",
            }
            for edge in result["edges"][:25]
            if isinstance(edge, dict)
        ]
        payload["count"] = result.get("count", len(result.get("edges") or []))
    if isinstance(result.get("deg_analysis"), dict):
        deg_analysis = result["deg_analysis"]
        payload["deg_analysis"] = {
            "status": deg_analysis.get("status"),
            "log2fold": deg_analysis.get("log2fold"),
            "padj": deg_analysis.get("padj"),
            "thresholds_applied": deg_analysis.get("thresholds_applied"),
            "genes": deg_analysis.get("genes", [])[:20] if isinstance(deg_analysis.get("genes"), list) else [],
            "rows": len(deg_analysis.get("rows", [])) if isinstance(deg_analysis.get("rows"), list) else 0,
        }
    if isinstance(result.get("srp_metadata"), list):
        payload["srp_metadata"] = [
            {
                "srp_id": row.get("srp_id"),
                "dee2_row_count": row.get("dee2_row_count"),
                "sra_run_count": row.get("sra_run_count"),
                "geo_series": row.get("geo_series", [])[:10] if isinstance(row.get("geo_series"), list) else [],
                "descriptions": row.get("descriptions", [])[:8] if isinstance(row.get("descriptions"), list) else [],
                "field_summaries": row.get("field_summaries", {}),
                "metadata_preview": row.get("metadata_preview", [])[:10] if isinstance(row.get("metadata_preview"), list) else [],
            }
            for row in result["srp_metadata"][:5]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("srp_metadata_result"), dict):
        srp_metadata_result = result["srp_metadata_result"]
        payload["srp_metadata_result"] = {
            "status": srp_metadata_result.get("status"),
            "srp_ids": srp_metadata_result.get("srp_ids", [])[:20] if isinstance(srp_metadata_result.get("srp_ids"), list) else [],
            "srp_metadata": srp_metadata_result.get("srp_metadata", [])[:5] if isinstance(srp_metadata_result.get("srp_metadata"), list) else [],
        }
    if isinstance(result.get("deg_gene_records"), list):
        payload["deg_gene_records"] = [
            {
                "gene": row.get("gene"),
                "pvalue": row.get("pvalue"),
                "pdj": row.get("pdj"),
                "log2FoldChange": row.get("log2FoldChange"),
            }
            for row in result["deg_gene_records"][:20]
            if isinstance(row, dict)
        ]
    if isinstance(result.get("upregulated_genes"), list):
        payload["upregulated_genes"] = result["upregulated_genes"][:50]
    if isinstance(result.get("downregulated_genes"), list):
        payload["downregulated_genes"] = result["downregulated_genes"][:50]
    if isinstance(result.get("selected_genes"), list):
        payload["selected_genes"] = result["selected_genes"][:50]
    if isinstance(result.get("enrichr"), dict):
        libs = result["enrichr"].get("libraries")
        if isinstance(libs, dict):
            payload["enrichr"] = {
                lib: [
                    {
                        "term": term.get("term"),
                        "p_value": term.get("p_value"),
                        "adjusted_p_value": term.get("adjusted_p_value"),
                        "combined_score": term.get("combined_score"),
                        "overlapping_genes": term.get("overlapping_genes"),
                    }
                    for term in terms[:3]
                    if isinstance(term, dict)
                ]
                for lib, terms in libs.items()
                if isinstance(terms, list)
            }
    if result.get("cypher"):
        payload["cypher"] = result["cypher"]

    if result.get("answer"):
        payload["answer"] = result["answer"]
    if result.get("pyvis_html_path"):
        payload["pyvis_html_path"] = result["pyvis_html_path"]
    if result.get("kegg_pathway_path"):
        payload["kegg_pathway_path"] = result["kegg_pathway_path"]
    if result.get("volcano_plot_path"):
        payload["volcano_plot_path"] = result["volcano_plot_path"]
    if isinstance(result.get("kegg_enrichr_results"), list):
        payload["kegg_enrichr_results"] = result["kegg_enrichr_results"][:5]
    if isinstance(result.get("selected_pathway"), dict):
        payload["selected_pathway"] = {
            "library": result["selected_pathway"].get("library"),
            "term": result["selected_pathway"].get("term"),
            "rank": result["selected_pathway"].get("rank"),
            "overlapping_genes": result["selected_pathway"].get("overlapping_genes"),
        }

    return payload or {"keys": sorted(result.keys())}


def _infer_analysis_arm(state: AgentState) -> str:
    arm = str(state.get("analysis_arm") or "").strip().lower()
    if arm in {"general", "srp", "srp_metadata", "disease", "memory_rwr", "pathway", "visualize", "primekg", "opentargets", "memory_lookup", "state_lookup", "memory_slice", "l1000cds2", "pubchem", "research_literature", "literature", "hypothesis", "druggability", "pdb_visualizer"}:
        return arm
    if state.get("pdb_visualization_result"):
        return "pdb_visualizer"
    if state.get("druggability_result"):
        return "druggability"
    if state.get("srp_metadata_result"):
        return "srp_metadata"
    if state.get("visualization_result"):
        return "visualize"
    if state.get("hypothesis_result"):
        return "hypothesis"
    if state.get("enrichr"):
        return "pathway"
    if state.get("memory_lookup_result"):
        return "memory_lookup"
    if state.get("state_lookup_result"):
        return "state_lookup"
    if state.get("memory_slice_result"):
        return "memory_slice"
    if state.get("primekg_result"):
        return "primekg"
    if state.get("rwr_genes"):
        if state.get("memory_deg_genes") or state.get("memory_deg_gene_records"):
            return "memory_rwr"
        if state.get("openalex_papers") or state.get("openalex_genes") or state.get("disease_name"):
            return "disease"
        return "general"
    if state.get("l1000cds2_result"):
        return "l1000cds2"
    if state.get("pubchem_result"):
        return "pubchem"
    if state.get("opentargets_result"):
        return "opentargets"
    if state.get("deg_analysis"):
        return "srp"
    if state.get("openalex_papers") or state.get("openalex_genes") or state.get("rwr_genes") or state.get("disease_name"):
        return "disease"
    if state.get("memory_deg_genes") and state.get("rwr_genes"):
        return "memory_rwr"
    return "general"


def _build_tool_list_text() -> str:
    lines = []
    for tool_obj in TOOL_SCHEMAS:
        lines.append(f"- {tool_obj.name}: {tool_obj.description}")
    return "\n".join(lines)


def _is_gemini_family_model() -> bool:
    return is_gemini_family_provider()


def _normalize_srp_ids(value: Any) -> list[str]:
    raw_values: list[Any] = []
    if isinstance(value, dict):
        raw_values = list(value.get("srp_ids") or [])
    elif isinstance(value, list):
        raw_values = value
    elif isinstance(value, str):
        raw_values = extract_srp_ids_from_text(value)

    normalized: list[str] = []
    for item in raw_values:
        token = str(item or "").strip().upper()
        if token.startswith("SRP") and token not in normalized:
            normalized.append(token)
    return normalized


def _build_system_prompt(state: AgentState) -> str:
    query = _compact_text(state.get("query"), limit=400)
    memory_summary = _compact_text(state.get("memory_summary"), limit=500) or "No prior memory."
    state_snapshot = {
        "analysis_arm": _infer_analysis_arm(state),
        "step_count": int(state.get("step_count") or 0),
        "genes": state.get("genes") or [],
        "srp_ids": state.get("srp_ids") or [],
        "control_name": state.get("control_name") or state.get("memory_control_name") or "",
        "test_name": state.get("test_name") or state.get("memory_test_name") or "",
        "disease_name": state.get("disease_name") or "",
        "memory_disease_name": state.get("memory_disease_name") or "",
        "deg_gene_count": len(state.get("deg_genes") or []),
        "upregulated_gene_count": len(state.get("upregulated_genes") or []),
        "downregulated_gene_count": len(state.get("downregulated_genes") or []),
        "openalex_gene_count": len(state.get("openalex_genes") or []),
        "hypothesis_count": len((_ensure_dict(state.get("hypothesis_result")).get("hypotheses") or [])),
        "literature_source_status": _ensure_dict(state.get("literature_source_status")),
        "memory_deg_gene_count": len(state.get("memory_deg_genes") or []),
        "memory_upregulated_gene_count": len(state.get("memory_upregulated_genes") or []),
        "memory_downregulated_gene_count": len(state.get("memory_downregulated_genes") or []),
        "has_srp_metadata": bool(_ensure_dict(state.get("srp_metadata_result") or state.get("memory_srp_metadata_result"))),
        "memory_enrichr_libraries": sorted(list((_ensure_dict(state.get("enrichr") or state.get("memory_enrichr")).get("libraries") or {}).keys())),
        "memory_l1000_hits": len((_ensure_dict(state.get("l1000cds2_result") or state.get("memory_l1000cds2_result")).get("top_drugs") or [])),
        "has_pubchem_result": bool(_ensure_dict(state.get("pubchem_result") or state.get("memory_pubchem_result"))),
        "has_druggability_result": bool(_ensure_dict(state.get("druggability_result") or state.get("memory_druggability_result"))),
        "has_pdb_visualization_result": bool(_ensure_dict(state.get("pdb_visualization_result") or state.get("memory_pdb_visualization_result"))),
        "rwr_gene_count": len(state.get("rwr_genes") or []),
        "has_graph": bool(isinstance(state.get("graph"), nx.Graph) and state["graph"].number_of_nodes() > 0),
        "graph_summary": _graph_summary(state.get("graph") if isinstance(state.get("graph"), nx.Graph) else None),
        "recent_tools": (state.get("tool_history") or [])[-5:],
    }

    gemini_block = ""
    if _is_gemini_family_model():
        gemini_block = (
            "Gemini/Gemma tool-calling instructions:\n"
            "- Prefer explicit structured tool arguments over leaving values inside prose.\n"
            "- When DEG analysis is needed and any SRP ID plus exact control/test labels are present in the query or memory, call `deg_analysis` immediately.\n"
            "- If exact control/test labels are missing, call `srp_metadata` first and include the SRP IDs in `srp_ids`.\n"
            "- Put SRP IDs in `srp_ids` as an uppercase JSON array such as [\"SRP277202\",\"SRP123456\"].\n"
            "- Also include `text` with the full user request so downstream normalization can recover cohort labels.\n"
            "- If cohort labels are uncertain, use empty strings instead of inventing them.\n"
            "- For stored-memory questions, prefer `memory_lookup`, `state_lookup`, or `memory_slice` over a free-text answer.\n"
            "- Use `state_lookup` when the user asks what is stored, asks for the value of a field, or asks for counts/lengths of stored variables.\n"
            "- Use `memory_slice` when the user asks for top N or bottom N items from a stored state field.\n"
            "- Use `memory_lookup` when the user asks for overlap genes, membership checks, or intersections between stored pathway and DEG results.\n"
            "- Do not describe the intended tool call in prose when the correct next action is to emit the tool call.\n\n"
        )

    return (
        "You are the orchestration layer for a gene expression analysis agent.\n"
        "Your job is to choose the single best next action for the current turn.\n"
        "Think like an agent supervisor: inspect the live state, decide whether to answer or call exactly one tool, then reassess after the tool returns.\n"
        "Do not plan a long sequence in text. Make the next grounded move.\n"
        "You may chain multiple specialists across loop iterations when the user asks for a real analysis workflow or when a previous specialist created the prerequisite state for the next requested step.\n"
        "A credible chain is short, stateful, and evidence-seeking: each specialist must consume current state or produce a result/artifact needed by the user's request.\n"
        "If the query is only a simple greeting, acknowledgment, or casual conversational turn, answer directly without tools.\n"
        "Do not answer biomedical or technical questions directly from remembered state alone.\n"
        "Use remembered state to choose and parameterize tools, not to bypass tool execution or rendered outputs.\n"
        "If the query needs technical analysis, prefer specialist tools over unsupported free-form reasoning.\n\n"
        "Specialist guidance:\n"
        f"{TOOL_USE_INSTRUCTIONS}\n\n"
        f"{gemini_block}"
        "Available tools and what they do:\n"
        f"{_build_tool_list_text()}\n\n"
        f"Current user query: {query}\n"
        f"Memory summary: {memory_summary}\n"
        f"Current state snapshot: {json.dumps(state_snapshot, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "Decision rules:\n"
        "- Choose only from the listed specialist tools when a tool is needed.\n"
        "- Make the smallest correct next decision rather than narrating a full workflow.\n"
        "- After a specialist returns, either call the next necessary specialist or answer from the accumulated structured state.\n"
        "- Do not repeat a specialist that has already satisfied the same part of the current query unless new inputs were created and repetition is clearly useful.\n"
        "- Prefer `synthesize_technical_response` when the requested specialist chain is complete and the answer should integrate multiple technical results.\n"
        "- Prefer using state and memory to parameterize tools instead of answering from memory alone.\n"
        "- Reuse stored genes, DEG results, disease names, pathway results, and graph state as tool inputs whenever they already satisfy prerequisites.\n"
        "- If a required input is missing, choose the tool that can recover it instead of asking the user unless the gap cannot be inferred or recovered.\n"
        "- For pathway enrichment, prefer stored DEG genes and respect up/down regulation cues.\n"
        "- For DEG analysis, extract control, test, and SRP identifiers from the query or memory before running the tool.\n"
        "- If a DEG request has SRP IDs but no exact control/test labels, call `srp_metadata` instead of `deg_analysis`.\n"
        "- If SRP IDs are visible in the user request, they must appear in the `deg_analysis.srp_ids` tool argument as a list of strings.\n"
        "- If the user asks about stored memory, state variables, list lengths, or literal stored values, prefer `state_lookup`.\n"
        "- If the user asks for top N or bottom N values from stored state, prefer `memory_slice`.\n"
        "- If the user asks about overlap genes, intersections, or gene membership inside stored pathway/GO/DEG results, prefer `memory_lookup`.\n"
        "- If the user asks for druggability, protein pockets, binding pockets, DoGSite, DOGSite, or structure-based target assessment for a gene, call `druggability`.\n"
        "- If the user asks to fetch, view, show, or visualize a protein/PDB/AlphaFold structure without druggability scoring, call `pdb_visualizer`.\n"
        "- For visualization, use stored pathway overlaps, RWR targets, DEG rows, or graphs whenever available.\n"
        "- Do not invent unavailable data, hidden evidence, or tool outputs.\n"
        "- Stop using tools once the user's question is sufficiently answered with a tool-backed result.\n"
        "- Keep any direct answer concise, technically accurate, and limited to simple conversational turns.\n"
    )


def _get_bound_llm():
    return get_llm().bind_tools(TOOL_SCHEMAS)


def _prepare_context(state: AgentState) -> AgentState:
    messages = list(state.get("messages") or [])
    if not messages:
        query = str(state.get("query") or "")
        messages = [HumanMessage(content=query)]

    update: AgentState = {
        "messages": messages,
        "step_count": int(state.get("step_count") or 0),
        "tool_history": list(state.get("tool_history") or []),
        "memory_summary": str(state.get("memory_summary") or ""),
    }
    if state.get("query"):
        update["query"] = str(state.get("query") or "")
    if state.get("memory_deg_genes") is not None:
        update["memory_deg_genes"] = list(state.get("memory_deg_genes") or [])
    if state.get("memory_upregulated_genes") is not None:
        update["memory_upregulated_genes"] = list(state.get("memory_upregulated_genes") or [])
    if state.get("memory_downregulated_genes") is not None:
        update["memory_downregulated_genes"] = list(state.get("memory_downregulated_genes") or [])
    if state.get("memory_deg_analysis") is not None:
        update["memory_deg_analysis"] = _ensure_dict(state.get("memory_deg_analysis"))
    if state.get("memory_deg_gene_records") is not None:
        update["memory_deg_gene_records"] = list(state.get("memory_deg_gene_records") or [])
    if state.get("memory_srp_metadata_result") is not None:
        update["memory_srp_metadata_result"] = _ensure_dict(state.get("memory_srp_metadata_result"))
    if state.get("memory_control_name") is not None:
        update["memory_control_name"] = str(state.get("memory_control_name") or "")
    if state.get("memory_test_name") is not None:
        update["memory_test_name"] = str(state.get("memory_test_name") or "")
    if state.get("memory_enrichr") is not None:
        update["memory_enrichr"] = _ensure_dict(state.get("memory_enrichr"))
    if state.get("memory_rwr_seed_genes") is not None:
        update["memory_rwr_seed_genes"] = list(state.get("memory_rwr_seed_genes") or [])
    if state.get("memory_rwr_genes") is not None:
        update["memory_rwr_genes"] = list(state.get("memory_rwr_genes") or [])
    if state.get("memory_disease_name") is not None:
        update["memory_disease_name"] = str(state.get("memory_disease_name") or "")
    if state.get("memory_openalex_genes") is not None:
        update["memory_openalex_genes"] = list(state.get("memory_openalex_genes") or [])
    if state.get("memory_opentargets_results") is not None:
        update["memory_opentargets_results"] = list(state.get("memory_opentargets_results") or [])
    if state.get("memory_l1000cds2_result") is not None:
        update["memory_l1000cds2_result"] = _ensure_dict(state.get("memory_l1000cds2_result"))
    if state.get("memory_pubchem_result") is not None:
        update["memory_pubchem_result"] = _ensure_dict(state.get("memory_pubchem_result"))
    if state.get("memory_hypothesis_result") is not None:
        update["memory_hypothesis_result"] = _ensure_dict(state.get("memory_hypothesis_result"))
    if state.get("memory_druggability_result") is not None:
        update["memory_druggability_result"] = _ensure_dict(state.get("memory_druggability_result"))
    if state.get("memory_pdb_visualization_result") is not None:
        update["memory_pdb_visualization_result"] = _ensure_dict(state.get("memory_pdb_visualization_result"))
    if state.get("memory_lookup_result") is not None:
        update["memory_lookup_result"] = _ensure_dict(state.get("memory_lookup_result"))
    if state.get("state_lookup_result") is not None:
        update["state_lookup_result"] = _ensure_dict(state.get("state_lookup_result"))
    if state.get("memory_slice_result") is not None:
        update["memory_slice_result"] = _ensure_dict(state.get("memory_slice_result"))
    if state.get("literature_source_status") is not None:
        update["literature_source_status"] = _ensure_dict(state.get("literature_source_status"))
    if state.get("literature_query") is not None:
        update["literature_query"] = str(state.get("literature_query") or "")
    if state.get("l1000cds2_result") is not None:
        update["l1000cds2_result"] = _ensure_dict(state.get("l1000cds2_result"))
    if state.get("pubchem_result") is not None:
        update["pubchem_result"] = _ensure_dict(state.get("pubchem_result"))
    if state.get("hypothesis_result") is not None:
        update["hypothesis_result"] = _ensure_dict(state.get("hypothesis_result"))
    if state.get("druggability_result") is not None:
        update["druggability_result"] = _ensure_dict(state.get("druggability_result"))
    if state.get("pdb_visualization_result") is not None:
        update["pdb_visualization_result"] = _ensure_dict(state.get("pdb_visualization_result"))
    return update


def _agent(state: AgentState) -> AgentState:
    _trace_tool_call("llm_agent")

    unsupported_message = _unsupported_deg_request_message(state.get("query"))
    if unsupported_message:
        return {
            "messages": [AIMessage(content=unsupported_message)],
            "answer": unsupported_message,
            "should_finalize": True,
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_pdb_visualizer(state):
        query_text = str(state.get("query") or "")
        query_genes = extract_genes_from_text(query_text, mode="strict")
        pdb_ids = re.findall(r"\b[0-9][A-Za-z0-9]{3}\b", query_text)
        gene = str((query_genes or state.get("genes") or state.get("memory_deg_genes") or [""])[0] or "").strip()
        forced_call = {
            "name": "pdb_visualizer",
            "args": {
                "gene": gene,
                "pdb_id": str((pdb_ids or [""])[0] or "").strip(),
            },
            "id": "forced_pdb_visualizer_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_druggability(state):
        query_text = str(state.get("query") or "")
        query_genes = extract_genes_from_text(query_text, mode="strict")
        gene = str((query_genes or state.get("genes") or state.get("memory_deg_genes") or [""])[0] or "").strip()
        forced_call = {
            "name": "druggability",
            "args": {
                "gene": gene,
                "top_n": _parse_top_n_from_text(query_text) or 3,
            },
            "id": "forced_druggability_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_srp_metadata(state):
        query_text = str(state.get("query") or "")
        forced_call = {
            "name": "srp_metadata",
            "args": {
                "srp_ids": _normalize_srp_ids(state.get("srp_ids")) or _normalize_srp_ids(query_text),
                "text": query_text,
            },
            "id": "forced_srp_metadata_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_deg_analysis(state):
        query_text = str(state.get("query") or "")
        parsed_groups = _extract_deg_group_labels_from_text(query_text)
        forced_call = {
            "name": "deg_analysis",
            "args": {
                "srp_ids": _normalize_srp_ids(state.get("srp_ids")) or _normalize_srp_ids(query_text),
                "control_name": str(state.get("control_name") or parsed_groups.get("control_name") or state.get("memory_control_name") or ""),
                "test_name": str(state.get("test_name") or parsed_groups.get("test_name") or state.get("memory_test_name") or ""),
                "text": query_text,
            },
            "id": "forced_deg_analysis_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_memory_slice_for_research_query(state):
        query_text = str(state.get("query") or "")
        requested_field = _memory_slice_field_from_query(query_text)
        forced_call = {
            "name": "memory_slice",
            "args": {
                "fields": [requested_field] if requested_field else [],
                "top_n": _parse_top_n_from_text(query_text),
                "text": query_text,
            },
            "id": "forced_memory_slice_for_research_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_stored_pathway_visualization(state):
        query_text = str(state.get("query") or "")
        forced_call = {
            "name": "visualize",
            "args": {
                "visualization_type": "kegg",
                "pathway_term": _extract_requested_pathway_name(query_text),
                "text": query_text,
            },
            "id": "forced_stored_pathway_visualization_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_volcano_visualization(state):
        query_text = str(state.get("query") or "")
        forced_call = {
            "name": "visualize",
            "args": {
                "visualization_type": "volcano",
                "text": query_text,
            },
            "id": "forced_volcano_visualization_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_pathway_rwr(state):
        query_text = str(state.get("query") or "")
        forced_genes = _query_gene_candidates(query_text) or _enrichr_overlap_gene_candidates(
            state,
            query=query_text,
            limit=_parse_top_n_from_text(query_text),
        )
        forced_call = {
            "name": "rwr_analysis",
            "args": {
                "genes": forced_genes,
                "analysis_arm": "memory_rwr",
                "text": query_text,
            },
            "id": "forced_pathway_rwr_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_rwr_visualization(state):
        query_text = str(state.get("query") or "")
        forced_call = {
            "name": "visualize",
            "args": {
                "visualization_type": "network",
                "text": query_text,
            },
            "id": "forced_rwr_visualization_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_pathway_tool(state):
        forced_genes = [
            str(value).strip().upper()
            for value in extract_genes_from_text(str(state.get("query") or ""), mode="strict")
            if str(value).strip()
        ]
        forced_call = {
            "name": "pathway",
            "args": {
                "genes": forced_genes,
                "direction": _deg_direction_from_query(str(state.get("query") or "")),
                "gene_limit": _parse_top_n_from_text(str(state.get("query") or "")),
                "term_limit": 10,
                "text": str(state.get("query") or ""),
            },
            "id": "forced_pathway_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_memory_lookup(state):
        forced_call = {
            "name": "memory_lookup",
            "args": {
                "pathway_term": _extract_requested_pathway_name(str(state.get("query") or "")),
                "direction": _deg_direction_from_query(str(state.get("query") or "")),
                "top_n": _parse_top_n_from_text(str(state.get("query") or "")),
                "text": str(state.get("query") or ""),
            },
            "id": "forced_memory_lookup_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_literature_tool(state):
        forced_genes = [
            str(value).strip().upper()
            for value in extract_genes_from_text(str(state.get("query") or ""), mode="strict")
            if str(value).strip()
        ]
        forced_call = {
            "name": "literature",
            "args": {
                "disease_name": str(state.get("disease_name") or state.get("memory_disease_name") or ""),
                "genes": forced_genes,
                "top_n": 20,
                "text": str(state.get("query") or ""),
            },
            "id": "forced_literature_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _hypothesis_requested(state.get("query")):
        forced_genes = [
            str(value).strip().upper()
            for value in extract_genes_from_text(str(state.get("query") or ""), mode="strict")
            if str(value).strip()
        ]
        forced_call = {
            "name": "hypothesis",
            "args": {
                "hypothesis_goal": str(state.get("query") or ""),
                "genes": forced_genes,
                "disease_name": str(state.get("disease_name") or state.get("memory_disease_name") or ""),
                "hypothesis_count": 3,
                "text": str(state.get("query") or ""),
            },
            "id": "forced_hypothesis_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    if _should_force_research_literature_tool(state):
        forced_genes = [
            str(value).strip().upper()
            for value in extract_genes_from_text(str(state.get("query") or ""), mode="strict")
            if str(value).strip()
        ]
        forced_call = {
            "name": "research_literature",
            "args": {
                "user_query": str(state.get("query") or ""),
                "disease_name": str(state.get("disease_name") or state.get("memory_disease_name") or ""),
                "genes": forced_genes,
                "top_n": 20,
            },
            "id": "forced_research_literature_call",
            "type": "tool_call",
        }
        response = AIMessage(content="", tool_calls=[forced_call])
        return {
            "messages": [response],
            "step_count": int(state.get("step_count") or 0) + 1,
        }

    llm = _get_bound_llm()
    messages = [SystemMessage(content=_build_system_prompt(state)), *list(state.get("messages") or [])]
    response = llm.invoke(messages)

    update: AgentState = {
        "messages": [response],
        "step_count": int(state.get("step_count") or 0) + 1,
    }

    if not getattr(response, "tool_calls", None):
        query_text = str(state.get("query") or "")
        if (
            not _is_simple_conversational_query(query_text)
            and (_should_force_research_literature_tool(state) or _should_force_literature_tool(state))
        ):
            fallback_genes = [
                str(value).strip().upper()
                for value in extract_genes_from_text(query_text, mode="strict")
                if str(value).strip()
            ]
            fallback_call = {
                "name": "research_literature",
                "args": {
                    "user_query": query_text,
                    "disease_name": str(state.get("disease_name") or state.get("memory_disease_name") or ""),
                    "genes": fallback_genes,
                    "top_n": 20,
                },
                "id": "fallback_research_literature_call",
                "type": "tool_call",
            }
            return {
                "messages": [AIMessage(content="", tool_calls=[fallback_call])],
                "step_count": int(state.get("step_count") or 0) + 1,
            }
        update["answer"] = _compact_text(_message_content_text(getattr(response, "content", "")), limit=4000)
        update["should_finalize"] = True
    return update


def _route_after_agent(state: AgentState) -> str:
    if int(state.get("step_count") or 0) >= MAX_AGENT_STEPS:
        return "finalize"
    ai_message = _latest_ai_message(list(state.get("messages") or []))
    if not ai_message or not getattr(ai_message, "tool_calls", None):
        return "finalize"

    tool_name = str(ai_message.tool_calls[0].get("name", "")).strip()
    if tool_name == "literature" and _literature_call_count(state) >= MAX_LITERATURE_CALLS_PER_QUERY:
        return "finalize"
    if tool_name in TOOL_EXECUTORS:
        return tool_name
    return "finalize"


def _route_after_tool(state: AgentState) -> str:
    return "finalize" if state.get("should_finalize") else "agent"


def _run_extract_genes(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text") or state.get("query") or "")
    mode = str(args.get("mode") or "strict")
    whitelist = args.get("whitelist")
    whitelist_set = None
    if isinstance(whitelist, list):
        whitelist_set = {str(value).strip().upper() for value in whitelist if str(value).strip()}
    genes = extract_genes_from_text(text, whitelist=whitelist_set, mode=mode)
    return {"genes": genes}


def _run_extract_srp_ids(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    text = str(args.get("text") or state.get("query") or "")
    return {"srp_ids": _normalize_srp_ids(text)}


def _run_extract_deg_groups(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    control_name = " ".join(str(args.get("control_name") or state.get("control_name") or state.get("memory_control_name") or "").split()).strip()
    test_name = " ".join(str(args.get("test_name") or state.get("test_name") or state.get("memory_test_name") or "").split()).strip()
    if control_name and test_name:
        return {"control_name": control_name, "test_name": test_name}

    text = str(args.get("text") or state.get("query") or "")
    parsed_groups = _extract_deg_group_labels_from_text(text)
    control_name = control_name or str(parsed_groups.get("control_name") or "").strip()
    test_name = test_name or str(parsed_groups.get("test_name") or "").strip()
    if control_name and test_name:
        return {"control_name": control_name, "test_name": test_name}

    llm = get_llm()
    response = llm.invoke(
        [
            (
                "system",
                "You are a normalization step inside a biomedical agent workflow. "
                "Extract DEG comparison groups from the user request. "
                "Return exactly one minified JSON object with only the keys `control_name` and `test_name`. "
                "Do not wrap the JSON in markdown, code fences, or commentary. "
                "Use this schema exactly: {\"control_name\":\"...\",\"test_name\":\"...\"}. "
                "Interpret synonyms such as control, baseline, healthy, normal, untreated, disease, case, treated, or condition when the comparison is implied. "
                "Preserve the original human-readable cohort labels exactly when possible. "
                "If either group is missing or ambiguous, return an empty string for that field in the JSON object. "
                "Do not add explanations, reasoning, or extra keys.",
            ),
            ("user", text),
        ]
    )
    try:
        parsed = _ensure_dict(parse_json_object(_message_content_text(getattr(response, "content", "")) or "{}"))
    except Exception:
        parsed = {}
    control_name = " ".join(str(parsed.get("control_name") or "").split()).strip()
    test_name = " ".join(str(parsed.get("test_name") or "").split()).strip()
    return {"control_name": control_name, "test_name": test_name}


def _run_identify_disease(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query") or state.get("query") or "")
    disease_result = identify_disease_from_query(query)
    disease_name = disease_result.get("disease", "") or state.get("memory_disease_name") or ""
    return {"disease_name": disease_name}


def _run_opentargets_association(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(state.get("query") or "")
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    genes_arg = args.get("genes")
    genes: list[str] = []
    if isinstance(genes_arg, list):
        genes = [str(value).strip().upper() for value in genes_arg if str(value).strip()]
    if not genes:
        genes = _memory_slice_gene_candidates(state)
    if not genes and _memory_gene_query_requested(query):
        direction = _deg_direction_from_query(query)
        deg_records = _memory_slice_deg_records(state) or state.get("deg_gene_records") or state.get("memory_deg_gene_records")
        genes = _genes_from_deg_records_by_direction(
            deg_records,
            direction=direction,
            top_n=top_n if isinstance(top_n, int) and top_n > 0 else 500,
        )
        if not genes:
            genes = [
                str(value).strip().upper()
                for value in (state.get("deg_genes") or state.get("memory_deg_genes") or state.get("genes") or [])
                if str(value).strip()
            ]
            if isinstance(top_n, int) and top_n > 0:
                genes = genes[:top_n]

    gene = str(
        args.get("gene")
        or state.get("disease_gene")
        or (state.get("genes") or state.get("memory_deg_genes") or [""])[0]
        or ""
    )
    if not gene:
        extracted = extract_genes_from_text(str(state.get("query") or ""), mode="strict")
        gene = str((extracted or [""])[0] or "")
    disease = str(
        args.get("disease")
        or args.get("disease_name")
        or state.get("disease_name")
        or state.get("memory_disease_name")
        or ""
    )
    if not disease and _memory_gene_query_requested(query):
        disease = _disease_from_association_query(query)
        if not disease:
            disease_result = identify_disease_from_query(query)
            disease = str(disease_result.get("disease") or "").strip()
    if _drug_association_query_requested(query) and not genes:
        result = find_drugs_for_gene(gene)
        result["association_kind"] = "gene_drug"
    elif genes and not disease:
        if len(genes) == 1:
            result = find_diseases_for_gene(genes[0])
        else:
            results = [find_diseases_for_gene(value) for value in genes]
            result = {
                "status": "ok",
                "genes": genes,
                "associated": any(item.get("associated") for item in results if isinstance(item, dict)),
                "results": results,
                "message": f"Retrieved top OpenTargets disease associations for {len(results)} genes.",
            }
    elif genes:
        result = check_gene_list_disease_associations(genes, disease)
        result["gene_set_source"] = "stored_deg_genes" if _memory_gene_query_requested(query) else "tool_args"
        if isinstance(top_n, int) and top_n > 0:
            result["top_n"] = top_n
        result["direction"] = _deg_direction_from_query(query)
    elif not disease:
        result = find_diseases_for_gene(gene)
    else:
        result = check_gene_disease_association(gene, disease)
    result["analysis_arm"] = "opentargets"
    return result


def _extract_drug_name_from_query(text: str | None) -> str:
    query = " ".join(str(text or "").split()).strip()
    patterns = (
        r"\bpubchem\s+(?:for|of)\s+(.+?)(?:\?|$)",
        r"\bdrug\s+(?:name\s+)?(.+?)(?:\?|$)",
        r"\bcompound\s+(?:name\s+)?(.+?)(?:\?|$)",
        r"\bfor\s+drug\s+(.+?)(?:\?|$)",
        r"\bfor\s+compound\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = " ".join(str(match.group(1) or "").split()).strip(" .,:;")
        candidate = re.sub(
            r"\b(what|which|show|tell|identify|retrieve|get|give|using|from|pubchem|pathways?|genes?|diseases?)\b",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = " ".join(candidate.split()).strip(" .,:;")
        if candidate:
            return candidate
    return ""


def _extract_pert_id_from_query(text: str | None) -> str:
    query = str(text or "")
    match = re.search(r"\b(BRD[-_][A-Z0-9-]+)\b", query, flags=re.IGNORECASE)
    if match:
        return str(match.group(1) or "").strip().upper().replace("_", "-")
    return ""


def _resolve_l1000_gene_lists(
    state: AgentState,
    args: dict[str, Any],
    *,
    query: str,
) -> tuple[list[str], list[str], str]:
    up_arg = args.get("up_genes")
    down_arg = args.get("down_genes")
    up_genes = [str(value).strip().upper() for value in up_arg if str(value).strip()] if isinstance(up_arg, list) else []
    down_genes = [str(value).strip().upper() for value in down_arg if str(value).strip()] if isinstance(down_arg, list) else []
    if up_genes and down_genes:
        return list(dict.fromkeys(up_genes)), list(dict.fromkeys(down_genes)), "tool_args"

    gene_limit = args.get("gene_limit")
    if isinstance(gene_limit, str) and gene_limit.isdigit():
        gene_limit = int(gene_limit)
    if not isinstance(gene_limit, int) or gene_limit <= 0:
        gene_limit = 500

    records = _memory_slice_deg_records(state) or state.get("deg_gene_records") or state.get("memory_deg_gene_records")
    if records:
        return (
            _genes_from_deg_records_by_direction(records, direction="up", top_n=gene_limit),
            _genes_from_deg_records_by_direction(records, direction="down", top_n=gene_limit),
            "memory_slice" if _memory_slice_deg_records(state) else "stored_deg_genes",
        )

    sliced_genes = _memory_slice_gene_candidates(state)
    if sliced_genes:
        midpoint = max(1, len(sliced_genes) // 2)
        return sliced_genes[:midpoint], sliced_genes[midpoint:], "memory_slice_split"

    query_genes = [str(value).strip().upper() for value in extract_genes_from_text(query, mode="strict") if str(value).strip()]
    if query_genes:
        midpoint = max(1, len(query_genes) // 2)
        return query_genes[:midpoint], query_genes[midpoint:], "query_gene_split"

    return [], [], "missing"


def _run_l1000cds2_query(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("text") or state.get("query") or "")
    up_genes, down_genes, gene_set_source = _resolve_l1000_gene_lists(state, args, query=query)

    cell_lines_arg = args.get("cell_lines")
    cell_lines = [str(value).strip().upper() for value in cell_lines_arg if str(value).strip()] if isinstance(cell_lines_arg, list) else []
    if not cell_lines:
        cell_lines = _extract_cell_lines_from_text(query)

    result_limit = args.get("result_limit")
    if result_limit is None:
        result_limit = _parse_top_n_from_text(query)
    if isinstance(result_limit, str) and result_limit.isdigit():
        result_limit = int(result_limit)
    if not isinstance(result_limit, int) or result_limit <= 0:
        result_limit = 20

    result = query_l1000cds2(
        up_genes=up_genes,
        down_genes=down_genes,
        cell_lines=cell_lines,
        aggravate=bool(args.get("aggravate")) if args.get("aggravate") is not None else _l1000_mode_from_query(query),
        combination=bool(args.get("combination", False)),
        share=bool(args.get("share", False)),
        db_version=str(args.get("db_version") or "latest"),
        result_limit=result_limit,
    )
    result["analysis_arm"] = "l1000cds2"
    result["gene_set_source"] = gene_set_source
    result["result_limit"] = result_limit
    return result


def _run_pubchem_drug_lookup(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("text") or state.get("query") or "")
    drug_name = " ".join(str(args.get("drug_name") or "").split()).strip()
    pert_id = " ".join(str(args.get("pert_id") or "").split()).strip().upper()
    if not pert_id:
        pert_id = _extract_pert_id_from_query(query)
    if not drug_name:
        drug_name = _extract_drug_name_from_query(query)
    if not drug_name and isinstance(state.get("pubchem_result"), dict):
        drug_name = str((state.get("pubchem_result") or {}).get("drug_name") or "").strip()
    if not pert_id and isinstance(state.get("pubchem_result"), dict):
        pert_id = str((state.get("pubchem_result") or {}).get("pert_id") or "").strip().upper()
    if not drug_name and isinstance(state.get("memory_pubchem_result"), dict):
        drug_name = str((state.get("memory_pubchem_result") or {}).get("drug_name") or "").strip()
    if not pert_id and isinstance(state.get("memory_pubchem_result"), dict):
        pert_id = str((state.get("memory_pubchem_result") or {}).get("pert_id") or "").strip().upper()

    result = query_pubchem_drug(drug_name=drug_name, pert_id=pert_id)
    result["analysis_arm"] = "pubchem"
    return result


# def _run_primekg_query(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
#     query = str(state.get("query") or "")
#     top_n = args.get("top_n")
#     if top_n is None:
#         top_n = _parse_top_n_from_text(query)
#     if isinstance(top_n, str) and top_n.isdigit():
#         top_n = int(top_n)

#     source_terms = args.get("source_terms") or args.get("genes") or args.get("entities")
#     if isinstance(source_terms, str):
#         source_terms = [source_terms]
#     if not isinstance(source_terms, list):
#         source_terms = []

#     if not source_terms and args.get("gene"):
#         source_terms = [str(args.get("gene"))]

#     if not source_terms and _memory_gene_query_requested(query):
#         source_terms = _genes_from_deg_records_by_direction(
#             state.get("deg_gene_records") or state.get("memory_deg_gene_records"),
#             direction=_deg_direction_from_query(query),
#             top_n=top_n if isinstance(top_n, int) and top_n > 0 else 50,
#         )
#         if not source_terms:
#             source_terms = list(state.get("deg_genes") or state.get("memory_deg_genes") or state.get("genes") or [])
#             if isinstance(top_n, int) and top_n > 0:
#                 source_terms = source_terms[:top_n]

#     if not source_terms:
#         source_terms = extract_genes_from_text(query, mode="strict")

#     target_terms = args.get("target_terms")
#     if isinstance(target_terms, str):
#         target_terms = [target_terms]
#     if not isinstance(target_terms, list):
#         target_terms = []

#     disease = str(args.get("disease") or args.get("disease_name") or "").strip()
#     if not disease and ("associated with" in query.lower() or "linked to" in query.lower()):
#         disease = _disease_from_association_query(query)
#     if disease and source_terms and not target_terms:
#         target_terms = [disease]
#     elif disease and not source_terms:
#         source_terms = [disease]

#     source_types = args.get("source_types")
#     target_types = args.get("target_types")
#     if not isinstance(source_types, list) or not source_types:
#         source_types = ["disease"] if disease and source_terms == [disease] else (["gene/protein"] if source_terms else [])
#     if not isinstance(target_types, list) or not target_types:
#         target_types = _primekg_target_types_from_query(query)
#     relation_terms = args.get("relation_terms")
#     result = query_primekg(
#         source_terms=[str(value).strip().upper() for value in source_terms if str(value).strip()],
#         target_terms=[str(value).strip() for value in target_terms if str(value).strip()],
#         source_types=source_types if isinstance(source_types, list) else [],
#         target_types=target_types if isinstance(target_types, list) else [],
#         relation_terms=relation_terms if isinstance(relation_terms, list) else [],
#         limit=int(args.get("limit") or 50),
#     )
#     result["analysis_arm"] = "primekg"
#     if _memory_gene_query_requested(query):
#         result["gene_set_source"] = "stored_deg_genes"
#         result["direction"] = _deg_direction_from_query(query)
#     return result

def _run_primekg_query(
    state: AgentState,
    args: dict[str, Any]
):
    question = (
        args.get("question")
        or state.get("query")
        or ""
    )

    focus_genes = [
        str(value).strip().upper()
        for value in extract_genes_from_text(question, mode="strict")
        if str(value).strip()
    ]
    return query_primekg(question, focus_genes=focus_genes)

def _run_fetch_openalex(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(state.get("query") or args.get("question") or args.get("text") or "")
    disease_name = str(
        args.get("disease_name")
        or args.get("disease")
        or state.get("disease_name")
        or state.get("memory_disease_name")
        or ""
    )
    top_n = int(args.get("top_n") or 20)

    arg_genes = [
        str(value).strip().upper()
        for value in _tool_arg_list(args.get("genes"))
        if str(value).strip()
    ]
    query_genes = [
        str(value).strip().upper()
        for value in extract_genes_from_text(query, mode="strict")
        if str(value).strip()
    ]
    memory_genes = (
        _literature_state_gene_candidates(state)
        if (_literature_followup_requested(query) and _literature_memory_gene_requested(query))
        or (not disease_name and _literature_followup_requested(query) and not query_genes)
        else []
    )
    literature_genes = list(dict.fromkeys(arg_genes + query_genes + memory_genes))

    openalex_result = fetch_openalex_papers_and_genes(
        disease_name,
        top_n=top_n,
        user_query=query or disease_name,
        genes=literature_genes,
    )
    genes = openalex_result.get("genes", [])
    return {
        "analysis_arm": "disease",
        "disease_name": openalex_result.get("disease", disease_name),
        "openalex_papers": openalex_result.get("papers", []),
        "ranked_openalex_papers": openalex_result.get("ranked_papers", []),
        "openalex_genes": genes,
        "literature_key_points": openalex_result.get("key_points", []),
        "literature_references": openalex_result.get("references", []),
        "literature_summary": openalex_result.get("literature_summary", ""),
        "literature_source_status": openalex_result.get("source_status", {}),
        "literature_query": openalex_result.get("query", query or disease_name),
        "genes": _merge_unique(state.get("genes"), genes),
    }


def _run_hypothesis(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    memory_state = {
        "memory_control_name": str(state.get("memory_control_name") or ""),
        "memory_test_name": str(state.get("memory_test_name") or ""),
        "memory_deg_genes": list(state.get("memory_deg_genes") or []),
        "memory_upregulated_genes": list(state.get("memory_upregulated_genes") or []),
        "memory_downregulated_genes": list(state.get("memory_downregulated_genes") or []),
        "memory_deg_analysis": _ensure_dict(state.get("memory_deg_analysis")),
        "memory_deg_gene_records": list(state.get("memory_deg_gene_records") or []),
        "memory_enrichr": _ensure_dict(state.get("memory_enrichr")),
        "memory_rwr_seed_genes": list(state.get("memory_rwr_seed_genes") or []),
        "memory_rwr_genes": list(state.get("memory_rwr_genes") or []),
        "memory_disease_name": str(state.get("memory_disease_name") or ""),
        "memory_openalex_genes": list(state.get("memory_openalex_genes") or []),
        "memory_opentargets_results": list(state.get("memory_opentargets_results") or []),
        "memory_l1000cds2_result": _ensure_dict(state.get("memory_l1000cds2_result")),
        "memory_pubchem_result": _ensure_dict(state.get("memory_pubchem_result")),
        "memory_lookup_result": _ensure_dict(state.get("memory_lookup_result")),
        "memory_slice_result": _ensure_dict(state.get("memory_slice_result")),
    }
    result = generate_experimental_hypotheses_safe(
        user_query=str(state.get("query") or args.get("text") or ""),
        hypothesis_goal=str(args.get("hypothesis_goal") or args.get("validation_goal") or state.get("query") or ""),
        disease_name=str(args.get("disease_name") or state.get("disease_name") or state.get("memory_disease_name") or ""),
        genes=list(args.get("genes") or []),
        conversation_messages=list(state.get("messages") or []),
        memory_state=memory_state,
        hypothesis_count=int(args.get("hypothesis_count") or 3),
        include_references=False,
    )
    result["analysis_arm"] = "hypothesis"
    result["should_finalize"] = True
    return result


def _run_research_literature(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    user_query = str(args.get("user_query") or args.get("text") or state.get("query") or "").strip()
    genes = args.get("genes")
    if not isinstance(genes, list) or not genes:
        sliced_genes = _memory_slice_gene_candidates(state)
        genes = sliced_genes if sliced_genes else _literature_state_gene_candidates(state)
    result = run_publication_research_assistant_safe(
        user_query,
        disease_name=str(args.get("disease_name") or state.get("disease_name") or state.get("memory_disease_name") or ""),
        genes=genes,
        top_n=int(args.get("top_n") or 20),
    )
    result["analysis_arm"] = "research_literature"
    result["should_finalize"] = True
    return result
def _run_deg_analysis(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    srp_ids = _normalize_srp_ids(args.get("srp_ids"))
    if not srp_ids:
        srp_ids = _normalize_srp_ids(state.get("srp_ids"))
    if not srp_ids:
        srp_ids = _normalize_srp_ids(_run_extract_srp_ids(state, args).get("srp_ids"))
    group_result = _run_extract_deg_groups(state, args)
    control_name = str(group_result.get("control_name") or "").strip()
    test_name = str(group_result.get("test_name") or "").strip()
    query = str(args.get("text") or state.get("query") or "")
    log2fold, padj = _parse_deg_thresholds(query, args)
    deg_result = run_deg_r_analysis(
        srp_ids=srp_ids,
        control_name=control_name,
        test_name=test_name,
        log2fold=log2fold,
        padj=padj,
    )
    deg_genes = deg_result.get("genes", [])
    deg_rows = deg_result.get("rows", [])
    deg_gene_records: list[dict[str, Any]] = []
    if isinstance(deg_rows, list):
        for row in deg_rows:
            if not isinstance(row, dict):
                continue
            try:
                row_log2fc = abs(float(row.get("log2FoldChange")))
            except Exception:
                row_log2fc = None
            try:
                row_padj = float(row.get("pdj") or row.get("padj"))
            except Exception:
                row_padj = None
            if row_log2fc is not None and row_log2fc < log2fold:
                continue
            if row_padj is not None and row_padj >= padj:
                continue
            gene = row.get("hgnc_symbol") or row.get("external_gene_name") or row.get("Ensembl") or row.get("entrezgene_accession") or ""
            gene = str(gene).strip()
            if not gene:
                continue
            deg_gene_records.append(
                {
                    "gene": gene,
                    "pvalue": row.get("pvalue"),
                    "pdj": row.get("pdj") or row.get("padj"),
                    "log2FoldChange": row.get("log2FoldChange"),
                    "description": row.get("description"),
                }
            )
    deg_genes = [str(row.get("gene") or "").strip() for row in deg_gene_records if str(row.get("gene") or "").strip()]
    display_rows = [
        {
            "gene": row.get("gene"),
            "log2FoldChange": row.get("log2FoldChange"),
            "pvalue": row.get("pvalue"),
            "description": row.get("description"),
        }
        for row in deg_gene_records
    ]
    upregulated_rows = sorted(
        (row for row in display_rows if _safe_float(row.get("log2FoldChange")) > 0),
        key=lambda row: _safe_float(row.get("log2FoldChange")),
        reverse=True,
    )
    downregulated_rows = sorted(
        (row for row in display_rows if _safe_float(row.get("log2FoldChange")) < 0),
        key=lambda row: _safe_float(row.get("log2FoldChange")),
    )
    deg_result["rows"] = display_rows
    deg_result["upregulated_rows"] = upregulated_rows
    deg_result["downregulated_rows"] = downregulated_rows
    deg_result["genes"] = deg_genes
    deg_result["log2fold"] = log2fold
    deg_result["padj"] = padj
    upregulated_genes = _genes_from_deg_records_by_direction(deg_gene_records, direction="up")
    downregulated_genes = _genes_from_deg_records_by_direction(deg_gene_records, direction="down")
    return {
        "analysis_arm": "srp",
        "srp_ids": srp_ids,
        "control_name": control_name,
        "test_name": test_name,
        "log2fold": log2fold,
        "padj": padj,
        "deg_analysis": deg_result,
        "deg_genes": deg_genes,
        "upregulated_genes": upregulated_genes,
        "downregulated_genes": downregulated_genes,
        "deg_gene_records": deg_gene_records,
        "genes": _merge_unique(state.get("genes"), deg_genes),
    }


def _run_srp_metadata(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    srp_ids = _normalize_srp_ids(args.get("srp_ids"))
    if not srp_ids:
        srp_ids = _normalize_srp_ids(state.get("srp_ids"))
    if not srp_ids:
        srp_ids = _normalize_srp_ids(args.get("text") or state.get("query") or "")
    return fetch_srp_metadata_summary_safe(
        srp_ids=srp_ids,
        text=str(args.get("text") or state.get("query") or ""),
        species=str(args.get("species") or "hsapiens"),
        max_dee2_rows=int(args.get("max_dee2_rows") or 5000),
        max_biosamples=int(args.get("max_biosamples") or 80),
    )


def _run_druggability(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    gene = str(args.get("gene") or "").strip()
    if not gene:
        extracted = extract_genes_from_text(str(state.get("query") or ""), mode="strict")
        gene = str((extracted or [""])[0] or "").strip()
    if not gene:
        candidates = []
        if isinstance(args.get("genes"), list):
            candidates.extend(args.get("genes") or [])
        candidates.extend(state.get("genes") or [])
        candidates.extend(state.get("memory_deg_genes") or [])
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                gene = text
                break

    return run_druggability_analysis_safe(
        gene=gene,
        organism_id=int(args.get("organism_id") or 9606),
        chain=str(args.get("chain") or ""),
        ligand=str(args.get("ligand") or ""),
        top_n=int(args.get("top_n") or 3),
        output_dir=args.get("output_dir"),
        dogsite_timeout_seconds=int(args.get("dogsite_timeout_seconds") or 900),
        poll_interval_seconds=int(args.get("poll_interval_seconds") or 15),
        pdbfixer_ph=float(args.get("pdbfixer_ph") or 7.0),
    )


def _run_pdb_visualizer(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    gene = str(args.get("gene") or "").strip()
    pdb_id = str(args.get("pdb_id") or "").strip()
    uniprot_id = str(args.get("uniprot_id") or "").strip()
    protein = str(args.get("protein") or "").strip()
    if not any((gene, pdb_id, uniprot_id, protein)):
        query = str(state.get("query") or "")
        pdb_ids = re.findall(r"\b[0-9][A-Za-z0-9]{3}\b", query)
        if pdb_ids:
            pdb_id = pdb_ids[0]
        extracted = extract_genes_from_text(query, mode="strict")
        if extracted:
            gene = str(extracted[0])
    if not any((gene, pdb_id, uniprot_id, protein)):
        candidates = list(state.get("genes") or []) + list(state.get("memory_deg_genes") or [])
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                gene = text
                break

    return run_pdb_visualization_safe(
        gene=gene,
        protein=protein,
        uniprot_id=uniprot_id,
        pdb_id=pdb_id,
        organism_id=int(args.get("organism_id") or 9606),
        output_dir=args.get("output_dir"),
    )


def _run_build_string_graph(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    genes = args.get("genes")
    if not isinstance(genes, list) or not genes:
        genes, gene_set_source = _resolve_rwr_source_genes(state, args, prefer_seed_genes=False)
    else:
        gene_set_source = "tool_args"
    genes = [str(value).strip().upper() for value in genes if str(value).strip()]
    if not genes:
        return {
            "status": "missing_seed_genes",
            "message": "No seed genes were available for STRING graph construction.",
            "graph": nx.Graph(),
            "genes": [],
            "rwr_seed_genes": [],
            "gene_set_source": gene_set_source,
        }

    info_path = str(args.get("info_path") or SETTINGS.string_info_path)
    links_path = str(args.get("links_path") or SETTINGS.string_links_path)
    info_resolved = Path(SETTINGS.resolve_path(info_path))
    links_resolved = Path(SETTINGS.resolve_path(links_path))
    if not info_resolved.exists() or not links_resolved.exists():
        missing = []
        if not info_resolved.exists():
            missing.append(str(info_resolved))
        if not links_resolved.exists():
            missing.append(str(links_resolved))
        return {
            "status": "string_files_missing",
            "message": "STRING graph files were not found: " + ", ".join(missing),
            "graph": nx.Graph(),
            "genes": genes,
            "rwr_seed_genes": genes,
            "gene_set_source": gene_set_source,
        }

    gene_to_id = load_gene_to_string_id(str(info_resolved))
    mapped_seed_genes = [gene for gene in genes if gene_to_id.get(gene)]
    unmapped_seed_genes = [gene for gene in genes if not gene_to_id.get(gene)]
    if not mapped_seed_genes:
        return {
            "status": "genes_not_in_string",
            "message": "None of the provided seed genes could be mapped into the local STRING index.",
            "graph": nx.Graph(),
            "genes": genes,
            "rwr_seed_genes": genes,
            "mapped_seed_genes": mapped_seed_genes,
            "unmapped_seed_genes": unmapped_seed_genes,
            "gene_set_source": gene_set_source,
        }

    graph = build_weighted_graph_from_string_files(
        genes=mapped_seed_genes,
        info_path=info_path,
        links_path=links_path,
        required_score=int(args.get("required_score") or SETTINGS.string_required_score),
        mode=str(args.get("mode") or SETTINGS.string_local_mode),
    )
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    update: dict[str, Any] = {
        "status": "ok" if graph.number_of_nodes() > 0 else "graph_empty",
        "message": (
            f"Built STRING graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
            if graph.number_of_nodes() > 0
            else "STRING graph construction completed, but no qualifying edges were found for the mapped seed genes."
        ),
        "graph": graph,
        "genes": mapped_seed_genes,
        "rwr_seed_genes": mapped_seed_genes,
        "mapped_seed_genes": mapped_seed_genes,
        "unmapped_seed_genes": unmapped_seed_genes,
        "gene_set_source": gene_set_source,
    }
    if analysis_arm:
        update["analysis_arm"] = analysis_arm
    return update


def _run_top_rwr(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return {
            "status": "missing_graph",
            "message": "RWR could not run because no STRING graph is available.",
            "rwr_genes": [],
            "rwr_seed_genes": list(state.get("genes") or []),
        }

    query = str(args.get("text") or state.get("query") or "")
    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    seed_genes = args.get("seed_genes")
    if not isinstance(seed_genes, list) or not seed_genes:
        seed_genes, gene_set_source = _resolve_rwr_source_genes(state, args, prefer_seed_genes=True)
    else:
        gene_set_source = "tool_args"
    seed_genes = [str(value).strip().upper() for value in seed_genes if str(value).strip()]
    if isinstance(top_n, int) and top_n > 0:
        seed_genes = seed_genes[:top_n]
    if not seed_genes:
        return {
            "status": "missing_seed_genes",
            "message": "RWR could not run because no usable seed genes were available.",
            "rwr_genes": [],
            "rwr_seed_genes": [],
            "gene_set_source": gene_set_source,
        }

    rwr = top_rwr_genes(
        graph,
        seed_genes,
        top_k=int(top_n or args.get("top_k") or 30),
        restart_prob=float(args.get("restart_prob") or 0.5),
        exclude=args.get("exclude"),
        exclude_hubs=bool(args.get("exclude_hubs", True)),
    )
    update: dict[str, Any] = {
        "status": "ok" if rwr else "no_rwr_hits",
        "message": (
            f"RWR completed and returned {len(rwr)} ranked genes."
            if rwr
            else "RWR ran, but no ranked genes were returned after filtering."
        ),
        "rwr_genes": rwr,
        "rwr_seed_genes": seed_genes,
        "direction": direction,
        "gene_set_source": gene_set_source,
    }
    if isinstance(top_n, int) and top_n > 0:
        update["top_n"] = top_n
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    if gene_set_source in {"tool_args", "query_genes"}:
        update["analysis_arm"] = "general"
    elif analysis_arm in {"general", "disease", "memory_rwr"}:
        update["analysis_arm"] = analysis_arm
    elif state.get("memory_deg_genes") or state.get("memory_deg_gene_records"):
        update["analysis_arm"] = "memory_rwr"
    elif state.get("disease_name") or state.get("openalex_papers") or state.get("openalex_genes"):
        update["analysis_arm"] = "disease"
    else:
        update["analysis_arm"] = "general"
    return update


def _run_enrichr(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    analysis_arm = str(args.get("analysis_arm") or state.get("analysis_arm") or "").strip().lower()
    query = str(args.get("text") or state.get("query") or "")
    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    if direction == "both":
        direction = "all"
    genes = args.get("genes")
    gene_set_source = "tool_args" if isinstance(genes, list) and genes else ""
    gene_limit = args.get("gene_limit")
    if gene_limit is None:
        gene_limit = args.get("top_n")
    if gene_limit is None:
        gene_limit = _parse_top_n_from_text(query)
    if isinstance(gene_limit, str) and gene_limit.isdigit():
        gene_limit = int(gene_limit)
    term_limit = args.get("term_limit")
    if term_limit is None:
        term_limit = args.get("result_limit")
    if term_limit is None:
        term_limit = 10
    if isinstance(term_limit, str) and term_limit.isdigit():
        term_limit = int(term_limit)
    if isinstance(genes, list) and isinstance(gene_limit, int) and gene_limit > 0:
        genes = genes[:gene_limit]
    if not isinstance(genes, list) or not genes:
        sliced_genes = _memory_slice_gene_candidates(state)
        if sliced_genes and _should_use_memory_slice_for_current_query(state, query, requested_limit=gene_limit):
            genes = sliced_genes[:gene_limit] if isinstance(gene_limit, int) and gene_limit > 0 else sliced_genes
            gene_set_source = "memory_slice"
        elif analysis_arm == "srp":
            genes = _stored_deg_genes_by_direction(
                state,
                direction=direction,
                top_n=gene_limit,
            )
            gene_set_source = "stored_deg_genes"
        else:
            genes = _merge_unique(
                _stored_deg_genes_by_direction(
                    state,
                    direction=direction,
                    top_n=gene_limit,
                ),
                state.get("genes"),
                [gene for gene, _ in (state.get("rwr_genes") or [])],
            )
            gene_set_source = "stored_deg_genes" if _stored_deg_genes_by_direction(state, direction=direction, top_n=gene_limit) else "state_fallback"

    background = [str(g).strip().upper() for g in _tool_arg_list(args.get("background_genes")) if str(g).strip()]
    if not background:
        background = list((state.get("graph") or nx.Graph()).nodes()) if isinstance(state.get("graph"), nx.Graph) else []
    if analysis_arm == "srp":
        background = _stored_deg_genes_by_direction(state, direction="all") or list(genes)

    return {
        "direction": direction,
        "analysis_arm": "pathway",
        "gene_limit": gene_limit,
        "term_limit": term_limit,
        "selected_genes": list(genes or []),
        "input_gene_count": len(list(genes or [])),
        "gene_set_source": gene_set_source or ("stored_deg_genes" if _memory_gene_query_requested(query) or state.get("deg_gene_records") or state.get("memory_deg_gene_records") else "tool_args"),
        "enrichr": enrichr_pathways(
            genes,
            top_n=int(term_limit or 10),
            background_genes=background,
        )
    }


def _visualization_gene_set(
    state: AgentState,
    args: dict[str, Any],
    *,
    query: str,
) -> tuple[list[str], str, str, int | None]:
    direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
    top_n = args.get("top_n")
    if top_n is None:
        top_n = _parse_top_n_from_text(query)
    if isinstance(top_n, str) and top_n.isdigit():
        top_n = int(top_n)

    explicit_genes = [str(g).strip().upper() for g in _tool_arg_list(args.get("genes")) if str(g).strip()]
    if explicit_genes:
        if isinstance(top_n, int) and top_n > 0:
            explicit_genes = explicit_genes[:top_n]
        return explicit_genes, "tool_args", direction, top_n

    sliced_genes = _memory_slice_gene_candidates(state)
    if sliced_genes:
        if isinstance(top_n, int) and top_n > 0:
            sliced_genes = sliced_genes[:top_n]
        return sliced_genes, "memory_slice", direction, top_n

    deg_records = _memory_slice_deg_records(state) or state.get("deg_gene_records") or state.get("memory_deg_gene_records")
    deg_genes = _genes_from_deg_records_by_direction(
        deg_records,
        direction=direction,
        top_n=top_n,
    )
    if deg_genes:
        return deg_genes, "stored_deg_genes", direction, top_n

    fallback = _merge_unique(
        state.get("genes"),
        [gene for gene, _ in (state.get("rwr_genes") or [])],
        state.get("openalex_genes"),
        state.get("memory_openalex_genes"),
    )
    if isinstance(top_n, int) and top_n > 0:
        fallback = fallback[:top_n]
    return fallback, "state_fallback", direction, top_n


def _memory_rwr_targets(state: AgentState, *, top_k: int = 20) -> list[str]:
    targets = state.get("rwr_genes") or state.get("memory_rwr_genes") or []
    out: list[str] = []
    for row in targets[:top_k]:
        if isinstance(row, (list, tuple)) and row:
            gene = str(row[0]).strip().upper()
        elif isinstance(row, dict):
            gene = str(row.get("gene") or "").strip().upper()
        else:
            gene = ""
        if gene:
            out.append(gene)
    return out


def _run_visualize(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("text") or state.get("query") or "")
    visualization_type = str(args.get("visualization_type") or args.get("kind") or "").strip().lower()
    if not visualization_type:
        lowered = query.lower()
        if "volcano" in lowered:
            visualization_type = "volcano"
        elif "kegg" in lowered:
            visualization_type = "kegg"
        else:
            visualization_type = "network"

    if visualization_type == "network":
        requested_graph = args.get("graph")
        graph = requested_graph if isinstance(requested_graph, nx.Graph) else state.get("graph")
        graph_source = "args.graph" if isinstance(requested_graph, nx.Graph) else "state.graph"
        rebuilt_graph = False
        seed_genes = [
            str(g).strip().upper()
            for g in (
                args.get("genes")
                or state.get("rwr_seed_genes")
                or state.get("memory_rwr_seed_genes")
                or state.get("genes")
                or []
            )
            if str(g).strip()
        ]
        top_targets = _memory_rwr_targets(state, top_k=int(args.get("top_k") or 20))
        allowed_nodes = _merge_unique(seed_genes, top_targets)
        if (not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0) and allowed_nodes:
            graph = build_weighted_graph_from_string_files(
                genes=allowed_nodes,
                info_path=str(args.get("info_path") or SETTINGS.string_info_path),
                links_path=str(args.get("links_path") or SETTINGS.string_links_path),
                required_score=int(args.get("required_score") or SETTINGS.string_required_score),
                mode=str(args.get("mode") or SETTINGS.string_local_mode),
            )
            rebuilt_graph = True
            graph_source = "rebuilt_from_rwr_memory"
        result = build_network_visualization(
            graph,
            output_path=str(args.get("output_path") or "pyvis_network.html"),
            select_top_degree=int(args.get("select_top_degree") or 300),
            allowed_nodes=allowed_nodes,
            seed_genes=seed_genes,
            rwr_genes=top_targets,
        )
        result["visualization_type"] = "network"
        result["seed_genes"] = seed_genes
        result["top_targets"] = top_targets
        result["rwr_seed_genes"] = seed_genes
        result["rwr_genes"] = state.get("rwr_genes") or state.get("memory_rwr_genes") or []
        result["rebuilt_graph_from_memory"] = rebuilt_graph
        result["graph_source"] = graph_source
        if isinstance(graph, nx.Graph) and graph.number_of_nodes() > 0:
            result["graph"] = graph
        return result

    if visualization_type == "kegg":
        pathway_term = str(args.get("pathway_term") or "").strip()
        if not pathway_term:
            pathway_term = _extract_requested_pathway_name(query)
        selected_term, selected_library, selected_rank = _find_enrichr_term_from_state(
            state,
            pathway_term,
            query=query,
            library_filter={"KEGG_2021_Human"},
        )
        if not selected_term:
            selected_term, selected_library, selected_rank = _find_enrichr_term_from_state(
                state,
                pathway_term,
                query=query,
            )
        selected_term_genes = _term_overlapping_genes(selected_term)
        if selected_term_genes:
            genes = selected_term_genes
            gene_set_source = "stored_pathway_overlapping_genes"
            direction = str(args.get("direction") or _deg_direction_from_query(query) or "all").strip().lower()
            top_n = len(genes)
        else:
            genes, gene_set_source, direction, top_n = _visualization_gene_set(state, args, query=query)
        selected_library_text = str(selected_library or "").strip().lower()
        kegg_rank = (
            int(selected_rank)
            if selected_rank and selected_library_text == "kegg_2021_human"
            else int(args.get("kegg_rank") or 1)
        )
        result = build_kegg_pathway_visualization(
            genes,
            output_path=str(args.get("output_path") or "kegg_pathway.png"),
            kegg_rank=kegg_rank,
            species=str(args.get("species") or "human"),
            pathway_term=pathway_term,
        )
        result["visualization_type"] = "kegg"
        result["gene_set_source"] = gene_set_source
        result["direction"] = direction
        result["top_n"] = top_n
        if pathway_term:
            result["requested_pathway_term"] = pathway_term
        if selected_term:
            result["selected_pathway"] = {
                "library": selected_library,
                "term": (
                    selected_term.get("term")
                    or selected_term.get("path_name")
                    or selected_term.get("term_name")
                    or selected_term.get("name")
                    or selected_term.get("Path")
                    or selected_term.get("Term")
                ),
                "overlapping_genes": _term_overlapping_genes(selected_term),
                "rank": selected_rank,
            }
        return result

    if visualization_type == "volcano":
        deg_rows = state.get("deg_gene_records") or state.get("memory_deg_gene_records") or []
        result = build_volcano_plot(
            deg_rows,
            output_path=str(args.get("output_path") or "deg_volcano.html"),
            pvalue_threshold=float(args.get("pvalue_threshold") or 0.05),
            log2fc_threshold=float(args.get("log2fc_threshold") or 1.0),
        )
        result["visualization_type"] = "volcano"
        return result

    return {
        "status": "unsupported_visualization",
        "message": f"Unsupported visualization type: {visualization_type}",
        "visualization_type": visualization_type,
    }


def _visualization_answer(result: dict[str, Any]) -> str:
    visualization_type = str(result.get("visualization_type") or "").strip().lower()
    status = str(result.get("status") or "").strip().lower()
    output_path = str(
        result.get("pyvis_html_path")
        or result.get("kegg_pathway_path")
        or result.get("volcano_plot_path")
        or ""
    ).strip()
    if status != "ok":
        if output_path:
            status = "ok"
        else:
            return str(result.get("message") or "Visualization could not be generated.").strip()

    if visualization_type == "network":
        node_count = int(result.get("visualized_node_count") or 0)
        edge_count = int(result.get("visualized_edge_count") or 0)
        seed_count = len(result.get("seed_genes") or [])
        target_count = len(result.get("top_targets") or [])
        html_path = str(result.get("pyvis_html_path") or "").strip()
        parts = [
            "Successfully generated the interactive network visualization.",
        ]
        if node_count or edge_count:
            parts.append(f"The network includes {node_count} nodes and {edge_count} edges.")
        if seed_count:
            parts.append(f"Seed genes highlighted: {seed_count}.")
        if target_count:
            parts.append(f"RWR result genes highlighted: {target_count}.")
        if html_path:
            parts.append(f"Output saved to: {html_path}.")
        return " ".join(parts)

    if visualization_type == "kegg":
        path = str(result.get("kegg_pathway_path") or "").strip()
        message = "Successfully generated the KEGG pathway visualization."
        return f"{message} Output saved to: {path}." if path else message

    if visualization_type == "volcano":
        path = str(result.get("volcano_plot_path") or "").strip()
        points = int(result.get("points") or 0)
        message = "Successfully generated the DEG volcano plot."
        detail = f" Points plotted: {points}." if points else ""
        suffix = f" Output saved to: {path}." if path else ""
        return f"{message}{detail}{suffix}"

    message = "Successfully generated the requested visualization."
    return f"{message} Output saved to: {output_path}." if output_path else message


def _run_synthesize(state: AgentState, args: dict[str, Any]) -> dict[str, Any]:
    graph = state.get("graph")
    answer = synthesize_technical_response(
        user_query=str(args.get("user_query") or state.get("query") or ""),
        analysis_arm=str(args.get("analysis_arm") or state.get("analysis_arm") or _infer_analysis_arm(state)).strip().lower(),
        seed_genes=list(args.get("seed_genes") or state.get("genes") or []),
        srp_ids=list(args.get("srp_ids") or state.get("srp_ids") or []),
        disease_name=str(args.get("disease_name") or state.get("disease_name") or ""),
        deg_analysis=_ensure_dict(state.get("opentargets_result") or state.get("primekg_result") or state.get("deg_analysis")),
        rwr_genes=list(state.get("rwr_genes") or []),
        graph=graph if isinstance(graph, nx.Graph) else nx.Graph(),
        enrichr=_ensure_dict(state.get("enrichr") or state.get("memory_enrichr")),
        literature_papers=list(state.get("openalex_papers") or []),
        ranked_literature_papers=list(state.get("ranked_openalex_papers") or []),
        literature_key_points=list(state.get("literature_key_points") or []),
        literature_references=list(state.get("literature_references") or []),
        literature_summary=str(state.get("literature_summary") or ""),
        memory_lookup_result=_ensure_dict(state.get("memory_lookup_result")),
        state_lookup_result=_ensure_dict(state.get("state_lookup_result")),
        memory_slice_result=_ensure_dict(state.get("memory_slice_result")),
    )
    return {
        "answer": answer,
        "analysis_arm": str(args.get("analysis_arm") or state.get("analysis_arm") or "disease").strip().lower(),
        "should_finalize": True,
    }


def _specialist_history_update(state: AgentState, tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    history = list(state.get("tool_history") or [])
    history.append({"tool": tool_name, "args": args, "result": _serialize_tool_result(result)})
    return {"tool_history": history}


def _execute_tool_runner(tool_name: str, runner: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return normalize_tool_result(tool_name, runner())
    except Exception as exc:
        return tool_error_result(
            tool_name,
            f"{tool_name} failed: {sanitize_exception_message(exc)}",
        )


def _preserve_existing_graph_for_lookup(state: AgentState, update: dict[str, Any]) -> dict[str, Any]:
    existing_graph = state.get("graph")
    incoming_graph = update.get("graph")
    if (
        isinstance(existing_graph, nx.Graph)
        and existing_graph.number_of_nodes() > 0
        and not (isinstance(incoming_graph, nx.Graph) and incoming_graph.number_of_nodes() > 0)
    ):
        update = dict(update)
        update.pop("graph", None)
    return update


def _tool_observations(state: AgentState, call: dict[str, Any] | None, tool_name: str, result: dict[str, Any]) -> list[ToolMessage]:
    ai_message = _latest_ai_message(list(state.get("messages") or []))
    tool_calls = list(getattr(ai_message, "tool_calls", []) or [])
    if not tool_calls and call:
        tool_calls = [call]

    selected_id = str(call.get("id")) if call and call.get("id") else ""
    messages: list[ToolMessage] = []
    for tool_call in tool_calls:
        call_id = str(tool_call.get("id") or "")
        if not call_id:
            continue
        name = str(tool_call.get("name") or tool_name)
        if selected_id and call_id == selected_id:
            payload = _serialize_tool_result(result)
        else:
            payload = {
                "status": "deferred",
                "message": (
                    "This tool call was not executed in this step because the agent executes "
                    "one specialist at a time. Call it again if it is still needed."
                ),
            }
        messages.append(
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str),
                name=name,
                tool_call_id=call_id,
            )
        )
    return messages


def _specialist_node(tool_name: str) -> Callable[[AgentState], AgentState]:
    def node(state: AgentState) -> AgentState:
        call = _latest_tool_call(state, tool_name)
        args = dict(call.get("args") or {}) if call else {}
        _trace_tool_call(tool_name)

        if tool_name == "srp_metadata":
            result = _execute_tool_runner("srp_metadata", lambda: _run_srp_metadata(state, args))
            update = _specialist_history_update(state, "srp_metadata", args, result)
            update = {**update, **result}
            update["analysis_arm"] = "srp_metadata"
            update["srp_metadata_result"] = result
            update["memory_srp_metadata_result"] = result
            update["should_finalize"] = True
            state = {**state, **update}
            return {**state, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "deg_analysis":
            result = _execute_tool_runner("extract_srp_ids_from_text", lambda: _run_extract_srp_ids(state, args))
            update = _specialist_history_update(state, "extract_srp_ids_from_text", args, result)
            update = {**update, **result}
            state = {**state, **update}

            result = _execute_tool_runner("run_deg_r_analysis", lambda: _run_deg_analysis(state, args))
            update = _specialist_history_update(state, "run_deg_r_analysis", args, result)
            update = {**update, **result}
            update["memory_control_name"] = str(result.get("control_name") or "")
            update["memory_test_name"] = str(result.get("test_name") or "")
            update["memory_deg_analysis"] = _ensure_dict(result.get("deg_analysis"))
            update["memory_deg_genes"] = list(result.get("deg_genes") or [])
            update["memory_upregulated_genes"] = list(result.get("upregulated_genes") or [])
            update["memory_downregulated_genes"] = list(result.get("downregulated_genes") or [])
            update["memory_deg_gene_records"] = list(result.get("deg_gene_records") or [])
            state = {**state, **update}
            return {**state, **result, "analysis_arm": "srp", "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "pathway":
            result = _execute_tool_runner("enrichr_pathways", lambda: _run_enrichr(state, args))
            update = _specialist_history_update(state, "enrichr_pathways", args, result)
            update = {**update, **result}
            update["memory_enrichr"] = _ensure_dict(result.get("enrichr"))
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "rwr_analysis":
            build_result = _execute_tool_runner("build_weighted_graph_from_string_files", lambda: _run_build_string_graph(state, args))
            update = _specialist_history_update(state, "build_weighted_graph_from_string_files", args, build_result)
            update = {**update, **build_result}
            state = {**state, **update}

            rwr_result = _execute_tool_runner("top_rwr_genes", lambda: _run_top_rwr(state, args))
            update = _specialist_history_update(state, "top_rwr_genes", args, rwr_result)
            update = {**update, **rwr_result}
            graph_obj = state.get("graph")
            if isinstance(graph_obj, nx.Graph) and graph_obj.number_of_nodes() > 0:
                seed_genes = [str(g).strip().upper() for g in rwr_result.get("rwr_seed_genes", []) if str(g).strip()]
                rwr_hits = [
                    str(row[0]).strip().upper()
                    for row in (rwr_result.get("rwr_genes") or [])[:20]
                    if isinstance(row, (list, tuple)) and row and str(row[0]).strip()
                ]
                try:
                    visualization_result = build_network_visualization(
                        graph_obj,
                        output_path=str(args.get("output_path") or "pyvis_network.html"),
                        select_top_degree=int(args.get("select_top_degree") or 300),
                        allowed_nodes=None,
                        seed_genes=seed_genes,
                        rwr_genes=rwr_hits,
                    )
                    visualization_result["visualization_type"] = "network"
                    visualization_result["seed_genes"] = seed_genes
                    visualization_result["top_targets"] = rwr_hits
                    update["visualization_result"] = visualization_result
                    if visualization_result.get("pyvis_html_path"):
                        update["pyvis_html_path"] = visualization_result.get("pyvis_html_path")
                    if visualization_result.get("status") == "ok":
                        update["message"] = (
                            f"{str(rwr_result.get('message') or '').strip()} "
                            f"Generated network visualization at {visualization_result.get('pyvis_html_path')}."
                        ).strip()
                except Exception as exc:
                    update["visualization_result"] = {
                        "status": "visualization_failed",
                        "message": sanitize_exception_message(exc),
                        "visualization_type": "network",
                    }
            state = {**state, **update}
            return {**state, "messages": _tool_observations(state, call, tool_name, update)}

        if tool_name == "visualize":
            result = _execute_tool_runner("visualize", lambda: _run_visualize(state, args))
            update = _specialist_history_update(state, "visualize", args, result)
            update = {**update, **result}
            visual_answer = _visualization_answer(result)
            update["visualization_result"] = result
            update["answer"] = visual_answer
            update["message"] = visual_answer if str(result.get("status") or "").strip().lower() == "ok" else str(result.get("message") or visual_answer)
            if isinstance(result.get("graph"), nx.Graph):
                update["graph"] = result["graph"]
            if isinstance(result.get("rwr_seed_genes"), list):
                update["memory_rwr_seed_genes"] = list(result.get("rwr_seed_genes") or [])
            if isinstance(result.get("rwr_genes"), list):
                update["memory_rwr_genes"] = list(result.get("rwr_genes") or [])
            update["should_finalize"] = True
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "hypothesis":
            result = _execute_tool_runner("hypothesis", lambda: _run_hypothesis(state, args))
            update = _specialist_history_update(state, "hypothesis", args, result)
            update = {**update, **result}
            update["analysis_arm"] = "hypothesis"
            update["hypothesis_result"] = result
            update["memory_hypothesis_result"] = result
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "memory_lookup":
            result = _execute_tool_runner("memory_lookup", lambda: _run_memory_lookup(state, args))
            update = _specialist_history_update(state, "memory_lookup", args, result)
            update = {**update, **result}
            update["memory_lookup_result"] = result
            update = _preserve_existing_graph_for_lookup(state, update)
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "state_lookup":
            try:
                result, fallback_update = _run_state_lookup_with_memory_fallback(state, args)
            except Exception as exc:
                result = tool_error_result(
                    "state_lookup",
                    f"state_lookup failed: {sanitize_exception_message(exc)}",
                )
                fallback_update = {"state_lookup_result": result}
            update = _specialist_history_update(state, "state_lookup", args, result)
            update = {**update, **result}
            update = {**update, **fallback_update}
            update["state_lookup_result"] = fallback_update.get("state_lookup_result", result)
            if result.get("analysis_arm") == "memory_slice":
                update["memory_slice_result"] = result
            elif result.get("analysis_arm") == "memory_lookup":
                update["memory_lookup_result"] = result
            update = _preserve_existing_graph_for_lookup(state, update)
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "memory_slice":
            result = _execute_tool_runner("memory_slice", lambda: _run_memory_slice(state, args))
            update = _specialist_history_update(state, "memory_slice", args, result)
            update = {**update, **result}
            update["memory_slice_result"] = result
            if _should_chain_research_after_memory_slice(state, result):
                update["should_finalize"] = False
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "literature":
            disease_result = _execute_tool_runner("identify_disease_from_query", lambda: _run_identify_disease(state, args))
            update = _specialist_history_update(state, "identify_disease_from_query", args, disease_result)
            update = {**update, **disease_result}
            state = {**state, **update}

            openalex_result = _execute_tool_runner("fetch_openalex_papers_and_genes", lambda: _run_fetch_openalex(state, args))
            update = _specialist_history_update(state, "fetch_openalex_papers_and_genes", args, openalex_result)
            update = {**update, **openalex_result}
            state = {**state, **update}

            gene_result = _execute_tool_runner("extract_genes_from_text", lambda: _run_extract_genes(state, args))
            update = _specialist_history_update(state, "extract_genes_from_text", args, gene_result)
            update = {**update, **gene_result}
            state = {**state, **update}
            return {**state, **openalex_result, "messages": _tool_observations(state, call, tool_name, openalex_result)}

        if tool_name == "research_literature":
            result = _execute_tool_runner("research_literature", lambda: _run_research_literature(state, args))
            update = _specialist_history_update(state, "run_literature_agent", args, result)
            update = {**update, **result}
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "identify_disease_from_query":
            result = _execute_tool_runner("identify_disease_from_query", lambda: _run_identify_disease(state, args))
            update = _specialist_history_update(state, "identify_disease_from_query", args, result)
            update = {**update, **result}
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "primekg_query":
            result = _execute_tool_runner("primekg_query", lambda: _run_primekg_query(state, args))
            update = _specialist_history_update(state, "primekg_query", args, result)
            update = {**update, **result}
            update["primekg_result"] = result
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "opentargets_association":
            result = _execute_tool_runner("opentargets_association", lambda: _run_opentargets_association(state, args))
            update = _specialist_history_update(state, "opentargets_association", args, result)
            update = {**update, **result}
            update["opentargets_result"] = result
            if result.get("gene"):
                update["disease_gene"] = str(result.get("gene") or "").strip().upper()
            if result.get("status") == "ok":
                history = list(state.get("memory_opentargets_results") or [])
                history.append(result)
                update["memory_opentargets_results"] = history
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "l1000cds2_query":
            result = _execute_tool_runner("l1000cds2_query", lambda: _run_l1000cds2_query(state, args))
            update = _specialist_history_update(state, "l1000cds2_query", args, result)
            update = {**update, **result}
            update["analysis_arm"] = "l1000cds2"
            update["l1000cds2_result"] = result
            if result.get("status") == "ok":
                update["memory_l1000cds2_result"] = result
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "pubchem_drug_lookup":
            result = _execute_tool_runner("pubchem_drug_lookup", lambda: _run_pubchem_drug_lookup(state, args))
            update = _specialist_history_update(state, "pubchem_drug_lookup", args, result)
            update = {**update, **result}
            update["analysis_arm"] = "pubchem"
            update["pubchem_result"] = result
            if result.get("status") == "ok":
                update["memory_pubchem_result"] = result
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "druggability":
            result = _execute_tool_runner("druggability", lambda: _run_druggability(state, args))
            update = _specialist_history_update(state, "druggability", args, result)
            update = {**update, **result}
            update["analysis_arm"] = "druggability"
            update["druggability_result"] = result
            update["memory_druggability_result"] = result
            if result.get("gene"):
                update["genes"] = _merge_unique(state.get("genes"), [str(result.get("gene") or "").strip().upper()])
            update["answer"] = str(result.get("answer") or result.get("message") or "").strip()
            update["should_finalize"] = True
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "pdb_visualizer":
            result = _execute_tool_runner("pdb_visualizer", lambda: _run_pdb_visualizer(state, args))
            update = _specialist_history_update(state, "pdb_visualizer", args, result)
            update = {**update, **result}
            update["analysis_arm"] = "pdb_visualizer"
            update["pdb_visualization_result"] = result
            update["memory_pdb_visualization_result"] = result
            if result.get("gene"):
                update["genes"] = _merge_unique(state.get("genes"), [str(result.get("gene") or "").strip().upper()])
            update["answer"] = str(result.get("answer") or result.get("message") or "").strip()
            update["should_finalize"] = True
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        if tool_name == "synthesize_technical_response":
            result = _execute_tool_runner("synthesize_technical_response", lambda: _run_synthesize(state, args))
            update = _specialist_history_update(state, "synthesize_technical_response", args, result)
            update = {**update, **result}
            update["should_finalize"] = True
            return {**state, **update, "messages": _tool_observations(state, call, tool_name, result)}

        return state

    return node


def _finalize(state: AgentState) -> AgentState:
    answer = str(state.get("answer") or "").strip()
    if not answer:
        ai_message = _latest_ai_message(list(state.get("messages") or []))
        if ai_message and not getattr(ai_message, "tool_calls", None):
            answer = _compact_text(_message_content_text(getattr(ai_message, "content", "")), limit=4000)

    if state.get("tool_history"):
        analysis_arm = _infer_analysis_arm(state)
        if analysis_arm == "srp_metadata" and answer:
            graph = state.get("graph")
            meta = {
                "analysis_arm": analysis_arm,
                "is_followup": bool(state.get("is_followup", False)),
                "route_rationale": state.get("route_rationale", ""),
                "srp_ids": list(state.get("srp_ids") or []),
                "memory_srp_metadata_result": _ensure_dict(state.get("memory_srp_metadata_result")),
                "srp_metadata_result": _ensure_dict(state.get("srp_metadata_result")),
                "tool_history": list(state.get("tool_history") or [])[-10:],
            }
            return {
                "answer": answer,
                "meta": meta,
                "analysis_arm": analysis_arm,
                "graph": graph if isinstance(graph, nx.Graph) else None,
            }
        if analysis_arm == "druggability" and answer:
            graph = state.get("graph")
            meta = {
                "analysis_arm": analysis_arm,
                "is_followup": bool(state.get("is_followup", False)),
                "route_rationale": state.get("route_rationale", ""),
                "genes": list(state.get("genes") or []),
                "druggability_result": _ensure_dict(state.get("druggability_result")),
                "memory_druggability_result": _ensure_dict(state.get("memory_druggability_result")),
                "tool_history": list(state.get("tool_history") or [])[-10:],
            }
            return {
                "answer": answer,
                "meta": meta,
                "analysis_arm": analysis_arm,
                "graph": graph if isinstance(graph, nx.Graph) else None,
            }
        if analysis_arm == "pdb_visualizer" and answer:
            graph = state.get("graph")
            meta = {
                "analysis_arm": analysis_arm,
                "is_followup": bool(state.get("is_followup", False)),
                "route_rationale": state.get("route_rationale", ""),
                "genes": list(state.get("genes") or []),
                "pdb_visualization_result": _ensure_dict(state.get("pdb_visualization_result")),
                "memory_pdb_visualization_result": _ensure_dict(state.get("memory_pdb_visualization_result")),
                "tool_history": list(state.get("tool_history") or [])[-10:],
            }
            return {
                "answer": answer,
                "meta": meta,
                "analysis_arm": analysis_arm,
                "graph": graph if isinstance(graph, nx.Graph) else None,
            }
        if analysis_arm in {"memory_lookup", "state_lookup", "memory_slice"}:
            lookup_result = (
                state.get("memory_lookup_result")
                if analysis_arm == "memory_lookup"
                else (state.get("state_lookup_result") if analysis_arm == "state_lookup" else state.get("memory_slice_result"))
            )
            answer = synthesize_technical_response(
                user_query=str(state.get("query") or ""),
                analysis_arm=analysis_arm,
                seed_genes=list(state.get("genes") or []),
                srp_ids=list(state.get("srp_ids") or []),
                disease_name=str(state.get("disease_name") or state.get("memory_disease_name") or ""),
                deg_analysis=_ensure_dict(lookup_result),
                rwr_genes=list(state.get("rwr_genes") or []),
                graph=state.get("graph") if isinstance(state.get("graph"), nx.Graph) else nx.Graph(),
                enrichr=_ensure_dict(state.get("enrichr") or state.get("memory_enrichr")),
                literature_papers=list(state.get("openalex_papers") or []),
                ranked_literature_papers=list(state.get("ranked_openalex_papers") or []),
                literature_key_points=list(state.get("literature_key_points") or []),
                literature_references=list(state.get("literature_references") or []),
                literature_summary=str(state.get("literature_summary") or ""),
                memory_lookup_result=_ensure_dict(state.get("memory_lookup_result")),
                state_lookup_result=_ensure_dict(state.get("state_lookup_result")),
                memory_slice_result=_ensure_dict(state.get("memory_slice_result")),
            )
            graph = state.get("graph")
            meta = {
                "analysis_arm": analysis_arm,
                "is_followup": bool(state.get("is_followup", False)),
                "route_rationale": state.get("route_rationale", ""),
                "srp_ids": list(state.get("srp_ids") or []),
                "control_name": str(state.get("control_name") or ""),
                "test_name": str(state.get("test_name") or ""),
                "memory_control_name": str(state.get("memory_control_name") or ""),
                "memory_test_name": str(state.get("memory_test_name") or ""),
                "memory_deg_genes": list(state.get("memory_deg_genes") or []),
                "memory_upregulated_genes": list(state.get("memory_upregulated_genes") or []),
                "memory_downregulated_genes": list(state.get("memory_downregulated_genes") or []),
                "memory_deg_analysis": _ensure_dict(state.get("memory_deg_analysis")),
                "memory_deg_gene_records": list(state.get("memory_deg_gene_records") or []),
                "memory_srp_metadata_result": _ensure_dict(state.get("memory_srp_metadata_result")),
                "memory_disease_name": str(state.get("memory_disease_name") or ""),
                "memory_openalex_genes": list(state.get("memory_openalex_genes") or []),
                "memory_opentargets_results": list(state.get("memory_opentargets_results") or []),
                "memory_l1000cds2_result": _ensure_dict(state.get("memory_l1000cds2_result")),
                "memory_pubchem_result": _ensure_dict(state.get("memory_pubchem_result")),
                "disease_name": str(state.get("disease_name") or ""),
                "disease_gene": str(state.get("disease_gene") or ""),
                "memory_lookup_result": _ensure_dict(state.get("memory_lookup_result")),
                "state_lookup_result": _ensure_dict(state.get("state_lookup_result")),
                "memory_slice_result": _ensure_dict(state.get("memory_slice_result")),
                "openalex_papers": list(state.get("openalex_papers") or []),
                "ranked_openalex_papers": list(state.get("ranked_openalex_papers") or []),
                "openalex_genes": list(state.get("openalex_genes") or []),
                "literature_key_points": list(state.get("literature_key_points") or []),
                "literature_references": list(state.get("literature_references") or []),
                "literature_summary": str(state.get("literature_summary") or ""),
                "literature_source_status": _ensure_dict(state.get("literature_source_status")),
                "literature_query": str(state.get("literature_query") or ""),
                "primekg_result": _ensure_dict(state.get("primekg_result")),
                "opentargets_result": _ensure_dict(state.get("opentargets_result")),
                "l1000cds2_result": _ensure_dict(state.get("l1000cds2_result")),
                "pubchem_result": _ensure_dict(state.get("pubchem_result")),
                "deg_analysis": _ensure_dict(state.get("deg_analysis")),
                "srp_metadata_result": _ensure_dict(state.get("srp_metadata_result")),
                "deg_genes": list(state.get("deg_genes") or []),
                "upregulated_genes": list(state.get("upregulated_genes") or []),
                "downregulated_genes": list(state.get("downregulated_genes") or []),
                "deg_gene_records": list(state.get("deg_gene_records") or []),
                "genes": list(state.get("genes") or []),
                "rwr_seed_genes": list(state.get("rwr_seed_genes") or []),
                "network": _graph_summary(graph if isinstance(graph, nx.Graph) else None),
                "rwr_genes": list(state.get("rwr_genes") or []),
                "enrichr": _ensure_dict(state.get("enrichr")),
                "pyvis_html_path": str(state.get("pyvis_html_path") or ""),
                "kegg_pathway_path": str(state.get("kegg_pathway_path") or ""),
                "volcano_plot_path": str(state.get("volcano_plot_path") or ""),
                "visualization_result": _ensure_dict(state.get("visualization_result")),
                "tool_history": list(state.get("tool_history") or [])[-10:],
            }
            return {
                "answer": answer,
                "meta": meta,
                "analysis_arm": analysis_arm,
                "graph": graph if isinstance(graph, nx.Graph) else None,
            }
        if analysis_arm == "hypothesis":
            graph = state.get("graph")
            branch_answer = str(state.get("answer") or answer or "").strip()
            if not branch_answer:
                branch_answer = synthesize_technical_response(
                    user_query=str(state.get("query") or ""),
                    analysis_arm=analysis_arm,
                    seed_genes=list(state.get("genes") or []),
                    srp_ids=list(state.get("srp_ids") or []),
                    disease_name=str(state.get("disease_name") or state.get("memory_disease_name") or ""),
                    deg_analysis=_ensure_dict(state.get("hypothesis_result")),
                    rwr_genes=list(state.get("rwr_genes") or []),
                    graph=graph if isinstance(graph, nx.Graph) else nx.Graph(),
                    enrichr=_ensure_dict(state.get("enrichr") or state.get("memory_enrichr")),
                    literature_papers=list(state.get("openalex_papers") or []),
                    ranked_literature_papers=list(state.get("ranked_openalex_papers") or []),
                    literature_key_points=list(state.get("literature_key_points") or []),
                    literature_references=list(state.get("literature_references") or []),
                    literature_summary=str(state.get("literature_summary") or ""),
                    memory_lookup_result=_ensure_dict(state.get("memory_lookup_result")),
                    state_lookup_result=_ensure_dict(state.get("state_lookup_result")),
                    memory_slice_result=_ensure_dict(state.get("memory_slice_result")),
                )
            meta = {
                "analysis_arm": analysis_arm,
                "is_followup": bool(state.get("is_followup", False)),
                "route_rationale": state.get("route_rationale", ""),
                "memory_control_name": str(state.get("memory_control_name") or ""),
                "memory_test_name": str(state.get("memory_test_name") or ""),
                "memory_deg_genes": list(state.get("memory_deg_genes") or []),
                "memory_upregulated_genes": list(state.get("memory_upregulated_genes") or []),
                "memory_downregulated_genes": list(state.get("memory_downregulated_genes") or []),
                "memory_deg_analysis": _ensure_dict(state.get("memory_deg_analysis")),
                "memory_deg_gene_records": list(state.get("memory_deg_gene_records") or []),
                "memory_srp_metadata_result": _ensure_dict(state.get("memory_srp_metadata_result")),
                "memory_disease_name": str(state.get("memory_disease_name") or ""),
                "memory_openalex_genes": list(state.get("memory_openalex_genes") or []),
                "memory_l1000cds2_result": _ensure_dict(state.get("memory_l1000cds2_result")),
                "memory_pubchem_result": _ensure_dict(state.get("memory_pubchem_result")),
                "memory_hypothesis_result": _ensure_dict(state.get("memory_hypothesis_result")),
                "disease_name": str(state.get("disease_name") or ""),
                "openalex_papers": list(state.get("openalex_papers") or []),
                "ranked_openalex_papers": list(state.get("ranked_openalex_papers") or []),
                "literature_key_points": list(state.get("literature_key_points") or []),
                "literature_references": list(state.get("literature_references") or []),
                "literature_summary": str(state.get("literature_summary") or ""),
                "literature_source_status": _ensure_dict(state.get("literature_source_status")),
                "hypothesis_result": _ensure_dict(state.get("hypothesis_result")),
                "tool_history": list(state.get("tool_history") or [])[-10:],
            }
            return {
                "answer": branch_answer,
                "meta": meta,
                "analysis_arm": analysis_arm,
                "graph": graph if isinstance(graph, nx.Graph) else None,
            }
        if analysis_arm in {"research_literature", "literature"}:
            graph = state.get("graph")
            branch_answer = synthesize_technical_response(
                user_query=str(state.get("query") or ""),
                analysis_arm=analysis_arm,
                seed_genes=list(state.get("genes") or []),
                srp_ids=list(state.get("srp_ids") or []),
                disease_name=str(state.get("disease_name") or state.get("memory_disease_name") or ""),
                deg_analysis=_ensure_dict(state.get("deg_analysis")),
                rwr_genes=list(state.get("rwr_genes") or []),
                graph=graph if isinstance(graph, nx.Graph) else nx.Graph(),
                enrichr=_ensure_dict(state.get("enrichr") or state.get("memory_enrichr")),
                literature_papers=list(state.get("openalex_papers") or []),
                ranked_literature_papers=list(state.get("ranked_openalex_papers") or []),
                literature_key_points=list(state.get("literature_key_points") or []),
                literature_references=list(state.get("literature_references") or []),
                literature_summary=str(state.get("literature_summary") or state.get("answer") or answer or ""),
                memory_lookup_result=_ensure_dict(state.get("memory_lookup_result")),
                state_lookup_result=_ensure_dict(state.get("state_lookup_result")),
                memory_slice_result=_ensure_dict(state.get("memory_slice_result")),
            )
            meta = {
                "analysis_arm": analysis_arm,
                "is_followup": bool(state.get("is_followup", False)),
                "route_rationale": state.get("route_rationale", ""),
                "srp_ids": list(state.get("srp_ids") or []),
                "control_name": str(state.get("control_name") or ""),
                "test_name": str(state.get("test_name") or ""),
                "memory_control_name": str(state.get("memory_control_name") or ""),
                "memory_test_name": str(state.get("memory_test_name") or ""),
                "memory_deg_genes": list(state.get("memory_deg_genes") or []),
                "memory_upregulated_genes": list(state.get("memory_upregulated_genes") or []),
                "memory_downregulated_genes": list(state.get("memory_downregulated_genes") or []),
                "memory_deg_analysis": _ensure_dict(state.get("memory_deg_analysis")),
                "memory_deg_gene_records": list(state.get("memory_deg_gene_records") or []),
                "memory_disease_name": str(state.get("memory_disease_name") or ""),
                "memory_openalex_genes": list(state.get("memory_openalex_genes") or []),
                "memory_opentargets_results": list(state.get("memory_opentargets_results") or []),
                "memory_l1000cds2_result": _ensure_dict(state.get("memory_l1000cds2_result")),
                "memory_pubchem_result": _ensure_dict(state.get("memory_pubchem_result")),
                "memory_hypothesis_result": _ensure_dict(state.get("memory_hypothesis_result")),
                "disease_name": str(state.get("disease_name") or ""),
                "disease_gene": str(state.get("disease_gene") or ""),
                "state_lookup_result": _ensure_dict(state.get("state_lookup_result")),
                "memory_slice_result": _ensure_dict(state.get("memory_slice_result")),
                "openalex_papers": list(state.get("openalex_papers") or []),
                "ranked_openalex_papers": list(state.get("ranked_openalex_papers") or []),
                "openalex_genes": list(state.get("openalex_genes") or []),
                "literature_key_points": list(state.get("literature_key_points") or []),
                "literature_references": list(state.get("literature_references") or []),
                "literature_summary": str(state.get("literature_summary") or ""),
                "literature_source_status": _ensure_dict(state.get("literature_source_status")),
                "literature_query": str(state.get("literature_query") or ""),
                "primekg_result": _ensure_dict(state.get("primekg_result")),
                "opentargets_result": _ensure_dict(state.get("opentargets_result")),
                "l1000cds2_result": _ensure_dict(state.get("l1000cds2_result")),
                "pubchem_result": _ensure_dict(state.get("pubchem_result")),
                "hypothesis_result": _ensure_dict(state.get("hypothesis_result")),
                "memory_lookup_result": _ensure_dict(state.get("memory_lookup_result")),
                "deg_analysis": _ensure_dict(state.get("deg_analysis")),
                "srp_metadata_result": _ensure_dict(state.get("srp_metadata_result")),
                "deg_genes": list(state.get("deg_genes") or []),
                "upregulated_genes": list(state.get("upregulated_genes") or []),
                "downregulated_genes": list(state.get("downregulated_genes") or []),
                "deg_gene_records": list(state.get("deg_gene_records") or []),
                "genes": list(state.get("genes") or []),
                "rwr_seed_genes": list(state.get("rwr_seed_genes") or []),
                "network": _graph_summary(graph if isinstance(graph, nx.Graph) else None),
                "rwr_genes": list(state.get("rwr_genes") or []),
                "enrichr": _ensure_dict(state.get("enrichr")),
                "pyvis_html_path": str(state.get("pyvis_html_path") or ""),
                "kegg_pathway_path": str(state.get("kegg_pathway_path") or ""),
                "volcano_plot_path": str(state.get("volcano_plot_path") or ""),
                "visualization_result": _ensure_dict(state.get("visualization_result")),
                "tool_history": list(state.get("tool_history") or [])[-10:],
            }
            return {
                "answer": branch_answer,
                "meta": meta,
                "analysis_arm": analysis_arm,
                "graph": graph if isinstance(graph, nx.Graph) else None,
            }
        specialist_payload = _ensure_dict(state.get("deg_analysis"))
        if analysis_arm == "primekg":
            specialist_payload = _ensure_dict(state.get("primekg_result"))
        elif analysis_arm == "l1000cds2":
            specialist_payload = _ensure_dict(state.get("l1000cds2_result") or state.get("memory_l1000cds2_result"))
        elif analysis_arm == "pubchem":
            specialist_payload = _ensure_dict(state.get("pubchem_result") or state.get("memory_pubchem_result"))
        elif analysis_arm == "opentargets":
            specialist_payload = _ensure_dict(state.get("opentargets_result"))
        elif analysis_arm == "druggability":
            specialist_payload = _ensure_dict(state.get("druggability_result") or state.get("memory_druggability_result"))
        elif analysis_arm == "pdb_visualizer":
            specialist_payload = _ensure_dict(state.get("pdb_visualization_result") or state.get("memory_pdb_visualization_result"))
        answer = synthesize_technical_response(
            user_query=str(state.get("query") or ""),
            analysis_arm=analysis_arm,
            seed_genes=list(state.get("genes") or []),
            srp_ids=list(state.get("srp_ids") or []),
            disease_name=str(state.get("disease_name") or ""),
            deg_analysis=specialist_payload,
            rwr_genes=list(state.get("rwr_genes") or []),
            graph=state.get("graph") if isinstance(state.get("graph"), nx.Graph) else nx.Graph(),
            enrichr=_ensure_dict(state.get("enrichr") or state.get("memory_enrichr")),
            literature_papers=list(state.get("openalex_papers") or []),
            ranked_literature_papers=list(state.get("ranked_openalex_papers") or []),
            literature_key_points=list(state.get("literature_key_points") or []),
            literature_references=list(state.get("literature_references") or []),
            literature_summary=str(state.get("literature_summary") or ""),
            memory_lookup_result=_ensure_dict(state.get("memory_lookup_result")),
            state_lookup_result=_ensure_dict(state.get("state_lookup_result")),
            memory_slice_result=_ensure_dict(state.get("memory_slice_result")),
        )

    analysis_arm = _infer_analysis_arm(state)
    graph = state.get("graph")
    if not answer:
        specialist_payload = _ensure_dict(state.get("deg_analysis"))
        if analysis_arm == "primekg":
            specialist_payload = _ensure_dict(state.get("primekg_result"))
        elif analysis_arm == "l1000cds2":
            specialist_payload = _ensure_dict(state.get("l1000cds2_result") or state.get("memory_l1000cds2_result"))
        elif analysis_arm == "pubchem":
            specialist_payload = _ensure_dict(state.get("pubchem_result") or state.get("memory_pubchem_result"))
        elif analysis_arm == "opentargets":
            specialist_payload = _ensure_dict(state.get("opentargets_result"))
        elif analysis_arm == "druggability":
            specialist_payload = _ensure_dict(state.get("druggability_result") or state.get("memory_druggability_result"))
        elif analysis_arm == "pdb_visualizer":
            specialist_payload = _ensure_dict(state.get("pdb_visualization_result") or state.get("memory_pdb_visualization_result"))
        elif analysis_arm == "memory_lookup":
            specialist_payload = _ensure_dict(state.get("memory_lookup_result"))
        elif analysis_arm == "state_lookup":
            specialist_payload = _ensure_dict(state.get("state_lookup_result"))
        elif analysis_arm == "memory_slice":
            specialist_payload = _ensure_dict(state.get("memory_slice_result"))
        answer = synthesize_technical_response(
            user_query=str(state.get("query") or ""),
            analysis_arm=analysis_arm,
            seed_genes=list(state.get("genes") or []),
            srp_ids=list(state.get("srp_ids") or []),
            disease_name=str(state.get("disease_name") or state.get("memory_disease_name") or ""),
            deg_analysis=specialist_payload,
            rwr_genes=list(state.get("rwr_genes") or []),
            graph=graph if isinstance(graph, nx.Graph) else nx.Graph(),
            enrichr=_ensure_dict(state.get("enrichr") or state.get("memory_enrichr")),
            literature_papers=list(state.get("openalex_papers") or []),
            ranked_literature_papers=list(state.get("ranked_openalex_papers") or []),
            literature_key_points=list(state.get("literature_key_points") or []),
            literature_references=list(state.get("literature_references") or []),
            literature_summary=str(state.get("literature_summary") or ""),
            memory_lookup_result=_ensure_dict(state.get("memory_lookup_result")),
            state_lookup_result=_ensure_dict(state.get("state_lookup_result")),
            memory_slice_result=_ensure_dict(state.get("memory_slice_result")),
        )
    meta = {
        "analysis_arm": analysis_arm,
        "is_followup": bool(state.get("is_followup", False)),
        "route_rationale": state.get("route_rationale", ""),
        "srp_ids": list(state.get("srp_ids") or []),
        "control_name": str(state.get("control_name") or ""),
        "test_name": str(state.get("test_name") or ""),
        "memory_control_name": str(state.get("memory_control_name") or ""),
        "memory_test_name": str(state.get("memory_test_name") or ""),
        "memory_deg_genes": list(state.get("memory_deg_genes") or []),
        "memory_upregulated_genes": list(state.get("memory_upregulated_genes") or []),
        "memory_downregulated_genes": list(state.get("memory_downregulated_genes") or []),
        "memory_deg_analysis": _ensure_dict(state.get("memory_deg_analysis")),
        "memory_deg_gene_records": list(state.get("memory_deg_gene_records") or []),
        "memory_srp_metadata_result": _ensure_dict(state.get("memory_srp_metadata_result")),
        "memory_disease_name": str(state.get("memory_disease_name") or ""),
        "memory_openalex_genes": list(state.get("memory_openalex_genes") or []),
        "memory_opentargets_results": list(state.get("memory_opentargets_results") or []),
        "memory_l1000cds2_result": _ensure_dict(state.get("memory_l1000cds2_result")),
        "memory_pubchem_result": _ensure_dict(state.get("memory_pubchem_result")),
        "memory_hypothesis_result": _ensure_dict(state.get("memory_hypothesis_result")),
        "disease_name": str(state.get("disease_name") or ""),
        "disease_gene": str(state.get("disease_gene") or ""),
        "state_lookup_result": _ensure_dict(state.get("state_lookup_result")),
        "memory_slice_result": _ensure_dict(state.get("memory_slice_result")),
        "openalex_papers": list(state.get("openalex_papers") or []),
        "ranked_openalex_papers": list(state.get("ranked_openalex_papers") or []),
        "openalex_genes": list(state.get("openalex_genes") or []),
        "literature_key_points": list(state.get("literature_key_points") or []),
        "literature_references": list(state.get("literature_references") or []),
        "literature_summary": str(state.get("literature_summary") or ""),
        "literature_source_status": _ensure_dict(state.get("literature_source_status")),
        "literature_query": str(state.get("literature_query") or ""),
        "primekg_result": _ensure_dict(state.get("primekg_result")),
        "opentargets_result": _ensure_dict(state.get("opentargets_result")),
        "l1000cds2_result": _ensure_dict(state.get("l1000cds2_result")),
        "pubchem_result": _ensure_dict(state.get("pubchem_result")),
        "hypothesis_result": _ensure_dict(state.get("hypothesis_result")),
        "memory_lookup_result": _ensure_dict(state.get("memory_lookup_result")),
        "deg_analysis": _ensure_dict(state.get("deg_analysis")),
        "srp_metadata_result": _ensure_dict(state.get("srp_metadata_result")),
        "deg_genes": list(state.get("deg_genes") or []),
        "upregulated_genes": list(state.get("upregulated_genes") or []),
        "downregulated_genes": list(state.get("downregulated_genes") or []),
        "deg_gene_records": list(state.get("deg_gene_records") or []),
        "genes": list(state.get("genes") or []),
        "rwr_seed_genes": list(state.get("rwr_seed_genes") or []),
        "network": _graph_summary(graph if isinstance(graph, nx.Graph) else None),
        "rwr_genes": list(state.get("rwr_genes") or []),
        "enrichr": _ensure_dict(state.get("enrichr")),
        "pyvis_html_path": str(state.get("pyvis_html_path") or ""),
        "kegg_pathway_path": str(state.get("kegg_pathway_path") or ""),
        "volcano_plot_path": str(state.get("volcano_plot_path") or ""),
        "visualization_result": _ensure_dict(state.get("visualization_result")),
        "tool_history": list(state.get("tool_history") or [])[-10:],
    }
    return {
        "answer": answer,
        "meta": meta,
        "analysis_arm": analysis_arm,
        "graph": graph if isinstance(graph, nx.Graph) else None,
    }


def build_app():
    graph = StateGraph(AgentState)

    graph.add_node("prepare_context", _prepare_context)
    graph.add_node("agent", _agent)
    graph.add_node("finalize", _finalize)

    for tool_name in TOOL_EXECUTORS:
        graph.add_node(tool_name, _specialist_node(tool_name))

    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {tool_name: tool_name for tool_name in TOOL_EXECUTORS} | {"finalize": "finalize"},
    )
    for tool_name in TOOL_EXECUTORS:
        graph.add_conditional_edges(
            tool_name,
            _route_after_tool,
            {"agent": "agent", "finalize": "finalize"},
        )
    graph.add_edge("finalize", END)

    return graph.compile()


def _tool_arg_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# New specialist checklist:
# 1. Add a tool schema here so the orchestrator can call it.
# 2. Add its runtime handler in `_specialist_node(...)` and include any state updates.
# 3. Register the tool name in `TOOL_EXECUTORS` so routing can reach it.
# 4. Update `TOOL_USE_INSTRUCTIONS`, prompt guidance, and `_finalize(...)` metadata if the tool introduces new memory/results.
# 5. Extend `AgentState` when the tool stores new fields across turns.
TOOL_SCHEMAS = [
    tool(
        "extract_genes_from_text",
        description="Extract candidate gene symbols from free text only. Parameters: `text`, optional `mode` (`strict` preferred) and optional `whitelist`. Does not validate disease associations or run analysis.",
        return_direct=False,
    )(lambda text, mode="strict", whitelist=None: {"genes": extract_genes_from_text(text, whitelist={str(value).strip().upper() for value in whitelist} if isinstance(whitelist, list) else None, mode=mode)}),
    tool(
        "extract_srp_ids_from_text",
        description="Extract SRP accession identifiers from text only. Parameters: `text`. It does not resolve GEO/GSE IDs, download DEE2 data, or run DEG analysis.",
        return_direct=False,
    )(lambda text: {"srp_ids": extract_srp_ids_from_text(text)}),
    tool(
        "identify_disease_from_query",
        description="Infer a disease name from the user query. Parameters: `query`. It does not validate ontology IDs, diagnose disease, or establish gene-disease associations.",
        return_direct=False,
    )(lambda query: identify_disease_from_query(query)),
    tool(
        "literature",
        description="Retrieve literature evidence through the configured OpenAlex/PubMed/Google Scholar pipeline. Parameters: `disease_name`, `genes`, `top_n`, `text`. Use when the user asks to find/search/check evidence for a statement, claim, mechanism, biomarker, gene-disease link, or explicitly asks for PubMed/OpenAlex/Google Scholar/paper searches. Cannot access paywalled full text, guarantee systematic-review completeness, provide clinical advice, or query unsupported literature sources.",
        return_direct=False,
    )(lambda disease_name="", genes=None, top_n=20, text=None: {"disease_name": disease_name, "genes": list(genes or []), "top_n": int(top_n), "text": text}),
    tool(
        "research_literature",
        description="Generate a literature-style answer with references from model knowledge. Parameters: `user_query`, optional `disease_name`, `genes`, `top_n`. Use for broad research, investigate, review, explain, summarize, overview, or what-is-known-about requests that do not ask to find/search/check evidence for a specific statement and do not explicitly request PubMed/OpenAlex/Google Scholar/paper search. Does not perform live retrieval; references are best-effort and should not be described as newly searched or verified.",
        return_direct=False,
    )(lambda user_query, disease_name="", genes=None, top_n=20: {"user_query": user_query, "disease_name": disease_name, "genes": list(genes or []), "top_n": int(top_n)}),
    tool(
        "deg_analysis",
        description="""
        Run the only supported differential expression workflow: DEE2 SRP datasets analyzed with DESeq2.

        Args:
            srp_ids: List of DEE2/SRA project accessions such as
                    ["SRP277202"]. Required.
            control_name: Control cohort label such as
                    "Healthy lung tissue". Required.
            test_name: Test cohort label such as
                    "COPD lung tissue". Required.
            log2fold: Optional absolute log2 fold-change threshold.
            padj: Optional adjusted p-value/FDR threshold.
            text: Optional text containing control, test, and SRP IDs.

        Returns:
            Differentially expressed genes and statistics.

        Cannot run EdgeR, limma/Lemma, voom, NOISeq, custom count matrices,
        uploaded files, GEO-only IDs without SRP IDs, single-cell analysis,
        proteomics, custom design formulas, or batch correction.
        """,
        return_direct=False,
    )(
        lambda srp_ids=None, control_name=None, test_name=None, log2fold=1.0, padj=0.05, text=None: {
            "srp_ids": srp_ids,
            "control_name": control_name,
            "test_name": test_name,
            "log2fold": log2fold,
            "padj": padj,
            "text": text,
        }
    ),
    tool(
        "srp_metadata",
        description="""
        Retrieve DEE2/SRA metadata for SRP accessions before DEG analysis.

        Args:
            srp_ids: List of SRP accessions such as ["SRP277202"].
            text: Optional full user request containing SRP IDs.
            species: DEE2 species key; defaults to hsapiens.
            max_dee2_rows: Maximum matching DEE2 rows to keep.
            max_biosamples: Maximum BioSample records to inspect.

        Returns:
            DEE2 descriptions and metadata values under treatment, sample_name,
            and disease so the user can choose exact control/test labels.

        Does not run DEG analysis or choose the contrast automatically.
        """,
        return_direct=False,
    )(
        lambda srp_ids=None, text=None, species="hsapiens", max_dee2_rows=5000, max_biosamples=80: {
            "srp_ids": _tool_arg_list(srp_ids),
            "text": text,
            "species": species,
            "max_dee2_rows": max_dee2_rows,
            "max_biosamples": max_biosamples,
        }
    ),
    tool(
        "rwr_analysis",
        description="Perform Random Walk with Restart target prioritization over the configured local STRING network. Parameters: `genes`, optional `top_k`/`top_n`, `restart_prob`, STRING file paths/score settings. Inputs must be seed genes from user text or stored DEG/literature/memory results. Cannot use BioGRID, IntAct, tissue-specific or directed causal networks, or live-downloaded networks.",
        return_direct=False,
    )(lambda genes, info_path=SETTINGS.string_info_path, links_path=SETTINGS.string_links_path, required_score=SETTINGS.string_required_score, mode=SETTINGS.string_local_mode: {
        "graph": build_weighted_graph_from_string_files(
            genes=list(genes or []),
            info_path=info_path,
            links_path=links_path,
            required_score=int(required_score),
            mode=mode,
        )
    }),
    tool(
        "top_rwr_genes",
        description="Rank genes using Random Walk with Restart on the current in-memory STRING graph. Parameters: `seed_genes`, optional `top_k`, `restart_prob`. Requires an existing graph; does not build or fetch networks itself.",
        return_direct=False,
    )(lambda seed_genes, top_k=30, restart_prob=0.5: {"rwr_genes": seed_genes, "top_k": top_k, "restart_prob": restart_prob}),
    tool(
        "pathway",
        description="Run over-representation enrichment with Enrichr/gget using configured Reactome, KEGG Human, and GO libraries. Parameters: `genes`, `direction` (`up`, `down`, `all`), `gene_limit`, `term_limit`, optional `background_genes`, `text`. Prefer stored DEG genes for DEG follow-ups. Cannot run GSEA, MSigDB/custom GMT, clusterProfiler/ReactomePA, non-human KEGG unless implemented, or unsupported enrichment algorithms.",
        return_direct=False,
    )(lambda genes=None, direction="all", gene_limit=None, term_limit=10, background_genes=None, text=None: {
        "genes": list(genes or []),
        "direction": direction,
        "gene_limit": gene_limit,
        "term_limit": int(term_limit),
        "background_genes": list(background_genes or []),
        "text": text,
    }),
    tool(
        "visualize",
        description="Create only supported visualizations. Parameters include `visualization_type` (`network`, `kegg`, or `volcano`), `genes`, `top_n`, `direction`, `output_path`, `kegg_rank`, `species`, `select_top_degree`, `pvalue_threshold`, `log2fc_threshold`, `pathway_term`, `text`. `network` uses STRING/PyVis, `kegg` uses KEGG/gget, and `volcano` uses stored DEG rows. Cannot create heatmaps, PCA, UMAP, boxplots, survival plots, circos plots, dashboards, or arbitrary custom figures.",
        return_direct=False,
    )(
        lambda visualization_type, genes=None, top_n=None, direction=None, output_path=None, kegg_rank=1, species="human", select_top_degree=300, pvalue_threshold=0.05, log2fc_threshold=1.0, pathway_term=None, text=None: {
            "visualization_type": visualization_type,
            "genes": list(genes or []),
            "top_n": top_n,
            "direction": direction,
            "output_path": output_path,
            "kegg_rank": kegg_rank,
            "species": species,
            "select_top_degree": select_top_degree,
            "pvalue_threshold": pvalue_threshold,
            "log2fc_threshold": log2fc_threshold,
            "pathway_term": pathway_term,
            "text": text,
        }
    ),
    tool(
        "hypothesis",
        description="Generate plausible biomedical hypotheses and conceptual validation experiment ideas from conversation history and stored memory only. Parameters: `hypothesis_goal`, `genes`, `disease_name`, `hypothesis_count`, `text`. Can suggest experiment designs, readouts, controls, expected observations, interpretation, caveats, rationale, and key assumptions. Cannot validate hypotheses against external sources, search literature, cite references, assess novelty, retrieve new evidence, or provide step-by-step wet-lab protocols.",
        return_direct=False,
    )(
        lambda hypothesis_goal=None, validation_goal=None, genes=None, disease_name="", hypothesis_count=3, text=None: {
            "hypothesis_goal": hypothesis_goal or validation_goal or text or "",
            "genes": list(genes or []),
            "disease_name": disease_name,
            "hypothesis_count": hypothesis_count,
            "text": text,
        }
    ),
    tool(
        "memory_lookup",
        description="Answer lookup/matching questions from current chat memory only. Parameters: `pathway_term`, `direction`, `top_n`, `text`. Use for stored pathway/GO overlap genes, DEG membership, and intersections. Cannot run new analysis, search external data, or infer unstored results.",
        return_direct=False,
    )(
        lambda pathway_term=None, direction=None, top_n=None, text=None: {
            "pathway_term": pathway_term,
            "direction": direction,
            "top_n": top_n,
            "text": text,
        }
    ),
    tool(
        "state_lookup",
        description="Inspect literal agent state fields, values, and counts. Parameters: `fields`, `mode`, `max_items`, `text`. Use when the user asks what is stored or wants raw values/lengths. Cannot interpret biology or create missing results.",
        return_direct=False,
    )(
        lambda fields=None, mode="both", max_items=25, text=None: {
            "fields": list(fields or []),
            "mode": mode,
            "max_items": max_items,
            "text": text,
        }
    ),
    tool(
        "memory_slice",
        description="Select top N and/or bottom N items from a stored list-like state field and make that slice reusable. Parameters: `fields`, `top_n`, `bottom_n`, `text`. Uses existing list order only; cannot statistically rank or sort unless the stored list already encodes that order.",
        return_direct=False,
    )(
        lambda fields=None, top_n=None, bottom_n=None, text=None: {
            "fields": list(fields or []),
            "top_n": top_n,
            "bottom_n": bottom_n,
            "text": text,
        }
    ),
    tool(
        "primekg_query",
        description="""
        Query the configured local PrimeKG/Neo4j graph using natural language.

        Use for:
        - disease to gene relationships
        - gene to pathway relationships
        - drug to target relationships
        - disease to drug relationships
        - phenotype relationships
        - indirect graph questions that require 2-hop or 3-hop joins
        - "what connects X and Y" style graph exploration

        Input:
            question: Natural language question.

        Returns:
            Answer plus generated Cypher query and raw graph results.

        Constraints:
            Read-only PrimeKG relationships only. Cannot answer from live web
            knowledge, write to Neo4j, change schema, or invent labels and
            relationship types outside the configured PrimeKG schema.
        """,
        return_direct=False,
    )(
        lambda question: {"question": question}
    ),
    tool(
        "opentargets_association",
        description="Query OpenTargets for gene-disease associations or gene-linked drugs. Parameters: `gene`, `genes`, `disease`/`disease_name`. Genes are standardized to Ensembl IDs with MyGene. Cannot run pathway enrichment, DEG analysis, PubChem chemistry lookup, PrimeKG paths, or full clinical trial interpretation.",
        return_direct=False,
    )(lambda gene=None, genes=None, disease=None, disease_name=None: {"gene": gene or "", "genes": list(genes or []), "disease": disease or disease_name or ""}),
    tool(
        "l1000cds2_query",
        description="Query L1000CDS2 for small molecules using separate up-regulated and down-regulated gene lists. Parameters: `up_genes`, `down_genes`, optional `cell_lines`, `aggravate` for mimic mode, `combination`, `share`, `db_version`, `gene_limit`, `result_limit`, `text`. Default is reversal mode. Cannot query CMap APIs outside L1000CDS2, infer mechanisms without returned signatures, or run from a single undirected gene list unless a split is explicit or stored DEG directions exist.",
        return_direct=False,
    )(
        lambda up_genes=None, down_genes=None, cell_lines=None, aggravate=None, combination=False, share=False, db_version="latest", gene_limit=500, result_limit=20, text=None: {
            "up_genes": list(up_genes or []),
            "down_genes": list(down_genes or []),
            "cell_lines": list(cell_lines or []),
            "aggravate": aggravate,
            "combination": combination,
            "share": share,
            "db_version": db_version,
            "gene_limit": gene_limit,
            "result_limit": result_limit,
            "text": text,
        }
    ),
    tool(
        "pubchem_drug_lookup",
        description="Query PubChem for a drug/compound by name, `pert_desc`, or BRD-like `pert_id`. Parameters: `drug_name`, `pert_id`, `text`. Retrieves compound records, properties, synonyms, descriptions, and annotations; genes/pathways/diseases may be mentioned only when supported by PubChem text. Cannot run L1000 signatures, target validation, clinical efficacy analysis, or non-compound biomedical graph queries.",
        return_direct=False,
    )(
        lambda drug_name=None, pert_id=None, text=None: {
            "drug_name": drug_name or "",
            "pert_id": pert_id or "",
            "text": text,
        }
    ),
    tool(
        "druggability",
        description="""
        Run structure-backed druggability and binding-pocket analysis for one gene.

        Args:
            gene: Gene symbol such as EGFR, TP53, or BRCA1. Required.
            organism_id: NCBI taxonomy id for UniProt resolution; defaults to
                9606 for human.
            chain: Optional protein chain to pass to DoGSite; empty means all.
            ligand: Optional ligand identifier for DoGSite; empty means none.
            top_n: Number of top pockets to return and download files for.
            output_dir: Optional local output directory.
            dogsite_timeout_seconds: Maximum time to wait for DoGSite.
            poll_interval_seconds: Seconds between DoGSite polling requests.
            pdbfixer_ph: pH used when adding hydrogens with PDBFixer.

        Workflow:
            Resolve gene to UniProt, download the best available RCSB PDB,
            otherwise download AlphaFold coordinates, sanitize/add hydrogens
            with PDBFixer when available, upload to ProteinsPlus, submit
            DoGSiteScorer, wait for the result table, and download top pocket
            residue/map files.
        """,
        return_direct=False,
    )(
        lambda gene, organism_id=9606, chain="", ligand="", top_n=3, output_dir=None, dogsite_timeout_seconds=900, poll_interval_seconds=15, pdbfixer_ph=7.0: {
            "gene": gene,
            "organism_id": organism_id,
            "chain": chain,
            "ligand": ligand,
            "top_n": top_n,
            "output_dir": output_dir,
            "dogsite_timeout_seconds": dogsite_timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "pdbfixer_ph": pdbfixer_ph,
        }
    ),
    tool(
        "pdb_visualizer",
        description="""
        Fetch and visualize one protein structure without running pocket scoring.

        Args:
            gene: Gene symbol such as CRISPLD2, EGFR, or TP53.
            protein: Optional protein label when the user names a protein.
            uniprot_id: Optional UniProt accession such as Q9H0B8.
            pdb_id: Optional direct RCSB PDB ID such as 1A2B.
            organism_id: NCBI taxonomy id for UniProt resolution; defaults to
                9606 for human.
            output_dir: Optional local output directory.

        Workflow:
            Resolve gene/protein to UniProt when needed, prefer an RCSB PDB,
            otherwise download the latest AlphaFold PDB through the AlphaFold
            API, save the PDB, and generate an interactive 3Dmol HTML viewer.
        """,
        return_direct=False,
    )(
        lambda gene="", protein="", uniprot_id="", pdb_id="", organism_id=9606, output_dir=None: {
            "gene": gene,
            "protein": protein,
            "uniprot_id": uniprot_id,
            "pdb_id": pdb_id,
            "organism_id": organism_id,
            "output_dir": output_dir,
        }
    ),
    tool(
        "synthesize_technical_response",
        description="Write the final user-facing technical summary from available structured analysis state. Use after the requested specialist work is complete. It cannot create new evidence, run tools, or fill missing fields by guessing.",
        return_direct=False,
    )(lambda user_query=None, analysis_arm="disease", seed_genes=None, srp_ids=None, disease_name="", deg_analysis=None, rwr_genes=None, graph=None, enrichr=None: synthesize_technical_response(
        user_query=user_query,
        analysis_arm=analysis_arm,
        seed_genes=list(seed_genes or []),
        srp_ids=list(srp_ids or []),
        disease_name=disease_name,
        deg_analysis=_ensure_dict(deg_analysis),
        rwr_genes=list(rwr_genes or []),
        graph=graph if isinstance(graph, nx.Graph) else nx.Graph(),
        enrichr=_ensure_dict(enrichr),
    )),
]


TOOL_EXECUTORS: dict[str, Callable[[AgentState, dict[str, Any]], dict[str, Any]]] = {
    "deg_analysis": lambda state, args: {},
    "srp_metadata": lambda state, args: {},
    "pathway": lambda state, args: {},
    "rwr_analysis": lambda state, args: {},
    "literature": lambda state, args: {},
    "research_literature": lambda state, args: {},
    "identify_disease_from_query": lambda state, args: {},
    "visualize": lambda state, args: {},
    "hypothesis": lambda state, args: {},
    "memory_lookup": lambda state, args: {},
    "state_lookup": lambda state, args: {},
    "memory_slice": lambda state, args: {},
    "primekg_query": lambda state, args: {},
    "opentargets_association": lambda state, args: {},
    "l1000cds2_query": lambda state, args: {},
    "pubchem_drug_lookup": lambda state, args: {},
    "druggability": lambda state, args: {},
    "pdb_visualizer": lambda state, args: {},
    "synthesize_technical_response": lambda state, args: {},
}
