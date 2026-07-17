from __future__ import annotations

import csv
import io
from typing import Any

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage
try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def find_dotenv(*args, **kwargs) -> str:
        return ""

    def load_dotenv(*args, **kwargs) -> bool:
        return False

from gea_agent.config import SETTINGS

try:
    from gea_agent.agent.graph import build_app
    APP_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - runtime guard for partial environments
    build_app = None
    APP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


load_dotenv(find_dotenv(usecwd=True), override=False)

st.set_page_config(page_title="GEA Agent", layout="wide")
st.title("GEA Agent")
st.caption("Gene expression analysis assistant with memory-aware tool routing.")

MEMORY_DEFAULTS: dict[str, Any] = {
    "messages": [],
    "last_graph": None,
    "last_meta": None,
    "last_error": "",
    "memory_deg_genes": [],
    "memory_upregulated_genes": [],
    "memory_downregulated_genes": [],
    "memory_deg_analysis": {},
    "memory_deg_gene_records": [],
    "memory_control_name": "",
    "memory_test_name": "",
    "memory_enrichr": {},
    "memory_rwr_seed_genes": [],
    "memory_rwr_genes": [],
    "memory_disease_name": "",
    "memory_openalex_genes": [],
    "memory_opentargets_results": [],
    "memory_l1000cds2_result": {},
    "memory_pubchem_result": {},
    "memory_hypothesis_result": {},
    "memory_slice_result": {},
}


@st.cache_resource(show_spinner=False)
def _get_compiled_app():
    if build_app is None:
        raise RuntimeError(APP_IMPORT_ERROR or "The agent graph could not be imported.")
    app_graph = build_app()
    if SETTINGS.streamlit_draw_graph:
        try:
            app_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        except Exception:
            pass
    return app_graph


def _init_session_state() -> None:
    for key, value in MEMORY_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value


def _reset_memory(*, keep_messages: bool) -> None:
    messages = list(st.session_state.messages) if keep_messages else []
    for key, value in MEMORY_DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value
    st.session_state.messages = messages


def _build_memory_summary() -> str:
    parts: list[str] = []

    if st.session_state.memory_deg_genes:
        parts.append(f"Stored DEG genes available: {len(st.session_state.memory_deg_genes)}.")
    if st.session_state.memory_deg_gene_records:
        parts.append(f"Stored DEG gene records available: {len(st.session_state.memory_deg_gene_records)}.")
    if st.session_state.memory_control_name or st.session_state.memory_test_name:
        parts.append(
            f"Stored DEG comparison: control={st.session_state.memory_control_name or 'NA'}, "
            f"test={st.session_state.memory_test_name or 'NA'}."
        )
    if st.session_state.memory_enrichr:
        libraries = st.session_state.memory_enrichr.get("libraries", {})
        if isinstance(libraries, dict) and libraries:
            parts.append(f"Stored pathway results available: {len(libraries)} libraries.")
    if st.session_state.memory_rwr_seed_genes:
        parts.append(f"Stored RWR seed genes available: {len(st.session_state.memory_rwr_seed_genes)}.")
    if st.session_state.memory_rwr_genes:
        parts.append(f"Stored RWR targets available: {len(st.session_state.memory_rwr_genes)}.")
    if st.session_state.memory_disease_name:
        parts.append(f"Last disease context: {st.session_state.memory_disease_name}.")
    if st.session_state.memory_openalex_genes:
        parts.append(f"Stored disease literature genes available: {len(st.session_state.memory_openalex_genes)}.")
    if st.session_state.memory_opentargets_results:
        parts.append(f"Stored OpenTargets results available: {len(st.session_state.memory_opentargets_results)}.")
    if st.session_state.memory_l1000cds2_result:
        top_drugs = st.session_state.memory_l1000cds2_result.get("top_drugs", [])
        if isinstance(top_drugs, list):
            parts.append(f"Stored L1000CDS2 hits available: {len(top_drugs)}.")
    if st.session_state.memory_pubchem_result:
        cid = st.session_state.memory_pubchem_result.get("cid")
        if cid:
            parts.append(f"Stored PubChem result available for CID {cid}.")
    if st.session_state.memory_hypothesis_result:
        hypotheses = st.session_state.memory_hypothesis_result.get("hypotheses", [])
        if isinstance(hypotheses, list) and hypotheses:
            parts.append(f"Stored experimental hypotheses available: {len(hypotheses)}.")
    if st.session_state.memory_slice_result:
        field = st.session_state.memory_slice_result.get("field")
        selected = st.session_state.memory_slice_result.get("selected_values", [])
        if field and isinstance(selected, list):
            parts.append(f"Stored memory slice available from {field}: {len(selected)} selected items.")

    recent_messages = st.session_state.messages[-4:]
    if recent_messages:
        transcript = []
        for message in recent_messages:
            role = message.get("role", "")
            content = " ".join(str(message.get("content", "")).split())
            if content:
                transcript.append(f"{role}: {content[:160]}")
        if transcript:
            parts.append("Recent turns: " + " | ".join(transcript))

    return "\n".join(parts)


def _invoke_state_from_session(prompt: str) -> dict[str, Any]:
    history_messages = []
    for entry in st.session_state.messages:
        role = str(entry.get("role") or "").strip().lower()
        content = str(entry.get("content") or "")
        if not content.strip():
            continue
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            history_messages.append(AIMessage(content=content))

    invoke_state: dict[str, Any] = {
        "query": prompt,
        "messages": history_messages,
        "memory_summary": _build_memory_summary(),
    }
    for key in MEMORY_DEFAULTS:
        if key.startswith("memory_") and st.session_state.get(key):
            invoke_state[key] = st.session_state.get(key)
    if st.session_state.get("memory_slice_result"):
        invoke_state["memory_slice_result"] = st.session_state.get("memory_slice_result")
    return invoke_state


def _update_memory_from_meta(meta: dict[str, Any]) -> None:
    if meta.get("analysis_arm") == "srp":
        st.session_state.memory_deg_genes = meta.get("deg_genes", []) if isinstance(meta.get("deg_genes", []), list) else []
        st.session_state.memory_upregulated_genes = meta.get("upregulated_genes", []) if isinstance(meta.get("upregulated_genes", []), list) else []
        st.session_state.memory_downregulated_genes = meta.get("downregulated_genes", []) if isinstance(meta.get("downregulated_genes", []), list) else []
        st.session_state.memory_deg_analysis = meta.get("deg_analysis", {}) if isinstance(meta.get("deg_analysis", {}), dict) else {}
        st.session_state.memory_deg_gene_records = meta.get("deg_gene_records", []) if isinstance(meta.get("deg_gene_records", []), list) else []
        st.session_state.memory_control_name = str(meta.get("control_name") or "")
        st.session_state.memory_test_name = str(meta.get("test_name") or "")
    if isinstance(meta.get("enrichr"), dict):
        st.session_state.memory_enrichr = meta.get("enrichr", {})
    st.session_state.memory_rwr_seed_genes = meta.get("rwr_seed_genes", []) if isinstance(meta.get("rwr_seed_genes", []), list) else []
    st.session_state.memory_rwr_genes = meta.get("rwr_genes", []) if isinstance(meta.get("rwr_genes", []), list) else []
    if meta.get("analysis_arm") in {"disease", "memory_rwr"}:
        disease_name = meta.get("disease_name", "")
        if isinstance(disease_name, str):
            st.session_state.memory_disease_name = disease_name
        openalex_genes = meta.get("openalex_genes", [])
        if isinstance(openalex_genes, list):
            st.session_state.memory_openalex_genes = openalex_genes
    opentargets_result = meta.get("opentargets_result")
    if isinstance(opentargets_result, dict) and opentargets_result:
        history = list(st.session_state.memory_opentargets_results or [])
        history.append(opentargets_result)
        st.session_state.memory_opentargets_results = history[-20:]
    l1000_result = meta.get("l1000cds2_result")
    if isinstance(l1000_result, dict) and l1000_result:
        st.session_state.memory_l1000cds2_result = l1000_result
    pubchem_result = meta.get("pubchem_result")
    if isinstance(pubchem_result, dict) and pubchem_result:
        st.session_state.memory_pubchem_result = pubchem_result
    hypothesis_result = meta.get("hypothesis_result")
    if isinstance(hypothesis_result, dict) and hypothesis_result:
        st.session_state.memory_hypothesis_result = hypothesis_result
    memory_slice_result = meta.get("memory_slice_result")
    if isinstance(memory_slice_result, dict) and memory_slice_result:
        st.session_state.memory_slice_result = memory_slice_result


def _csv_bytes(rows: list[dict[str, Any]], columns: list[str] | None = None) -> bytes:
    if not rows:
        return b""
    resolved_columns = columns or sorted({str(key) for row in rows for key in row.keys()})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=resolved_columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized = {}
        for column in resolved_columns:
            value = row.get(column, "")
            if isinstance(value, list):
                normalized[column] = "; ".join(str(item) for item in value)
            elif isinstance(value, dict):
                normalized[column] = str(value)
            else:
                normalized[column] = "" if value is None else value
        writer.writerow(normalized)
    return buffer.getvalue().encode("utf-8")


def _deg_download_rows(meta: dict[str, Any], deg_analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    records = meta.get("deg_gene_records")
    if isinstance(records, list) and records:
        return [row for row in records if isinstance(row, dict)]
    rows = deg_analysis.get("rows") if isinstance(deg_analysis, dict) else []
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _pathway_download_rows(enrichr: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(enrichr, dict):
        return []
    libs = enrichr.get("libraries")
    if not isinstance(libs, dict):
        return []
    rows: list[dict[str, Any]] = []
    for library, terms in libs.items():
        if not isinstance(terms, list):
            continue
        for rank, term in enumerate(terms, start=1):
            if not isinstance(term, dict):
                continue
            overlapping = term.get("overlapping_genes") or term.get("genes") or []
            rows.append(
                {
                    "library": library,
                    "rank": rank,
                    "term": term.get("term") or term.get("t"),
                    "p_value": term.get("p_value") if term.get("p_value") is not None else term.get("p"),
                    "adjusted_p_value": (
                        term.get("adjusted_p_value")
                        if term.get("adjusted_p_value") is not None
                        else term.get("adj")
                    ),
                    "combined_score": (
                        term.get("combined_score")
                        if term.get("combined_score") is not None
                        else term.get("cs")
                    ),
                    "overlapping_genes": overlapping if isinstance(overlapping, list) else [],
                    "n_overlap_genes": term.get("n_overlap_genes") or (len(overlapping) if isinstance(overlapping, list) else ""),
                }
            )
    return rows


def _l1000_download_rows(l1000_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(l1000_result, dict):
        return []
    top_drugs = l1000_result.get("top_drugs")
    if not isinstance(top_drugs, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in top_drugs:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "drug": row.get("name"),
                "pert_id": row.get("pert_id"),
                "best_rank": row.get("best_rank"),
                "best_score": row.get("best_score"),
                "signature_count": row.get("signature_count"),
                "cell_lines": row.get("cell_lines") if isinstance(row.get("cell_lines"), list) else [],
            }
        )
    return rows


def _render_downloads(meta: dict[str, Any], graph: nx.Graph | None) -> None:
    deg_analysis = meta.get("deg_analysis")
    enrichr = meta.get("enrichr")
    l1000_result = meta.get("l1000cds2_result")
    volcano_plot_path = meta.get("volcano_plot_path")

    deg_rows = _deg_download_rows(meta, deg_analysis if isinstance(deg_analysis, dict) else None)
    pathway_rows = _pathway_download_rows(enrichr if isinstance(enrichr, dict) else None)
    l1000_rows = _l1000_download_rows(l1000_result if isinstance(l1000_result, dict) else None)
    has_graph = isinstance(graph, nx.Graph) and graph.number_of_nodes() > 0
    has_volcano = isinstance(volcano_plot_path, str) and bool(volcano_plot_path.strip())

    if not any((deg_rows, pathway_rows, l1000_rows, has_graph, has_volcano)):
        return

    download_specs: list[dict[str, Any]] = []
    if deg_rows:
        download_specs.append(
            {
                "label": "DEG genes CSV",
                "data": _csv_bytes(
                    deg_rows,
                    ["gene", "log2FoldChange", "pvalue", "description"],
                ),
                "file_name": "deg_genes.csv",
                "mime": "text/csv",
                "key": "download_deg_genes_csv",
            }
        )
    if has_volcano:
        try:
            with open(str(volcano_plot_path), "rb") as handle:
                download_specs.append(
                    {
                        "label": "Volcano plot",
                        "data": handle.read(),
                        "file_name": "deg_volcano.png",
                        "mime": "image/png",
                        "key": "download_volcano_png",
                    }
                )
        except FileNotFoundError:
            st.caption("Volcano plot file is not available for download.")
    if l1000_rows:
        download_specs.append(
            {
                "label": "L1000 table CSV",
                "data": _csv_bytes(l1000_rows, ["drug", "pert_id", "best_rank", "best_score", "signature_count", "cell_lines"]),
                "file_name": "l1000cds2_results.csv",
                "mime": "text/csv",
                "key": "download_l1000_csv",
            }
        )
    if pathway_rows:
        download_specs.append(
            {
                "label": "Pathway CSV",
                "data": _csv_bytes(
                    pathway_rows,
                    ["library", "rank", "term", "p_value", "adjusted_p_value", "combined_score", "overlapping_genes", "n_overlap_genes"],
                ),
                "file_name": "pathway_enrichment.csv",
                "mime": "text/csv",
                "key": "download_pathway_csv",
            }
        )
    if has_graph:
        buff = io.BytesIO()
        nx.write_graphml(graph, buff)
        download_specs.append(
            {
                "label": "STRING graph GraphML",
                "data": buff.getvalue(),
                "file_name": "string_network.graphml",
                "mime": "application/graphml+xml",
                "key": "download_string_graphml",
            }
        )

    if not download_specs:
        return

    st.subheader("Downloads")
    cols = st.columns([1.2] * len(download_specs) + [4])
    for col, spec in zip(cols, download_specs):
        with col:
            st.download_button(
                spec["label"],
                data=spec["data"],
                file_name=spec["file_name"],
                mime=spec["mime"],
                use_container_width=True,
                key=spec["key"],
            )


def _render_technical_tables(meta: dict[str, Any], graph: nx.Graph | None) -> None:
    analysis_arm = meta.get("analysis_arm") or "disease"
    disease_name = meta.get("disease_name")
    openalex_papers = meta.get("openalex_papers")
    ranked_papers = meta.get("ranked_openalex_papers")
    literature_references = meta.get("literature_references")
    deg_analysis = meta.get("deg_analysis")
    rwr = meta.get("rwr_genes")
    enrichr = meta.get("enrichr")
    l1000_result = meta.get("l1000cds2_result")
    pubchem_result = meta.get("pubchem_result")
    hypothesis_result = meta.get("hypothesis_result")

    if analysis_arm != "srp" and isinstance(disease_name, str) and disease_name:
        st.subheader("Disease query")
        st.caption(disease_name)

    if analysis_arm != "srp" and isinstance(ranked_papers, list) and ranked_papers:
        st.subheader("Top literature hits")
        preview = []
        for paper in ranked_papers[:5]:
            if not isinstance(paper, dict):
                continue
            preview.append(
                {
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "relevance": paper.get("relevance"),
                    "reason": paper.get("reason"),
                }
            )
        if preview:
            st.table(preview)
    elif analysis_arm != "srp" and isinstance(openalex_papers, list) and openalex_papers:
        st.subheader("OpenAlex papers")
        st.caption(f"{len(openalex_papers)} papers scanned for genes.")
        st.table([{"title": paper.get("title"), "year": paper.get("year")} for paper in openalex_papers[:5] if isinstance(paper, dict)])

    if analysis_arm != "srp" and isinstance(literature_references, list) and literature_references:
        st.subheader("References")
        preview = []
        for ref in literature_references[:10]:
            if not isinstance(ref, dict):
                continue
            preview.append(
                {
                    "title": ref.get("title"),
                    "authors": ref.get("authors"),
                    "journal": ref.get("journal"),
                    "year": ref.get("year"),
                    "doi": ref.get("doi"),
                    "pmid": ref.get("pmid"),
                    "source": ref.get("source"),
                    "note": ref.get("note"),
                }
            )
        if preview:
            st.table(preview)

    if isinstance(deg_analysis, dict) and analysis_arm == "srp":
        rows = deg_analysis.get("rows")
        genes = deg_analysis.get("genes", [])
        st.subheader("Differentially expressed genes")
        status = deg_analysis.get("status")
        message = deg_analysis.get("message")
        if deg_analysis.get("log2fold") not in (None, "") or deg_analysis.get("padj") not in (None, ""):
            st.caption(
                f"Used thresholds: log2fold={deg_analysis.get('log2fold', 1.0)}, padj={deg_analysis.get('padj', 0.05)}"
            )
        if status and status != "ok":
            st.info(str(message or "DEG output is not ready yet."))
        elif isinstance(message, str) and message.strip():
            st.caption(message)
        if isinstance(genes, list) and genes:
            st.caption(f"{len(genes)} genes detected from the DEG output.")
        up_rows = deg_analysis.get("upregulated_rows")
        down_rows = deg_analysis.get("downregulated_rows")
        if isinstance(up_rows, list) and up_rows:
            st.markdown("**Top up-regulated genes**")
            st.table(up_rows[:10])
        if isinstance(down_rows, list) and down_rows:
            st.markdown("**Top down-regulated genes**")
            st.table(down_rows[:10])
        if not (isinstance(up_rows, list) and up_rows) and not (isinstance(down_rows, list) and down_rows) and isinstance(rows, list) and rows:
            st.table(rows[:10])

    if analysis_arm != "srp" and isinstance(rwr, list) and rwr:
        st.subheader("Random Walk with Restart")
        st.table([{"gene": g, "score": float(s)} for g, s in rwr[:20]])

    if isinstance(enrichr, dict):
        libs = enrichr.get("libraries")
        if isinstance(libs, dict) and libs:
            st.subheader("Pathway enrichment")
            tab_names = list(libs.keys())
            tabs = st.tabs([str(n) for n in tab_names])
            for tab, lib in zip(tabs, tab_names):
                with tab:
                    terms = libs.get(lib)
                    if not isinstance(terms, list) or not terms:
                        st.info("No enriched terms returned.")
                        continue
                    rows = []
                    for t in terms:
                        if not isinstance(t, dict):
                            continue
                        rows.append(
                            {
                                "term": t.get("term"),
                                "p_value": t.get("p_value"),
                                "adjusted_p_value": t.get("adjusted_p_value"),
                                "combined_score": t.get("combined_score"),
                                "overlap_genes": ", ".join(list(t.get("overlapping_genes") or [])[:10]),
                            }
                        )
                    st.table(rows[:10])

    if isinstance(l1000_result, dict) and l1000_result:
        st.subheader("L1000CDS2 drug matches")
        requested_cell_lines = l1000_result.get("requested_cell_lines")
        if isinstance(requested_cell_lines, list) and requested_cell_lines:
            st.caption("Cell lines: " + ", ".join(str(value) for value in requested_cell_lines if str(value).strip()))
        top_drugs = l1000_result.get("top_drugs")
        if isinstance(top_drugs, list) and top_drugs:
            st.table(
                [
                    {
                        "drug": row.get("name"),
                        "pert_id": row.get("pert_id"),
                        "best_rank": row.get("best_rank"),
                        "best_score": row.get("best_score"),
                        "cell_lines": ", ".join(row.get("cell_lines") or []),
                        "signature_count": row.get("signature_count"),
                    }
                    for row in top_drugs[:20]
                    if isinstance(row, dict)
                ]
            )
        else:
            st.info(str(l1000_result.get("message") or "No L1000CDS2 drug matches were returned."))

    if isinstance(pubchem_result, dict) and pubchem_result:
        st.subheader("PubChem compound record")
        cid = pubchem_result.get("cid")
        title = pubchem_result.get("title") or pubchem_result.get("drug_name")
        matched_query = pubchem_result.get("matched_query")
        matched_strategy = pubchem_result.get("matched_strategy")
        pert_id = pubchem_result.get("pert_id")
        caption_parts = []
        if title:
            caption_parts.append(str(title))
        if cid:
            caption_parts.append(f"CID {cid}")
        if pert_id:
            caption_parts.append(f"pert_id {pert_id}")
        if matched_query:
            caption_parts.append(f"matched query {matched_query}")
        if matched_strategy:
            caption_parts.append(f"match type {matched_strategy}")
        if caption_parts:
            st.caption(" | ".join(caption_parts))

        properties = pubchem_result.get("properties")
        if isinstance(properties, dict) and properties:
            st.table(
                [
                    {"property": str(key), "value": "" if value is None else str(value)}
                    for key, value in properties.items()
                    if value not in (None, "")
                ]
            )

        synonyms = pubchem_result.get("synonyms")
        if isinstance(synonyms, list) and synonyms:
            st.caption("Synonyms")
            st.table([{"synonym": value} for value in synonyms[:25] if str(value).strip()])

        annotation_lines = pubchem_result.get("annotation_lines")
        if isinstance(annotation_lines, list) and annotation_lines:
            st.caption("Annotation snippets used for synthesis")
            st.table([{"annotation": value} for value in annotation_lines[:25] if str(value).strip()])

    if isinstance(hypothesis_result, dict) and hypothesis_result:
        st.subheader("Experimental hypotheses")
        summary = hypothesis_result.get("hypothesis_summary")
        if isinstance(summary, str) and summary.strip():
            st.caption(summary)
        hypotheses = hypothesis_result.get("hypotheses")
        if isinstance(hypotheses, list) and hypotheses:
            st.table(
                [
                    {
                        "title": row.get("title"),
                        "rationale": row.get("rationale"),
                        "experiment_design": row.get("experiment_design"),
                        "expected_observation": row.get("expected_observation"),
                        "existing_evidence": row.get("existing_evidence"),
                    }
                    for row in hypotheses[:10]
                    if isinstance(row, dict)
                ]
            )

    _render_downloads(meta, graph)

    pyvis_html_path = meta.get("pyvis_html_path")
    if isinstance(pyvis_html_path, str) and pyvis_html_path:
        try:
            with open(pyvis_html_path, "r", encoding="utf-8") as f:
                st.subheader("Network visualization")
                components.html(f.read(), height=850, scrolling=True)
        except FileNotFoundError:
            st.warning("PyVis visualization file was not found.")

    kegg_pathway_path = meta.get("kegg_pathway_path")
    if isinstance(kegg_pathway_path, str) and kegg_pathway_path:
        st.subheader("KEGG pathway visualization")
        st.image(kegg_pathway_path, caption=kegg_pathway_path)

    volcano_plot_path = meta.get("volcano_plot_path")
    if isinstance(volcano_plot_path, str) and volcano_plot_path:
        st.subheader("Volcano plot")
        st.image(volcano_plot_path, caption=volcano_plot_path)

    tool_history = meta.get("tool_history")
    if isinstance(tool_history, list) and tool_history:
        st.subheader("Tool trace")
        recent_history = tool_history[-20:]
        for index, entry in enumerate(recent_history, start=1):
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool", f"tool_{index}"))
            with st.expander(f"{index}. {tool_name}", expanded=index == len(recent_history)):
                args = entry.get("args")
                result = entry.get("result")
                if isinstance(args, dict) and args:
                    st.caption("Args")
                    st.json(args)
                if isinstance(result, dict) and result:
                    st.caption("Result")
                    st.json(result)


def _render_sidebar(meta: dict[str, Any]) -> None:
    st.sidebar.header("Session")
    if st.sidebar.button("Clear chat and memory", use_container_width=True):
        _reset_memory(keep_messages=False)
        st.rerun()
    if st.sidebar.button("Clear memory only", use_container_width=True):
        _reset_memory(keep_messages=True)
        st.rerun()

    st.sidebar.caption(f"LLM provider: {SETTINGS.llm_provider}")
    st.sidebar.caption(f"STRING mode: {SETTINGS.string_local_mode}")

    analysis_arm = meta.get("analysis_arm") or "disease"
    deg_analysis = meta.get("deg_analysis")
    rwr_genes = meta.get("rwr_genes")
    if analysis_arm == "srp" and isinstance(deg_analysis, dict):
        deg_genes = deg_analysis.get("genes", [])
        if isinstance(deg_genes, list):
            st.sidebar.metric("DEG genes", len(deg_genes))
    if isinstance(rwr_genes, list) and rwr_genes:
        st.sidebar.metric("RWR targets", len(rwr_genes))

    net = meta.get("network")
    if isinstance(net, dict):
        st.sidebar.metric("Nodes", int(net.get("nodes", 0)))
        st.sidebar.metric("Edges", int(net.get("edges", 0)))
        top = net.get("top_degree", [])
        if isinstance(top, list) and top:
            st.sidebar.caption("Top degree")
            st.sidebar.table(top)

    tool_history = meta.get("tool_history")
    if isinstance(tool_history, list) and tool_history:
        st.sidebar.caption("Latest tool calls")
        for entry in tool_history[-5:]:
            if isinstance(entry, dict):
                st.sidebar.write(str(entry.get("tool", "unknown")))

    if st.session_state.last_error:
        st.sidebar.error(st.session_state.last_error)


_init_session_state()
if build_app is None:
    st.error("The agent runtime dependencies are incomplete in this environment.")
    st.code(APP_IMPORT_ERROR)
    st.stop()

app = _get_compiled_app()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a biomedical question, request a literature answer with references, run DEG follow-ups, or query stored pathway/RWR results.")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last_error = ""
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke(_invoke_state_from_session(prompt))
                answer = str(result.get("answer") or "").strip() or "No answer was generated."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_graph = result.get("graph")
                st.session_state.last_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
                _update_memory_from_meta(st.session_state.last_meta)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                st.session_state.last_error = error_text
                st.error("The agent hit an error while processing this request.")
                st.code(error_text)

meta = st.session_state.last_meta
graph = st.session_state.last_graph
_render_sidebar(meta if isinstance(meta, dict) else {})
if isinstance(meta, dict) and meta:
    st.divider()
    st.header("Technical results")
    _render_technical_tables(meta, graph if isinstance(graph, nx.Graph) else None)
