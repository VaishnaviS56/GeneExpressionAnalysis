from __future__ import annotations

import io
from typing import Any

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
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
    invoke_state: dict[str, Any] = {
        "query": prompt,
        "memory_summary": _build_memory_summary(),
    }
    for key in MEMORY_DEFAULTS:
        if key.startswith("memory_") and st.session_state.get(key):
            invoke_state[key] = st.session_state.get(key)
    return invoke_state


def _update_memory_from_meta(meta: dict[str, Any]) -> None:
    if meta.get("analysis_arm") == "srp":
        st.session_state.memory_deg_genes = meta.get("deg_genes", []) if isinstance(meta.get("deg_genes", []), list) else []
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


def _render_technical_tables(meta: dict[str, Any], graph: nx.Graph | None) -> None:
    analysis_arm = meta.get("analysis_arm") or "disease"
    disease_name = meta.get("disease_name")
    openalex_papers = meta.get("openalex_papers")
    ranked_papers = meta.get("ranked_openalex_papers")
    deg_analysis = meta.get("deg_analysis")
    rwr = meta.get("rwr_genes")
    enrichr = meta.get("enrichr")

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

    if isinstance(deg_analysis, dict) and analysis_arm == "srp":
        rows = deg_analysis.get("rows")
        genes = deg_analysis.get("genes", [])
        st.subheader("Differentially expressed genes")
        status = deg_analysis.get("status")
        message = deg_analysis.get("message")
        if status and status != "ok":
            st.info(str(message or "DEG output is not ready yet."))
        if isinstance(genes, list) and genes:
            st.caption(f"{len(genes)} genes detected from the DEG output.")
        if isinstance(rows, list) and rows:
            st.table(rows[:10])

        deg_gene_records = meta.get("deg_gene_records")
        if isinstance(deg_gene_records, list) and deg_gene_records:
            st.caption("DEG genes with p-values preserved for downstream analysis.")
            st.table(deg_gene_records[:10])

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
                    st.table(rows)

    if isinstance(graph, nx.Graph) and graph.number_of_nodes() > 0:
        st.subheader("Downloads")
        buff = io.BytesIO()
        nx.write_graphml(graph, buff)
        st.download_button(
            "Download STRING graph (GraphML)",
            data=buff.getvalue(),
            file_name="string_network.graphml",
            mime="application/graphml+xml",
        )

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
    if analysis_arm == "srp" and isinstance(deg_analysis, dict):
        deg_genes = deg_analysis.get("genes", [])
        if isinstance(deg_genes, list):
            st.sidebar.metric("DEG genes", len(deg_genes))

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
meta = st.session_state.last_meta if isinstance(st.session_state.last_meta, dict) else {}
_render_sidebar(meta)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a biomedical question, run DEG follow-ups, or query stored pathway/RWR results.")
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
if isinstance(meta, dict) and meta:
    st.divider()
    st.header("Technical results")
    _render_technical_tables(meta, graph if isinstance(graph, nx.Graph) else None)
