from __future__ import annotations

import io

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import find_dotenv, load_dotenv

from gea_agent.agent.graph import build_app


load_dotenv(find_dotenv(usecwd=True), override=False)

st.set_page_config(page_title="GEA Agent", layout="wide")
st.title("GEA Agent (Planner-Executor Orchestrator + STRING + NetworkX)")

app = build_app()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if "app" not in st.session_state:
    st.session_state.app = build_app()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_graph" not in st.session_state:
    st.session_state.last_graph = None
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None
if "memory_deg_genes" not in st.session_state:
    st.session_state.memory_deg_genes = []
if "memory_deg_analysis" not in st.session_state:
    st.session_state.memory_deg_analysis = {}
if "memory_deg_gene_records" not in st.session_state:
    st.session_state.memory_deg_gene_records = []
if "memory_disease_name" not in st.session_state:
    st.session_state.memory_disease_name = ""
if "memory_openalex_genes" not in st.session_state:
    st.session_state.memory_openalex_genes = []
if "memory_opentargets_results" not in st.session_state:
    st.session_state.memory_opentargets_results = []


def _build_memory_summary() -> str:
    parts: list[str] = []

    if st.session_state.memory_deg_genes:
        parts.append(f"Stored DEG genes available: {len(st.session_state.memory_deg_genes)}.")
    if st.session_state.memory_deg_gene_records:
        parts.append(f"Stored DEG gene records available: {len(st.session_state.memory_deg_gene_records)}.")
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


def _render_technical_tables(meta: dict, graph: nx.Graph | None):
    if not isinstance(meta, dict):
        return

    analysis_arm = meta.get("analysis_arm") or "disease"
    disease_name = meta.get("disease_name")
    openalex_papers = meta.get("openalex_papers")
    deg_analysis = meta.get("deg_analysis")
    rwr = meta.get("rwr_genes")
    enrichr = meta.get("enrichr")

    if analysis_arm != "srp" and isinstance(disease_name, str) and disease_name:
        st.subheader("Disease query")
        st.caption(disease_name)

    if analysis_arm != "srp" and isinstance(openalex_papers, list) and openalex_papers:
        st.subheader("OpenAlex papers")
        st.caption(f"{len(openalex_papers)} papers scanned for genes.")
        max_gene_count = 0
        for paper in openalex_papers:
            if not isinstance(paper, dict):
                continue
            genes = paper.get("genes", [])
            if isinstance(genes, list):
                max_gene_count = max(max_gene_count, len(genes))
        filtered_papers = [
            paper
            for paper in openalex_papers
            if isinstance(paper, dict)
            and isinstance(paper.get("genes"), list)
            and len(paper.get("genes", [])) == max_gene_count
            and max_gene_count > 0
        ]
        st.caption(f"Showing only papers with the most genes identified: {max_gene_count}.")
        preview = []
        for paper in filtered_papers[:5]:
            preview.append(
                {
                    "title": paper.get("title"),
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
        if status and status != "ok":
            st.info(str(message or "DEG output is not ready yet."))
        if isinstance(genes, list) and genes:
            st.caption(f"{len(genes)} genes detected from the DEG output.")
        if isinstance(rows, list) and rows:
            st.table(rows[:10])
        elif status == "ok":
            st.info("The DEG CSV was loaded, but no rows were found.")

        deg_gene_records = meta.get("deg_gene_records")
        if isinstance(deg_gene_records, list) and deg_gene_records:
            st.caption("DEG genes with p-values preserved for downstream analysis.")
            st.table(deg_gene_records[:10])

    if analysis_arm != "srp" and isinstance(rwr, list) and rwr:
        st.subheader("Random Walk with Restart (Top 20)")
        st.table([{"gene": g, "score": float(s)} for g, s in rwr[:20]])

    if isinstance(enrichr, dict):
        libs = enrichr.get("libraries")
        if isinstance(libs, dict) and libs:
            st.subheader("Pathway enrichment (Enrichr via gget)")
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
                else:
                    st.info("No compact tool result recorded.")


def _render_sidebar():
    st.sidebar.header("Network")

    meta = st.session_state.last_meta or {}
    if not isinstance(meta, dict):
        meta = {}

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
            if not isinstance(entry, dict):
                continue
            st.sidebar.write(str(entry.get("tool", "unknown")))


_render_sidebar()

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question... (include genes like TP53, EGFR for technical mode)")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    invoke_state = {"query": prompt}
    invoke_state["memory_summary"] = _build_memory_summary()
    if st.session_state.memory_deg_genes:
        invoke_state["memory_deg_genes"] = st.session_state.memory_deg_genes
    if st.session_state.memory_deg_analysis:
        invoke_state["memory_deg_analysis"] = st.session_state.memory_deg_analysis
    if st.session_state.memory_deg_gene_records:
        invoke_state["memory_deg_gene_records"] = st.session_state.memory_deg_gene_records
    if st.session_state.memory_disease_name:
        invoke_state["memory_disease_name"] = st.session_state.memory_disease_name
    if st.session_state.memory_openalex_genes:
        invoke_state["memory_openalex_genes"] = st.session_state.memory_openalex_genes
    if st.session_state.memory_opentargets_results:
        invoke_state["memory_opentargets_results"] = st.session_state.memory_opentargets_results

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.app.invoke(invoke_state)
            answer = result.get("answer", "")
            st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_graph = result.get("graph")
            st.session_state.last_meta = result.get("meta")
            meta = result.get("meta") or {}
            if isinstance(meta, dict) and meta.get("analysis_arm") == "srp":
                st.session_state.memory_deg_genes = meta.get("deg_genes", []) if isinstance(meta.get("deg_genes", []), list) else []
                st.session_state.memory_deg_analysis = meta.get("deg_analysis", {}) if isinstance(meta.get("deg_analysis", {}), dict) else {}
                st.session_state.memory_deg_gene_records = meta.get("deg_gene_records", []) if isinstance(meta.get("deg_gene_records", []), list) else []
            if isinstance(meta, dict) and meta.get("analysis_arm") in {"disease", "memory_rwr"}:
                disease_name = meta.get("disease_name", "")
                if isinstance(disease_name, str):
                    st.session_state.memory_disease_name = disease_name
                openalex_genes = meta.get("openalex_genes", [])
                if isinstance(openalex_genes, list):
                    st.session_state.memory_openalex_genes = openalex_genes
            if isinstance(meta, dict) and meta.get("opentargets_result"):
                result = meta.get("opentargets_result")
                if isinstance(result, dict):
                    history = list(st.session_state.memory_opentargets_results or [])
                    history.append(result)
                    st.session_state.memory_opentargets_results = history[-20:]

    st.rerun()

# Persistent technical results (stay visible across reruns)
meta = st.session_state.last_meta
graph = st.session_state.last_graph
if isinstance(meta, dict):
    st.divider()
    st.header("Technical results")
    _render_technical_tables(meta, graph if isinstance(graph, nx.Graph) else None)
