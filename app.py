from __future__ import annotations

import io

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from dotenv import find_dotenv, load_dotenv

from gea_agent.agent.graph import build_app


load_dotenv(find_dotenv(usecwd=True), override=False)

st.set_page_config(page_title="GEA Agent", layout="wide")
st.title("GEA Agent (LangGraph + STRING + NetworkX)")

if "app" not in st.session_state:
    st.session_state.app = build_app()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_graph" not in st.session_state:
    st.session_state.last_graph = None
if "last_meta" not in st.session_state:
    st.session_state.last_meta = None


def _render_technical_tables(meta: dict, graph: nx.Graph | None):
    if not isinstance(meta, dict):
        return

    deg_analysis = meta.get("deg_analysis")
    rwr = meta.get("rwr_genes")
    enrichr = meta.get("enrichr")

    if isinstance(deg_analysis, dict):
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

    if isinstance(rwr, list) and rwr:
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


def _render_sidebar():
    st.sidebar.header("Network")

    meta = st.session_state.last_meta or {}
    if not isinstance(meta, dict):
        meta = {}

    deg_analysis = meta.get("deg_analysis")
    if isinstance(deg_analysis, dict):
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

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.app.invoke({"query": prompt})
            answer = result.get("answer", "")
            st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_graph = result.get("graph")
            st.session_state.last_meta = result.get("meta")

    st.rerun()

# Persistent technical results (stay visible across reruns)
meta = st.session_state.last_meta
graph = st.session_state.last_graph
if isinstance(meta, dict):
    st.divider()
    st.header("Technical results")
    _render_technical_tables(meta, graph if isinstance(graph, nx.Graph) else None)
