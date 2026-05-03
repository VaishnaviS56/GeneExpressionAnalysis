from __future__ import annotations

import io
import math

import altair as alt
import networkx as nx
import streamlit as st
from dotenv import load_dotenv

from gea_agent.agent.graph import build_app


load_dotenv()

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


def _flatten_enrichr(enrichr: dict) -> list[dict]:
    libs = enrichr.get("libraries") if isinstance(enrichr, dict) else None
    if not isinstance(libs, dict):
        return []

    rows: list[dict] = []
    for lib, terms in libs.items():
        if not isinstance(terms, list):
            continue
        for t in terms:
            if not isinstance(t, dict):
                continue
            term = t.get("term")
            if not term:
                continue
            adj_p = t.get("adjusted_p_value")
            combined = t.get("combined_score")
            try:
                adj_p_f = float(adj_p) if adj_p is not None else None
            except Exception:
                adj_p_f = None
            try:
                combined_f = float(combined) if combined is not None else None
            except Exception:
                combined_f = None

            rows.append(
                {
                    "library": str(lib),
                    "term": str(term),
                    "adj_p": adj_p_f,
                    "combined_score": combined_f,
                    "neglog10_adj_p": (-math.log10(adj_p_f) if adj_p_f and adj_p_f > 0 else None),
                }
            )
    return rows


def _render_pathway_viz(enrichr: dict):
    rows = _flatten_enrichr(enrichr)
    if not rows:
        st.info("No pathway results to visualize yet.")
        return

    libs = sorted({r["library"] for r in rows})
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_lib = st.selectbox("Pathway library", libs, index=0)
    with col2:
        metric = st.selectbox("Sort/plot by", ["-log10(adj p)", "combined score"], index=0)

    data = [r for r in rows if r["library"] == selected_lib]
    if metric == "combined score":
        data = sorted(data, key=lambda r: (r["combined_score"] is None, -(r["combined_score"] or 0.0)))
        x_field = "combined_score"
        x_title = "Combined score"
    else:
        data = sorted(data, key=lambda r: (r["neglog10_adj_p"] is None, -(r["neglog10_adj_p"] or 0.0)))
        x_field = "neglog10_adj_p"
        x_title = "-log10(adjusted p-value)"

    data = data[:15]
    if not data:
        st.info("No terms available for this library.")
        return

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y("term:N", sort="-x", title="Term"),
            x=alt.X(f"{x_field}:Q", title=x_title),
            tooltip=[
                alt.Tooltip("term:N"),
                alt.Tooltip("adj_p:Q", title="adj p", format=".2e"),
                alt.Tooltip("combined_score:Q", title="combined", format=".2f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Top terms from Enrichr (gget) results.")


def _render_sidebar():
    st.sidebar.header("Results")

    meta = st.session_state.last_meta or {}
    if not isinstance(meta, dict):
        meta = {}

    net = meta.get("network")
    if isinstance(net, dict):
        st.sidebar.subheader("Network")
        st.sidebar.metric("Nodes", int(net.get("nodes", 0)))
        st.sidebar.metric("Edges", int(net.get("edges", 0)))
        top = net.get("top_degree", [])
        if isinstance(top, list) and top:
            st.sidebar.caption("Top degree")
            st.sidebar.table(top)

    rwr = meta.get("rwr_genes")
    if isinstance(rwr, list) and rwr:
        st.sidebar.subheader("RWR (Top 20)")
        st.sidebar.table([{"gene": g, "score": float(s)} for g, s in rwr])

    graph = st.session_state.last_graph
    if isinstance(graph, nx.Graph) and graph.number_of_nodes() > 0:
        buff = io.BytesIO()
        nx.write_graphml(graph, buff)
        st.sidebar.download_button(
            "Download GraphML",
            data=buff.getvalue(),
            file_name="string_network.graphml",
            mime="application/graphml+xml",
        )


_render_sidebar()

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

    # Pathway visualization section (renders after a technical run)
    meta = st.session_state.last_meta
    if isinstance(meta, dict) and isinstance(meta.get("enrichr"), dict):
        with st.expander("Pathway visualization", expanded=True):
            _render_pathway_viz(meta.get("enrichr"))

    st.rerun()