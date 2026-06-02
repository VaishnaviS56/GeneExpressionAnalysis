from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from gea_agent.config import SETTINGS
from gea_agent.tools.pyvis_visualizer import build_pyvis_html
from gea_agent.tools.string_graph_cache import load_or_build_full_string_graph


st.set_page_config(page_title="STRING PyVis Viewer", layout="wide")
st.title("STRING Graph PyVis Viewer")
st.caption("This view loads the cached full STRING graph and visualizes it by default.")

graph = load_or_build_full_string_graph(
    info_path=SETTINGS.string_info_path,
    links_path=SETTINGS.string_links_path,
    required_score=SETTINGS.string_required_score,
    cache_path=SETTINGS.string_graph_cache_path,
    force_rebuild=SETTINGS.string_force_rebuild,
)

st.sidebar.header("Graph Summary")
st.sidebar.metric("Nodes", graph.number_of_nodes())
st.sidebar.metric("Edges", graph.number_of_edges())

html_path = build_pyvis_html(
    graph,
    title="STRING Network",
    output_path="pyvis_network.html",
    select_top_degree=300,
)

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

components.html(html, height=850, scrolling=True)

st.download_button(
    "Download PyVis HTML",
    data=Path(html_path).read_bytes(),
    file_name="pyvis_network.html",
    mime="text/html",
)
