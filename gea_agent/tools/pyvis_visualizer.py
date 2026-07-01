from __future__ import annotations

from pathlib import Path
from typing import Iterable

import networkx as nx


def build_pyvis_html(
    graph: nx.Graph,
    *,
    title: str = "STRING Network",
    output_path: str | None = None,
    notebook: bool = False,
    height: str = "800px",
    width: str = "100%",
    directed: bool = False,
    select_top_degree: int | None = 300,
) -> str:
    """
    Build a PyVis visualization for a NetworkX graph and return the HTML path.

    If `output_path` is provided, the HTML is written there. Otherwise a temp-ish
    file name in the current directory is used.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pyvis is not installed. Install it with `pip install pyvis`."
        ) from exc

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot visualize an empty graph.")

    net = Network(height=height, width=width, directed=directed, notebook=notebook, bgcolor="#0f172a", font_color="#e2e8f0")
    # net.heading = title

    nodes = list(graph.nodes())
    if select_top_degree is not None:
        select_top_degree = max(1, int(select_top_degree))
    if select_top_degree is not None and graph.number_of_nodes() > select_top_degree:
        ranked = sorted(graph.degree(), key=lambda item: item[1], reverse=True)
        keep = {node for node, _ in ranked[:select_top_degree]}
        keep_graph = graph.subgraph(keep).copy()
    else:
        keep_graph = graph

    degrees = dict(keep_graph.degree())
    max_degree = max(degrees.values()) if degrees else 1

    for node in keep_graph.nodes():
        degree = degrees.get(node, 0)
        size = 10 + (degree / max_degree) * 25 if max_degree else 10
        net.add_node(
            node,
            label=str(node),
            title=f"{node}<br>Degree: {degree}",
            size=size,
        )

    for u, v, attrs in keep_graph.edges(data=True):
        weight = float(attrs.get("weight", 1.0))
        net.add_edge(u, v, value=weight, title=f"Weight: {weight:.3f}")

    net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=1.0)
    net.set_options(
        """
        {
          "nodes": {
            "borderWidth": 0,
            "color": {
              "background": "#60a5fa",
              "highlight": {
                "background": "#f59e0b",
                "border": "#f59e0b"
              }
            },
            "font": {"size": 16}
          },
          "edges": {
            "color": {"color": "#94a3b8", "highlight": "#f59e0b"},
            "smooth": false
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -2500,
              "springLength": 120,
              "springConstant": 0.04,
              "damping": 0.5
            }
          }
        }
        """
    )

    html_path = output_path or "pyvis_network.html"
    Path(html_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    net.write_html(html_path)
    return str(Path(html_path).resolve())
