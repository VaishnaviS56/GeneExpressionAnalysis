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
    select_top_degree: int | None = None,
    seed_genes: Iterable[str] | None = None,
    rwr_genes: Iterable[str] | None = None,
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

    net = Network(height=height, width=width, directed=directed, notebook=notebook, bgcolor="#ffffff", font_color="#0f172a")
    # net.heading = title

    seed_gene_set = {str(node).strip().upper() for node in (seed_genes or []) if str(node).strip()}
    rwr_gene_set = {str(node).strip().upper() for node in (rwr_genes or []) if str(node).strip()}

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
    label_degree_cutoff = max_degree * 0.55 if max_degree else 0

    for node in keep_graph.nodes():
        degree = degrees.get(node, 0)
        size = 8 + (degree / max_degree) * 22 if max_degree else 8
        node_upper = str(node).strip().upper()
        is_seed = node_upper in seed_gene_set
        is_rwr = node_upper in rwr_gene_set
        show_label = is_seed or is_rwr or degree >= label_degree_cutoff
        color = {
            "background": "#6ea8fe",
            "border": "#d7e3f4",
            "highlight": {
                "background": "#3b82f6",
                "border": "#1d4ed8",
            },
        }
        border_width = 0
        if is_rwr:
            color = {
                "background": "#f59e0b",
                "border": "#fbbf24",
                "highlight": {
                    "background": "#f59e0b",
                    "border": "#fbbf24",
                },
            }
            border_width = 2
        if is_seed:
            color = {
                "background": "#22c55e",
                "border": "#86efac",
                "highlight": {
                    "background": "#22c55e",
                    "border": "#86efac",
                },
            }
            border_width = 4
            size += 8
        title_bits = [f"{node}", f"Degree: {degree}"]
        if is_seed:
            title_bits.append("Seed gene")
        if is_rwr:
            title_bits.append("RWR hit")
        net.add_node(
            node,
            label=str(node) if show_label else " ",
            title="<br>".join(title_bits),
            size=size,
            color=color,
            borderWidth=border_width,
            font={"size": 16 if show_label else 0, "strokeWidth": 4, "strokeColor": "#ffffff"},
        )

    for u, v, attrs in keep_graph.edges(data=True):
        weight = float(attrs.get("weight", 1.0))
        net.add_edge(u, v, value=weight, title=f"Weight: {weight:.3f}")

    net.force_atlas_2based(gravity=-45, central_gravity=0.015, spring_length=135, spring_strength=0.06, damping=0.55, overlap=1.0)
    net.set_options(
        """
        {
          "layout": {
            "improvedLayout": true
          },
          "nodes": {
            "borderWidth": 1,
            "color": {
              "background": "#6ea8fe",
              "border": "#d7e3f4",
              "highlight": {
                "background": "#3b82f6",
                "border": "#1d4ed8"
              }
            },
            "font": {"size": 14, "face": "Arial", "color": "#0f172a", "strokeWidth": 4, "strokeColor": "#ffffff"}
          },
          "edges": {
            "color": {"color": "rgba(148, 163, 184, 0.38)", "highlight": "#f59e0b"},
            "smooth": false,
            "width": 0.65,
            "selectionWidth": 1.25
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": false,
            "keyboard": false,
            "hideEdgesOnDrag": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 180,
              "fit": true
            },
            "barnesHut": {
              "gravitationalConstant": -3600,
              "springLength": 145,
              "springConstant": 0.035,
              "damping": 0.58,
              "avoidOverlap": 0.35
            }
          }
        }
        """
    )

    html_path = output_path or "pyvis_network.html"
    Path(html_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    net.write_html(html_path)
    legend = """
<div class="network-clean-legend">
  <span><i class="seed"></i>Seed genes</span>
  <span><i class="rwr"></i>RWR hits</span>
  <span><i class="other"></i>Network genes</span>
</div>
<style>
  html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #ffffff; font-family: Arial, sans-serif; }
  #mynetwork { border: 0 !important; height: 100vh !important; background: #ffffff !important; }
  .vis-tooltip { border: 1px solid #dbe4ec !important; border-radius: 8px !important; box-shadow: 0 12px 32px rgba(15, 23, 42, .16) !important; font-family: Arial, sans-serif !important; font-size: 12px !important; }
  .network-clean-legend { position: fixed; left: 14px; top: 12px; z-index: 5; display: flex; gap: 8px; flex-wrap: wrap; max-width: calc(100% - 28px); pointer-events: none; }
  .network-clean-legend span { display: inline-flex; align-items: center; gap: 6px; padding: 5px 8px; border: 1px solid #e2e8f0; border-radius: 999px; background: rgba(255,255,255,.92); color: #334155; font-size: 12px; }
  .network-clean-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 999px; }
  .network-clean-legend .seed { background: #22c55e; }
  .network-clean-legend .rwr { background: #f59e0b; }
  .network-clean-legend .other { background: #6ea8fe; }
</style>
"""
    html = Path(html_path).read_text(encoding="utf-8")
    Path(html_path).write_text(html.replace("</body>", f"{legend}</body>"), encoding="utf-8")
    return str(Path(html_path).resolve())
