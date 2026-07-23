from __future__ import annotations

import math
import re
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from gea_agent.tools.pyvis_visualizer import build_pyvis_html


def _deg_gene_label(row: pd.Series) -> str:
    for key in ("hgnc_symbol", "external_gene_name", "gene", "Gene", "symbol", "Ensembl"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _normalize_pathway_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _select_kegg_rank(
    results: list[dict[str, Any]],
    pathway_term: str | None,
    *,
    default_rank: int,
) -> tuple[int, dict[str, Any] | None]:
    desired = _normalize_pathway_label(pathway_term)
    max_rank = max(1, len(results))
    safe_default_rank = max(1, min(int(default_rank), max_rank))
    if not desired:
        return safe_default_rank, None

    best: tuple[int, dict[str, Any], int] | None = None
    for index, row in enumerate(results, start=1):
        if not isinstance(row, dict):
            continue
        label = str(
            row.get("path_name")
            or row.get("term")
            or row.get("term_name")
            or row.get("name")
            or row.get("Path")
            or row.get("Term")
            or ""
        ).strip()
        if not label:
            continue
        label_norm = _normalize_pathway_label(label)
        score = 0
        if label_norm == desired:
            score = 1000
        elif desired in label_norm or label_norm in desired:
            score = 800
        elif all(token in label_norm for token in desired.split() if token):
            score = 600
        if score <= 0:
            continue
        if best is None or score > best[2]:
            best = (index, row, score)

    if best is None:
        return safe_default_rank, None
    return best[0], best[1]


def build_network_visualization(
    graph,
    *,
    output_path: str = "pyvis_network.html",
    select_top_degree: int = 300,
    allowed_nodes: list[str] | None = None,
    seed_genes: list[str] | None = None,
    rwr_genes: list[str] | None = None,
) -> dict[str, Any]:
    if graph is None or getattr(graph, "number_of_nodes", lambda: 0)() == 0:
        return {
            "status": "missing_graph",
            "message": "No graph is available for network visualization.",
            "pyvis_html_path": "",
        }

    render_graph = graph
    if allowed_nodes:
        allowed = {str(node).strip().upper() for node in allowed_nodes if str(node).strip()}
        keep = [node for node in graph.nodes() if str(node).strip().upper() in allowed]
        if keep:
            candidate_graph = graph.subgraph(keep).copy()
            if candidate_graph.number_of_edges() > 0 or graph.number_of_edges() == 0:
                render_graph = candidate_graph

    html_path = build_pyvis_html(
        render_graph,
        output_path=output_path,
        select_top_degree=select_top_degree,
        seed_genes=seed_genes,
        rwr_genes=rwr_genes,
    )
    return {
        "status": "ok",
        "message": "Built PyVis network visualization.",
        "pyvis_html_path": html_path,
        "visualized_node_count": int(render_graph.number_of_nodes()),
        "visualized_edge_count": int(render_graph.number_of_edges()),
    }


def build_kegg_pathway_visualization(
    genes: list[str],
    *,
    output_path: str = "kegg_pathway.png",
    kegg_rank: int = 1,
    species: str = "human",
    pathway_term: str | None = None,
) -> dict[str, Any]:
    genes = [str(g).strip().upper() for g in genes if str(g).strip()]
    genes = list(dict.fromkeys(genes))
    if not genes:
        return {
            "status": "missing_genes",
            "message": "No genes are available for KEGG pathway visualization.",
            "kegg_pathway_path": "",
        }

    try:
        import gget
    except Exception as exc:
        return {
            "status": "dependency_missing",
            "message": f"gget is required for KEGG visualization: {exc}",
            "kegg_pathway_path": "",
        }

    output = str(Path(output_path).resolve())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    try:
        preview_results = gget.enrichr(
            genes=genes,
            database="KEGG_2021_Human",
            species=species,
            json=True,
            verbose=False,
        )
        preview_rows = preview_results if isinstance(preview_results, list) else []
        if not preview_rows:
            return {
                "status": "not_found",
                "message": "No KEGG enrichment results were returned for the selected genes, so a KEGG pathway image could not be generated.",
                "kegg_pathway_path": "",
                "genes": genes,
                "requested_pathway_term": str(pathway_term or "").strip(),
                "kegg_enrichr_results": [],
            }
        selected_rank, selected_pathway = _select_kegg_rank(
            preview_rows,
            pathway_term,
            default_rank=int(kegg_rank),
        )
        if pathway_term and not selected_pathway:
            available = []
            for row in preview_rows[:5]:
                if not isinstance(row, dict):
                    continue
                label = str(
                    row.get("path_name")
                    or row.get("term")
                    or row.get("term_name")
                    or row.get("name")
                    or row.get("Path")
                    or row.get("Term")
                    or ""
                ).strip()
                if label:
                    available.append(label)
            return {
                "status": "not_found",
                "message": (
                    f"No KEGG pathway matching '{pathway_term}' was found for the selected genes. "
                    + (f"Top KEGG candidates: {', '.join(available)}." if available else "")
                ).strip(),
                "kegg_pathway_path": "",
                "genes": genes,
                "requested_pathway_term": str(pathway_term or "").strip(),
                "kegg_enrichr_results": preview_rows,
            }
        results = gget.enrichr(
            genes=genes,
            database="KEGG_2021_Human",
            species=species,
            kegg_out=output,
            kegg_rank=selected_rank,
            json=True,
            verbose=False,
        )
    except Exception as exc:
        return {
            "status": "run_failed",
            "message": f"KEGG pathway visualization failed: {exc}",
            "kegg_pathway_path": "",
        }

    return {
        "status": "ok",
        "message": "Built KEGG pathway visualization.",
        "kegg_pathway_path": output,
        "kegg_enrichr_results": results if isinstance(results, list) else [],
        "genes": genes,
        "kegg_rank": selected_rank,
        "requested_pathway_term": str(pathway_term or "").strip(),
        "matched_pathway": selected_pathway if isinstance(selected_pathway, dict) else {},
    }


def build_volcano_plot(
    deg_rows: list[dict[str, Any]],
    *,
    output_path: str = "deg_volcano.html",
    pvalue_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
) -> dict[str, Any]:
    if not isinstance(deg_rows, list) or not deg_rows:
        return {
            "status": "missing_deg_rows",
            "message": "No DEG rows are available for volcano plotting.",
            "volcano_plot_path": "",
        }

    frame = pd.DataFrame(deg_rows)
    if frame.empty or "log2FoldChange" not in frame.columns or "pvalue" not in frame.columns:
        return {
            "status": "invalid_deg_rows",
            "message": "DEG rows are missing log2FoldChange or pvalue columns.",
            "volcano_plot_path": "",
        }

    frame["log2FoldChange"] = pd.to_numeric(frame["log2FoldChange"], errors="coerce")
    frame["pvalue"] = pd.to_numeric(frame["pvalue"], errors="coerce")
    frame = frame.dropna(subset=["log2FoldChange", "pvalue"]).copy()
    if frame.empty:
        return {
            "status": "invalid_deg_rows",
            "message": "No DEG rows contain numeric log2FoldChange and pvalue values.",
            "volcano_plot_path": "",
        }

    frame["neg_log10_p"] = -(frame["pvalue"].clip(lower=1e-300)).map(lambda value: float(math.log10(value)))
    frame["gene_label"] = frame.apply(_deg_gene_label, axis=1)
    frame["direction"] = "neutral"
    frame.loc[
        (frame["log2FoldChange"] >= float(log2fc_threshold)) & (frame["pvalue"] <= float(pvalue_threshold)),
        "direction",
    ] = "up"
    frame.loc[
        (frame["log2FoldChange"] <= -float(log2fc_threshold)) & (frame["pvalue"] <= float(pvalue_threshold)),
        "direction",
    ] = "down"

    try:
        from pyvis.network import Network
    except ImportError as exc:  # pragma: no cover - dependency guard
        return {
            "status": "dependency_missing",
            "message": f"pyvis is required for volcano visualization: {exc}",
            "volcano_plot_path": "",
        }

    colors = {"up": "#dc2626", "down": "#2563eb", "neutral": "#94a3b8"}
    direction_counts = {
        label: int((frame["direction"] == label).sum())
        for label in ("up", "down", "neutral")
    }

    output = str(Path(output_path).with_suffix(".html").resolve())
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    x_abs = max(
        float(frame["log2FoldChange"].abs().max()),
        abs(float(log2fc_threshold)) * 1.25,
        1.0,
    )
    ymax = float(frame["neg_log10_p"].max()) if not frame.empty else 0.0
    y_max = max(ymax, -float(math.log10(float(pvalue_threshold))) * 1.25, 1.0)
    canvas_left = -500.0
    canvas_right = 500.0
    canvas_top = -330.0
    canvas_bottom = 330.0

    def scale_x(value: Any) -> float:
        return (float(value) / x_abs) * canvas_right

    def scale_y(value: Any) -> float:
        return canvas_bottom - (float(value) / y_max) * (canvas_bottom - canvas_top)

    labeled = frame[
        (frame["direction"] != "neutral") & frame["gene_label"].astype(str).str.strip().ne("")
    ].copy()
    labeled = labeled.assign(abs_log2fc=frame["log2FoldChange"].abs())
    labeled = labeled.sort_values(["neg_log10_p", "abs_log2fc"], ascending=[False, False])
    labeled_genes = set(labeled["gene_label"].astype(str).str.upper().head(30))

    net = Network(
        height="760px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#0f172a",
        directed=False,
    )

    def add_line(line_id: str, x1: float, y1: float, x2: float, y2: float, *, color: str, dashes: bool = False) -> None:
        source = f"{line_id}_a"
        target = f"{line_id}_b"
        invisible = {
            "background": "rgba(255,255,255,0)",
            "border": "rgba(255,255,255,0)",
            "highlight": {
                "background": "rgba(255,255,255,0)",
                "border": "rgba(255,255,255,0)",
            },
        }
        net.add_node(source, label=" ", x=x1, y=y1, fixed=True, physics=False, size=1, color=invisible)
        net.add_node(target, label=" ", x=x2, y=y2, fixed=True, physics=False, size=1, color=invisible)
        net.add_edge(source, target, color=color, width=1, dashes=dashes, physics=False, smooth=False)

    def add_text(label_id: str, label: str, x: float, y: float, *, size: int = 15, color: str = "#334155") -> None:
        net.add_node(
            label_id,
            label=label,
            x=x,
            y=y,
            fixed=True,
            physics=False,
            shape="text",
            font={"size": size, "color": color, "face": "Arial", "strokeWidth": 3, "strokeColor": "#ffffff"},
        )

    def add_tick(line_id: str, x: float, y: float, *, orientation: str) -> None:
        if orientation == "x":
            add_line(line_id, x, y - 5, x, y + 5, color="#cbd5e1")
        else:
            add_line(line_id, x - 5, y, x + 5, y, color="#cbd5e1")

    threshold_y = scale_y(-float(math.log10(float(pvalue_threshold))))
    positive_threshold_x = scale_x(float(log2fc_threshold))
    negative_threshold_x = scale_x(-float(log2fc_threshold))
    zero_x = scale_x(0)
    zero_y = scale_y(0)
    add_line("x_axis", canvas_left, scale_y(0), canvas_right, scale_y(0), color="#cbd5e1")
    add_line("y_axis", scale_x(0), canvas_bottom, scale_x(0), canvas_top, color="#cbd5e1")
    add_line(
        "pvalue_threshold",
        canvas_left,
        threshold_y,
        canvas_right,
        threshold_y,
        color="#64748b",
        dashes=True,
    )
    add_line(
        "positive_log2fc_threshold",
        positive_threshold_x,
        canvas_bottom,
        positive_threshold_x,
        canvas_top,
        color="#64748b",
        dashes=True,
    )
    add_line(
        "negative_log2fc_threshold",
        negative_threshold_x,
        canvas_bottom,
        negative_threshold_x,
        canvas_top,
        color="#64748b",
        dashes=True,
    )

    x_ticks = sorted({-x_abs, -float(log2fc_threshold), 0.0, float(log2fc_threshold), x_abs})
    for index, tick in enumerate(x_ticks):
        tick_x = scale_x(tick)
        add_tick(f"x_tick_{index}", tick_x, zero_y, orientation="x")
        add_text(f"x_tick_label_{index}", f"{tick:.2g}", tick_x, zero_y + 28, size=13)

    y_tick_values = sorted({0.0, -float(math.log10(float(pvalue_threshold))), y_max})
    for index, tick in enumerate(y_tick_values):
        tick_y = scale_y(tick)
        add_tick(f"y_tick_{index}", zero_x, tick_y, orientation="y")
        add_text(f"y_tick_label_{index}", f"{tick:.2g}", zero_x - 34, tick_y, size=13)

    add_text("x_axis_label", "log2 Fold Change", 0, canvas_bottom + 58, size=17, color="#0f172a")
    add_text("y_axis_label", "-log10(p-value)", canvas_left - 55, scale_y(y_max / 2), size=17, color="#0f172a")
    add_text(
        "positive_threshold_label",
        f"log2FC >= {float(log2fc_threshold):.4g}",
        positive_threshold_x + 62,
        canvas_top + 34,
        size=14,
        color="#475569",
    )
    add_text(
        "negative_threshold_label",
        f"log2FC <= -{float(log2fc_threshold):.4g}",
        negative_threshold_x - 68,
        canvas_top + 34,
        size=14,
        color="#475569",
    )
    add_text(
        "pvalue_threshold_label",
        f"p <= {float(pvalue_threshold):.4g}",
        canvas_right - 64,
        threshold_y - 18,
        size=14,
        color="#475569",
    )

    for index, row in frame.reset_index(drop=True).iterrows():
        gene_label = str(row["gene_label"] or f"gene_{index + 1}").strip()
        direction = str(row["direction"])
        log2fc = float(row["log2FoldChange"])
        neg_log10_p = float(row["neg_log10_p"])
        pvalue = float(row["pvalue"])
        is_labeled = gene_label.upper() in labeled_genes
        label = gene_label if is_labeled else " "
        size = 9 if direction == "neutral" else 14
        if is_labeled:
            size = 18
        title = (
            f"{escape(gene_label)}; log2FoldChange: {log2fc:.4g}; p-value: {pvalue:.4g}; -log10(p): {neg_log10_p:.4g}; class: {escape(direction)}"
        )
        net.add_node(
            f"gene_{index}",
            label=label,
            title=title,
            x=scale_x(log2fc),
            y=scale_y(neg_log10_p),
            fixed=True,
            physics=False,
            shape="dot",
            size=size,
            color={
                "background": colors.get(direction, colors["neutral"]),
                "border": "#334155" if is_labeled else colors.get(direction, colors["neutral"]),
                "highlight": {
                    "background": "#f59e0b",
                    "border": "#0f172a",
                },
            },
            font={"size": 16 if is_labeled else 0, "vadjust": -18},
            borderWidth=2 if is_labeled else 0,
        )

    net.set_options(
        """
        {
          "layout": {"improvedLayout": false},
          "interaction": {
            "hover": true,
            "tooltipDelay": 80,
            "navigationButtons": true,
            "keyboard": true,
            "zoomView": true,
            "dragView": true
          },
          "physics": {"enabled": false},
          "nodes": {
            "scaling": {"min": 7, "max": 20},
            "font": {"face": "Arial", "color": "#0f172a", "strokeWidth": 3, "strokeColor": "#ffffff"}
          },
          "edges": {
            "color": {"inherit": false},
            "smooth": false,
            "selectionWidth": 1
          }
        }
        """
    )
    net.write_html(output)

    legend = f"""
<div class="volcano-overlay">
  <div class="volcano-title">Differential Expression Volcano Plot</div>
  <div class="volcano-axis volcano-x">log2 Fold Change</div>
  <div class="volcano-axis volcano-y">-log10(p-value)</div>
  <div class="volcano-legend">
    <span><i style="background:{colors['up']}"></i>Up ({direction_counts['up']})</span>
    <span><i style="background:{colors['down']}"></i>Down ({direction_counts['down']})</span>
    <span><i style="background:{colors['neutral']}"></i>Neutral ({direction_counts['neutral']})</span>
  </div>
  <div class="volcano-thresholds">p <= {float(pvalue_threshold):.4g}; |log2FC| >= {float(log2fc_threshold):.4g}</div>
</div>
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; background: #ffffff; }}
  #mynetwork {{ height: 760px !important; border: 0 !important; }}
  .volcano-overlay {{ pointer-events: none; position: fixed; inset: 0; color: #0f172a; }}
  .volcano-title {{ position: absolute; top: 14px; left: 22px; font-size: 18px; font-weight: 700; }}
  .volcano-axis {{ position: absolute; font-size: 13px; color: #334155; }}
  .volcano-x {{ left: 50%; bottom: 18px; transform: translateX(-50%); }}
  .volcano-y {{ top: 50%; left: -26px; transform: rotate(-90deg) translateX(-50%); transform-origin: left top; }}
  .volcano-legend {{ position: absolute; right: 18px; top: 16px; display: flex; gap: 12px; flex-wrap: wrap; justify-content: flex-end; max-width: 60%; font-size: 12px; }}
  .volcano-legend span {{ display: inline-flex; align-items: center; gap: 5px; background: rgba(255,255,255,.88); border: 1px solid #e2e8f0; border-radius: 999px; padding: 5px 8px; }}
  .volcano-legend i {{ width: 10px; height: 10px; border-radius: 999px; display: inline-block; }}
  .volcano-thresholds {{ position: absolute; left: 18px; bottom: 18px; font-size: 12px; color: #475569; background: rgba(255,255,255,.88); border: 1px solid #e2e8f0; border-radius: 999px; padding: 5px 8px; }}
</style>
"""
    html = Path(output).read_text(encoding="utf-8")
    Path(output).write_text(html.replace("</body>", f"{legend}</body>"), encoding="utf-8")

    return {
        "status": "ok",
        "message": "Built interactive DEG volcano plot.",
        "volcano_plot_path": output,
        "volcano_html_path": output,
        "points": int(len(frame)),
        "upregulated_points": direction_counts["up"],
        "downregulated_points": direction_counts["down"],
        "neutral_points": direction_counts["neutral"],
    }
