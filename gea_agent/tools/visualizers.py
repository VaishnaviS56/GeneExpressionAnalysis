from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from gea_agent.tools.pyvis_visualizer import build_pyvis_html


def build_network_visualization(
    graph,
    *,
    output_path: str = "pyvis_network.html",
    select_top_degree: int = 300,
    allowed_nodes: list[str] | None = None,
) -> dict[str, Any]:
    if graph is None or getattr(graph, "number_of_nodes", lambda: 0)() == 0:
        return {
            "status": "missing_graph",
            "message": "No graph is available for network visualization.",
            "pyvis_html_path": "",
        }

    if isinstance(allowed_nodes, list) and allowed_nodes:
        keep = {str(node).strip().upper() for node in allowed_nodes if str(node).strip()}
        mapped_keep = {node for node in graph.nodes() if str(node).strip().upper() in keep}
        if mapped_keep:
            graph = graph.subgraph(mapped_keep).copy()
            select_top_degree = max(len(mapped_keep), 1)
        if graph.number_of_nodes() == 0:
            return {
                "status": "missing_graph",
                "message": "No matching nodes were available for network visualization.",
                "pyvis_html_path": "",
            }

    html_path = build_pyvis_html(
        graph,
        output_path=output_path,
        select_top_degree=select_top_degree,
    )
    return {
        "status": "ok",
        "message": "Built PyVis network visualization.",
        "pyvis_html_path": html_path,
    }


def build_kegg_pathway_visualization(
    genes: list[str],
    *,
    output_path: str = "kegg_pathway.png",
    kegg_rank: int = 1,
    species: str = "human",
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
        results = gget.enrichr(
            genes=genes,
            database="KEGG_2021_Human",
            species=species,
            kegg_out=output,
            kegg_rank=int(kegg_rank),
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
        "kegg_rank": int(kegg_rank),
    }


def build_volcano_plot(
    deg_rows: list[dict[str, Any]],
    *,
    output_path: str = "deg_volcano.png",
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
    frame["direction"] = "neutral"
    frame.loc[
        (frame["log2FoldChange"] >= float(log2fc_threshold)) & (frame["pvalue"] <= float(pvalue_threshold)),
        "direction",
    ] = "up"
    frame.loc[
        (frame["log2FoldChange"] <= -float(log2fc_threshold)) & (frame["pvalue"] <= float(pvalue_threshold)),
        "direction",
    ] = "down"

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"up": "#dc2626", "down": "#2563eb", "neutral": "#94a3b8"}
    for label in ("neutral", "up", "down"):
        subset = frame[frame["direction"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["log2FoldChange"],
            subset["neg_log10_p"],
            s=18,
            alpha=0.75,
            c=colors[label],
            label=label,
        )

    ax.axvline(float(log2fc_threshold), color="#64748b", linestyle="--", linewidth=1)
    ax.axvline(-float(log2fc_threshold), color="#64748b", linestyle="--", linewidth=1)
    ax.axhline(-float(math.log10(float(pvalue_threshold))), color="#64748b", linestyle="--", linewidth=1)
    ax.set_title("Differential Expression Volcano Plot")
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(p-value)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)

    output = str(Path(output_path).resolve())
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "ok",
        "message": "Built DEG volcano plot.",
        "volcano_plot_path": output,
        "points": int(len(frame)),
    }
