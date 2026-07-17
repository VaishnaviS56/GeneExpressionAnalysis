from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
    ymax = float(frame["neg_log10_p"].max()) if not frame.empty else 0.0
    threshold_y = ymax * 0.98 if ymax > 0 else 0.1
    ax.text(
        float(log2fc_threshold),
        threshold_y,
        f"log2FC = {float(log2fc_threshold):.2f}",
        color="#000000",
        fontsize=8,
        ha="left",
        va="top",
        rotation=90,
    )
    ax.text(
        -float(log2fc_threshold),
        threshold_y,
        f"log2FC = -{float(log2fc_threshold):.2f}",
        color="#000000",
        fontsize=8,
        ha="right",
        va="top",
        rotation=90,
    )

    labeled = frame[
        (frame["direction"] != "neutral") & frame["gene_label"].astype(str).str.strip().ne("")
    ].copy()
    labeled = labeled.sort_values(["neg_log10_p", "log2FoldChange"], ascending=[False, False])
    for _, row in labeled.iterrows():
        x = float(row["log2FoldChange"])
        y = float(row["neg_log10_p"])
        dx = 3 if x >= 0 else -3
        ha = "left" if x >= 0 else "right"
        ax.annotate(
            row["gene_label"],
            xy=(x, y),
            xytext=(dx, 3),
            textcoords="offset points",
            fontsize=7,
            color="#000000",
            ha=ha,
            va="bottom",
        )

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
