from __future__ import annotations

import json
from typing import Any

import networkx as nx

from gea_agent.tools.llm import get_llm


def synthesize_technical_response(
    *,
    user_query: str,
    seed_genes: list[str],
    rwr_genes: list[tuple[str, float]],
    graph: nx.Graph,
    enrichr: dict[str, Any],
) -> str:
    llm = get_llm()

    payload = {
        "user_query": user_query,
        "seed_genes": seed_genes,
        "network": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()},
        "rwr_top_genes": rwr_genes,
        "enrichr": enrichr,
    }

    resp = llm.invoke(
        [
            (
                "system",
                "You are a bioinformatics assistant. Produce a clear, intuitive chat-friendly report.\n"
                "Use markdown with short sections and bullet points.\n\n"
                "Sections:\n"
                "- Detected genes (seed genes)\n"
                "- STRING network summary (nodes, edges)\n"
                "- Random Walk with Restart: top additional genes (show top 20 with scores)\n"
                "- Pathway enrichment (Enrichr via gget): for each library, list top terms with adjusted p-values\n\n"
                "Be explicit about what you did. If the network or enrichment is empty, explain why and suggest fixes.",
            ),
            ("user", json.dumps(payload, ensure_ascii=False)),
        ]
    )
    return getattr(resp, "content", "") or ""