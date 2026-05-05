import networkx as nx

from gea_agent.tools.random_walk_restart import top_rwr_genes


def test_rwr_ranks_neighbor_higher():
    g = nx.Graph()
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("B", "C", weight=1.0)
    g.add_edge("A", "D", weight=0.1)

    top = top_rwr_genes(g, ["A"], top_k=3, restart_prob=0.5)
    genes = [x[0] for x in top]

    assert "B" in genes
    assert genes.index("B") < genes.index("D")