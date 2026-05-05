from gea_agent.tools.build_graph import build_weighted_graph_from_string_edges


def test_build_graph_adds_weighted_edges():
    graph = build_weighted_graph_from_string_edges(
        [
            {"preferredName_A": "TP53", "preferredName_B": "EGFR", "score": 0.9},
            {"preferredName_A": "TP53", "preferredName_B": "EGFR", "score": 0.8},
        ]
    )
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph["TP53"]["EGFR"]["weight"] == 0.9

