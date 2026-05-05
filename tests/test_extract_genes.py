from gea_agent.tools.extract_genes import extract_genes_from_text


def test_extract_genes_basic():
    genes = extract_genes_from_text("Compare TP53 and EGFR expression in tumor vs normal.")
    assert "TP53" in genes
    assert "EGFR" in genes


def test_extract_genes_filters_common_tokens():
    genes = extract_genes_from_text("How and why is DNA repair important?")
    assert "DNA" not in genes

