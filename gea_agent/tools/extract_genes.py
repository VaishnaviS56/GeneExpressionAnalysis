import re

_GENE_TOKEN = re.compile(r"\b[a-zA-Z0-9-]{2,15}\b")

BLACKLIST = {
    "and", "or", "the", "with", "from", "this", "that",
    "what", "when", "why", "how", "gene", "genes",
    "rna", "dna", "gsea", "go", "kegg"
}


def extract_genes_from_text(text: str) -> list[str]:
    tokens = _GENE_TOKEN.findall(text or "")

    genes = []
    for token in tokens:
        token_clean = token.upper()

        if token_clean.lower() in BLACKLIST:
            continue

        # Heuristic: gene-like patterns
        if not re.match(r"^[A-Z0-9-]+$", token_clean):
            continue

        # Avoid pure numbers
        if token_clean.isdigit():
            continue

        if token_clean not in genes:
            genes.append(token_clean)

    return genes
