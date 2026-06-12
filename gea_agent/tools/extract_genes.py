import re

_GENE_TOKEN = re.compile(r"\b[a-zA-Z0-9-]{2,12}\b")

BLACKLIST = {
    "A",
    "AL",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BEEN",
    "BEING",
    "BY",
    "CDNA",
    "CELL",
    "CELLS",
    "CNV",
    "COVID",
    "DATA",
    "DNA",
    "EGG",
    "ET",
    "EXPRESSION",
    "FEMALE",
    "FIG",
    "FIGS",
    "FOR",
    "FROM",
    "GENE",
    "GENES",
    "GTP",
    "HUMAN",
    "I",
    "II",
    "III",
    "IN",
    "IS",
    "IV",
    "IX",
    "MALE",
    "MEN",
    "METHODS",
    "MRNA",
    "MOUSE",
    "MUTATION",
    "MUTATIONS",
    "NADH",
    "NGS",
    "NOT",
    "OF",
    "ON",
    "OR",
    "P",
    "PCR",
    "PROTEIN",
    "PROTEINS",
    "QPCR",
    "RAT",
    "RESULTS",
    "RNA",
    "RT",
    "SARS",
    "SNP",
    "TABLE",
    "TABLES",
    "THAT",
    "THE",
    "THESE",
    "THIS",
    "THOSE",
    "TISSUE",
    "TISSUES",
    "TO",
    "TRANSCRIPT",
    "TRANSCRIPTS",
    "V",
    "VARIANT",
    "VARIANTS",
    "VI",
    "VII",
    "VIII",
    "WAS",
    "WERE",
    "WITH",
    "WITHOUT",
    "WOMEN",
    "X",
}


def extract_genes_from_text(text: str) -> list[str]:
    try:
        tokens = _GENE_TOKEN.findall(text)
    except:
        raise ValueError

    genes = []
    for token in tokens:
        token_clean = token.upper()

        if token_clean in BLACKLIST:
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
