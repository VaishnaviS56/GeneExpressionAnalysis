from __future__ import annotations
import re
from pathlib import Path

# _GENE_TOKEN = re.compile(r"\b[a-zA-Z0-9-]{3,7}\b")

# BLACKLIST = {
#     "A",
#     "AL",
#     "AN",
#     "AND",
#     "ARE",
#     "AS",
#     "AT",
#     "BE",
#     "BEEN",
#     "BEING",
#     "BY",
#     "CDNA",
#     "CELL",
#     "CELLS",
#     "CNV",
#     "COVID",
#     "DATA",
#     "DNA",
#     "EGG",
#     "ET",
#     "EXPRESSION",
#     "FEMALE",
#     "FIG",
#     "FIGS",
#     "FOR",
#     "FROM",
#     "GENE",
#     "GENES",
#     "GTP",
#     "HUMAN",
#     "I",
#     "II",
#     "III",
#     "IN",
#     "IS",
#     "IV",
#     "IX",
#     "MALE",
#     "MEN",
#     "METHODS",
#     "MRNA",
#     "MOUSE",
#     "MUTATION",
#     "MUTATIONS",
#     "NADH",
#     "NGS",
#     "NOT",
#     "OF",
#     "ON",
#     "OR",
#     "P",
#     "PCR",
#     "PROTEIN",
#     "PROTEINS",
#     "QPCR",
#     "RAT",
#     "RESULTS",
#     "RNA",
#     "RT",
#     "SARS",
#     "SNP",
#     "TABLE",
#     "TABLES",
#     "THAT",
#     "THE",
#     "THESE",
#     "THIS",
#     "THOSE",
#     "TISSUE",
#     "TISSUES",
#     "TO",
#     "TRANSCRIPT",
#     "TRANSCRIPTS",
#     "V",
#     "VARIANT",
#     "VARIANTS",
#     "VI",
#     "VII",
#     "VIII",
#     "WAS",
#     "WERE",
#     "WITH",
#     "WITHOUT",
#     "WOMEN",
#     "X",
# }


# def extract_genes_from_text(text: str) -> list[str]:
#     try:
#         tokens = _GENE_TOKEN.findall(text)
#     except:
#         raise ValueError

#     genes = []
#     for token in tokens:
#         token_clean = token.upper()

#         if token_clean in BLACKLIST:
#             continue

#         # Heuristic: gene-like patterns
#         if not re.match(r"^[A-Z0-9-]+$", token_clean):
#             continue

#         # Avoid pure numbers
#         if token_clean.isdigit():
#             continue

#         if token_clean not in genes:
#             genes.append(token_clean)

#     return genes


_TOKEN_RE = re.compile(r"\b[A-Z0-9]{2,12}\b")
_HAS_LETTER_RE = re.compile(r"[A-Z]")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_ALL_DIGITS_RE = re.compile(r"^\d+$")

_GENE_WITH_DIGIT_RE = re.compile(r"\b[a-zA-Z0-9-]{3,7}\b")

# Capture alpha-only symbols in common textual contexts
_ALPHA_CONTEXT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bGENE\s+([A-Z]{2,10})\b"),
    re.compile(r"\b([A-Z]{2,10})\s+GENE\b"),
    re.compile(r"\b([A-Z]{2,10})\s*\("),
    re.compile(r"\b([A-Z]{2,10})\b\s*/\s*([A-Z]{2,10})\b"),
]

_STOPWORDS = {
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
    "OOCYTE",
    "OOCYTES",
    "OR",
    "OVARIES",
    "OVARY",
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
    "SPERM",
    "TABLE",
    "TABLES",
    "TESTES",
    "TESTIS",
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


def load_symbol_whitelist(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Gene symbols file not found: {p}")
    symbols: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        symbols.add(s.upper())
    return symbols or None


def _extract_alpha_symbols_from_context(up: str) -> set[str]:
    out: set[str] = set()
    for pat in _ALPHA_CONTEXT_PATTERNS:
        for m in pat.finditer(up):
            for g in m.groups():
                if not g:
                    continue
                sym = g.upper()
                if sym in _STOPWORDS:
                    continue
                out.add(sym)
    return out


def extract_genes_from_text(
    text: str | None,
    *,
    whitelist: set[str] | None = None,
    mode: str = "strict",
) -> list[str]:
    """Extract gene symbols from free text.

    Defaults to `strict` for precision.

    - If `whitelist` is provided, only whitelisted symbols are returned.
    - Without a whitelist:
      - strict: returns digit-containing symbols (e.g., BRCA1, TP53, IL6)
      - lenient: also returns alpha-only symbols found in symbol-like contexts
        (e.g., "SRY gene", "gene SRY", "SRY ( ... )", "SRY/FOXL2").

    For best results, provide an HGNC (or organism-specific) symbol whitelist.
    """
    if not text:
        return []

    mode = (mode or "strict").strip().lower()
    if mode not in {"strict", "lenient"}:
        raise ValueError("mode must be 'strict' or 'lenient'")

    hits: list[str] = []
    up = text.upper()

    alpha_context: set[str] = set()
    if whitelist is None and mode == "lenient":
        alpha_context = _extract_alpha_symbols_from_context(up)

    for m in _TOKEN_RE.finditer(up):
        sym = m.group(0)

        if sym in _STOPWORDS:
            continue
        if _ALL_DIGITS_RE.match(sym):
            continue
        if _YEAR_RE.match(sym):
            continue
        if not _HAS_LETTER_RE.search(sym):
            continue

        if whitelist is not None:
            if sym in whitelist:
                hits.append(sym)
            continue

        if _GENE_WITH_DIGIT_RE.match(sym):
            hits.append(sym)
            continue

        if mode == "lenient" and sym in alpha_context:
            hits.append(sym)

    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out

