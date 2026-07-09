from __future__ import annotations

from collections import OrderedDict
from typing import Any

from gea_agent.config import SETTINGS
from gea_agent.tools.http_utils import get_retrying_session


L1000CDS2_QUERY_URL = "https://maayanlab.cloud/L1000CDS2/query"


def _unique_upper(items: list[str] | None) -> list[str]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for item in items or []:
        value = str(item or "").strip().upper()
        if value:
            ordered[value] = None
    return list(ordered.keys())


def _normalize_overlap(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, genes in value.items():
        if isinstance(genes, str):
            normalized[str(key)] = [g.strip().upper() for g in genes.split(",") if g.strip()]
        elif isinstance(genes, list):
            normalized[str(key)] = [str(g).strip().upper() for g in genes if str(g).strip()]
    return normalized


def _normalize_signatures(
    rows: Any,
    *,
    requested_cell_lines: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []

    requested = {cell.upper() for cell in requested_cell_lines if str(cell).strip()}
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        cell_line = str(row.get("cell_id") or "").strip().upper()
        if requested and cell_line not in requested:
            continue

        normalized.append(
            {
                "rank": index,
                "score": row.get("score"),
                "perturbation": str(row.get("pert_desc") or "").strip(),
                "pert_id": row.get("pert_id"),
                "pubchem_id": row.get("pubchem_id"),
                "drugbank_id": row.get("drugchem_id"),
                "cell_line": cell_line,
                "dose": row.get("pert_dose"),
                "dose_unit": row.get("pert_dose_unit"),
                "time": row.get("pert_time"),
                "time_unit": row.get("pert_time_unit"),
                "sig_id": row.get("sig_id"),
                "overlap": _normalize_overlap(row.get("overlap")),
            }
        )
    return normalized


def _rank_drugs(signatures: list[dict[str, Any]], *, result_limit: int) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in signatures:
        name = str(row.get("perturbation") or "").strip()
        if not name:
            continue
        key = name.upper()
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = {
                "name": name,
                "best_rank": row.get("rank"),
                "best_score": row.get("score"),
                "pert_id": row.get("pert_id"),
                "pubchem_id": row.get("pubchem_id"),
                "drugbank_id": row.get("drugbank_id"),
                "cell_lines": [row.get("cell_line")] if row.get("cell_line") else [],
                "signature_count": 1,
                "example_signature": {
                    "sig_id": row.get("sig_id"),
                    "dose": row.get("dose"),
                    "dose_unit": row.get("dose_unit"),
                    "time": row.get("time"),
                    "time_unit": row.get("time_unit"),
                },
            }
            continue

        existing["signature_count"] = int(existing.get("signature_count") or 0) + 1
        cell_line = str(row.get("cell_line") or "").strip()
        if cell_line and cell_line not in existing["cell_lines"]:
            existing["cell_lines"].append(cell_line)

    return list(grouped.values())[: max(1, int(result_limit))]


def query_l1000cds2(
    *,
    up_genes: list[str],
    down_genes: list[str],
    cell_lines: list[str] | None = None,
    aggravate: bool = False,
    combination: bool = False,
    share: bool = False,
    db_version: str = "latest",
    result_limit: int = 20,
    timeout: int | None = None,
) -> dict[str, Any]:
    up = _unique_upper(up_genes)
    down = _unique_upper(down_genes)
    cells = _unique_upper(cell_lines)

    if not up or not down:
        return {
            "status": "missing_input",
            "message": "Both up-regulated and down-regulated gene lists are required for L1000CDS2 gene-set search.",
            "up_genes": up,
            "down_genes": down,
            "requested_cell_lines": cells,
            "top_drugs": [],
            "top_signatures": [],
        }

    payload = {
        "data": {
            # "upGenes": up,
            # "dnGenes": down,
            "upGenes": down,
            "dnGenes": up,
        },
        "config": {
            "aggravate": bool(aggravate),
            "searchMethod": "geneSet",
            "share": bool(share),
            "combination": bool(combination),
            "db-version": str(db_version or "latest"),
        },
        "meta": [{"key": "Cell", "value": cell} for cell in cells],
    }

    session = get_retrying_session()
    response = session.post(
        L1000CDS2_QUERY_URL,
        json=payload,
        timeout=int(timeout or SETTINGS.http_timeout_seconds),
    )
    try:
        response.raise_for_status()
    except Exception as exc:
        return {
            "status": "http_error",
            "message": f"L1000CDS2 request failed: {exc}",
            "http_status": getattr(response, "status_code", None),
            "up_genes": up,
            "down_genes": down,
            "requested_cell_lines": cells,
            "top_drugs": [],
            "top_signatures": [],
        }

    try:
        raw = response.json()
    except Exception as exc:
        return {
            "status": "parse_error",
            "message": f"L1000CDS2 returned a non-JSON response: {exc}",
            "http_status": getattr(response, "status_code", None),
            "up_genes": up,
            "down_genes": down,
            "requested_cell_lines": cells,
            "top_drugs": [],
            "top_signatures": [],
        }
    
    print(raw)

    signatures = _normalize_signatures(raw.get("topMeta"), requested_cell_lines=cells)
    top_drugs = _rank_drugs(signatures, result_limit=result_limit)
    if cells and not signatures:
        message = "L1000CDS2 returned results, but none matched the requested cell-line filter."
    elif top_drugs:
        message = "Retrieved L1000CDS2 drug matches."
    else:
        message = "L1000CDS2 returned no drug matches."

    return {
        "status": "ok" if isinstance(raw, dict) else "error",
        "message": message,
        "query_url": L1000CDS2_QUERY_URL,
        "mode": "mimic" if aggravate else "reverse",
        "up_genes": up,
        "down_genes": down,
        "up_gene_count": len(up),
        "down_gene_count": len(down),
        "requested_cell_lines": cells,
        "cell_line_filter_applied": bool(cells),
        "share_id": raw.get("shareId"),
        "top_drugs": top_drugs,
        "top_signatures": signatures[:50],
        "signature_count": len(signatures),
        "combinations": raw.get("combinations") if isinstance(raw.get("combinations"), list) else [],
        "raw_result": raw,
    }
