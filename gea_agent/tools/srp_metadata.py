from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from io import StringIO
from typing import Any

from gea_agent.config import SETTINGS
from gea_agent.tools.http_utils import get_retrying_session
from gea_agent.tools.result_utils import sanitize_exception_message, tool_error_result
from gea_agent.tools.srp_ids import extract_srp_ids_from_text


DEE2_METADATA_URL = "https://www.dee2.io/metadata/{species}_metadata.tsv"
SRA_RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo"
BIOSAMPLE_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BIOSAMPLE_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _normalize_srp_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, str):
        raw_values = extract_srp_ids_from_text(value)
    else:
        raw_values = []

    normalized: list[str] = []
    for value in raw_values:
        srp_id = str(value or "").strip().upper()
        if re.fullmatch(r"SRP\d+", srp_id) and srp_id not in normalized:
            normalized.append(srp_id)
    return normalized


def _compact_text(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _row_value(row: dict[str, Any], *keys: str) -> str:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.strip().lower())
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _fetch_dee2_metadata_rows(
    srp_ids: list[str],
    *,
    species: str,
    max_rows: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    session = get_retrying_session()
    url = DEE2_METADATA_URL.format(species=species)
    response = session.get(url, stream=True, timeout=SETTINGS.http_timeout_seconds)
    if response.status_code >= 400:
        return [], {
            "status": "http_error",
            "url": url,
            "status_code": response.status_code,
        }

    target = set(srp_ids)
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    scanned_rows = 0
    truncated = False

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        if header is None:
            header = line.rstrip("\n\r").split("\t")
            continue
        scanned_rows += 1
        parts = line.rstrip("\n\r").split("\t")
        row = {
            key: parts[index] if index < len(parts) else ""
            for index, key in enumerate(header)
        }
        if str(row.get("SRP_accession") or "").strip().upper() not in target:
            continue
        rows.append(row)
        if len(rows) >= max_rows:
            truncated = True
            break

    return rows, {
        "status": "ok",
        "url": url,
        "scanned_rows": scanned_rows,
        "matched_rows": len(rows),
        "truncated": truncated,
        "columns": header or [],
    }


def _fetch_sra_runinfo(srp_id: str) -> list[dict[str, str]]:
    response = get_retrying_session().get(
        SRA_RUNINFO_URL,
        params={"acc": srp_id},
        timeout=SETTINGS.http_timeout_seconds,
    )
    if response.status_code >= 400 or not response.text.strip():
        return []
    return [dict(row) for row in csv.DictReader(StringIO(response.text))]


def _parse_biosample_attributes(xml_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return attrs

    for attribute in root.findall(".//Attribute"):
        name = (
            attribute.attrib.get("attribute_name")
            or attribute.attrib.get("display_name")
            or attribute.attrib.get("harmonized_name")
            or ""
        )
        value = " ".join("".join(attribute.itertext()).split())
        if not name or not value:
            continue
        normalized_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if normalized_name and normalized_name not in attrs:
            attrs[normalized_name] = value

    for sample_id in root.findall(".//Id"):
        label = (
            sample_id.attrib.get("db_label")
            or sample_id.attrib.get("db")
            or sample_id.attrib.get("namespace")
            or ""
        )
        value = " ".join("".join(sample_id.itertext()).split())
        if not label or not value:
            continue
        normalized_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        if normalized_label and normalized_label not in attrs:
            attrs[normalized_label] = value

    return attrs


def _fetch_biosample_attributes(biosample_accession: str) -> dict[str, str]:
    biosample = str(biosample_accession or "").strip()
    if not biosample:
        return {}

    search = get_retrying_session().get(
        BIOSAMPLE_ESEARCH_URL,
        params={
            "db": "biosample",
            "term": f"{biosample}[accn]",
            "retmode": "json",
        },
        timeout=SETTINGS.http_timeout_seconds,
    )
    if search.status_code >= 400:
        return {}
    try:
        payload = search.json()
        ids = payload.get("esearchresult", {}).get("idlist", [])
    except Exception:
        ids = []
    if not ids:
        return {}

    fetch = get_retrying_session().get(
        BIOSAMPLE_EFETCH_URL,
        params={
            "db": "biosample",
            "id": ids[0],
            "retmode": "xml",
        },
        timeout=SETTINGS.http_timeout_seconds,
    )
    if fetch.status_code >= 400:
        return {}
    return _parse_biosample_attributes(fetch.text)


def _first_present(*values: Any) -> str:
    for value in values:
        text = " ".join(str(value or "").split()).strip()
        if text:
            return text
    return ""


def _summarize_values(rows: list[dict[str, Any]], column: str, *, limit: int = 30) -> dict[str, Any]:
    counter = Counter(
        str(row.get(column) or "").strip()
        for row in rows
        if str(row.get(column) or "").strip()
    )
    values = [
        {"value": value, "count": count}
        for value, count in counter.most_common(limit)
    ]
    return {
        "column": column,
        "unique_count": len(counter),
        "values": values,
    }


def _format_metadata_answer(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return str(result.get("message") or "SRP metadata could not be retrieved.").strip()

    lines: list[str] = ["**SRP Metadata Preview**"]
    lines.append(
        "I found metadata that can help you choose the exact DEG comparison labels. "
        "Use one value from the listed fields as `control_name` and one as `test_name` when you run DEG analysis."
    )

    for srp_result in result.get("srp_metadata", []):
        if not isinstance(srp_result, dict):
            continue
        srp_id = str(srp_result.get("srp_id") or "").strip()
        lines.append("")
        lines.append(f"**{srp_id}**")
        lines.append(
            f"DEE2 rows: {srp_result.get('dee2_row_count', 0)}; "
            f"SRA runs: {srp_result.get('sra_run_count', 0)}"
        )
        geo_series = srp_result.get("geo_series")
        if isinstance(geo_series, list) and geo_series:
            lines.append("GEO series: " + ", ".join(str(value) for value in geo_series[:10]))

        descriptions = srp_result.get("descriptions")
        if isinstance(descriptions, list) and descriptions:
            lines.append("")
            lines.append("Descriptions:")
            for row in descriptions[:8]:
                if isinstance(row, dict):
                    text = str(row.get("description") or "").strip()
                    count = row.get("count")
                    if text:
                        lines.append(f"- {text}" + (f" ({count} run(s))" if count else ""))

        field_summaries = srp_result.get("field_summaries")
        if isinstance(field_summaries, dict):
            for field in ("treatment", "sample_name", "disease"):
                summary = field_summaries.get(field)
                if not isinstance(summary, dict):
                    continue
                values = summary.get("values")
                if not isinstance(values, list) or not values:
                    continue
                lines.append("")
                lines.append(f"`{field}` values:")
                for item in values[:20]:
                    if not isinstance(item, dict):
                        continue
                    value = str(item.get("value") or "").strip()
                    count = item.get("count")
                    if value:
                        lines.append(f"- {value}" + (f" ({count} run(s))" if count else ""))

        preview_rows = srp_result.get("metadata_preview")
        if isinstance(preview_rows, list) and preview_rows:
            lines.append("")
            lines.append("Metadata preview:")
            lines.append("| Run | BioSample | treatment | sample_name | disease |")
            lines.append("|---|---|---|---|---|")
            for row in preview_rows[:10]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        _compact_text(row.get(key), limit=70).replace("|", "\\|")
                        for key in ("run", "biosample", "treatment", "sample_name", "disease")
                    )
                    + " |"
                )

    lines.append("")
    lines.append("**Next Step**")
    lines.append("Send the exact control and test labels from the metadata values above, and I can run the DEE2/DESeq2 DEG workflow.")
    return "\n".join(lines).strip()


def fetch_srp_metadata_summary(
    *,
    srp_ids: list[str] | None = None,
    text: str | None = None,
    species: str = "hsapiens",
    max_dee2_rows: int = 5000,
    max_biosamples: int = 80,
) -> dict[str, Any]:
    resolved_srp_ids = _normalize_srp_ids(srp_ids or []) or _normalize_srp_ids(text or "")
    if not resolved_srp_ids:
        return {
            "status": "no_srp_ids",
            "analysis_arm": "srp_metadata",
            "srp_ids": [],
            "srp_metadata": [],
            "message": "No SRP IDs were provided for metadata discovery.",
            "answer": "No SRP IDs were provided for metadata discovery.",
            "should_finalize": True,
        }

    dee2_rows, dee2_status = _fetch_dee2_metadata_rows(
        resolved_srp_ids,
        species=str(species or "hsapiens").strip() or "hsapiens",
        max_rows=max(1, int(max_dee2_rows or 5000)),
    )

    dee2_by_srp: dict[str, list[dict[str, str]]] = defaultdict(list)
    dee2_by_run: dict[str, dict[str, str]] = {}
    for row in dee2_rows:
        srp = str(row.get("SRP_accession") or "").strip().upper()
        run = str(row.get("SRR_accession") or "").strip().upper()
        if srp:
            dee2_by_srp[srp].append(row)
        if run:
            dee2_by_run[run] = row

    srp_metadata: list[dict[str, Any]] = []
    for srp_id in resolved_srp_ids:
        srp_dee2_rows = dee2_by_srp.get(srp_id, [])
        dee2_runs = {
            str(row.get("SRR_accession") or "").strip().upper()
            for row in srp_dee2_rows
            if str(row.get("SRR_accession") or "").strip()
        }

        runinfo_rows = _fetch_sra_runinfo(srp_id)
        if dee2_runs:
            runinfo_rows = [
                row
                for row in runinfo_rows
                if str(row.get("Run") or "").strip().upper() in dee2_runs
            ]

        biosample_values = []
        for row in runinfo_rows:
            biosample = str(row.get("BioSample") or "").strip()
            if biosample and biosample not in biosample_values:
                biosample_values.append(biosample)
            if len(biosample_values) >= max(1, int(max_biosamples or 80)):
                break
        biosample_attrs = {
            biosample: _fetch_biosample_attributes(biosample)
            for biosample in biosample_values
        }

        metadata_rows: list[dict[str, Any]] = []
        for row in runinfo_rows:
            run = str(row.get("Run") or "").strip().upper()
            biosample = str(row.get("BioSample") or "").strip()
            attrs = biosample_attrs.get(biosample, {})
            dee2_row = dee2_by_run.get(run, {})
            metadata_rows.append(
                {
                    "run": run,
                    "biosample": biosample,
                    "geo_series": _row_value(dee2_row, "GEO_series"),
                    "qc_summary": _row_value(dee2_row, "QC_summary"),
                    "description": _first_present(
                        _row_value(dee2_row, "experiment_title"),
                        _row_value(dee2_row, "sample_title"),
                        _row_value(dee2_row, "study_title"),
                        row.get("Experiment"),
                    ),
                    "treatment": _first_present(
                        _row_value(dee2_row, "treatment"),
                        row.get("treatment"),
                        attrs.get("treatment"),
                        attrs.get("treatment_protocol"),
                    ),
                    "sample_name": _first_present(
                        _row_value(dee2_row, "sample_name"),
                        _row_value(dee2_row, "sample"),
                        _row_value(dee2_row, "source_name"),
                        row.get("SampleName"),
                        row.get("sample_name"),
                        attrs.get("sample_name"),
                        attrs.get("sample"),
                        attrs.get("source_name"),
                    ),
                    "disease": _first_present(
                        _row_value(dee2_row, "disease"),
                        _row_value(dee2_row, "disease_state"),
                        row.get("disease"),
                        attrs.get("disease"),
                        attrs.get("disease_state"),
                        attrs.get("phenotype"),
                        attrs.get("diagnosis"),
                    ),
                    "source_name": _first_present(attrs.get("source_name"), row.get("LibraryName")),
                    "biosample_attributes": attrs,
                }
            )

        description_counts = Counter(
            row["description"]
            for row in metadata_rows
            if str(row.get("description") or "").strip()
        )
        if not description_counts:
            description_counts = Counter(
                _first_present(
                    _row_value(row, "experiment_title"),
                    _row_value(row, "sample_title"),
                    _row_value(row, "study_title"),
                )
                for row in srp_dee2_rows
                if _first_present(
                    _row_value(row, "experiment_title"),
                    _row_value(row, "sample_title"),
                    _row_value(row, "study_title"),
                )
            )

        srp_metadata.append(
            {
                "srp_id": srp_id,
                "dee2_row_count": len(srp_dee2_rows),
                "sra_run_count": len(runinfo_rows),
                "bio_sample_count": len(biosample_values),
                "bio_sample_fetch_limit": int(max_biosamples or 80),
                "geo_series": sorted(
                    {
                        str(row.get("GEO_series") or "").strip()
                        for row in srp_dee2_rows
                        if str(row.get("GEO_series") or "").strip()
                    }
                ),
                "descriptions": [
                    {"description": _compact_text(value, limit=260), "count": count}
                    for value, count in description_counts.most_common(20)
                ],
                "field_summaries": {
                    "treatment": _summarize_values(metadata_rows, "treatment"),
                    "sample_name": _summarize_values(metadata_rows, "sample_name"),
                    "disease": _summarize_values(metadata_rows, "disease"),
                },
                "metadata_preview": [
                    {
                        key: row.get(key, "")
                        for key in ("run", "biosample", "description", "treatment", "sample_name", "disease")
                    }
                    for row in metadata_rows[:50]
                ],
            }
        )

    status = "ok" if any(row.get("dee2_row_count") or row.get("sra_run_count") for row in srp_metadata) else "not_found"
    result: dict[str, Any] = {
        "status": status,
        "analysis_arm": "srp_metadata",
        "srp_ids": resolved_srp_ids,
        "species": species,
        "dee2_source": dee2_status,
        "srp_metadata": srp_metadata,
        "message": (
            "Fetched SRP metadata for cohort-label discovery."
            if status == "ok"
            else "No matching DEE2/SRA metadata rows were found for the supplied SRP IDs."
        ),
        "should_finalize": True,
    }
    result["answer"] = _format_metadata_answer(result)
    return result


def fetch_srp_metadata_summary_safe(
    *,
    srp_ids: list[str] | None = None,
    text: str | None = None,
    species: str = "hsapiens",
    max_dee2_rows: int = 5000,
    max_biosamples: int = 80,
) -> dict[str, Any]:
    try:
        return fetch_srp_metadata_summary(
            srp_ids=srp_ids,
            text=text,
            species=species,
            max_dee2_rows=max_dee2_rows,
            max_biosamples=max_biosamples,
        )
    except Exception as exc:
        return tool_error_result(
            "srp_metadata",
            f"SRP metadata discovery failed: {sanitize_exception_message(exc)}",
            analysis_arm="srp_metadata",
            srp_ids=_normalize_srp_ids(srp_ids or []) or _normalize_srp_ids(text or ""),
            srp_metadata=[],
            should_finalize=True,
        )
