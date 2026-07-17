from __future__ import annotations

import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any
import os

from gea_agent.config import SETTINGS


_DEG_COLUMNS = [
    "Ensembl",
    "hgnc_symbol",
    "entrezgene_id",
    "entrezgene_accession",
    "external_gene_name",
    "description",
    "log2FoldChange",
    "pvalue",
    "pdj",
]

_SAVED_DEG_COLUMNS = [
    "hgnc_symbol",
    "entrezgene_id",
    "entrezgene_accession",
    "description",
    "log2FoldChange",
    "pvalue",
]

_DEG_NUMERIC_COLUMNS = {"log2FoldChange"}


def _placeholder_path(value: str) -> bool:
    return "PLACEHOLDER" in value.upper()


def _normalize_windows_env_path(value: str) -> str:
    cleaned = value.strip().strip('"').strip("'")
    cleaned = cleaned.replace("\x08", "\\")
    return cleaned


def _resolve_rscript_executable(value: str) -> str | None:
    value = _normalize_windows_env_path(value)
    candidate = Path(value)
    if candidate.exists():
        return str(candidate)

    if candidate.suffix.lower() == ".exe" and candidate.name.lower() == "r.exe":
        sibling = candidate.with_name("Rscript.exe")
        if sibling.exists():
            return str(sibling)

    resolved = shutil.which(value)
    if resolved:
        return resolved

    return None


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        clean_key = str(key).strip()
        clean_value = "" if value is None else str(value).strip()
        normalized[clean_key] = clean_value
    if "pdj" not in normalized and "padj" in normalized:
        normalized["pdj"] = normalized.get("padj", "")
    return normalized


def _truncate_decimal_text(value: Any, places: int = 3) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    factor = 10**places
    truncated = math.trunc(number * factor) / factor
    return f"{truncated:.{places}f}"


def _format_deg_numeric_columns(row: dict[str, str]) -> dict[str, str]:
    formatted = dict(row)
    for column in _DEG_NUMERIC_COLUMNS:
        if column in formatted:
            formatted[column] = _truncate_decimal_text(formatted.get(column, ""))
    return formatted


def _write_clean_deg_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SAVED_DEG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in _SAVED_DEG_COLUMNS})


def _gene_label(row: dict[str, str]) -> str:
    for key in ("hgnc_symbol", "external_gene_name", "Ensembl", "entrezgene_accession"):
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def _read_deg_csv(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {
            "status": "missing_output",
            "output_csv_path": str(csv_path),
            "genes": [],
            "rows": [],
            "message": "DEG output CSV was not found.",
        }

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        genes: list[str] = []

        for raw_row in reader:
            row = _format_deg_numeric_columns(_normalize_row(raw_row))
            rows.append({column: row.get(column, "") for column in _DEG_COLUMNS})

            label = _gene_label(row)
            if label and label not in genes:
                genes.append(label)

    _write_clean_deg_csv(csv_path, rows)

    return {
        "status": "ok",
        "output_csv_path": str(csv_path),
        "genes": genes,
        "rows": rows,
        "message": f"Loaded {len(rows)} DEG rows.",
    }


def run_deg_r_analysis(
    *,
    srp_ids: list[str] | None = None,
    control_name: str | None = None,
    test_name: str | None = None,
    log2fold: float = 1.0,
    padj: float = 0.05,
) -> dict[str, Any]:
    """
    Run the hard-coded R DEG workflow and parse the resulting CSV.

    The script path, supporting files folder, and output CSV are configured in
    `gea_agent.config.Settings` so they can be filled in later without changing
    the code.
    """
    output_path = Path(SETTINGS.deg_output_csv_path)
    script_path = Path(SETTINGS.deg_r_script_path)
    supporting_dir = Path(SETTINGS.deg_supporting_files_dir)
    executable = _resolve_rscript_executable(SETTINGS.rscript_executable)

    script_path = Path(os.path.join(os.getcwd(), script_path))
    supporting_dir = Path(os.path.join(os.getcwd(), supporting_dir))
    output_path = Path(os.path.join(os.getcwd(), output_path))

    if _placeholder_path(SETTINGS.deg_r_script_path) or _placeholder_path(SETTINGS.deg_output_csv_path):
        return {
            "status": "not_configured",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "log2fold": float(log2fold),
            "padj": float(padj),
            "genes": [],
            "rows": [],
            "message": "DEG R paths are still placeholders.",
        }

    if not srp_ids:
        return {
            "status": "no_srp_ids",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "log2fold": float(log2fold),
            "padj": float(padj),
            "genes": [],
            "rows": [],
            "message": "No SRP ids were provided to the DEG runner.",
        }

    control_name = " ".join(str(control_name or "").split()).strip()
    test_name = " ".join(str(test_name or "").split()).strip()
    if not control_name or not test_name:
        return {
            "status": "missing_groups",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "log2fold": float(log2fold),
            "padj": float(padj),
            "genes": [],
            "rows": [],
            "message": "Both control_name and test_name are required for the DEG runner.",
        }

    if not executable:
        return {
            "status": "rscript_missing",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "log2fold": float(log2fold),
            "padj": float(padj),
            "genes": [],
            "rows": [],
            "message": f"Could not resolve the R executable from: {SETTINGS.rscript_executable}",
        }

    if not script_path.exists():
        return {
            "status": "missing_script",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "log2fold": float(log2fold),
            "padj": float(padj),
            "genes": [],
            "rows": [],
            "message": "DEG R script was not found.",
        }

    supporting_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(srp_ids)
        print(
            f"[tool] DEG subprocess: {executable} {script_path} "
            f"\"{control_name}\" \"{test_name}\" {' '.join(str(s) for s in srp_ids or [])}"
        )
        subprocess.run(
            [executable, str(script_path), control_name, test_name, str(log2fold), str(padj), *[str(s) for s in srp_ids or []]],
            cwd=str(supporting_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return {
            "status": "rscript_missing",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "log2fold": float(log2fold),
            "padj": float(padj),
            "genes": [],
            "rows": [],
            "message": f"Could not find the Rscript executable: {exc}",
        }
    except subprocess.CalledProcessError as exc:
        print("[tool] DEG subprocess failed")
        print("[tool] stdout:")
        print(exc.stdout or "")
        print("[tool] stderr:")
        print(exc.stderr or "")

        return {
            "status": "run_failed",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "log2fold": float(log2fold),
            "padj": float(padj),
            "genes": [],
            "rows": [],
            "message": (exc.stderr or exc.stdout or "DEG R script failed."),
        }

    result = _read_deg_csv(output_path)
    result["log2fold"] = float(log2fold)
    result["padj"] = float(padj)
    result["thresholds_applied_post_hoc"] = True
    result["message"] = (
        str(result.get("message") or "").strip()
        + " Requested log2fold/padj thresholds were applied in Python after the R script completed because the script itself hardcodes those values."
    ).strip()
    return result
