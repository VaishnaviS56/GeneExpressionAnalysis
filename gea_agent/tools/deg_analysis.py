from __future__ import annotations

import csv
import subprocess
import shutil
from pathlib import Path
from typing import Any

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
    return normalized


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
            row = _normalize_row(raw_row)
            rows.append({column: row.get(column, "") for column in _DEG_COLUMNS})

            label = _gene_label(row)
            if label and label not in genes:
                genes.append(label)

    return {
        "status": "ok",
        "output_csv_path": str(csv_path),
        "genes": genes,
        "rows": rows,
        "message": f"Loaded {len(rows)} DEG rows.",
    }


def run_deg_r_analysis() -> dict[str, Any]:
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
    print(SETTINGS.rscript_executable)

    if _placeholder_path(SETTINGS.deg_r_script_path) or _placeholder_path(SETTINGS.deg_output_csv_path):
        return {
            "status": "not_configured",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "genes": [],
            "rows": [],
            "message": "DEG R paths are still placeholders.",
        }

    if not executable:
        print(f"Could not resolve Rscript executable from: {SETTINGS.rscript_executable}")
        return {
            "status": "rscript_missing",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
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
            "genes": [],
            "rows": [],
            "message": "DEG R script was not found.",
        }

    supporting_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [executable, str(script_path)],
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
            "genes": [],
            "rows": [],
            "message": f"Could not find the Rscript executable: {exc}",
        }
    except subprocess.CalledProcessError as exc:

        return {
            "status": "run_failed",
            "script_path": str(script_path),
            "supporting_files_dir": str(supporting_dir),
            "output_csv_path": str(output_path),
            "genes": [],
            "rows": [],
            "message": (exc.stderr or exc.stdout or "DEG R script failed."),
        }

    return _read_deg_csv(output_path)
