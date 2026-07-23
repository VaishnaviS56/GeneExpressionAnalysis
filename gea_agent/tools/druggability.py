from __future__ import annotations

import csv
import io
import re
import time
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from gea_agent.config import SETTINGS
from gea_agent.tools.http_utils import get_retrying_session
from gea_agent.tools.result_utils import sanitize_exception_message, tool_error_result


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
PROTEINS_PLUS_BASE_URL = "https://proteins.plus"
PROTEINS_PLUS_UPLOAD_URL = f"{PROTEINS_PLUS_BASE_URL}/api/pdb_files_rest"
DOGSITE_URL = f"{PROTEINS_PLUS_BASE_URL}/api/dogsite_rest"


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_gene(gene: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "", _safe_text(gene)).upper()


def _druggability_dir(gene: str, output_dir: str | None = None) -> Path:
    root = Path(SETTINGS.resolve_path(output_dir or "druggability_results"))
    path = root / _normalize_gene(gene)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pdb_visualization_dir(label: str, output_dir: str | None = None) -> Path:
    root = Path(SETTINGS.resolve_path(output_dir or "pdb_visualizations"))
    path = root / _normalize_gene(label or "protein")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_text(url: str, path: Path) -> str:
    response = get_retrying_session().get(url, timeout=SETTINGS.http_timeout_seconds)
    response.raise_for_status()
    text = response.text
    path.write_text(text, encoding="utf-8", errors="replace")
    return str(path)


def _resolve_uniprot_accession(gene: str, organism_id: int = 9606) -> dict[str, Any]:
    params = {
        "query": f'(gene_exact:{gene}) AND (organism_id:{organism_id})',
        "format": "json",
        "fields": "accession,id,gene_names,protein_name,organism_name",
        "size": "5",
    }
    response = get_retrying_session().get(
        UNIPROT_SEARCH_URL,
        params=params,
        timeout=SETTINGS.http_timeout_seconds,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return {"status": "not_found", "message": f"No UniProt accession found for {gene}."}

    primary = results[0]
    accession = _safe_text(primary.get("primaryAccession"))
    return {
        "status": "ok",
        "uniprot_id": accession,
        "uniprot_entry": _safe_text(primary.get("uniProtkbId")),
        "protein_name": _safe_text((primary.get("proteinDescription") or {}).get("recommendedName", {}).get("fullName", {}).get("value")),
        "gene": gene,
    }


def _find_rcsb_pdb_for_uniprot(uniprot_id: str) -> dict[str, Any]:
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "request_options": {
            "return_all_hits": False,
            "paginate": {"start": 0, "rows": 10},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
        "return_type": "entry",
    }
    response = get_retrying_session().post(
        RCSB_SEARCH_URL,
        json=query,
        timeout=SETTINGS.http_timeout_seconds,
    )
    if response.status_code == 204:
        return {"status": "not_found", "pdb_id": ""}
    response.raise_for_status()
    result_set = response.json().get("result_set") or []
    for item in result_set:
        pdb_id = _safe_text(item.get("identifier")).upper()
        if re.fullmatch(r"[0-9A-Z]{4}", pdb_id):
            return {"status": "ok", "pdb_id": pdb_id}
    return {"status": "not_found", "pdb_id": ""}


def _get_structure(gene: str, uniprot_id: str, out_dir: Path) -> dict[str, Any]:
    rcsb_result = _find_rcsb_pdb_for_uniprot(uniprot_id)
    pdb_id = _safe_text(rcsb_result.get("pdb_id")).upper()
    if pdb_id:
        path = out_dir / f"{gene}_{pdb_id}.pdb"
        _download_text(RCSB_PDB_DOWNLOAD_URL.format(pdb_id=pdb_id), path)
        return {
            "status": "ok",
            "source": "rcsb",
            "pdb_id": pdb_id,
            "pdb_path": str(path),
            "message": f"Downloaded RCSB PDB structure {pdb_id}.",
        }

    path = out_dir / f"{gene}_{uniprot_id}_alphafold.pdb"
    alphafold = _find_alphafold_pdb_url(uniprot_id)
    alphafold_url = _safe_text(alphafold.get("pdb_url"))
    if not alphafold_url:
        return {
            "status": "not_found",
            "source": "none",
            "pdb_id": "",
            "pdb_path": "",
            "alphafold": alphafold,
            "message": f"No RCSB PDB or AlphaFold PDB model found for UniProt {uniprot_id}.",
        }
    response = get_retrying_session().get(
        alphafold_url,
        timeout=SETTINGS.http_timeout_seconds,
    )
    if response.status_code == 404:
        return {
            "status": "not_found",
            "source": "none",
            "pdb_id": "",
            "pdb_path": "",
            "alphafold": alphafold,
            "message": f"No RCSB PDB or AlphaFold PDB model found for UniProt {uniprot_id}.",
        }
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8", errors="replace")
    return {
        "status": "ok",
        "source": "alphafold",
        "pdb_id": "",
        "pdb_path": str(path),
        "alphafold": alphafold,
        "message": f"Downloaded AlphaFold PDB model for UniProt {uniprot_id}.",
    }


def _find_alphafold_pdb_url(uniprot_id: str) -> dict[str, Any]:
    try:
        response = get_retrying_session().get(
            ALPHAFOLD_API_URL.format(uniprot_id=uniprot_id),
            timeout=SETTINGS.http_timeout_seconds,
        )
        if response.status_code < 400:
            payload = response.json()
            records = payload if isinstance(payload, list) else []
            if records and isinstance(records[0], dict):
                record = records[0]
                pdb_url = _safe_text(record.get("pdbUrl"))
                if pdb_url:
                    return {
                        "status": "ok",
                        "pdb_url": pdb_url,
                        "latest_version": record.get("latestVersion"),
                        "model_entity_id": record.get("modelEntityId"),
                        "source": "api",
                    }
                latest = record.get("latestVersion")
                if latest:
                    version_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{int(latest)}.pdb"
                    return {
                        "status": "ok",
                        "pdb_url": version_url,
                        "latest_version": latest,
                        "model_entity_id": record.get("modelEntityId"),
                        "source": "api_latest_version",
                    }
    except Exception as exc:
        api_error = sanitize_exception_message(exc)
    else:
        api_error = ""

    for version in range(6, 0, -1):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.pdb"
        try:
            response = get_retrying_session().head(url, timeout=SETTINGS.http_timeout_seconds)
        except Exception:
            continue
        if response.status_code == 200:
            return {
                "status": "ok",
                "pdb_url": url,
                "latest_version": version,
                "model_entity_id": f"AF-{uniprot_id}-F1",
                "source": "version_probe",
            }

    return {
        "status": "not_found",
        "pdb_url": "",
        "message": (
            f"AlphaFold did not expose a downloadable PDB URL for UniProt {uniprot_id}."
            + (f" API error: {api_error}" if api_error else "")
        ),
    }


def _fix_pdb_with_pdbfixer(input_path: Path, output_path: Path, ph: float = 7.0) -> dict[str, Any]:
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except Exception as exc:
        return {
            "status": "dependency_missing",
            "message": (
                "PDBFixer/OpenMM is not installed. Install `pdbfixer` and `openmm` "
                f"to run sanitization. Import error: {sanitize_exception_message(exc)}"
            ),
            "fixed_pdb_path": str(input_path),
            "used_original": True,
        }

    fixer = PDBFixer(filename=str(input_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=True)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    with output_path.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle)
    return {
        "status": "ok",
        "message": "PDBFixer sanitized the structure and added hydrogens.",
        "fixed_pdb_path": str(output_path),
        "used_original": False,
    }


def _extract_location_id(payload: dict[str, Any]) -> str:
    for key in ("id", "pdbCode", "pdb_code", "pdbFile", "pdb_file_id"):
        value = _safe_text(payload.get(key))
        if value:
            return value
    location = _safe_text(payload.get("location"))
    if location:
        return location.rstrip("/").rsplit("/", 1)[-1]
    return ""


def _upload_custom_pdb(pdb_path: Path) -> dict[str, Any]:
    with pdb_path.open("rb") as handle:
        response = get_retrying_session().post(
            PROTEINS_PLUS_UPLOAD_URL,
            files={"pdb_file[pathvar]": (pdb_path.name, handle, "chemical/x-pdb")},
            headers={"Accept": "application/json"},
            timeout=max(SETTINGS.http_timeout_seconds, 60),
        )
    try:
        payload = response.json()
    except Exception:
        payload = {"text": response.text}
    if response.status_code >= 400:
        return {
            "status": "error",
            "status_code": response.status_code,
            "message": _safe_text(payload.get("message") or payload.get("error") or response.text),
            "payload": payload,
        }
    upload_id = _extract_location_id(payload)
    return {
        "status": "ok" if upload_id else "missing_upload_id",
        "upload_id": upload_id,
        "status_code": response.status_code,
        "payload": payload,
        "message": f"Uploaded PDB to ProteinsPlus as {upload_id}." if upload_id else "ProteinsPlus upload did not return a usable structure id.",
    }


def _submit_dogsite(pdb_code: str, *, chain: str = "", ligand: str = "") -> dict[str, Any]:
    response = get_retrying_session().post(
        DOGSITE_URL,
        json={
            "dogsite": {
                "pdbCode": pdb_code,
                "analysisDetail": "1",
                "bindingSitePredictionGranularity": "1",
                "ligand": ligand or "",
                "chain": chain or "",
            }
        },
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        timeout=SETTINGS.http_timeout_seconds,
    )
    try:
        payload = response.json()
    except Exception:
        payload = {"text": response.text}
    if response.status_code >= 400:
        return {
            "status": "error",
            "status_code": response.status_code,
            "message": _safe_text(payload.get("message") or payload.get("error") or response.text),
            "payload": payload,
        }
    location = _safe_text(payload.get("location"))
    job_id = location.rstrip("/").rsplit("/", 1)[-1] if location else _safe_text(payload.get("id"))
    return {
        "status": "ok" if job_id else "missing_job_id",
        "job_id": job_id,
        "location": location,
        "status_code": response.status_code,
        "payload": payload,
        "message": f"Submitted DoGSite job {job_id}." if job_id else "DoGSite did not return a job id.",
    }


def _poll_dogsite(job_id: str, *, timeout_seconds: int, poll_interval_seconds: int) -> dict[str, Any]:
    deadline = time.time() + max(1, timeout_seconds)
    url = f"{DOGSITE_URL}/{job_id}"
    last_payload: dict[str, Any] = {}
    while time.time() < deadline:
        response = get_retrying_session().get(
            url,
            headers={"Accept": "application/json"},
            timeout=SETTINGS.http_timeout_seconds,
        )
        try:
            payload = response.json()
        except Exception:
            payload = {"text": response.text}
        last_payload = payload
        if response.status_code == 200 and str(payload.get("status_code") or "200") == "200":
            return {"status": "ok", "job_id": job_id, "payload": payload}
        if response.status_code not in {200, 202}:
            return {
                "status": "error",
                "job_id": job_id,
                "status_code": response.status_code,
                "message": _safe_text(payload.get("message") or payload.get("error") or response.text),
                "payload": payload,
            }
        time.sleep(max(1, poll_interval_seconds))
    return {
        "status": "timeout",
        "job_id": job_id,
        "message": f"DoGSite job did not finish within {timeout_seconds} seconds.",
        "payload": last_payload,
    }


def _download_result_file(value: Any, out_dir: Path, filename: str) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    if text.startswith(("http://", "https://", "/")):
        url = urljoin(PROTEINS_PLUS_BASE_URL, text)
        response = get_retrying_session().get(url, timeout=SETTINGS.http_timeout_seconds)
        response.raise_for_status()
        path = out_dir / filename
        path.write_bytes(response.content)
        return str(path)
    path = out_dir / filename
    path.write_text(str(value), encoding="utf-8", errors="replace")
    return str(path)


def _parse_result_table(table_text: str, *, top_n: int) -> list[dict[str, Any]]:
    clean = table_text.strip()
    if not clean:
        return []
    sample = clean[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
    except Exception:
        dialect = csv.excel_tab
    rows = list(csv.DictReader(io.StringIO(clean), dialect=dialect))
    pockets: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        compact = {_safe_text(key): _safe_text(value) for key, value in row.items() if _safe_text(key)}
        if not compact:
            continue
        pocket_name = (
            compact.get("name")
            or compact.get("Name")
            or compact.get("pocket")
            or compact.get("Pocket")
            or compact.get("site")
            or compact.get("Site")
            or f"pocket_{index}"
        )
        score = _first_float(
            compact,
            "drugScore",
            "DrugScore",
            "druggability_score",
            "druggability",
            "score",
            "Score",
        )
        volume = _first_float(compact, "volume", "Volume", "vol", "Vol")
        pockets.append(
            {
                "rank": index,
                "name": pocket_name,
                "drug_score": score,
                "volume": volume,
                "properties": compact,
            }
        )

    pockets.sort(
        key=lambda item: (
            item.get("drug_score") is not None,
            float(item.get("drug_score") or 0.0),
            float(item.get("volume") or 0.0),
        ),
        reverse=True,
    )
    for rank, pocket in enumerate(pockets, start=1):
        pocket["rank"] = rank
    return pockets[: max(1, top_n)]


def _first_float(row: dict[str, str], *keys: str) -> float | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value in (None, ""):
            continue
        try:
            return float(str(value).replace(",", "."))
        except Exception:
            continue
    return None


def _collect_outputs(payload: dict[str, Any], out_dir: Path, *, top_n: int) -> dict[str, Any]:
    result_table_path = _download_result_file(payload.get("result_table"), out_dir, "dogsite_result_table.txt")
    table_text = Path(result_table_path).read_text(encoding="utf-8", errors="replace") if result_table_path else ""
    top_pockets = _parse_result_table(table_text, top_n=top_n)

    residues = payload.get("residues") if isinstance(payload.get("residues"), list) else []
    maps = payload.get("pockets") if isinstance(payload.get("pockets"), list) else []
    for index, pocket in enumerate(top_pockets):
        file_rank = index + 1
        if index < len(residues):
            pocket["residue_file"] = _download_result_file(residues[index], out_dir, f"pocket_{file_rank}_residues.pdb")
        if index < len(maps):
            pocket["map_file"] = _download_result_file(maps[index], out_dir, f"pocket_{file_rank}_map.ccp4")

    descriptor_path = _download_result_file(
        payload.get("descriptor_explanation"),
        out_dir,
        "dogsite_descriptor_explanation.txt",
    )
    return {
        "result_table_path": result_table_path,
        "descriptor_explanation_path": descriptor_path,
        "top_pockets": top_pockets,
    }


def _format_druggability_answer(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return _safe_text(result.get("message") or "Druggability analysis failed.")

    lines = [
        f"**Druggability Results for {result.get('gene')}**",
        f"Structure source: {result.get('structure_source')} "
        + (f"({result.get('pdb_id')})" if result.get("pdb_id") else f"(UniProt {result.get('uniprot_id')})"),
        f"Sanitized PDB: `{result.get('fixed_pdb_path')}`",
        f"DoGSite job: `{result.get('dogsite_job_id')}`",
    ]
    result_table = _safe_text(result.get("result_table_path"))
    if result_table:
        lines.append(f"Result table: `{result_table}`")
    viewer = _safe_text(result.get("pdb_viewer_html_path"))
    if viewer:
        lines.append(f"PDB viewer: `{viewer}`")

    pockets = result.get("top_pockets") if isinstance(result.get("top_pockets"), list) else []
    if pockets:
        lines.extend(["", "Top pockets:"])
        lines.append("| Rank | Pocket | Drug score | Volume | Residue file | Map file |")
        lines.append("|---|---|---:|---:|---|---|")
        for pocket in pockets:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(pocket.get("rank") or ""),
                        _safe_text(pocket.get("name")).replace("|", "\\|"),
                        "" if pocket.get("drug_score") is None else f"{float(pocket.get('drug_score')):.3g}",
                        "" if pocket.get("volume") is None else f"{float(pocket.get('volume')):.3g}",
                        f"`{pocket.get('residue_file')}`" if pocket.get("residue_file") else "",
                        f"`{pocket.get('map_file')}`" if pocket.get("map_file") else "",
                    ]
                )
                + " |"
            )
    else:
        lines.append("")
        lines.append("DoGSite finished, but no pocket rows could be parsed from the result table.")
    return "\n".join(lines).strip()


def _format_pdb_visualization_answer(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return _safe_text(result.get("message") or "PDB visualization failed.")

    label = result.get("gene") or result.get("protein") or result.get("uniprot_id") or result.get("pdb_id") or "protein"
    lines = [
        f"**PDB Visualization for {label}**",
        f"Structure source: {result.get('structure_source')} "
        + (f"({result.get('pdb_id')})" if result.get("pdb_id") else f"(UniProt {result.get('uniprot_id')})"),
        f"Protein PDB: `{result.get('pdb_path')}`",
    ]
    viewer = _safe_text(result.get("pdb_viewer_html_path"))
    if viewer:
        lines.append(f"Interactive viewer: `{viewer}`")
    return "\n".join(lines).strip()


def _read_pdb_for_viewer(path: Any) -> str:
    text_path = _safe_text(path)
    if not text_path:
        return ""
    try:
        return Path(text_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def build_pdb_pocket_viewer(
    *,
    protein_pdb_path: str,
    top_pockets: list[dict[str, Any]],
    output_path: str,
    title: str = "Protein Pocket Viewer",
) -> dict[str, Any]:
    protein_pdb = _read_pdb_for_viewer(protein_pdb_path)
    if not protein_pdb:
        return {
            "status": "missing_protein_pdb",
            "message": "Protein PDB file is not available for viewer generation.",
            "pdb_viewer_html_path": "",
        }

    pocket_entries: list[dict[str, str]] = []
    for pocket in top_pockets:
        if not isinstance(pocket, dict):
            continue
        residue_pdb = _read_pdb_for_viewer(pocket.get("residue_file"))
        if not residue_pdb:
            continue
        pocket_entries.append(
            {
                "rank": str(pocket.get("rank") or len(pocket_entries) + 1),
                "name": _safe_text(pocket.get("name")) or f"Pocket {len(pocket_entries) + 1}",
                "drug_score": "" if pocket.get("drug_score") is None else f"{float(pocket.get('drug_score')):.4g}",
                "volume": "" if pocket.get("volume") is None else f"{float(pocket.get('volume')):.4g}",
                "pdb": residue_pdb,
                "residue_file": _safe_text(pocket.get("residue_file")),
                "map_file": _safe_text(pocket.get("map_file")),
            }
        )

    output = Path(output_path).with_suffix(".html").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    html = _pdb_viewer_html(
        title=title,
        protein_pdb=protein_pdb,
        pocket_entries=pocket_entries,
        protein_pdb_path=str(Path(protein_pdb_path).resolve()),
    )
    output.write_text(html, encoding="utf-8")
    return {
        "status": "ok",
        "message": "Built interactive PDB pocket viewer.",
        "pdb_viewer_html_path": str(output),
        "visualized_pocket_count": len(pocket_entries),
    }


def _json_script_string(value: str) -> str:
    import json

    return json.dumps(value)


def _pdb_viewer_html(
    *,
    title: str,
    protein_pdb: str,
    pocket_entries: list[dict[str, str]],
    protein_pdb_path: str,
) -> str:
    import json

    pockets_json = json.dumps(pocket_entries)
    protein_json = json.dumps(protein_pdb)
    safe_title = escape(title)
    safe_protein_path = escape(protein_pdb_path)
    default_pocket_name = escape(pocket_entries[0]["name"] if pocket_entries else "No pocket residue PDB available")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #5b6475;
      --line: #d7dce5;
      --panel: #f7f8fb;
      --accent: #0f766e;
      --pocket: #e11d48;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
    }}
    main {{ min-width: 0; min-height: 100vh; position: relative; }}
    h1 {{ font-size: 20px; line-height: 1.25; margin: 0 0 8px; }}
    .caption {{ color: var(--muted); font-size: 12px; line-height: 1.45; word-break: break-word; }}
    .controls {{ display: grid; gap: 10px; margin: 18px 0; }}
    button, select {{
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 13px;
    }}
    button {{ cursor: pointer; text-align: left; }}
    button.active {{ border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }}
    .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .metric {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 8px;
      min-height: 52px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 11px; }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 15px; }}
    #viewer {{ position: absolute; inset: 0; }}
    .viewer-label {{
      position: absolute;
      left: 18px;
      bottom: 16px;
      background: rgba(255,255,255,.9);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
      color: var(--muted);
      pointer-events: none;
    }}
    .legend {{ display: grid; gap: 8px; margin-top: 14px; font-size: 12px; color: var(--muted); }}
    .legend span {{ display: inline-flex; align-items: center; gap: 8px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
    @media (max-width: 760px) {{
      body {{ grid-template-columns: 1fr; grid-template-rows: auto 72vh; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      main {{ min-height: 72vh; }}
    }}
  </style>
</head>
<body>
  <aside>
    <h1>{safe_title}</h1>
    <div class="caption">Protein PDB: {safe_protein_path}</div>
    <div class="controls">
      <select id="styleSelect" aria-label="Protein style">
        <option value="cartoon">Cartoon + transparent surface</option>
        <option value="surface">Surface only</option>
        <option value="stick">Protein sticks</option>
      </select>
      <div id="pocketButtons"></div>
      <button id="showAll" type="button">Show all pockets</button>
      <button id="resetView" type="button">Reset view</button>
    </div>
    <div class="row">
      <div class="metric"><span>Selected pocket</span><strong id="selectedName">{default_pocket_name}</strong></div>
      <div class="metric"><span>Drug score</span><strong id="selectedScore">NA</strong></div>
    </div>
    <div class="row" style="margin-top:8px">
      <div class="metric"><span>Volume</span><strong id="selectedVolume">NA</strong></div>
      <div class="metric"><span>Pockets loaded</span><strong>{len(pocket_entries)}</strong></div>
    </div>
    <div class="legend">
      <span><i class="swatch" style="background:#7aa6d8"></i>Protein</span>
      <span><i class="swatch" style="background:#e11d48"></i>Selected pocket residues</span>
      <span><i class="swatch" style="background:#f59e0b"></i>Other pocket residues</span>
    </div>
  </aside>
  <main>
    <div id="viewer"></div>
    <div class="viewer-label">Drag to rotate. Scroll to zoom. Pocket residues are overlaid on the full protein structure.</div>
  </main>
  <script>
    const proteinPdb = {protein_json};
    const pockets = {pockets_json};
    const viewer = $3Dmol.createViewer('viewer', {{ backgroundColor: 'white' }});
    const colors = ['#e11d48', '#f59e0b', '#0f766e', '#7c3aed', '#0891b2', '#ea580c'];
    let selectedIndex = pockets.length ? 0 : -1;

    function setProteinStyle(styleName) {{
      viewer.removeAllSurfaces();
      viewer.setStyle({{}}, {{}});
      if (styleName === 'surface') {{
        viewer.setStyle({{}}, {{ line: {{ color: '#9aa7ba', linewidth: 0.4 }} }});
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{ opacity: 0.78, color: '#9ec5e8' }}, {{}});
      }} else if (styleName === 'stick') {{
        viewer.setStyle({{}}, {{ stick: {{ radius: 0.13, color: '#7aa6d8' }} }});
      }} else {{
        viewer.setStyle({{}}, {{ cartoon: {{ color: '#7aa6d8' }} }});
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{ opacity: 0.18, color: '#9ec5e8' }}, {{}});
      }}
    }}

    function renderPocket(index) {{
      viewer.removeAllModels();
      viewer.removeAllSurfaces();
      viewer.addModel(proteinPdb, 'pdb');
      setProteinStyle(document.getElementById('styleSelect').value);

      pockets.forEach((pocket, i) => {{
        if (!pocket.pdb) return;
        const model = viewer.addModel(pocket.pdb, 'pdb');
        const isSelected = index === -1 || i === index;
        const color = i === index ? colors[0] : colors[(i % (colors.length - 1)) + 1];
        model.setStyle({{}}, {{
          stick: {{ color, radius: isSelected ? 0.26 : 0.16 }},
          sphere: {{ color, radius: isSelected ? 0.42 : 0.24 }}
        }});
      }});

      document.querySelectorAll('[data-pocket-index]').forEach((button) => {{
        button.classList.toggle('active', Number(button.dataset.pocketIndex) === index);
      }});
      const selected = pockets[index] || {{}};
      document.getElementById('selectedName').textContent = index === -1 ? 'All pockets' : (selected.name || 'Pocket');
      document.getElementById('selectedScore').textContent = index === -1 ? 'mixed' : (selected.drug_score || 'NA');
      document.getElementById('selectedVolume').textContent = index === -1 ? 'mixed' : (selected.volume || 'NA');
      viewer.zoomTo();
      viewer.render();
    }}

    function buildControls() {{
      const box = document.getElementById('pocketButtons');
      box.innerHTML = '';
      if (!pockets.length) {{
        const empty = document.createElement('div');
        empty.className = 'caption';
        empty.textContent = 'No pocket residue PDB files were available in the DoGSite result.';
        box.appendChild(empty);
        return;
      }}
      pockets.forEach((pocket, index) => {{
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.pocketIndex = String(index);
        button.textContent = `${{pocket.rank || index + 1}}. ${{pocket.name || 'Pocket'}}`;
        button.addEventListener('click', () => {{
          selectedIndex = index;
          renderPocket(index);
        }});
        box.appendChild(button);
      }});
    }}

    document.getElementById('styleSelect').addEventListener('change', () => renderPocket(selectedIndex));
    document.getElementById('showAll').addEventListener('click', () => {{
      selectedIndex = -1;
      renderPocket(-1);
    }});
    document.getElementById('resetView').addEventListener('click', () => {{
      viewer.zoomTo();
      viewer.render();
    }});
    window.addEventListener('resize', () => viewer.resize());

    buildControls();
    renderPocket(selectedIndex);
  </script>
</body>
</html>
"""


def run_druggability_analysis(
    *,
    gene: str,
    organism_id: int = 9606,
    chain: str = "",
    ligand: str = "",
    top_n: int = 3,
    output_dir: str | None = None,
    dogsite_timeout_seconds: int = 900,
    poll_interval_seconds: int = 15,
    pdbfixer_ph: float = 7.0,
) -> dict[str, Any]:
    gene_symbol = _normalize_gene(gene)
    if not gene_symbol:
        return tool_error_result(
            "druggability",
            "A gene symbol is required for druggability analysis.",
            analysis_arm="druggability",
            should_finalize=True,
        )

    out_dir = _druggability_dir(gene_symbol, output_dir)
    uniprot = _resolve_uniprot_accession(gene_symbol, organism_id=organism_id)
    if uniprot.get("status") != "ok":
        return {
            "status": "not_found",
            "analysis_arm": "druggability",
            "gene": gene_symbol,
            "message": _safe_text(uniprot.get("message")),
            "answer": _safe_text(uniprot.get("message")),
            "should_finalize": True,
        }

    structure = _get_structure(gene_symbol, str(uniprot["uniprot_id"]), out_dir)
    if structure.get("status") != "ok":
        result = {
            **structure,
            "analysis_arm": "druggability",
            "gene": gene_symbol,
            "uniprot_id": uniprot.get("uniprot_id"),
            "should_finalize": True,
        }
        result["answer"] = _format_druggability_answer(result)
        return result

    raw_path = Path(str(structure["pdb_path"]))
    fixed_path = out_dir / f"{raw_path.stem}_fixed.pdb"
    fix_result = _fix_pdb_with_pdbfixer(raw_path, fixed_path, ph=float(pdbfixer_ph))
    fixed_pdb_path = Path(str(fix_result.get("fixed_pdb_path") or raw_path))

    upload = _upload_custom_pdb(fixed_pdb_path)
    pdb_code_for_dogsite = _safe_text(upload.get("upload_id"))
    if not pdb_code_for_dogsite and structure.get("source") == "rcsb":
        pdb_code_for_dogsite = _safe_text(structure.get("pdb_id"))
    if not pdb_code_for_dogsite:
        result = {
            "status": "error",
            "analysis_arm": "druggability",
            "gene": gene_symbol,
            "uniprot_id": uniprot.get("uniprot_id"),
            "structure": structure,
            "pdbfixer": fix_result,
            "proteins_plus_upload": upload,
            "message": _safe_text(upload.get("message") or "ProteinsPlus did not return a usable custom PDB id."),
            "should_finalize": True,
        }
        result["answer"] = _format_druggability_answer(result)
        return result

    submit = _submit_dogsite(pdb_code_for_dogsite, chain=chain, ligand=ligand)
    if submit.get("status") != "ok":
        result = {
            "status": "error",
            "analysis_arm": "druggability",
            "gene": gene_symbol,
            "uniprot_id": uniprot.get("uniprot_id"),
            "structure": structure,
            "pdbfixer": fix_result,
            "proteins_plus_upload": upload,
            "dogsite_submit": submit,
            "message": _safe_text(submit.get("message") or "DoGSite job submission failed."),
            "should_finalize": True,
        }
        result["answer"] = _format_druggability_answer(result)
        return result

    dogsite = _poll_dogsite(
        str(submit["job_id"]),
        timeout_seconds=int(dogsite_timeout_seconds),
        poll_interval_seconds=int(poll_interval_seconds),
    )
    if dogsite.get("status") != "ok":
        result = {
            "status": dogsite.get("status") or "error",
            "analysis_arm": "druggability",
            "gene": gene_symbol,
            "uniprot_id": uniprot.get("uniprot_id"),
            "structure": structure,
            "pdbfixer": fix_result,
            "proteins_plus_upload": upload,
            "dogsite_submit": submit,
            "dogsite_result": dogsite,
            "message": _safe_text(dogsite.get("message") or "DoGSite job did not complete."),
            "should_finalize": True,
        }
        result["answer"] = _format_druggability_answer(result)
        return result

    outputs = _collect_outputs(dogsite.get("payload") or {}, out_dir, top_n=int(top_n or 3))
    viewer_result = build_pdb_pocket_viewer(
        protein_pdb_path=str(fixed_pdb_path),
        top_pockets=outputs.get("top_pockets") if isinstance(outputs.get("top_pockets"), list) else [],
        output_path=str(out_dir / f"{gene_symbol}_pocket_viewer.html"),
        title=f"{gene_symbol} Protein Pocket Viewer",
    )
    result = {
        "status": "ok",
        "analysis_arm": "druggability",
        "gene": gene_symbol,
        "uniprot_id": uniprot.get("uniprot_id"),
        "uniprot": uniprot,
        "structure_source": structure.get("source"),
        "pdb_id": structure.get("pdb_id"),
        "raw_pdb_path": structure.get("pdb_path"),
        "fixed_pdb_path": str(fixed_pdb_path),
        "pdbfixer": fix_result,
        "proteins_plus_upload": upload,
        "dogsite_job_id": submit.get("job_id"),
        "dogsite_submit": submit,
        "dogsite_result": dogsite,
        "output_dir": str(out_dir),
        "pdb_viewer": viewer_result,
        "pdb_viewer_html_path": viewer_result.get("pdb_viewer_html_path", ""),
        "should_finalize": True,
        **outputs,
    }
    result["message"] = f"DoGSite druggability analysis completed for {gene_symbol}."
    result["answer"] = _format_druggability_answer(result)
    return result


def run_pdb_visualization(
    *,
    gene: str = "",
    protein: str = "",
    uniprot_id: str = "",
    pdb_id: str = "",
    organism_id: int = 9606,
    output_dir: str | None = None,
) -> dict[str, Any]:
    query_label = _normalize_gene(gene or protein or uniprot_id or pdb_id)
    if not query_label:
        return tool_error_result(
            "pdb_visualizer",
            "A gene symbol, UniProt accession, protein label, or PDB ID is required for PDB visualization.",
            analysis_arm="pdb_visualizer",
            should_finalize=True,
        )

    out_dir = _pdb_visualization_dir(query_label, output_dir)
    resolved_uniprot: dict[str, Any] = {}
    structure: dict[str, Any]

    direct_pdb = _safe_text(pdb_id or "")
    if not direct_pdb and re.fullmatch(r"[0-9][A-Za-z0-9]{3}", query_label):
        direct_pdb = query_label

    if direct_pdb:
        direct_pdb = direct_pdb.upper()
        path = out_dir / f"{query_label}_{direct_pdb}.pdb"
        _download_text(RCSB_PDB_DOWNLOAD_URL.format(pdb_id=direct_pdb), path)
        structure = {
            "status": "ok",
            "source": "rcsb",
            "pdb_id": direct_pdb,
            "pdb_path": str(path),
            "message": f"Downloaded RCSB PDB structure {direct_pdb}.",
        }
    else:
        resolved_accession = _safe_text(uniprot_id)
        if not resolved_accession:
            resolved_uniprot = _resolve_uniprot_accession(query_label, organism_id=organism_id)
            if resolved_uniprot.get("status") != "ok":
                result = {
                    "status": "not_found",
                    "analysis_arm": "pdb_visualizer",
                    "gene": query_label,
                    "message": _safe_text(resolved_uniprot.get("message")),
                    "answer": _safe_text(resolved_uniprot.get("message")),
                    "should_finalize": True,
                }
                return result
            resolved_accession = str(resolved_uniprot.get("uniprot_id") or "")
        structure = _get_structure(query_label, resolved_accession, out_dir)

    if structure.get("status") != "ok":
        result = {
            **structure,
            "analysis_arm": "pdb_visualizer",
            "gene": query_label if not direct_pdb else "",
            "protein": protein,
            "uniprot_id": uniprot_id or resolved_uniprot.get("uniprot_id", ""),
            "should_finalize": True,
        }
        result["answer"] = _format_pdb_visualization_answer(result)
        return result

    protein_path = str(structure.get("pdb_path") or "")
    viewer_result = build_pdb_pocket_viewer(
        protein_pdb_path=protein_path,
        top_pockets=[],
        output_path=str(out_dir / f"{query_label}_pdb_viewer.html"),
        title=f"{query_label} Protein Structure Viewer",
    )
    result = {
        "status": "ok",
        "analysis_arm": "pdb_visualizer",
        "gene": query_label if not direct_pdb else "",
        "protein": protein,
        "uniprot_id": uniprot_id or resolved_uniprot.get("uniprot_id", ""),
        "uniprot": resolved_uniprot,
        "structure_source": structure.get("source"),
        "pdb_id": structure.get("pdb_id"),
        "pdb_path": protein_path,
        "raw_pdb_path": protein_path,
        "pdb_viewer": viewer_result,
        "pdb_viewer_html_path": viewer_result.get("pdb_viewer_html_path", ""),
        "output_dir": str(out_dir),
        "should_finalize": True,
    }
    result["message"] = f"Built PDB visualization for {query_label}."
    result["answer"] = _format_pdb_visualization_answer(result)
    return result


def run_druggability_analysis_safe(**kwargs: Any) -> dict[str, Any]:
    try:
        return run_druggability_analysis(**kwargs)
    except Exception as exc:
        gene = _normalize_gene(kwargs.get("gene"))
        return tool_error_result(
            "druggability",
            f"Druggability analysis failed: {sanitize_exception_message(exc)}",
            analysis_arm="druggability",
            gene=gene,
            should_finalize=True,
        )


def run_pdb_visualization_safe(**kwargs: Any) -> dict[str, Any]:
    try:
        return run_pdb_visualization(**kwargs)
    except Exception as exc:
        label = _normalize_gene(kwargs.get("gene") or kwargs.get("protein") or kwargs.get("uniprot_id") or kwargs.get("pdb_id"))
        return tool_error_result(
            "pdb_visualizer",
            f"PDB visualization failed: {sanitize_exception_message(exc)}",
            analysis_arm="pdb_visualizer",
            gene=label,
            should_finalize=True,
        )
