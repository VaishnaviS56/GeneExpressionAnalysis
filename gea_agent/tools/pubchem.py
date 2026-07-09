from __future__ import annotations

from typing import Any
from urllib.parse import quote

from gea_agent.config import SETTINGS
from gea_agent.tools.http_utils import get_retrying_session


PUBCHEM_PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PUG_VIEW_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"


def _request_json(url: str) -> dict[str, Any] | None:
    try:
        response = get_retrying_session().get(
            url,
            timeout=SETTINGS.http_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _first_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _first_string(item)
            if text:
                return text
    if isinstance(value, dict):
        for key in ("String", "StringWithMarkup", "Number", "Boolean", "Value", "Name"):
            text = _first_string(value.get(key))
            if text:
                return text
    return ""


def _extract_section_text(section: dict[str, Any], lines: list[str]) -> None:
    heading = str(section.get("TOCHeading") or "").strip()
    information = section.get("Information")
    if isinstance(information, list):
        for item in information:
            if not isinstance(item, dict):
                continue
            value = item.get("Value")
            text = ""
            if isinstance(value, dict):
                for key in ("StringWithMarkup", "String", "Number"):
                    text = _first_string(value.get(key))
                    if text:
                        break
            if not text:
                text = _first_string(item)
            if text:
                lines.append(f"{heading}: {text}" if heading else text)

    for subsection in section.get("Section") or []:
        if isinstance(subsection, dict):
            _extract_section_text(subsection, lines)


def _annotation_lines(payload: dict[str, Any] | None, *, limit: int = 120) -> list[str]:
    if not isinstance(payload, dict):
        return []
    record = payload.get("Record")
    if not isinstance(record, dict):
        return []
    lines: list[str] = []
    for section in record.get("Section") or []:
        if isinstance(section, dict):
            _extract_section_text(section, lines)
    deduped: list[str] = []
    for line in lines:
        text = " ".join(str(line).split())
        if text and text not in deduped:
            deduped.append(text)
    return deduped[:limit]


def _extract_cids(payload: dict[str, Any] | None) -> list[int]:
    if not isinstance(payload, dict):
        return []
    identifier_list = payload.get("IdentifierList") or {}
    cids = identifier_list.get("CID") or []
    if not isinstance(cids, list):
        return []
    cleaned = []
    for value in cids:
        try:
            cleaned.append(int(value))
        except Exception:
            continue
    return cleaned


def _resolve_cid(search_terms: list[tuple[str, str]]) -> tuple[int | None, list[int], str, str, str]:
    seen_terms: list[tuple[str, str]] = []
    for raw_term, mode in search_terms:
        term = " ".join(str(raw_term or "").split()).strip()
        mode = str(mode or "").strip().lower()
        key = (term, mode)
        if not term or key in seen_terms:
            continue
        seen_terms.append(key)
        encoded = quote(term)
        if mode == "registry":
            payload = _request_json(f"{PUBCHEM_PUG_BASE}/compound/xref/RegistryID/{encoded}/cids/JSON")
        else:
            payload = _request_json(f"{PUBCHEM_PUG_BASE}/compound/name/{encoded}/cids/JSON")
        if not isinstance(payload, dict):
            continue

        cleaned = _extract_cids(payload)
        if cleaned:
            return cleaned[0], cleaned, "", term, mode

    return None, [], "No PubChem compound ID was found for the provided name or identifier.", "", ""


def query_pubchem_drug(drug_name: str = "", pert_id: str = "") -> dict[str, Any]:
    drug_name = " ".join(str(drug_name or "").split()).strip()
    pert_id = " ".join(str(pert_id or "").split()).strip().upper()
    if not drug_name and not pert_id:
        return {
            "status": "missing_input",
            "drug_name": "",
            "pert_id": "",
            "message": "A drug name or pert_id is required.",
        }

    cid, candidate_cids, cid_message, matched_query, matched_strategy = _resolve_cid(
        [
            (pert_id, "registry"),
            (pert_id, "name"),
            (drug_name, "name"),
        ]
    )
    if not cid:
        return {
            "status": "not_found",
            "drug_name": drug_name,
            "pert_id": pert_id,
            "matched_query": matched_query,
            "matched_strategy": matched_strategy,
            "candidate_cids": candidate_cids,
            "message": cid_message or "PubChem did not return a CID.",
        }

    property_fields = ",".join(
        [
            "Title",
            "MolecularFormula",
            "MolecularWeight",
            "CanonicalSMILES",
            "IsomericSMILES",
            "InChI",
            "InChIKey",
            "XLogP",
            "TPSA",
            "HBondDonorCount",
            "HBondAcceptorCount",
            "RotatableBondCount",
            "Complexity",
        ]
    )
    properties_payload = _request_json(
        f"{PUBCHEM_PUG_BASE}/compound/cid/{cid}/property/{property_fields}/JSON"
    )
    synonyms_payload = _request_json(f"{PUBCHEM_PUG_BASE}/compound/cid/{cid}/synonyms/JSON")
    description_payload = _request_json(f"{PUBCHEM_PUG_BASE}/compound/cid/{cid}/description/JSON")
    record_payload = _request_json(f"{PUBCHEM_PUG_BASE}/compound/cid/{cid}/record/JSON")
    annotation_payload = _request_json(f"{PUBCHEM_PUG_VIEW_BASE}/data/compound/{cid}/JSON")

    property_rows = (((properties_payload or {}).get("PropertyTable") or {}).get("Properties") or [])
    properties = property_rows[0] if isinstance(property_rows, list) and property_rows and isinstance(property_rows[0], dict) else {}

    info_list = (((synonyms_payload or {}).get("InformationList") or {}).get("Information") or [])
    synonyms = []
    if isinstance(info_list, list):
        for item in info_list:
            if not isinstance(item, dict):
                continue
            for synonym in item.get("Synonym") or []:
                text = str(synonym or "").strip()
                if text and text not in synonyms:
                    synonyms.append(text)

    descriptions = []
    info_list = (((description_payload or {}).get("InformationList") or {}).get("Information") or [])
    if isinstance(info_list, list):
        for item in info_list:
            if not isinstance(item, dict):
                continue
            description = str(item.get("Description") or "").strip()
            if description and description not in descriptions:
                descriptions.append(description)

    annotation_lines = _annotation_lines(annotation_payload)
    title = str(properties.get("Title") or drug_name or pert_id).strip() or drug_name or pert_id

    return {
        "status": "ok",
        "analysis_arm": "pubchem",
        "drug_name": drug_name,
        "pert_id": pert_id,
        "matched_query": matched_query,
        "matched_strategy": matched_strategy,
        "title": title,
        "cid": cid,
        "candidate_cids": candidate_cids,
        "properties": properties,
        "synonyms": synonyms[:100],
        "descriptions": descriptions[:20],
        "annotation_lines": annotation_lines,
        "record_available": isinstance(record_payload, dict),
        "message": "Retrieved PubChem compound data.",
        "raw_result": {
            "properties": properties_payload or {},
            "synonyms": synonyms_payload or {},
            "description": description_payload or {},
            "record": record_payload or {},
            "annotations": annotation_payload or {},
        },
    }
