from __future__ import annotations

import re


_SRP_PATTERN = re.compile(r"\bSRP\d+\b", re.IGNORECASE)


def extract_srp_ids_from_text(text: str) -> list[str]:
    ids = []
    for match in _SRP_PATTERN.finditer(text or ""):
        value = match.group(0).upper()
        if value not in ids:
            ids.append(value)
    return ids
