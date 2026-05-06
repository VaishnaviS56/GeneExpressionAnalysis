from __future__ import annotations

from typing import Literal, TypedDict


class QueryClassification(TypedDict):
    kind: Literal["general", "technical"]
    rationale: str
    genes: list[str]
