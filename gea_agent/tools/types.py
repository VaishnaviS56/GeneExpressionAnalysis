from __future__ import annotations

from typing import Literal, TypedDict


class QueryClassification(TypedDict):
    kind: Literal["general", "technical"]
    rationale: str
    genes: list[str]


class StringEdge(TypedDict):
    preferredName_A: str
    preferredName_B: str
    score: float  # 0..1 combined score normalized

