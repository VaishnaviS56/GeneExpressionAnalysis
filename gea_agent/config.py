from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # STRING config
    string_species: int = int(os.getenv("STRING_SPECIES", "9606"))  # Homo sapiens
    string_required_score: int = int(os.getenv("STRING_REQUIRED_SCORE", "700"))  # 0.700
    string_limit: int = int(os.getenv("STRING_LIMIT", "200"))


SETTINGS = Settings()

