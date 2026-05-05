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

    # Local STRING files (downloaded)
    # Point these to your downloaded files (repo-relative or absolute paths).
    string_info_path: str = os.getenv(
        "STRING_INFO_PATH",
        r"9606.protein.info.v12.0.txt\9606.protein.info.v12.0.txt",
    )
    string_links_path: str = os.getenv(
        "STRING_LINKS_PATH",
        r"9606.protein.links.v12.0.min900.txt\9606.protein.links.v12.0.min900.txt",
    )

    # How much of the network to load from the links file:
    # - seed_1hop: only edges touching seed genes
    # - seed_1hop_closed: seed_1hop plus edges among discovered nodes (2 passes)
    # - full: all edges (can be large)
    string_local_mode: str = os.getenv("STRING_LOCAL_MODE", "seed_1hop_closed")


SETTINGS = Settings()