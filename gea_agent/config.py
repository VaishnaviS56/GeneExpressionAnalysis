from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def find_dotenv(*args, **kwargs) -> str:
        return ""

    def load_dotenv(*args, **kwargs) -> bool:
        return False


load_dotenv(find_dotenv(usecwd=True), override=False)


class Settings:
    def __init__(self) -> None:
        env_file = find_dotenv(usecwd=True)
        self.repo_root = Path(env_file).resolve().parent if env_file else Path.cwd().resolve()

        self.llm_provider: str = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        self.groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3-27b-it.gguf:latest")
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

        self.http_timeout_seconds: int = int(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))
        self.http_max_retries: int = int(os.getenv("HTTP_MAX_RETRIES", "3"))
        self.http_backoff_factor: float = float(os.getenv("HTTP_BACKOFF_FACTOR", "0.5"))

        self.string_required_score: int = int(os.getenv("STRING_REQUIRED_SCORE", "900"))

        self.neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password: str = os.getenv("NEO4J_PASSWORD", "primekg123")
        self.neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j").strip()

        self.deg_r_script_path: str = self.resolve_path(
            os.getenv("DEG_R_SCRIPT_PATH", r"GEA\dee2_t2d.R")
        )
        self.deg_supporting_files_dir: str = self.resolve_path(
            os.getenv("DEG_SUPPORTING_FILES_DIR", r"GEA")
        )
        self.deg_output_csv_path: str = self.resolve_path(
            os.getenv("DEG_OUTPUT_CSV_PATH", r"GEA\DEG_T2D_LFC1.csv")
        )
        self.rscript_executable: str = os.getenv(
            "RSCRIPT_EXECUTABLE",
            os.getenv("R_EXECUTABLE", "Rscript"),
        )

        self.string_info_path: str = self.resolve_path(
            os.getenv(
                "STRING_INFO_PATH",
                r"9606.protein.info.v12.0.txt\9606.protein.info.v12.0.txt",
            )
        )
        self.string_links_path: str = self.resolve_path(
            os.getenv(
                "STRING_LINKS_PATH",
                r"9606.protein.links.v12.0.min900.txt\9606.protein.links.v12.0.min900.txt",
            )
        )
        self.string_local_mode: str = os.getenv("STRING_LOCAL_MODE", "seed_1hop_closed")
        self.string_graph_cache_path: str = self.resolve_path(
            os.getenv("STRING_GRAPH_CACHE_PATH", r"string_full_graph.pkl")
        )
        self.string_force_rebuild: bool = os.getenv("STRING_FORCE_REBUILD", "0") == "1"

        self.primekg_csv_path: str = self.resolve_path(os.getenv("PRIMEKG_CSV_PATH", "kg.csv"))
        self.streamlit_draw_graph: bool = os.getenv("STREAMLIT_DRAW_GRAPH", "0") == "1"

    def resolve_path(self, raw_path: str) -> str:
        path = Path(str(raw_path or "").strip())
        if not path:
            return str(self.repo_root)
        if path.is_absolute():
            return str(path)
        return str((self.repo_root / path).resolve())


SETTINGS = Settings()
