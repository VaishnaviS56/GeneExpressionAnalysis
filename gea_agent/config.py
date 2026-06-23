from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv(usecwd=True), override=False)


class Settings:
    # groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    string_required_score: int = int(os.getenv("STRING_REQUIRED_SCORE", "900"))

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "primekg123"

    # deg_r_script_path: str = os.getenv(
    #     "DEG_R_SCRIPT_PATH",
    #     r"C:\Vaishanvi\TCS\GEA\dee2_data.R",
    # )
    # deg_supporting_files_dir: str = os.getenv(
    #     "DEG_SUPPORTING_FILES_DIR",
    #     r"C:\Vaishanvi\TCS\GEA",
    # )
    # deg_output_csv_path: str = os.getenv(
    #     "DEG_OUTPUT_CSV_PATH",
    #     r"C:\Vaishanvi\TCS\GEA\\DEG_T2D_LFC1.csv",
    # )
    deg_r_script_path: str = os.getenv(
        "DEG_R_SCRIPT_PATH",
        r"GEA\dee2_t2d.R",
    )
    deg_supporting_files_dir: str = os.getenv(
        "DEG_SUPPORTING_FILES_DIR",
        r"GEA",
    )
    deg_output_csv_path: str = os.getenv(
        "DEG_OUTPUT_CSV_PATH",
        r"GEA\DEG_T2D_LFC1.csv",
    )
    rscript_executable: str = os.getenv(
        "RSCRIPT_EXECUTABLE",
        os.getenv("R_EXECUTABLE", "Rscript"),
    )

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
    # string_local_mode: str = os.getenv("STRING_LOCAL_MODE", "full")

    # Cache for full STRING graph (pickle)
    string_graph_cache_path: str = os.getenv("STRING_GRAPH_CACHE_PATH", r"string_full_graph.pkl")
    string_force_rebuild: bool = os.getenv("STRING_FORCE_REBUILD", "0") == "1"

    primekg_csv_path: str = os.getenv("PRIMEKG_CSV_PATH", "kg.csv")


SETTINGS = Settings()
