from __future__ import annotations

from functools import lru_cache
import os

from gea_agent.config import SETTINGS

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional dependency
    ChatGroq = None

try:
    from langchain_mistralai import ChatMistralAI
except Exception:  # pragma: no cover - optional dependency
    ChatMistralAI = None


def _provider_candidates() -> list[str]:
    requested = SETTINGS.llm_provider
    if requested and requested != "auto":
        return [requested]

    candidates: list[str] = []
    if ChatMistralAI is not None and str(os.getenv("MISTRAL_API_KEY") or "").strip():
        candidates.append("mistral")
    if ChatGroq is not None and str(os.getenv("GROQ_API_KEY") or "").strip():
        candidates.append("groq")

    return candidates or ["mistral", "groq"]


@lru_cache(maxsize=1)
def get_llm():
    errors: list[str] = []

    for provider in _provider_candidates():
        if provider == "mistral":
            if ChatMistralAI is None:
                errors.append("Mistral provider requested but `langchain_mistralai` is not installed.")
                continue
            try:
                return ChatMistralAI(
                    model=SETTINGS.mistral_model,
                    temperature=SETTINGS.temperature,
                )
            except Exception as exc:
                errors.append(f"Failed to initialize Mistral: {exc}")
                continue

        if provider == "groq":
            if ChatGroq is None:
                errors.append("Groq provider requested but `langchain_groq` is not installed.")
                continue
            try:
                return ChatGroq(
                    model=SETTINGS.groq_model,
                    temperature=SETTINGS.temperature,
                )
            except Exception as exc:
                errors.append(f"Failed to initialize Groq: {exc}")
                continue

        errors.append(f"Unsupported LLM provider: {provider}")

    provider_hint = (
        "Set `LLM_PROVIDER` to `mistral` or `groq`, and provide the matching API key "
        "(`MISTRAL_API_KEY` or `GROQ_API_KEY`)."
    )
    details = " | ".join(errors) if errors else "No provider could be initialized."
    raise RuntimeError(f"Unable to initialize an LLM client. {provider_hint} Details: {details}")
