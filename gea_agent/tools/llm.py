from __future__ import annotations

from functools import lru_cache
import json
import os
import re
from typing import Any

import httpx

from gea_agent.config import SETTINGS

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional dependency
    ChatGroq = None

try:
    from langchain_mistralai import ChatMistralAI
except Exception:  # pragma: no cover - optional dependency
    ChatMistralAI = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None


class LLMConnectivityError(RuntimeError):
    """Raised when every configured LLM provider fails due to connectivity issues."""


def is_gemini_family_provider() -> bool:
    provider = str(SETTINGS.llm_provider or "").strip().lower()
    model_name = str(getattr(SETTINGS, "gemini_model", "") or "").strip().lower()
    return provider == "gemini" or "gemini" in model_name or "gemma" in model_name


def parse_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}

    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1].strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _provider_candidates() -> list[str]:
    requested = SETTINGS.llm_provider
    if requested and requested != "auto":
        return [requested]

    candidates: list[str] = []
    if ChatGoogleGenerativeAI is not None and str(os.getenv("GOOGLE_API_KEY") or "").strip():
        candidates.append("gemini")
    if ChatMistralAI is not None and str(os.getenv("MISTRAL_API_KEY") or "").strip():
        candidates.append("mistral")
    if ChatGroq is not None and str(os.getenv("GROQ_API_KEY") or "").strip():
        candidates.append("groq")

    return candidates or ["gemini", "mistral", "groq"]


def _timeout_seconds() -> float:
    return max(5.0, float(getattr(SETTINGS, "http_timeout_seconds", 30) or 30))


def _build_provider_specs() -> list[dict[str, Any]]:
    timeout = _timeout_seconds()
    specs: list[dict[str, Any]] = []

    for provider in _provider_candidates():
        if provider == "gemini":
            if ChatGoogleGenerativeAI is None:
                specs.append(
                    {
                        "name": "gemini",
                        "factory_error": "Gemini provider requested but `langchain_google_genai` is not installed.",
                    }
                )
                continue
            specs.append(
                {
                    "name": "gemini",
                    "factory": lambda: ChatGoogleGenerativeAI(
                        model=SETTINGS.gemini_model,
                        temperature=SETTINGS.temperature,
                        request_timeout=timeout,
                        retries=0,
                    ),
                }
            )
            continue

        if provider == "mistral":
            if ChatMistralAI is None:
                specs.append(
                    {
                        "name": "mistral",
                        "factory_error": "Mistral provider requested but `langchain_mistralai` is not installed.",
                    }
                )
                continue
            specs.append(
                {
                    "name": "mistral",
                    "factory": lambda: ChatMistralAI(
                        model_name=SETTINGS.mistral_model,
                        temperature=SETTINGS.temperature,
                        timeout=int(timeout),
                        max_retries=0,
                    ),
                }
            )
            continue

        if provider == "groq":
            if ChatGroq is None:
                specs.append(
                    {
                        "name": "groq",
                        "factory_error": "Groq provider requested but `langchain_groq` is not installed.",
                    }
                )
                continue
            specs.append(
                {
                    "name": "groq",
                    "factory": lambda: ChatGroq(
                        model=SETTINGS.groq_model,
                        temperature=SETTINGS.temperature,
                        timeout=timeout,
                        max_retries=0,
                        http_client=httpx.Client(timeout=timeout, trust_env=False),
                        http_async_client=httpx.AsyncClient(timeout=timeout, trust_env=False),
                    ),
                }
            )
            continue

        specs.append({"name": provider, "factory_error": f"Unsupported LLM provider: {provider}"})

    return specs


def _is_connectivity_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        message = str(current).lower()
        if any(
            token in message
            for token in (
                "winerror 10013",
                "connection error",
                "connecterror",
                "unable to connect",
                "socket",
                "timed out",
                "timeout",
                "temporarily unavailable",
                "connection refused",
                "network is unreachable",
                "dns",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _format_llm_failure(errors: list[str], connectivity_failures: int) -> str:
    provider_hint = (
        "Set `LLM_PROVIDER` to `gemini`, `mistral`, or `groq`, and provide the matching API key "
        "(`GOOGLE_API_KEY`, `MISTRAL_API_KEY`, or `GROQ_API_KEY`). "
        "Optionally set `GEMINI_MODEL`, `MISTRAL_MODEL`, or `GROQ_MODEL`."
    )
    details = " | ".join(errors) if errors else "No provider could be initialized."
    if connectivity_failures:
        return (
            "Unable to reach any configured LLM provider. Outbound HTTPS from this environment appears to be blocked "
            "or denied, so LLM-backed features cannot complete network calls right now. "
            f"Details: {details}"
        )
    return f"Unable to initialize an LLM client. {provider_hint} Details: {details}"


class ResilientLLM:
    def __init__(
        self,
        provider_specs: list[dict[str, Any]],
        bound_tools: Any | None = None,
    ) -> None:
        self._provider_specs = provider_specs
        self._bound_tools = bound_tools
        self._clients: dict[str, Any] = {}

    def bind_tools(self, tools: Any):
        return ResilientLLM(self._provider_specs, bound_tools=tools)

    def _get_provider_client(self, spec: dict[str, Any]):
        name = str(spec["name"])
        if name in self._clients:
            return self._clients[name]

        factory_error = spec.get("factory_error")
        if factory_error:
            raise RuntimeError(str(factory_error))

        factory = spec.get("factory")
        if not callable(factory):
            raise RuntimeError(f"Unsupported LLM provider: {name}")

        client = factory()
        self._clients[name] = client
        return client

    def invoke(self, *args: Any, **kwargs: Any):
        errors: list[str] = []
        connectivity_failures = 0

        for spec in self._provider_specs:
            name = str(spec["name"])
            try:
                client = self._get_provider_client(spec)
                runnable = client.bind_tools(self._bound_tools) if self._bound_tools is not None else client
                return runnable.invoke(*args, **kwargs)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                if _is_connectivity_error(exc):
                    connectivity_failures += 1
                continue

        message = _format_llm_failure(errors, connectivity_failures)
        if connectivity_failures:
            raise LLMConnectivityError(message)
        raise RuntimeError(message)


@lru_cache(maxsize=1)
def get_llm():
    return ResilientLLM(_build_provider_specs())
