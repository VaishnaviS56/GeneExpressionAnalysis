from __future__ import annotations

from langchain_groq import ChatGroq

from gea_agent.config import SETTINGS


def get_llm() -> ChatGroq:
    return ChatGroq(model=SETTINGS.groq_model, temperature=SETTINGS.temperature)

