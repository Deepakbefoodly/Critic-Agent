"""
LangChain model factory.

All agents import get_critic_llm() or get_fast_llm() from here.
Switching providers (e.g. OpenAI, Anthropic) only requires changing this file.
"""

import os
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from config import get_settings


def _make_llm(model_name: str, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    api_key = settings.google_api_key or os.getenv("GOOGLE_API_KEY", "")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=False,
    )


@lru_cache(maxsize=1)
def get_critic_llm() -> ChatGoogleGenerativeAI:
    """Full-quality model for critic scoring and synthesis."""
    return _make_llm(get_settings().critic_model)


@lru_cache(maxsize=1)
def get_fast_llm() -> ChatGoogleGenerativeAI:
    """Cheap + fast model for rubric building and gap detection."""
    return _make_llm(get_settings().fast_model)