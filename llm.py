"""OpenRouter-backed chat model configuration."""

import os

from langchain_openai import ChatOpenAI

AVAILABLE_MODELS = {
    "OpenAI GPT-4o mini": "openai/gpt-4o-mini",
    "Google Gemini Flash 2.0": "google/gemini-2.0-flash-001",
    "Anthropic Claude 3.5 Haiku": "anthropic/claude-3.5-haiku",
}
DEFAULT_MODEL = "openai/gpt-4o-mini"


def get_llm(model: str) -> ChatOpenAI:
    """Create a chat model configured to use OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        default_headers={"X-Title": "Hopscotch Support"},
    )
