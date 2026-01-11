import os
from typing import Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .deepseek.client import build_deepseek_chat_model
from .gemini.client import build_gemini_chat_model
from .groq.client import build_groq_chat_model
from .mistral.client import build_mistral_chat_model
from .openai.client import build_openai_chat_model

Builder = Callable[..., BaseChatModel]

BUILDERS: Dict[str, Builder] = {
    "deepseek": build_deepseek_chat_model,
    "gemini": build_gemini_chat_model,
    "groq": build_groq_chat_model,
    "mistral": build_mistral_chat_model,
    "openai": build_openai_chat_model,
}

MODEL_ENV_VARS = {
    "deepseek": "DEEPSEEK_MODEL_NAME",
    "gemini": "GEMINI_MODEL_NAME",
    "groq": "GROQ_MODEL_NAME",
    "mistral": "MISTRAL_MODEL_NAME",
    "openai": "OPENAI_MODEL_NAME",
}


def available_text_llms() -> list[str]:
    """Return the list of registered providers."""
    return sorted(BUILDERS.keys())


def create_text_llm(provider: str, **kwargs) -> BaseChatModel:
    """
    Instantiate a text-to-text LLM using the registered factory builders.

    Args:
        provider: LLM provider name (mistral | gemini | groq | openai).
        **kwargs: Extra keyword arguments forwarded to the underlying builder.

    Returns:
        A ``BaseChatModel`` ready for use in LangChain pipelines.
    """
    if not provider:
        raise ValueError(
            "Provider is required. Must be one of: deepseek, mistral, gemini, groq, openai"
        )

    key = provider.lower()
    try:
        builder = BUILDERS[key]
    except KeyError as exc:
        available = ", ".join(available_text_llms())
        raise ValueError(
            f"Unsupported text LLM provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc

    return builder(**kwargs)


def resolve_text_model_name(provider: str) -> str | None:
    """
    Return the provider-specific model name from environment variable.
    
    Args:
        provider: LLM provider name (mistral | gemini | groq | openai).
    
    Returns:
        Model name from environment variable, or None if not set.
    """
    if not provider:
        return None
    
    env_var = MODEL_ENV_VARS.get(provider.lower())
    return os.getenv(env_var) if env_var else None
