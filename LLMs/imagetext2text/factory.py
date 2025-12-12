import os
from typing import Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .pixtral.client import build_pixtral_chat_model

Builder = Callable[..., BaseChatModel]

BUILDERS: Dict[str, Builder] = {
    "pixtral": build_pixtral_chat_model,
}

MODEL_ENV_VARS = {
    "pixtral": "PIXTRAL_MODEL_NAME",
}


def available_vision_llms() -> list[str]:
    """Return the list of registered vision LLM providers."""
    return sorted(BUILDERS.keys())


def create_vision_llm(provider: str, **kwargs) -> BaseChatModel:
    """
    Instantiate a vision LLM using the registered factory builders.

    Args:
        provider: Vision LLM provider name (pixtral).
        **kwargs: Extra keyword arguments forwarded to the underlying builder.

    Returns:
        A ``BaseChatModel`` ready for use in LangChain pipelines with vision support.
    """
    if not provider:
        raise ValueError(
            "Provider is required. Must be one of: pixtral"
        )

    key = provider.lower()
    try:
        builder = BUILDERS[key]
    except KeyError as exc:
        available = ", ".join(available_vision_llms())
        raise ValueError(
            f"Unsupported vision LLM provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc

    return builder(**kwargs)


def resolve_vision_model_name(provider: str) -> str | None:
    """
    Return the provider-specific model name from environment variable.
    
    Args:
        provider: Vision LLM provider name (pixtral).
    
    Returns:
        Model name from environment variable, or None if not set.
    """
    if not provider:
        return None
    
    env_var = MODEL_ENV_VARS.get(provider.lower())
    return os.getenv(env_var) if env_var else None

