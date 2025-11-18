import os
from typing import Callable, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from .groq.client import build_groq_chat_model
from .mistral.client import build_mistral_chat_model
from .openai.client import build_openai_chat_model

Builder = Callable[..., BaseChatModel]

BUILDERS: Dict[str, Builder] = {
    "groq": build_groq_chat_model,
    "mistral": build_mistral_chat_model,
    "openai": build_openai_chat_model,
}

MODEL_ENV_VARS = {
    "groq": "GROQ_MODEL_NAME",
    "mistral": "MISTRAL_MODEL_NAME",
    "openai": "OPENAI_MODEL_NAME",
}


def available_text_llms() -> list[str]:
    """Return the list of registered providers."""
    return sorted(BUILDERS.keys())


def create_text_llm(provider: str | None = None, **kwargs) -> BaseChatModel:
    """
    Instantiate a text-to-text LLM using the registered factory builders.

    Args:
        provider: Optional provider override. Falls back to the ``TEXT_LLM_PROVIDER``
            environment variable, then to ``mistral``.
        **kwargs: Extra keyword arguments forwarded to the underlying builder.

    Returns:
        A ``BaseChatModel`` ready for use in LangChain pipelines.
    """
    choice = provider or kwargs.pop("provider", None)
    if not choice:
        choice = os.getenv("TEXT_LLM_PROVIDER", "mistral")

    key = choice.lower()
    try:
        builder = BUILDERS[key]
    except KeyError as exc:
        available = ", ".join(available_text_llms())
        raise ValueError(
            f"Unsupported text LLM provider '{choice}'. "
            f"Available providers: {available}"
        ) from exc

    return builder(**kwargs)


def resolve_text_model_name(provider: str | None = None) -> str | None:
    """
    Return the provider-specific model override env (if any).
    """
    choice = provider or os.getenv("TEXT_LLM_PROVIDER", "")
    env_var = MODEL_ENV_VARS.get(choice.lower())
    return os.getenv(env_var) if env_var else None
