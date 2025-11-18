import os

from langchain_groq import ChatGroq

DEFAULT_MODEL = "kimi-k2-instruct-0905"


def build_groq_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatGroq:
    """
    Build a ``ChatGroq`` client configured for Groq's Kimi models.
    """
    model = model_name or os.getenv("GROQ_MODEL_NAME", DEFAULT_MODEL)
    api_key = kwargs.pop("api_key", None) or os.getenv("GROQ_API_KEY")

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key

    return ChatGroq(**client_kwargs)

