import os

from langchain_openai import ChatOpenAI

CHEAPEST_MODEL = "gpt-4o-mini"


def build_openai_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatOpenAI:
    """
    Build a ``ChatOpenAI`` client that defaults to OpenAI's most affordable
    GPT-4o-mini tier while still allowing overrides via env vars or kwargs.
    """
    model = model_name or os.getenv("OPENAI_MODEL_NAME", CHEAPEST_MODEL)
    api_key = kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY")

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key

    return ChatOpenAI(**client_kwargs)

