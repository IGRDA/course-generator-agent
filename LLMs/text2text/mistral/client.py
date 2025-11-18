import os

from langchain_mistralai import ChatMistralAI

DEFAULT_MODEL = "mistral-small-latest"


def build_mistral_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatMistralAI:
    """
    Build a ``ChatMistralAI`` client for text generation tasks.

    Notes:
        Pixtral uses a dedicated API key, so this helper only reads
        ``MISTRAL_API_KEY`` for text models. Override ``mistral_api_key`` in
        kwargs if you need to supply a different credential.
    """
    model = model_name or os.getenv("MISTRAL_MODEL_NAME", DEFAULT_MODEL)
    mistral_api_key = (
        kwargs.pop("mistral_api_key", None)
        or kwargs.pop("api_key", None)
        or os.getenv("MISTRAL_API_KEY")
    )

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if mistral_api_key:
        client_kwargs["mistral_api_key"] = mistral_api_key

    return ChatMistralAI(**client_kwargs)

