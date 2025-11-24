import os

from langchain_openai import ChatOpenAI


def build_openai_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatOpenAI:
    """
    Build a ``ChatOpenAI`` client that requires the model name to be provided
    either as a parameter or via the OPENAI_MODEL_NAME environment variable.
    """
    model = model_name or os.getenv("OPENAI_MODEL_NAME")
    if not model:
        raise ValueError("OPENAI_MODEL_NAME environment variable must be set")
    api_key = kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY")

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key

    return ChatOpenAI(**client_kwargs)

