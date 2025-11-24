import os

from langchain_groq import ChatGroq


def build_groq_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatGroq:
    """
    Build a ``ChatGroq`` client that requires the model name to be provided
    either as a parameter or via the GROQ_MODEL_NAME environment variable.
    """
    model = model_name or os.getenv("GROQ_MODEL_NAME")
    if not model:
        raise ValueError("GROQ_MODEL_NAME environment variable must be set")
    api_key = kwargs.pop("api_key", None) or os.getenv("GROQ_API_KEY")

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key

    return ChatGroq(**client_kwargs)

