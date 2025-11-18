import os

from langchain_google_genai import ChatGoogleGenerativeAI

CHEAPEST_MODEL = "gemini-1.5-flash"


def build_gemini_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    """
    Build a ``ChatGoogleGenerativeAI`` client that defaults to Gemini's most
    affordable Flash tier while still allowing overrides via env vars or kwargs.
    """
    model = model_name or os.getenv("GEMINI_MODEL_NAME", CHEAPEST_MODEL)
    api_key = (
        kwargs.pop("google_api_key", None)
        or kwargs.pop("api_key", None)
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        client_kwargs["google_api_key"] = api_key

    return ChatGoogleGenerativeAI(**client_kwargs)


