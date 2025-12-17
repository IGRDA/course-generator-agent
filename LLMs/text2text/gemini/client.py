import os

from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[Gemini] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatGemini(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with exponential backoff retry and logging."""
    
    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        def _call():
            return super(ChatGemini, self)._generate(*args, **kwargs)
        return _call()
    
    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        async def _acall():
            return await super(ChatGemini, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_gemini_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    """
    Build a ``ChatGemini`` client that requires the model name to be provided
    either as a parameter or via the GEMINI_MODEL_NAME environment variable.
    """
    model = model_name or os.getenv("GEMINI_MODEL_NAME")
    if not model:
        raise ValueError("GEMINI_MODEL_NAME environment variable must be set")
    api_key = (
        kwargs.pop("google_api_key", None)
        or kwargs.pop("api_key", None)
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        client_kwargs["google_api_key"] = api_key

    return ChatGemini(**client_kwargs)


