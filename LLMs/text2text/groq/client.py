import os

from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[Groq] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatGroqWithRetry(ChatGroq):
    """ChatGroq with exponential backoff retry and logging."""
    
    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        def _call():
            return super(ChatGroqWithRetry, self)._generate(*args, **kwargs)
        return _call()
    
    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        async def _acall():
            return await super(ChatGroqWithRetry, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_groq_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatGroq:
    """
    Build a ``ChatGroqWithRetry`` client that requires the model name to be provided
    either as a parameter or via the GROQ_MODEL_NAME environment variable.
    """
    model = model_name or os.getenv("GROQ_MODEL_NAME")
    if not model:
        raise ValueError("GROQ_MODEL_NAME environment variable must be set")
    api_key = kwargs.pop("api_key", None) or os.getenv("GROQ_API_KEY")

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key

    return ChatGroqWithRetry(**client_kwargs)

