import os

from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[OpenAI] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatOpenAIWithRetry(ChatOpenAI):
    """ChatOpenAI with exponential backoff retry and logging."""
    
    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        def _call():
            return super(ChatOpenAIWithRetry, self)._generate(*args, **kwargs)
        return _call()
    
    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        async def _acall():
            return await super(ChatOpenAIWithRetry, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_openai_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatOpenAI:
    """
    Build a ``ChatOpenAIWithRetry`` client that requires the model name to be provided
    either as a parameter or via the OPENAI_MODEL_NAME environment variable.
    """
    model = model_name or os.getenv("OPENAI_MODEL_NAME")
    if not model:
        raise ValueError("OPENAI_MODEL_NAME environment variable must be set")
    api_key = kwargs.pop("api_key", None) or os.getenv("OPENAI_API_KEY")

    client_kwargs = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        client_kwargs["api_key"] = api_key

    return ChatOpenAIWithRetry(**client_kwargs)

