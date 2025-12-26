import os

from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[Mistral] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatMistral(ChatMistralAI):
    """ChatMistralAI with exponential backoff retry (20s-500s)."""
    
    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=8, min=8, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        def _call():
            return super(ChatMistral, self)._generate(*args, **kwargs)
        return _call()
    
    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=8, min=8, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        async def _acall():
            return await super(ChatMistral, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_mistral_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 5,
    **kwargs,
) -> ChatMistralAI:
    """Build a ChatMistralAI client with custom exponential backoff."""
    model = model_name or os.getenv("MISTRAL_MODEL_NAME")
    if not model:
        raise ValueError("MISTRAL_MODEL_NAME environment variable must be set")
    mistral_api_key = (
        kwargs.pop("mistral_api_key", None)
        or kwargs.pop("api_key", None)
        or os.getenv("MISTRAL_API_KEY")
    )

    return ChatMistral(
        model=model,
        temperature=temperature,
        max_retries=0,  # Disable internal retries
        mistral_api_key=mistral_api_key,
        timeout=360,
        **kwargs,
    )
