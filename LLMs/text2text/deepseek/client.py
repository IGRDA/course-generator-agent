import os

from langchain_deepseek import ChatDeepSeek
from tenacity import retry, stop_after_attempt, wait_exponential


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[DeepSeek] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatDeepSeekWithRetry(ChatDeepSeek):
    """ChatDeepSeek with exponential backoff retry."""
    
    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=10, min=10, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        def _call():
            return super(ChatDeepSeekWithRetry, self)._generate(*args, **kwargs)
        return _call()
    
    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=10, min=10, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        async def _acall():
            return await super(ChatDeepSeekWithRetry, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_deepseek_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    **kwargs,
) -> ChatDeepSeek:
    """
    Build a ChatDeepSeek client using the official langchain-deepseek package.
    Supports structured output with deepseek-chat (DeepSeek-V3).
    """
    model = model_name or os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    if not model:
        raise ValueError("DEEPSEEK_MODEL_NAME environment variable must be set")
    
    api_key = kwargs.pop("api_key", None) or os.getenv("DEEPSEEK_API_KEY")

    return ChatDeepSeekWithRetry(
        model=model,
        temperature=temperature,
        api_key=api_key,
        max_retries=0,  # Disable internal retries, we handle them
        **kwargs,
    )
