import os

from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential


def _log_retry(retry_state):
    """Log retry attempts to console."""
    print(f"[Pixtral] Retrying call (attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}")


class ChatPixtral(ChatMistralAI):
    """ChatMistralAI for Pixtral vision model with exponential backoff retry (20s-500s)."""
    
    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        def _call():
            return super(ChatPixtral, self)._generate(*args, **kwargs)
        return _call()
    
    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True,
            before_sleep=_log_retry
        )
        async def _acall():
            return await super(ChatPixtral, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_pixtral_chat_model(
    model_name: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 5,
    **kwargs,
) -> ChatMistralAI:
    """
    Build a ChatMistralAI client configured for Pixtral vision model.
    
    Pixtral is Mistral's vision model that can analyze images.
    Uses PIXTRAL_API_KEY or falls back to MISTRAL_API_KEY.
    
    Args:
        model_name: Model name override. Defaults to PIXTRAL_MODEL_NAME env var
                   or 'pixtral-large-latest'.
        temperature: Sampling temperature (default 0.1 for consistent scoring).
        max_retries: Number of retries (handled by custom exponential backoff).
        **kwargs: Additional arguments passed to ChatMistralAI.
    
    Returns:
        ChatPixtral instance configured for vision tasks.
    """
    model = model_name or os.getenv("PIXTRAL_MODEL_NAME", "pixtral-large-latest")
    
    # Try PIXTRAL_API_KEY first, then fall back to MISTRAL_API_KEY
    api_key = (
        kwargs.pop("pixtral_api_key", None)
        or kwargs.pop("mistral_api_key", None)
        or kwargs.pop("api_key", None)
        or os.getenv("PIXTRAL_API_KEY")
        or os.getenv("MISTRAL_API_KEY")
    )
    
    if not api_key:
        raise ValueError(
            "PIXTRAL_API_KEY or MISTRAL_API_KEY environment variable must be set"
        )

    return ChatPixtral(
        model=model,
        temperature=temperature,
        max_retries=0,  # Disable internal retries, we use custom exponential backoff
        mistral_api_key=api_key,
        **kwargs,
    )

