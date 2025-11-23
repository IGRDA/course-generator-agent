import os

from langchain_mistralai import ChatMistralAI
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_MODEL = "mistral-small-latest"


class RetryingChatMistralAI(ChatMistralAI):
    """ChatMistralAI with exponential backoff retry (20s-500s)."""
    
    def _generate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True
        )
        def _call():
            return super(RetryingChatMistralAI, self)._generate(*args, **kwargs)
        return _call()
    
    async def _agenerate(self, *args, **kwargs):
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=20, min=20, max=500),
            reraise=True
        )
        async def _acall():
            return await super(RetryingChatMistralAI, self)._agenerate(*args, **kwargs)
        return await _acall()


def build_mistral_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
    max_retries: int = 5,
    **kwargs,
) -> ChatMistralAI:
    """Build a ChatMistralAI client with custom exponential backoff."""
    model = model_name or os.getenv("MISTRAL_MODEL_NAME", DEFAULT_MODEL)
    mistral_api_key = (
        kwargs.pop("mistral_api_key", None)
        or kwargs.pop("api_key", None)
        or os.getenv("MISTRAL_API_KEY")
    )

    return RetryingChatMistralAI(
        model=model,
        temperature=temperature,
        max_retries=0,  # Disable internal retries
        mistral_api_key=mistral_api_key,
        **kwargs,
    )
