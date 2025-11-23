from typing import Callable, Dict, List

from .bing.client import search_videos as bing_search_videos

VideoSearchFunc = Callable[[str, int], List[dict]]

VIDEO_SEARCH_PROVIDERS: Dict[str, VideoSearchFunc] = {
    "bing": bing_search_videos,
}


def available_video_search_providers() -> list[str]:
    """Return the list of registered video search providers."""
    return sorted(VIDEO_SEARCH_PROVIDERS.keys())


def create_video_search(provider: str) -> VideoSearchFunc:
    """
    Get video search function for the specified provider.
    
    Args:
        provider: Video search provider name (bing).
        
    Returns:
        A video search function that accepts (query: str, max_results: int).
    """
    if not provider:
        raise ValueError("Provider is required. Must be one of: bing")
    
    key = provider.lower()
    try:
        return VIDEO_SEARCH_PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(available_video_search_providers())
        raise ValueError(
            f"Unsupported video search provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc

