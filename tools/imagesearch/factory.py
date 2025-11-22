from typing import Callable, Dict, List

from .ddg.client import search_images as ddg_search_images
from .openverse.client import search_images as openverse_search_images
from .bing.client import search_images as bing_search_images

ImageSearchFunc = Callable[[str, int], List[dict]]

IMAGE_SEARCH_PROVIDERS: Dict[str, ImageSearchFunc] = {
    "ddg": ddg_search_images,
    "openverse": openverse_search_images,
    "bing": bing_search_images,
}


def available_image_search_providers() -> list[str]:
    """Return the list of registered image search providers."""
    return sorted(IMAGE_SEARCH_PROVIDERS.keys())


def create_image_search(provider: str) -> ImageSearchFunc:
    """
    Get image search function for the specified provider.
    
    Args:
        provider: Image search provider name (ddg | openverse | bing).
        
    Returns:
        An image search function that accepts (query: str, max_results: int).
    """
    if not provider:
        raise ValueError("Provider is required. Must be one of: ddg, openverse, bing")
    
    key = provider.lower()
    try:
        return IMAGE_SEARCH_PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(available_image_search_providers())
        raise ValueError(
            f"Unsupported image search provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc

