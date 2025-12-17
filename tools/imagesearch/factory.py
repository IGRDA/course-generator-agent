from typing import Callable, Dict, List

from .ddg.client import search_images as ddg_search_images
from .bing.client import search_images as bing_search_images
from .freepik.client import search_images as freepik_search_images
from .google.client import search_images as google_search_images

ImageSearchFunc = Callable[[str, int], List[dict]]

IMAGE_SEARCH_PROVIDERS: Dict[str, ImageSearchFunc] = {
    "ddg": ddg_search_images,
    "bing": bing_search_images,
    "freepik": freepik_search_images,
    "google": google_search_images,
}


def available_image_search_providers() -> list[str]:
    """Return the list of registered image search providers."""
    return sorted(IMAGE_SEARCH_PROVIDERS.keys())


def create_image_search(provider: str) -> ImageSearchFunc:
    """
    Get image search function for the specified provider.
    
    Args:
        provider: Image search provider name (ddg | bing | freepik).
        
    Returns:
        An image search function that accepts (query: str, max_results: int).
    """
    if not provider:
        available = ", ".join(available_image_search_providers())
        raise ValueError(f"Provider is required. Available providers: {available}")
    
    key = provider.lower()
    try:
        return IMAGE_SEARCH_PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(available_image_search_providers())
        raise ValueError(
            f"Unsupported image search provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc

