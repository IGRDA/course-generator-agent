"""Tools module for course generator utilities."""

from .websearch import create_web_search, available_search_providers
from .imagesearch import create_image_search, available_image_search_providers
from .videosearch import create_video_search, available_video_search_providers

__all__ = [
    "create_web_search",
    "available_search_providers",
    "create_image_search",
    "available_image_search_providers",
    "create_video_search",
    "available_video_search_providers"
]
