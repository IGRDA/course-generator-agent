"""Video search factory with provider abstraction."""

from typing import Callable, Dict, List
from pydantic import BaseModel, Field

from .youtube.client import search_videos as youtube_search_videos
from .bing.client import search_videos as bing_search_videos


class VideoResult(BaseModel):
    """Video search result with full metadata."""
    title: str = Field(..., description="Video title")
    url: str = Field(..., description="YouTube video URL")
    duration: int = Field(..., description="Video duration in seconds")
    published_at: int = Field(..., description="Publication timestamp in milliseconds")
    thumbnail: str = Field(..., description="Thumbnail URL")
    channel: str = Field(..., description="Channel name")
    views: int = Field(..., description="View count")
    likes: int = Field(..., description="Like count")


VideoSearchFunc = Callable[[str, int], List[dict]]

VIDEO_SEARCH_PROVIDERS: Dict[str, VideoSearchFunc] = {
    "youtube": youtube_search_videos,  # Primary provider (yt-dlp, full metadata)
    "bing": bing_search_videos,  # Deprecated: minimal metadata, kept for fallback
}


def available_video_search_providers() -> list[str]:
    """Return the list of registered video search providers."""
    return sorted(VIDEO_SEARCH_PROVIDERS.keys())


def create_video_search(provider: str = "youtube") -> VideoSearchFunc:
    """
    Get video search function for the specified provider.
    
    Args:
        provider: Video search provider name. Options:
            - "youtube" (default): Full metadata via yt-dlp (title, url, duration,
              published_at, thumbnail, channel, views, likes)
            - "bing": Deprecated, minimal metadata (url only)
        
    Returns:
        A video search function that accepts (query: str, max_results: int).
    """
    if not provider:
        provider = "youtube"
    
    key = provider.lower()
    try:
        return VIDEO_SEARCH_PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(available_video_search_providers())
        raise ValueError(
            f"Unsupported video search provider '{provider}'. "
            f"Available providers: {available}"
        ) from exc
