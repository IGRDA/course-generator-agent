"""Openverse image search."""

from typing import List
import requests


def search_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on Openverse (Creative Commons).
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    try:
        url = "https://api.openverse.org/v1/images/"
        params = {"q": query, "page_size": max_results}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for image in data.get("results", []):
            results.append({
                "url": image["url"],
                "thumbnail_url": image.get("thumbnail") or image["url"],
                "description": image.get("title", ""),
                "author": image.get("creator", "Unknown"),
                "license": image.get("license", "")
            })
        
        return results
        
    except Exception as e:
        return [{"error": f"Openverse search failed: {str(e)}"}]

