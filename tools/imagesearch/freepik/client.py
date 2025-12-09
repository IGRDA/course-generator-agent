"""Freepik image search."""

import os
from typing import List
import requests

def search_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on Freepik.
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    api_key = os.environ.get("FREEPIK_API_KEY")
    if not api_key:
        return [{"error": "FREEPIK_API_KEY not found in environment variables"}]

    url = "https://api.freepik.com/v1/resources"
    
    headers = {
        "x-freepik-api-key": api_key,
        "Accept-Language": "en-US"
    }
    
    # 'limit' is the standard search parameter for Freepik resources endpoint
    params = {
        "term": query,
        "limit": max_results,
        "page": 1
    }
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    results = []
    
    items = data.get("data", [])
    for item in items:
        # Get the image preview URL
        image_source = item.get("image", {}).get("source", {})
        image_url = image_source.get("url")
        
        # If no direct image URL, skip
        if not image_url:
            continue
            
        results.append({
            "url": image_url,
            "thumbnail_url": image_url,
            "description": item.get("title", ""),
            "author": item.get("author", {}).get("name", "Unknown"),
            "page_url": item.get("url") # Additional context
        })
        
    return results

if __name__ == "__main__":
    # Test script
    query = "hacker"
    print(f"🔍 Searching images for: '{query}'")
    
    results = search_images(query, 5)
    
    if results and "error" in results[0]:
        print(f"❌ {results[0]['error']}")
    else:
        print(f"\n📸 Found {len(results)} images:\n")
        for i, img in enumerate(results, 1):
            print(f"{i}. {img.get('description', 'No description')}")
            print(f"   URL: {img['url']}")
            print(f"   Author: {img['author']}")
            print()

