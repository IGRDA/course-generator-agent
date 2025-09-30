"""Bing image search scraper tool (no API key required)."""

from typing import List
from langchain_core.tools import tool


def is_valid_image_url(url: str) -> bool:
    """Validate if URL points to an actual image."""
    if not url or not isinstance(url, str):
        return False
    
    # Must be HTTP/HTTPS
    if not url.startswith(('http://', 'https://')):
        return False
    
    # Common image extensions
    image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg')
    url_lower = url.lower()
    
    # Check extension or format parameter
    has_image_ext = any(ext in url_lower for ext in image_exts)
    has_format_param = 'format=jpg' in url_lower or 'format=png' in url_lower
    
    return has_image_ext or has_format_param


@tool
def search_bing_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on Bing by scraping search results.
    No API key required, no rate limits.
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    import requests
    from bs4 import BeautifulSoup
    import json
    import re
    
    try:
        url = "https://www.bing.com/images/search"
        params = {"q": query, "first": 1, "count": max_results * 2}
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Method 1: Parse JSON from 'm' attribute in anchor tags
        for a_tag in soup.find_all('a', class_='iusc'):
            if len(results) >= max_results:
                break
            
            m_attr = a_tag.get('m')
            if m_attr:
                try:
                    data = json.loads(m_attr)
                    image_url = data.get("murl", "")
                    
                    # Only add if it's a valid image URL
                    if image_url and is_valid_image_url(image_url):
                        results.append({
                            "url": image_url,
                            "thumbnail_url": data.get("turl", image_url),
                            "description": data.get("t", ""),
                            "author": "Unknown"
                        })
                except:
                    continue
        
        # Method 2: Parse from script tags containing image data
        if len(results) < max_results:
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'murl' in script.string:
                    # Find all image URLs in script
                    urls = re.findall(r'"murl":"(https?://[^"]+)"', script.string)
                    for img_url in urls:
                        if len(results) >= max_results:
                            break
                        if is_valid_image_url(img_url):
                            results.append({
                                "url": img_url,
                                "thumbnail_url": img_url,
                                "description": "",
                                "author": "Unknown"
                            })
        
        return results if results else [{"error": "No images found"}]
        
    except Exception as e:
        return [{"error": f"Bing search failed: {str(e)}"}]


if __name__ == "__main__":
    results = search_bing_images.invoke({"query": "petrov defense", "max_results": 10})
    print(f"Found {len(results)} images:")
    for i, img in enumerate(results, 1):
        if "error" in img:
            print(f"  {i}. Error: {img['error']}")
        else:
            print(f"  {i}. {img['description'][:50] if img['description'] else 'No description'}")
            print(f"     URL: {img['url'][:80]}...")
