"""Bing image search."""

from typing import List
import requests
from bs4 import BeautifulSoup
import json
import re


def search_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on Bing.
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    try:
        url = "https://www.bing.com/images/search"
        
        params = {
            "q": query,
            "form": "HDRSC2",
            "first": 1,
            "tsc": "ImageHoverTitle"
        }
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.bing.com/"
        }
        
        response = requests.get(url, params=params, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # Parse JSON from anchor tags
        for a_tag in soup.find_all('a', class_='iusc'):
            if len(results) >= max_results:
                break
            
            m_attr = a_tag.get('m')
            if m_attr:
                data = json.loads(m_attr)
                image_url = data.get("murl")
                if image_url:
                    results.append({
                        "url": image_url,
                        "thumbnail_url": data.get("turl", image_url),
                        "description": data.get("t", ""),
                        "author": data.get("purl", "Unknown")
                    })
        
        # Fallback: parse from script tags
        if not results:
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'murl' in script.string:
                    urls = re.findall(r'"murl":"(https?://[^"]+)"', script.string)
                    for image_url in urls[:max_results]:
                        results.append({
                            "url": image_url,
                            "thumbnail_url": image_url,
                            "description": "",
                            "author": "Unknown"
                        })
                    break
        
        return results[:max_results]
        
    except Exception as e:
        return [{"error": f"Bing image search failed: {str(e)}"}]

