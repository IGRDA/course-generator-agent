"""Freepik image search using Selenium web scraping."""

from typing import List
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

def _create_driver():
    """Create a Chrome driver with anti-detection settings."""
    chrome_options = Options()
    
    # Performance optimizations
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    
    # Bypass detection
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Realistic browser headers
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    chrome_options.add_argument('--accept-language=en-US,en;q=0.9')
    chrome_options.add_argument('--accept-encoding=gzip, deflate, br')
    
    # Additional anti-fingerprinting
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-plugins-discovery')
    chrome_options.add_argument('--start-maximized')
    
    driver = webdriver.Chrome(options=chrome_options)
    
    # Override navigator properties to avoid detection
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def search_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on Freepik using web scraping.
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    driver = None
    try:
        driver = _create_driver()
        
        # Build search URL
        search_url = f"https://www.freepik.com/search?format=search&query={quote(query)}"
        driver.get(search_url)
        
        # Wait for images to load
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "figure img, img[data-src]")))
        
        # Small delay to ensure dynamic content loads
        time.sleep(2)
        
        results = []
        
        # Find image elements - Freepik uses various selectors
        image_elements = driver.find_elements(By.CSS_SELECTOR, "figure img[src], img[data-src]")
        
        for img_element in image_elements[:max_results]:
            try:
                # Try to get the image URL from src or data-src
                image_url = img_element.get_attribute('src') or img_element.get_attribute('data-src')
                
                # Skip if no URL or if it's a placeholder/loading image
                if not image_url or 'placeholder' in image_url.lower() or 'loading' in image_url.lower():
                    continue
                
                # Get alt text as description
                description = img_element.get_attribute('alt') or "No description"
                
                # Try to get the parent link for page URL
                page_url = ""
                try:
                    parent_link = img_element.find_element(By.XPATH, "./ancestor::a")
                    page_url = parent_link.get_attribute('href') or ""
                except NoSuchElementException:
                    pass
                
                results.append({
                    "url": image_url,
                    "thumbnail_url": image_url,
                    "description": description,
                    "author": "Freepik",
                    "page_url": page_url
                })
                
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                # Skip problematic elements
                continue
        
        return results if results else [{"error": "No images found"}]
        
    except TimeoutException:
        return [{"error": "Timeout waiting for images to load. Freepik may be blocking requests."}]
    except Exception as e:
        return [{"error": f"Error scraping Freepik: {str(e)}"}]
    finally:
        if driver:
            driver.quit()

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

