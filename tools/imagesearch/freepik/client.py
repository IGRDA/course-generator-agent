"""Freepik image search using Playwright with thread-local browser instances for thread-safety."""

import atexit
import threading
from typing import List, Optional
from urllib.parse import quote

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, Playwright
from playwright_stealth import Stealth


class FreepikClient:
    """
    Thread-safe Freepik client with persistent browser per thread.
    
    Each thread gets its own browser instance stored in thread-local storage,
    ensuring thread-safety while maintaining browser reuse within each thread.
    """
    
    _local = threading.local()  # Thread-local storage for instances
    _instances_lock = threading.Lock()
    _all_instances: List["FreepikClient"] = []  # Track all instances for cleanup
    _cleanup_registered = False
    
    def __init__(self):
        """Initialize the client (use get_instance() instead)."""
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._request_count = 0
        self._max_requests_before_refresh = 100  # Refresh context to avoid memory issues
    
    @classmethod
    def get_instance(cls) -> "FreepikClient":
        """Get or create the thread-local instance."""
        # Check if this thread already has an instance
        instance = getattr(cls._local, 'instance', None)
        
        if instance is None:
            # Create new instance for this thread
            instance = cls()
            instance._initialize()
            cls._local.instance = instance
            
            # Track instance for cleanup
            with cls._instances_lock:
                cls._all_instances.append(instance)
                
                # Register cleanup on first instance creation
                if not cls._cleanup_registered:
                    atexit.register(cls._cleanup_all)
                    cls._cleanup_registered = True
        
        return instance
    
    def _initialize(self) -> None:
        """Initialize Playwright browser with anti-detection settings."""
        self._playwright = sync_playwright().start()
        
        # Launch browser with anti-detection options
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ]
        )
        
        self._create_context()
    
    def _create_context(self) -> None:
        """Create a new browser context with realistic settings."""
        if self._context:
            try:
                self._context.close()
            except Exception:
                pass
        
        self._context = self._browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
            }
        )
        
        self._page = self._context.new_page()
        
        # Apply stealth patches to avoid detection
        Stealth(navigator_platform_override='MacIntel').apply_stealth_sync(self._page)
        
        # Additional CDP-based anti-detection
        try:
            client = self._context.new_cdp_session(self._page)
            client.send('Network.setUserAgentOverride', {
                'userAgent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'platform': 'MacIntel',
            })
        except Exception:
            pass  # CDP might not be available in all configurations
        
        # Override navigator.webdriver via init script
        self._page.add_init_script('''
            Object.defineProperty(navigator, "webdriver", { get: () => undefined });
            window.chrome = { runtime: {} };
        ''')
        
        self._request_count = 0
    
    def _ensure_fresh_context(self) -> None:
        """Refresh context periodically to prevent memory issues."""
        self._request_count += 1
        if self._request_count >= self._max_requests_before_refresh:
            self._create_context()
    
    def search_images(self, query: str, max_results: int = 5) -> List[dict]:
        """
        Search for images on Freepik using web scraping.
        
        Args:
            query: Search query for images
            max_results: Maximum number of images to return (default: 5)
            
        Returns:
            List of image results with URLs and metadata
        """
        try:
            self._ensure_fresh_context()
            
            # Build search URL
            search_url = f"https://www.freepik.com/search?format=search&query={quote(query)}"
            
            # Navigate with faster wait strategy
            self._page.goto(search_url, wait_until='domcontentloaded', timeout=15000)
            
            # Wait for images to appear (faster than networkidle)
            try:
                self._page.wait_for_selector('figure img[src]', timeout=5000, state='attached')
            except Exception:
                pass  # Continue even if timeout - might still have images
            
            # Minimal additional wait for dynamic content
            self._page.wait_for_timeout(200)
            
            results = []
            
            # Find image elements
            image_elements = self._page.query_selector_all('figure img[src], img[data-src]')
            
            for img_element in image_elements[:max_results * 2]:  # Get more to filter
                try:
                    # Get image URL from src or data-src
                    image_url = img_element.get_attribute('src') or img_element.get_attribute('data-src')
                    
                    # Skip placeholders and loading images
                    if not image_url or 'placeholder' in image_url.lower() or 'loading' in image_url.lower():
                        continue
                    
                    # Get alt text as description
                    description = img_element.get_attribute('alt') or "No description"
                    
                    # Try to get parent link for page URL
                    page_url = ""
                    try:
                        page_url = img_element.evaluate('el => el.closest("a")?.href || ""')
                    except Exception:
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
                        
                except Exception:
                    continue
            
            return results if results else [{"error": "No images found"}]
            
        except Exception as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower():
                return [{"error": "Timeout waiting for images. Freepik may be blocking requests."}]
            return [{"error": f"Error scraping Freepik: {error_msg}"}]
    
    def close(self) -> None:
        """Clean up browser resources for this instance."""
        try:
            if self._page:
                self._page.close()
                self._page = None
            if self._context:
                self._context.close()
                self._context = None
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
        except Exception:
            pass
    
    @classmethod
    def _cleanup_all(cls) -> None:
        """Clean up all thread-local instances (called on program exit)."""
        with cls._instances_lock:
            for instance in cls._all_instances:
                try:
                    instance.close()
                except Exception:
                    pass
            cls._all_instances.clear()
    
    @classmethod
    def reset(cls) -> None:
        """Reset all instances (useful for testing)."""
        cls._cleanup_all()
        # Clear thread-local storage for current thread
        if hasattr(cls._local, 'instance'):
            del cls._local.instance


# Backward-compatible function API
def search_images(query: str, max_results: int = 5) -> List[dict]:
    """
    Search for images on Freepik using web scraping.
    
    This function maintains backward compatibility with the original API
    while using the optimized thread-local client internally.
    
    Args:
        query: Search query for images
        max_results: Maximum number of images to return (default: 5)
        
    Returns:
        List of image results with URLs and metadata
    """
    return FreepikClient.get_instance().search_images(query, max_results)


if __name__ == "__main__":
    import time
    
    # Test script with benchmark
    queries = ["hacker", "programming", "artificial intelligence", "chess strategy", "cybersecurity"]
    
    print("=" * 60)
    print("Freepik Image Search - Playwright Thread-Safe")
    print("=" * 60)
    
    total_start = time.time()
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 [{i}/{len(queries)}] Searching: '{query}'")
        
        start = time.time()
        results = search_images(query, 3)
        elapsed = time.time() - start
        
        if results and "error" in results[0]:
            print(f"   ❌ {results[0]['error']}")
        else:
            print(f"   ✓ Found {len(results)} images in {elapsed:.2f}s")
            for img in results[:2]:
                print(f"     - {img.get('description', 'No description')[:50]}...")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print(f"📊 Total time for {len(queries)} searches: {total_elapsed:.2f}s")
    print(f"   Average per search: {total_elapsed/len(queries):.2f}s")
    print(f"   (First search includes browser startup ~3s)")
    print("=" * 60)
    
    # Cleanup
    FreepikClient.reset()
    print("\n✅ Browser closed successfully")
