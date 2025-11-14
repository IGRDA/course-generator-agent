"""
Image Selector Agent - Agentic pattern for searching and selecting best images.

This agent follows best practices:
- Async operations for better performance
- Retry logic with exponential backoff
- Structured outputs using Pydantic
- Vision model integration for intelligent selection
- Clear separation of concerns
"""

import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI
from tools.imagesearch.bing_scraper import search_bing_images
from .prompts import image_evaluation_prompt


# Model configuration
VISION_MODEL_NAME = "pixtral-12b-2409"  # Mistral's vision model
MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME", "mistral-small-latest")


class ImageEvaluation(BaseModel):
    """Structured output for image evaluation."""
    relevance_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Relevance score from 0-100"
    )
    explanation: str = Field(
        ...,
        description="Brief explanation of the score"
    )


class ImageCandidate(BaseModel):
    """Represents a candidate image with metadata."""
    url: str = Field(..., description="Image URL")
    thumbnail_url: str = Field(..., description="Thumbnail URL")
    description: str = Field(default="", description="Image description from search")
    author: str = Field(default="Unknown", description="Image author/source")
    evaluation: Optional[ImageEvaluation] = Field(None, description="Vision model evaluation")
    content_length: int = Field(default=0, description="Image file size for tie-breaking")
    search_rank: int = Field(default=0, description="Original search position for tie-breaking")


class ImageSelectionResult(BaseModel):
    """Result of image selection process."""
    query: str = Field(..., description="Original search query")
    selected_image: Optional[ImageCandidate] = Field(None, description="Best selected image")
    all_candidates: List[ImageCandidate] = Field(default_factory=list, description="All evaluated candidates")
    error: Optional[str] = Field(None, description="Error message if selection failed")


# Initialize vision LLM for image evaluation
vision_llm = ChatMistralAI(
    model=VISION_MODEL_NAME,
    temperature=0.2,
    timeout=30,  # Timeout for slow image loading
)

# Create structured output LLM
structured_vision_llm = vision_llm.with_structured_output(ImageEvaluation)


def is_valid_image_url(url: str) -> bool:
    """Validate if URL is a valid HTTP/HTTPS image URL."""
    if not url or not isinstance(url, str) or len(url) < 10:
        return False
    if not url.startswith(('http://', 'https://')):
        return False
    # Check for common image extensions or image indicators
    image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.jfif')
    url_lower = url.lower()
    has_ext = any(ext in url_lower for ext in image_exts)
    has_img_indicator = any(x in url_lower for x in ['image', 'img', 'photo', 'pic', 'format='])
    return has_ext or has_img_indicator


async def verify_image_accessible(url: str, timeout: int = 8) -> tuple[bool, int]:
    """Check if image is accessible and return (is_accessible, content_length)."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # Try HEAD first
            try:
                async with session.head(url, timeout=timeout, allow_redirects=True) as response:
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', '').lower()
                        content_length = int(response.headers.get('Content-Length', 0))
                        if 'image' in content_type and content_length > 1000:
                            return True, content_length
            except:
                pass
            
            # If HEAD fails, try GET with range to actually fetch some bytes
            async with session.get(url, timeout=timeout, allow_redirects=True) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '').lower()
                    # Actually read first 2KB to verify it's real
                    chunk = await response.content.read(2048)
                    if len(chunk) > 1000 and 'image' in content_type:
                        # Check for image magic bytes
                        is_image = (
                            chunk[:2] == b'\xff\xd8' or  # JPEG
                            chunk[:8] == b'\x89PNG\r\n\x1a\n' or  # PNG
                            chunk[:6] in (b'GIF87a', b'GIF89a') or  # GIF
                            chunk[:4] == b'RIFF'  # WEBP
                        )
                        if is_image:
                            content_length = int(response.headers.get('Content-Length', len(chunk)))
                            return True, content_length
                return False, 0
    except Exception as e:
        return False, 0


async def retry_async_call(async_func, max_retries: int = 2):
    """Simplified retry with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await async_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(2 ** attempt)


async def evaluate_image(image_url: str, query: str, max_retries: int = 2) -> Optional[tuple[ImageEvaluation, int]]:
    """
    Evaluate image and return (evaluation, content_length) or None if invalid.
    Content length used for tie-breaking (prefer larger/better quality images).
    """
    # Validate URL format
    if not is_valid_image_url(image_url):
        print(f"   ⚠️  Invalid URL format")
        return None
    
    # Check if image is accessible and get size
    is_accessible, content_length = await verify_image_accessible(image_url)
    if not is_accessible:
        print(f"   ⚠️  Image not accessible or invalid")
        return None
    
    async def _evaluate():
        prompt = image_evaluation_prompt.format(query=query)
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url},
                ],
            }
        ]
        result = await structured_vision_llm.ainvoke(message)
        return result
    
    try:
        evaluation = await retry_async_call(_evaluate, max_retries=max_retries)
        return (evaluation, content_length)
    except Exception as e:
        print(f"   ✗ Evaluation error: {str(e)[:50]}")
        return None


async def evaluate_images_parallel(
    candidates: List[ImageCandidate],
    query: str,
    concurrency: int = 2,
    max_retries: int = 2
) -> List[ImageCandidate]:
    """Evaluate images in parallel. Only valid images get evaluations."""
    sem = asyncio.Semaphore(concurrency)
    
    async def _evaluate_one(candidate: ImageCandidate, idx: int):
        async with sem:
            print(f"🔍 [{idx + 1}/{len(candidates)}] {candidate.url[:60]}...")
            result = await evaluate_image(candidate.url, query, max_retries)
            if result:
                evaluation, content_length = result
                candidate.evaluation = evaluation
                candidate.content_length = content_length
                size_kb = content_length // 1024
                print(f"   ✓ Score: {evaluation.relevance_score}/100 ({size_kb}KB)")
            else:
                candidate.evaluation = None
                print(f"   ✗ Skipped (invalid or inaccessible)")
    
    await asyncio.gather(
        *(_evaluate_one(candidate, i) for i, candidate in enumerate(candidates))
    )
    
    return candidates


def select_best_candidate(candidates: List[ImageCandidate]) -> Optional[ImageCandidate]:
    """Select best candidate with proper tie-breaking."""
    # Filter: must have evaluation AND score >= 40 (good quality threshold)
    valid = [c for c in candidates if c.evaluation and c.evaluation.relevance_score >= 40]
    
    if not valid:
        # Try lower threshold
        valid = [c for c in candidates if c.evaluation and c.evaluation.relevance_score >= 30]
        if not valid:
            print(f"\n⚠️  No images met minimum quality threshold")
            return None
    
    # Sort with tie-breaking:
    # 1. Highest score (primary)
    # 2. Larger file size (better quality, secondary)
    # 3. Better search rank (original position, tertiary)
    valid.sort(
        key=lambda c: (
            c.evaluation.relevance_score,  # Primary: highest score
            c.content_length,               # Secondary: larger file (better quality)
            -c.search_rank                  # Tertiary: earlier in search results
        ),
        reverse=True
    )
    
    best = valid[0]
    size_kb = best.content_length // 1024
    print(f"\n🎯 Best: Score {best.evaluation.relevance_score}/100 ({size_kb}KB)")
    print(f"   {best.evaluation.explanation}")
    
    # Show if there were ties
    ties = [c for c in valid if c.evaluation.relevance_score == best.evaluation.relevance_score]
    if len(ties) > 1:
        print(f"   (Selected from {len(ties)} images with same score based on size/rank)")
    
    return best


async def select_best_image(
    query: str,
    max_results: int = 5,
    concurrency: int = 2,
    max_retries: int = 2
) -> ImageSelectionResult:
    """
    Main agent: Search images and select best match.
    
    Args:
        query: Search query
        max_results: Max images to retrieve (default 5)
        concurrency: Concurrent evaluations (default 2)
        max_retries: Retry attempts (default 2)
    
    Returns:
        ImageSelectionResult with best image or error
    """
    print(f"\n🚀 Image Selector Agent")
    print(f"   Query: '{query}'")
    print(f"   Max results: {max_results}, Concurrency: {concurrency}\n")
    
    try:
        # Step 1: Search
        print(f"📸 Searching Bing...")
        search_results = await asyncio.to_thread(
            search_bing_images.invoke,
            {"query": query, "max_results": max_results * 2}  # Get extra in case some fail
        )
        
        if not search_results or (len(search_results) == 1 and "error" in search_results[0]):
            error = search_results[0].get("error", "No results") if search_results else "No results"
            return ImageSelectionResult(query=query, error=f"Search failed: {error}")
        
        # Step 2: Create candidates (filter errors, preserve rank)
        candidates = [
            ImageCandidate(
                url=img["url"],
                thumbnail_url=img.get("thumbnail_url", img["url"]),
                description=img.get("description", ""),
                author=img.get("author", "Unknown"),
                search_rank=idx
            )
            for idx, img in enumerate(search_results)
            if "url" in img and "error" not in img and is_valid_image_url(img["url"])
        ][:max_results]  # Limit to max_results
        
        if not candidates:
            return ImageSelectionResult(query=query, error="No valid URLs found")
        
        print(f"✓ Found {len(candidates)} valid image URLs\n")
        
        # Step 3: Evaluate in parallel
        print(f"🤖 Evaluating with vision model...\n")
        evaluated = await evaluate_images_parallel(candidates, query, concurrency, max_retries)
        
        # Step 4: Select best
        print(f"\n🎯 Selecting best match...")
        best = select_best_candidate(evaluated)
        
        if not best:
            valid_count = sum(1 for c in evaluated if c.evaluation)
            return ImageSelectionResult(
                query=query,
                all_candidates=evaluated,
                error=f"No suitable images found ({valid_count}/{len(evaluated)} evaluated successfully)"
            )
        
        print(f"✅ Success!\n")
        return ImageSelectionResult(
            query=query,
            selected_image=best,
            all_candidates=evaluated
        )
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return ImageSelectionResult(query=query, error=str(e))


def select_best_image_sync(
    query: str,
    max_results: int = 5,
    concurrency: int = 2,
    max_retries: int = 2
) -> ImageSelectionResult:
    """Synchronous wrapper."""
    return asyncio.run(select_best_image(query, max_results, concurrency, max_retries))


if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "uniformly acelerated linear motion"
    result = select_best_image_sync(query=query, max_results=50)
    
    if result.error:
        print(f"\n❌ {result.error}")
    elif result.selected_image:
        print(f"\n✅ Selected: {result.selected_image.url}")
        print(f"\n📊 All results:")
        for i, c in enumerate(result.all_candidates, 1):
            score = c.evaluation.relevance_score if c.evaluation else "N/A"
            print(f"   {i}. {score} - {c.url[:70]}")
