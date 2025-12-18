from typing import Annotated, List, Tuple
from operator import add
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import re
import os
import json
from pydantic import BaseModel, Field
from main.state import CourseState, HtmlElement, ParagraphBlock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy
from LLMs.text2text import create_text_llm, resolve_text_model_name
from LLMs.imagetext2text import create_vision_llm, resolve_vision_model_name
from tools.imagesearch.factory import create_image_search
from .prompts import (
    image_query_prompt,
    ImageRankingScore,
    ImageRankingResult,
    IMAGE_RANKING_SYSTEM_PROMPT,
    create_image_ranking_prompt,
    FALLBACK_PARSE_PROMPT,
)


# ---- Global concurrency control ----
# Global semaphore to cap concurrent Pixtral calls across all blocks/sections.
# This prevents concurrency multiplication (sections * blocks * per-block ranking).
_VISION_CALL_SEMAPHORE: threading.Semaphore | None = None


def _set_vision_call_semaphore(max_concurrent_calls: int) -> None:
    global _VISION_CALL_SEMAPHORE
    _VISION_CALL_SEMAPHORE = threading.Semaphore(max(1, int(max_concurrent_calls)))


def _vision_invoke(vision_llm, messages):
    if _VISION_CALL_SEMAPHORE is None:
        return vision_llm.invoke(messages)
    _VISION_CALL_SEMAPHORE.acquire()
    try:
        return vision_llm.invoke(messages)
    finally:
        _VISION_CALL_SEMAPHORE.release()


def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _looks_like_invalid_image_error(exc: BaseException) -> bool:
    """True for Pixtral 400-style invalid image URL errors (permanent, should drop URL)."""
    s = str(exc)
    return ("Error response 400" in s) or ("invalid_request_file" in s) or ("code\":\"3310" in s) or ("code': '3310" in s)


def _extract_invalid_image_url(exc: BaseException) -> str | None:
    """
    Try to extract the offending image URL from Pixtral 400 errors.

    Examples seen:
    - \"Image, 'https://...png', could not be loaded as a valid image\"
    - \"File could not be fetched from url 'https://...'\"
    - \"Image is an animated GIF. Image URL: https://...gif\"
    """
    s = str(exc)

    m = re.search(r"Image URL:\s*(https?://\S+)", s)
    if m:
        return m.group(1).rstrip("',\"")

    m = re.search(r"from url '([^']+)'", s)
    if m:
        return m.group(1)

    m = re.search(r"Image,\s*'([^']+)'", s)
    if m:
        return m.group(1)

    return None


# ---- State for individual section task ----
class ImageGenerationTask(BaseModel):
    """State for processing a single section's images"""
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_title: str
    html_elements: List[HtmlElement]


# ---- Failed block info for retry queue ----
class FailedBlockInfo(BaseModel):
    """Information about a failed block for retry queue"""
    module_idx: int
    submodule_idx: int
    section_idx: int
    block_title: str
    block_obj: ParagraphBlock
    error: str


# ---- State for aggregating results ----
class ImageGenerationState(BaseModel):
    """State for the image generation subgraph"""
    course_state: CourseState
    completed_sections: Annotated[list[dict], add] = Field(default_factory=list)
    failed_blocks: Annotated[list[FailedBlockInfo], add] = Field(default_factory=list)
    total_sections: int = 0


def rank_images(
    image_urls: List[str],
    course_title: str,
    block_title: str,
    content_preview: str,
    text_llm_provider: str,
    vision_provider: str,
    imagetext2text_concurrency: int,
    batch_size: int,
) -> Tuple[str, ImageRankingScore]:
    """
    Rank multiple images using batched vision LLM calls and return the best one.

    This avoids making one Pixtral call per image. Instead, it scores images in batches
    (up to `batch_size` images per call) and selects the best image locally.
    
    Scoring rubric (max 12 points):
    - Alignment (0,1,2,4,8): How well image matches topic AND block title
    - No Watermark (0 or 2): Clean images score higher
    - No Text (0 or 2): Images without text score higher
    
    Args:
        image_urls: List of image URLs to rank
        course_title: Course topic for context
        block_title: Block title for context
        content_preview: Content preview for context
        text_llm_provider: Provider for fallback parsing (e.g., 'mistral')
        vision_provider: Vision LLM provider (default: 'pixtral')
        imagetext2text_concurrency: Max parallel Pixtral calls (default: 5)
    
    Returns:
        Tuple of (best_image_url, score)
    """
    if not image_urls:
        raise ValueError("No images to rank")
    
    if len(image_urls) == 1:
        # No need to rank a single image, return with default score
        return image_urls[0], ImageRankingScore(
            alignment=4, no_watermark=2, has_text=2, total=8
        )

    # If semaphore wasn't initialized at run start, initialize best-effort here.
    if _VISION_CALL_SEMAPHORE is None:
        _set_vision_call_semaphore(imagetext2text_concurrency)

    # Create vision LLM once per block ranking
    vision_model_name = resolve_vision_model_name(vision_provider)
    vision_kwargs = {"temperature": 0.1}
    if vision_model_name:
        vision_kwargs["model_name"] = vision_model_name
    vision_llm = create_vision_llm(provider=vision_provider, **vision_kwargs)

    scored_images: List[Tuple[str, ImageRankingScore]] = []
    for batch in _chunk_list(image_urls, batch_size):
        # We may need to drop invalid URLs and retry the same batch
        remaining = list(batch)

        while remaining:
            human_prompt = create_image_ranking_prompt(
                course_title=course_title,
                block_title=block_title,
                content_preview=content_preview,
                num_images=len(remaining),
            )

            content = [{"type": "text", "text": human_prompt}]
            for url in remaining:
                content.append({"type": "image_url", "image_url": {"url": url}})

            messages = [
                {"role": "system", "content": IMAGE_RANKING_SYSTEM_PROMPT},
                HumanMessage(content=content),
            ]

            try:
                response = _vision_invoke(vision_llm, messages)
                raw_response = response.content

                try:
                    json_str = raw_response
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0]
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0]
                    json_str = json_str.strip()

                    parsed_data = json.loads(json_str)
                    ranking_result = ImageRankingResult.model_validate(parsed_data)
                except (json.JSONDecodeError, Exception) as parse_error:
                    print(f"⚠️ Vision LLM output parsing failed: {parse_error}")
                    print(f"   Raw response: {raw_response[:500]}...")
                    print("   Attempting fallback parsing with text LLM...")
                    ranking_result = _fallback_parse_ranking(raw_response, len(remaining), text_llm_provider)

                # Ensure score count matches batch size
                if len(ranking_result.scores) != len(remaining):
                    while len(ranking_result.scores) < len(remaining):
                        ranking_result.scores.append(
                            ImageRankingScore(alignment=0, no_watermark=0, has_text=0, total=0)
                        )
                    ranking_result.scores = ranking_result.scores[: len(remaining)]

                for url, score in zip(remaining, ranking_result.scores):
                    score.total = score.alignment + score.no_watermark + score.has_text
                    scored_images.append((url, score))

                # Batch succeeded; move to next batch
                break

            except Exception as e:
                # For 400 invalid-image style errors, drop the offending URL and retry the batch.
                if _looks_like_invalid_image_error(e):
                    bad_url = _extract_invalid_image_url(e)
                    if bad_url and bad_url in remaining:
                        remaining.remove(bad_url)
                        continue
                    # Heuristic: drop any GIFs (Pixtral rejects animated GIFs)
                    gif_urls = [u for u in remaining if u.lower().endswith(".gif")]
                    if gif_urls:
                        remaining.remove(gif_urls[0])
                        continue

                # Give up on this remaining batch and assign zero scores
                print(f"❌ Vision LLM call failed for batch of {len(remaining)} images: {str(e)}")
                for url in remaining:
                    scored_images.append((url, ImageRankingScore(alignment=0, no_watermark=0, has_text=0, total=0)))
                break
    
    # Find best image by total score
    if not scored_images:
        # Fallback: return first URL with zero score
        return image_urls[0], ImageRankingScore(
            alignment=0, no_watermark=0, has_text=0, total=0
        )
    
    best_url, best_score = scored_images[0]
    for url, score in scored_images[1:]:
        if score.total > best_score.total:
            best_url = url
            best_score = score
    
    # Find index for logging
    best_idx = image_urls.index(best_url) + 1
    print(f"   🏆 Best image: #{best_idx} with score {best_score.total}/12")
    
    return best_url, best_score


def _fallback_parse_ranking(
    raw_text: str,
    num_images: int,
    text_llm_provider: str
) -> ImageRankingResult:
    """
    Fallback parser using text LLM when vision LLM output is malformed.
    
    Args:
        raw_text: Raw response from vision LLM
        num_images: Expected number of images/scores
        text_llm_provider: Provider for text LLM (e.g., 'mistral')
    
    Returns:
        Parsed ImageRankingResult
    """
    # Create text LLM for parsing
    model_name = "mistral"
    llm_kwargs = {"temperature": 0}  # Deterministic for parsing
    if model_name:
        llm_kwargs["model_name"] = "mistral-small-latest"
    text_llm = create_text_llm(provider=text_llm_provider, **llm_kwargs)
    
    # Create parsing chain
    parse_chain = FALLBACK_PARSE_PROMPT | text_llm | StrOutputParser()
    
    try:
        parsed_response = parse_chain.invoke({
            "num_images": num_images,
            "raw_text": raw_text
        })
        
        # Clean up and parse
        json_str = parsed_response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        json_str = json_str.strip()
        
        parsed_data = json.loads(json_str)
        return ImageRankingResult.model_validate(parsed_data)
        
    except Exception as e:
        print(f"⚠️ Fallback parsing also failed: {str(e)}")
        # Return default scores
        default_scores = [
            ImageRankingScore(alignment=0, no_watermark=0, has_text=0, total=0)
            for _ in range(num_images)
        ]
        return ImageRankingResult(scores=default_scores)


def plan_image_generation(state: ImageGenerationState) -> dict:
    """Count the total sections that need image generation."""
    section_count = 0
    
    for module in state.course_state.modules:
        for submodule in module.submodules:
            section_count += len(submodule.sections)
    
    print(f"📋 Planning image generation for {section_count} sections")
    
    return {"total_sections": section_count}


def continue_to_images(state: ImageGenerationState) -> list[Send]:
    """Fan-out: Create a Send for each section to process in parallel."""
    sends = []
    
    for m_idx, module in enumerate(state.course_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                # Only process sections that have HTML
                if section.html:
                    task = ImageGenerationTask(
                        course_state=state.course_state,
                        module_idx=m_idx,
                        submodule_idx=sm_idx,
                        section_idx=s_idx,
                        section_title=section.title,
                        html_elements=section.html
                    )
                    sends.append(Send("generate_section_images", task))
    
    return sends


def _process_single_block(
    block_obj: ParagraphBlock,
    course_title: str,
    llm_provider: str,
    image_provider: str,
    k_images: int,
    use_vision_ranking: bool,
    vision_provider: str,
    imagetext2text_concurrency: int,
    vision_ranking_batch_size: int,
    query_chain,
    search_images,
) -> bool:
    """
    Process a single block to add an image.
    
    Returns True if image was added successfully, False otherwise.
    """
    # Extract first paragraph content for context
    content_preview = ""
    for elem in block_obj.elements:
        if elem.type == "p":
            content_preview = elem.content[:200]  # First 200 chars
            break
    
    # Generate image query using LLM
    query = query_chain.invoke({
        "course_title": course_title,
        "block_title": block_obj.title,
        "content_preview": content_preview
    })
    query = query.strip().replace('*', '').replace('"', '').strip()
    
    # Search for K images using configured provider
    print(f"   🔍 Searching for {k_images} images: '{query}'")
    results = search_images(query, max_results=k_images)
    
    # Check for errors
    if not results:
        raise Exception(f"Image search ({image_provider}) returned no results for query '{query}'")
    
    # Filter out error results
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        error_msg = results[0].get("error", "Unknown error") if results else "No results"
        raise Exception(f"Image search ({image_provider}) failed for query '{query}': {error_msg}")
    
    # Extract image URLs
    image_urls = [r["url"] for r in valid_results]
    
    # Select image based on config
    if use_vision_ranking and len(image_urls) > 1:
        # Rank images and select best one using vision LLM
        print(f"   🎯 Ranking {len(image_urls)} images for '{block_obj.title}'...")
        best_url, best_score = rank_images(
            image_urls=image_urls,
            course_title=course_title,
            block_title=block_obj.title,
            content_preview=content_preview,
            text_llm_provider=llm_provider,
            vision_provider=vision_provider,
            imagetext2text_concurrency=imagetext2text_concurrency,
            batch_size=vision_ranking_batch_size,
        )
        
        # Add image to block with ranking metadata
        block_obj.image = {
            "type": "img",
            "query": query,
            "content": best_url,
            "ranking_score": best_score.total,
            "ranking_details": {
                "alignment": best_score.alignment,
                "no_watermark": best_score.no_watermark,
                "has_text": best_score.has_text,
            }
        }
    else:
        # Just use first image (no ranking)
        best_url = image_urls[0]
        block_obj.image = {
            "type": "img",
            "query": query,
            "content": best_url,
        }
    
    return True


def generate_section_images(
    state: ImageGenerationTask,
) -> dict:
    """
    Generate images for all interactive blocks in a single section.
    
    Processes blocks in parallel using ThreadPoolExecutor with image_blocks_concurrency.
    If use_vision_ranking is enabled in config, fetches K images for each block, 
    ranks them using vision LLM (in parallel with imagetext2text_concurrency), 
    and selects the best one based on rubric scoring.
    If disabled, fetches only 1 image and uses it directly.
    
    Args:
        state: ImageGenerationTask with section info
        num_images_to_fetch: Number of images to fetch for ranking (default: 5)
        vision_provider: Vision LLM provider for ranking (default: 'pixtral')
    """
    # Extract context and config
    llm_provider = state.course_state.config.text_llm_provider
    image_provider = state.course_state.config.image_search_provider
    course_title = state.course_state.title
    
    # Get config values (all required fields)
    use_vision_ranking = state.course_state.config.use_vision_ranking
    num_images_to_fetch = state.course_state.config.num_images_to_fetch
    vision_provider = state.course_state.config.vision_llm_provider
    image_blocks_concurrency = state.course_state.config.image_blocks_concurrency
    imagetext2text_concurrency = state.course_state.config.imagetext2text_concurrency
    vision_ranking_batch_size = state.course_state.config.vision_ranking_batch_size
    
    k_images = num_images_to_fetch if use_vision_ranking else 1
    
    # Create LLM for query generation
    model_name = "mistral"
    llm_kwargs = {"temperature": 0.1}  # Slightly creative for better queries
    if model_name:
        llm_kwargs["model_name"] = "mistral-small-latest"
    llm = create_text_llm(provider=llm_provider, **llm_kwargs)
    
    # Create query generation chain
    query_chain = image_query_prompt | llm | StrOutputParser()
    
    # Get the image search function from factory
    search_images = create_image_search(image_provider)
    
    # Interactive formats that need images
    interactive_formats = ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]
    
    # Collect all blocks that need images
    blocks_to_process: List[ParagraphBlock] = []
    
    for element in state.html_elements:
        if element.type in interactive_formats:
            if isinstance(element.content, list):
                for block in element.content:
                    if isinstance(block, (dict, ParagraphBlock)):
                        # Convert dict to ParagraphBlock if needed
                        if isinstance(block, dict):
                            block_obj = ParagraphBlock(**block)
                            # Replace the dict with the object in the list
                            idx = element.content.index(block)
                            element.content[idx] = block_obj
                        else:
                            block_obj = block
                        blocks_to_process.append(block_obj)
    
    if not blocks_to_process:
        print(f"✓ No blocks to process for Module {state.module_idx+1}, "
              f"Submodule {state.submodule_idx+1}, Section {state.section_idx+1}")
        return {
            "completed_sections": [{
                "module_idx": state.module_idx,
                "submodule_idx": state.submodule_idx,
                "section_idx": state.section_idx,
                "html_elements": state.html_elements
            }]
        }
    
    # Process blocks in parallel
    images_added = 0
    failed_blocks_list = []
    
    with ThreadPoolExecutor(max_workers=image_blocks_concurrency) as executor:
        future_to_block = {
            executor.submit(
                _process_single_block,
                block_obj,
                course_title,
                llm_provider,
                image_provider,
                k_images,
                use_vision_ranking,
                vision_provider,
                imagetext2text_concurrency,
                vision_ranking_batch_size,
                query_chain,
                search_images,
            ): block_obj
            for block_obj in blocks_to_process
        }
        
        for future in as_completed(future_to_block):
            block_obj = future_to_block[future]
            try:
                if future.result():
                    images_added += 1
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error generating image for block '{block_obj.title}': {error_msg}")
                # Collect failed block for retry queue instead of raising
                failed_blocks_list.append(FailedBlockInfo(
                    module_idx=state.module_idx,
                    submodule_idx=state.submodule_idx,
                    section_idx=state.section_idx,
                    block_title=block_obj.title,
                    block_obj=block_obj,
                    error=error_msg
                ))
    
    if failed_blocks_list:
        print(f"⚠️ {len(failed_blocks_list)} blocks failed, queued for retry later")
    
    print(f"✓ Generated {images_added} images for Module {state.module_idx+1}, "
          f"Submodule {state.submodule_idx+1}, Section {state.section_idx+1}")
    
    # Return the completed section info along with any failed blocks
    return {
        "completed_sections": [{
            "module_idx": state.module_idx,
            "submodule_idx": state.submodule_idx,
            "section_idx": state.section_idx,
            "html_elements": state.html_elements
        }],
        "failed_blocks": failed_blocks_list
    }


def reduce_images(state: ImageGenerationState) -> dict:
    """Fan-in: Aggregate all generated images back into the course state and retry failed blocks."""
    print(f"📦 Reducing {len(state.completed_sections)} sections with images")
    
    # Update course state with all generated images
    for section_info in state.completed_sections:
        m_idx = section_info["module_idx"]
        sm_idx = section_info["submodule_idx"]
        s_idx = section_info["section_idx"]
        
        section = state.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx]
        section.html = section_info["html_elements"]
    
    # Check if there are failed blocks to retry
    if state.failed_blocks:
        print(f"\n🔄 Retrying {len(state.failed_blocks)} failed blocks sequentially...")
        
        # Retry failed blocks one at a time (sequential, lower concurrency)
        still_failed = _retry_failed_blocks(
            state.failed_blocks,
            state.course_state,
        )
        
        if still_failed:
            print(f"⚠️ {len(still_failed)} blocks still failed after retry:")
            for fb in still_failed:
                print(f"   - [{fb.block_title}] in Module {fb.module_idx+1}.{fb.submodule_idx+1}.{fb.section_idx+1}")
            print("   (Continuing without these images)")
        else:
            print("✅ All retried blocks succeeded!")
    
    success_count = state.total_sections
    if state.failed_blocks:
        failed_after_retry = len([fb for fb in state.failed_blocks if fb.block_obj.image is None])
        if failed_after_retry > 0:
            print(f"✅ Image generation complete! ({failed_after_retry} blocks without images)")
        else:
            print(f"✅ All {state.total_sections} section images generated successfully!")
    else:
        print(f"✅ All {state.total_sections} section images generated successfully!")
    
    return {"course_state": state.course_state}


def _retry_failed_blocks(
    failed_blocks: List[FailedBlockInfo],
    course_state: CourseState,
) -> List[FailedBlockInfo]:
    """
    Retry failed blocks sequentially (one at a time) with fresh browser context.
    
    Args:
        failed_blocks: List of failed block info
        course_state: Current course state for config access
        
    Returns:
        List of blocks that still failed after retry
    """
    from tools.imagesearch.factory import create_image_search
    
    llm_provider = course_state.config.text_llm_provider
    image_provider = course_state.config.image_search_provider
    course_title = course_state.title
    
    use_vision_ranking = course_state.config.use_vision_ranking
    num_images_to_fetch = course_state.config.num_images_to_fetch
    vision_provider = course_state.config.vision_llm_provider
    imagetext2text_concurrency = course_state.config.imagetext2text_concurrency
    vision_ranking_batch_size = course_state.config.vision_ranking_batch_size
    
    k_images = num_images_to_fetch if use_vision_ranking else 1
    
    # Create LLM for query generation
    model_name = "mistral"
    llm_kwargs = {"temperature": 0.3}
    if model_name:
        llm_kwargs["model_name"] = "mistral-small-latest"
    llm = create_text_llm(provider=llm_provider, **llm_kwargs)
    
    # Create query generation chain
    query_chain = image_query_prompt | llm | StrOutputParser()
    
    # Get fresh image search function
    search_images = create_image_search(image_provider)
    
    still_failed = []
    
    for i, fb in enumerate(failed_blocks):
        print(f"   🔄 Retrying block {i+1}/{len(failed_blocks)}: '{fb.block_title}'...")
        
        # Wait between retries to avoid rate limiting
        if i > 0:
            import time
            time.sleep(2)  # 2 second delay between retries
        
        try:
            success = _process_single_block(
                fb.block_obj,
                course_title,
                llm_provider,
                image_provider,
                k_images,
                use_vision_ranking,
                vision_provider,
                imagetext2text_concurrency,
                vision_ranking_batch_size,
                query_chain,
                search_images,
            )
            
            if success:
                print(f"   ✅ Retry succeeded for '{fb.block_title}'")
            else:
                print(f"   ❌ Retry failed for '{fb.block_title}'")
                still_failed.append(fb)
                
        except Exception as e:
            print(f"   ❌ Retry error for '{fb.block_title}': {str(e)}")
            still_failed.append(fb)
    
    return still_failed


def build_image_generation_graph(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Build the image generation subgraph using Send for dynamic parallelization.
    
    Args:
        max_retries: Number of retries for each section generation
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
    """
    graph = StateGraph(ImageGenerationState)
    
    # Configure retry policy with exponential backoff
    retry_policy = RetryPolicy(
        max_attempts=max_retries,
        initial_interval=initial_delay,
        backoff_factor=backoff_factor,
        max_interval=60.0
    )
    
    # Add nodes
    graph.add_node("plan_image_generation", plan_image_generation)
    graph.add_node("generate_section_images", generate_section_images, retry=retry_policy)
    graph.add_node("reduce_images", reduce_images)
    
    # Add edges
    graph.add_edge(START, "plan_image_generation")
    graph.add_conditional_edges("plan_image_generation", continue_to_images, ["generate_section_images"])
    graph.add_edge("generate_section_images", "reduce_images")
    graph.add_edge("reduce_images", END)
    
    return graph.compile()


def generate_all_section_images(
    course_state: CourseState,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    use_vision_ranking: bool = True,
    num_images_to_fetch: int = None,
    vision_provider: str = None,
) -> CourseState:
    """
    Main function to generate images for all section HTML blocks using LangGraph Send pattern.
    
    If use_vision_ranking is True, fetches K images for each block, ranks them using 
    vision LLM (Pixtral), and selects the best one based on rubric scoring.
    If False, fetches only 1 image and uses it directly (faster, no vision LLM calls).
    
    Concurrency is controlled at three levels:
    - image_sections_concurrency: Max parallel sections (LangGraph max_concurrency)
    - image_blocks_concurrency: Max parallel blocks within a section
    - imagetext2text_concurrency: Max parallel Pixtral calls per block
    
    Args:
        course_state: CourseState with HTML structures filled
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for exponential backoff
        use_vision_ranking: Whether to use vision LLM for ranking (default: True)
        num_images_to_fetch: Number of images to fetch for ranking (default: 5, ignored if use_vision_ranking=False)
        vision_provider: Vision LLM provider for ranking (default: 'pixtral')
        
    Returns:
        Updated CourseState with all images added to HTML blocks
    """
    image_provider = course_state.config.image_search_provider
    
    # Get settings from config (required fields)
    if num_images_to_fetch is None:
        num_images_to_fetch = course_state.config.num_images_to_fetch
    if vision_provider is None:
        vision_provider = course_state.config.vision_llm_provider
    
    # Get concurrency settings from config (required fields)
    image_sections_concurrency = course_state.config.image_sections_concurrency
    image_blocks_concurrency = course_state.config.image_blocks_concurrency
    imagetext2text_concurrency = course_state.config.imagetext2text_concurrency
    vision_ranking_batch_size = course_state.config.vision_ranking_batch_size

    # Enforce a global cap on concurrent Pixtral calls across all blocks/sections.
    _set_vision_call_semaphore(imagetext2text_concurrency)
    
    print(f"🚀 Starting parallel image generation with provider={image_provider}, "
          f"max_retries={max_retries}, initial_delay={initial_delay}s, backoff_factor={backoff_factor}")
    print(f"   📊 Concurrency: sections={image_sections_concurrency}, "
          f"blocks={image_blocks_concurrency}, vision_calls={imagetext2text_concurrency}")
    
    if use_vision_ranking:
        print(f"   📸 Fetching {num_images_to_fetch} images per block, ranking with {vision_provider} (batch_size={vision_ranking_batch_size})")
    else:
        print("   📸 Vision ranking disabled: picking first image result")
    
    # Build the graph with retry configuration
    graph = build_image_generation_graph(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor
    )
    
    # Initialize state
    initial_state = ImageGenerationState(course_state=course_state)
    
    # Execute the graph with concurrency control
    result = graph.invoke(
        initial_state,
        config={"max_concurrency": image_sections_concurrency}
    )
    
    return result["course_state"]


if __name__ == "__main__":
    import json
    from main.state import CourseConfig
    
    INPUT_FILE = "/Users/inaki/Documents/Personal/course-generator-agent/output/Chess_masterclass_20251215_213954.json"
    # Generate OUTPUT_FILE from INPUT_FILE with _images suffix
    base, ext = os.path.splitext(INPUT_FILE)
    OUTPUT_FILE = f"{base}_images{ext}"
    
    # Hardcoded defaults - change these to override config values
    DEFAULT_LLM_PROVIDER = "mistral"
    DEFAULT_LANGUAGE = "English"
    DEFAULT_IMAGE_PROVIDER = "google"  # Options: bing | freepik | ddg
    USE_VISION_RANKING = False  # Set to True to rank images with Pixtral, False to pick first result
    NUM_IMAGES_TO_FETCH = 8  # Number of images to fetch for ranking (only used if USE_VISION_RANKING=True)
    VISION_PROVIDER = "pixtral"  # Vision LLM for ranking
    IMAGE_SECTIONS_CONCURRENCY = 2  # Number of sections to process in parallel
    IMAGE_BLOCKS_CONCURRENCY = 5  # Number of blocks to process in parallel within each section
    IMAGETEXT2TEXT_CONCURRENCY = 5  # Number of Pixtral calls in parallel for image scoring
    VISION_RANKING_BATCH_SIZE = 8
    
    print("="*60)
    print("Image Generator - Standalone Mode")
    print("="*60)
    print(f"Reading from: {INPUT_FILE}")
    
    # Load input JSON
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found")
        print(f"   Please create an {INPUT_FILE} file with CourseState JSON structure")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {INPUT_FILE}: {str(e)}")
        exit(1)
    
    # Parse as CourseState
    try:
        course_state = CourseState.model_validate(data)
        
        # Apply defaults if config is incomplete
        if not course_state.config:
            course_state.config = CourseConfig(
                title=course_state.title or "Untitled Course",
                text_llm_provider=DEFAULT_LLM_PROVIDER,
                image_search_provider=DEFAULT_IMAGE_PROVIDER,
                total_pages=10
            )
        else:
            # Override config with hardcoded defaults
            course_state.config.image_search_provider = DEFAULT_IMAGE_PROVIDER
            course_state.config.use_vision_ranking = USE_VISION_RANKING
            course_state.config.num_images_to_fetch = NUM_IMAGES_TO_FETCH
            course_state.config.vision_llm_provider = VISION_PROVIDER
            course_state.config.image_sections_concurrency = IMAGE_SECTIONS_CONCURRENCY
            course_state.config.image_blocks_concurrency = IMAGE_BLOCKS_CONCURRENCY
            course_state.config.imagetext2text_concurrency = IMAGETEXT2TEXT_CONCURRENCY
            course_state.config.vision_ranking_batch_size = VISION_RANKING_BATCH_SIZE
        
        if not course_state.language:
            course_state.language = DEFAULT_LANGUAGE
            
    except Exception as e:
        print(f"❌ Error: Failed to parse CourseState: {str(e)}")
        exit(1)
    
    # Count sections with HTML
    html_sections = 0
    total_sections = 0
    for module in course_state.modules:
        for submodule in module.submodules:
            for section in submodule.sections:
                total_sections += 1
                if section.html:
                    html_sections += 1
    
    print(f"📊 Course: {course_state.title}")
    print(f"   Language: {course_state.language}")
    print(f"   Modules: {len(course_state.modules)}")
    print(f"   Sections: {total_sections}")
    print(f"   Sections with HTML: {html_sections}")
    
    if html_sections == 0:
        print("⚠️  Warning: No sections have HTML content to add images to")
        print("   Exiting without changes")
        exit(0)
    
    # Generate images
    print(f"\n🚀 Starting image generation...")
    print(f"   Using LLM: {course_state.config.text_llm_provider}")
    print(f"   Image provider: {course_state.config.image_search_provider}")
    print(f"   Vision ranking: {'ENABLED' if USE_VISION_RANKING else 'DISABLED (picking first image)'}")
    if USE_VISION_RANKING:
        print(f"   Images to fetch per block: {NUM_IMAGES_TO_FETCH}")
        print(f"   Vision LLM for ranking: {VISION_PROVIDER}")
    print(f"   Concurrency: sections={IMAGE_SECTIONS_CONCURRENCY}, blocks={IMAGE_BLOCKS_CONCURRENCY}, vision_calls={IMAGETEXT2TEXT_CONCURRENCY}")
    
    try:
        updated_state = generate_all_section_images(
            course_state,
            max_retries=3,
            initial_delay=1.0,
            backoff_factor=2.0,
            use_vision_ranking=USE_VISION_RANKING,
            num_images_to_fetch=NUM_IMAGES_TO_FETCH,
            vision_provider=VISION_PROVIDER,
        )
    except Exception as e:
        print(f"\n❌ Error during image generation: {str(e)}")
        print("   No output file created")
        exit(1)
    
    # Save output
    print(f"\n💾 Saving to: {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(updated_state.model_dump_json(indent=2, by_alias=True))
        print(f"✅ Success! Images added to {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error: Failed to save output: {str(e)}")
        exit(1)
    
    # Print all image queries and scores
    print("\n" + "="*60)
    print("📝 All Generated Image Queries and Scores:")
    print("="*60)
    
    query_count = 0
    for m_idx, module in enumerate(updated_state.modules):
        for sm_idx, submodule in enumerate(module.submodules):
            for s_idx, section in enumerate(submodule.sections):
                if section.html:
                    section_has_images = False
                    for element in section.html:
                        if element.type in ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]:
                            if isinstance(element.content, list):
                                for block in element.content:
                                    if isinstance(block, (dict, ParagraphBlock)):
                                        block_obj = block if isinstance(block, ParagraphBlock) else ParagraphBlock(**block)
                                        if block_obj.image and "query" in block_obj.image:
                                            if not section_has_images:
                                                print(f"\n📍 Module {m_idx+1}.{sm_idx+1}.{s_idx+1}: {section.title}")
                                                section_has_images = True
                                            query_count += 1
                                            score = block_obj.image.get('ranking_score')
                                            score_str = f"score: {score}/12" if score is not None else "no ranking"
                                            print(f"   {query_count}. [{block_obj.title}] → \"{block_obj.image['query']}\" ({score_str})")
    
    print(f"\n✨ Total queries generated: {query_count}")
    print("="*60)
