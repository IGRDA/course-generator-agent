from typing import Annotated, List, Tuple
from operator import add
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


# ---- Default configuration ----
DEFAULT_NUM_IMAGES_TO_FETCH = 5
DEFAULT_VISION_PROVIDER = "pixtral"


# ---- State for individual section task ----
class ImageGenerationTask(BaseModel):
    """State for processing a single section's images"""
    course_state: CourseState
    module_idx: int
    submodule_idx: int
    section_idx: int
    section_title: str
    html_elements: List[HtmlElement]


# ---- State for aggregating results ----
class ImageGenerationState(BaseModel):
    """State for the image generation subgraph"""
    course_state: CourseState
    completed_sections: Annotated[list[dict], add] = Field(default_factory=list)
    total_sections: int = 0


def rank_images(
    image_urls: List[str],
    course_title: str,
    block_title: str,
    content_preview: str,
    text_llm_provider: str,
    vision_provider: str = DEFAULT_VISION_PROVIDER,
) -> Tuple[str, ImageRankingScore]:
    """
    Rank multiple images using a vision LLM and return the best one.
    
    Uses Pixtral (or configured vision LLM) to score images based on a rubric:
    - Alignment (0-5): How well image matches content
    - No Watermark (0 or 2): Clean images score higher
    - Has Text (0 or 1): Images with text get bonus
    - Style (0-2): Professional/educational style
    
    Falls back to text LLM for parsing if structured output fails.
    
    Args:
        image_urls: List of image URLs to rank
        course_title: Course topic for context
        block_title: Block title for context
        content_preview: Content preview for context
        text_llm_provider: Provider for fallback parsing (e.g., 'mistral')
        vision_provider: Vision LLM provider (default: 'pixtral')
    
    Returns:
        Tuple of (best_image_url, score)
    """
    if not image_urls:
        raise ValueError("No images to rank")
    
    if len(image_urls) == 1:
        # No need to rank a single image, return with default score
        return image_urls[0], ImageRankingScore(
            alignment=3, no_watermark=1, has_text=0, style=1, total=5
        )
    
    # Create vision LLM
    vision_model_name = resolve_vision_model_name(vision_provider)
    vision_kwargs = {"temperature": 0.1}  # Low temp for consistent scoring
    if vision_model_name:
        vision_kwargs["model_name"] = vision_model_name
    vision_llm = create_vision_llm(provider=vision_provider, **vision_kwargs)
    
    # Build multimodal message with all images
    human_prompt = create_image_ranking_prompt(
        course_title=course_title,
        block_title=block_title,
        content_preview=content_preview,
        num_images=len(image_urls)
    )
    
    # Create content list with text and images
    content = [{"type": "text", "text": human_prompt}]
    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })
    
    messages = [
        {"role": "system", "content": IMAGE_RANKING_SYSTEM_PROMPT},
        HumanMessage(content=content)
    ]
    
    # Try to get structured output from vision LLM
    try:
        response = vision_llm.invoke(messages)
        raw_response = response.content
        
        # Try to parse as JSON
        try:
            # Clean up response - extract JSON if wrapped in markdown
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
            
            # Fallback: Use text LLM to parse the raw response
            ranking_result = _fallback_parse_ranking(
                raw_response, 
                len(image_urls), 
                text_llm_provider
            )
        
        # Validate we have the right number of scores
        if len(ranking_result.scores) != len(image_urls):
            print(f"⚠️ Score count mismatch: got {len(ranking_result.scores)}, expected {len(image_urls)}")
            # Pad with default scores or truncate
            while len(ranking_result.scores) < len(image_urls):
                ranking_result.scores.append(ImageRankingScore(
                    alignment=0, no_watermark=0, has_text=0, style=0, total=0
                ))
            ranking_result.scores = ranking_result.scores[:len(image_urls)]
        
        # Find best image by total score
        best_idx = 0
        best_score = ranking_result.scores[0]
        for idx, score in enumerate(ranking_result.scores):
            # Recompute total to ensure consistency
            score.total = score.alignment + score.no_watermark + score.has_text + score.style
            if score.total > best_score.total:
                best_idx = idx
                best_score = score
        
        print(f"   🏆 Best image: #{best_idx + 1} with score {best_score.total}/10")
        return image_urls[best_idx], best_score
        
    except Exception as e:
        print(f"❌ Vision LLM call failed: {str(e)}")
        # Return first image as fallback
        return image_urls[0], ImageRankingScore(
            alignment=0, no_watermark=0, has_text=0, style=0, total=0
        )


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
    model_name = resolve_text_model_name(text_llm_provider)
    llm_kwargs = {"temperature": 0}  # Deterministic for parsing
    if model_name:
        llm_kwargs["model_name"] = model_name
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
            ImageRankingScore(alignment=0, no_watermark=0, has_text=0, style=0, total=0)
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


def generate_section_images(
    state: ImageGenerationTask,
    num_images_to_fetch: int = DEFAULT_NUM_IMAGES_TO_FETCH,
    vision_provider: str = DEFAULT_VISION_PROVIDER,
) -> dict:
    """
    Generate images for all interactive blocks in a single section.
    
    If use_vision_ranking is enabled in config, fetches K images for each block, 
    ranks them using vision LLM, and selects the best one based on rubric scoring.
    If disabled, fetches only 1 image and uses it directly.
    
    Args:
        state: ImageGenerationTask with section info
        num_images_to_fetch: Number of images to fetch for ranking (default: 5)
        vision_provider: Vision LLM provider for ranking (default: 'pixtral')
    """
    # Extract context
    llm_provider = state.course_state.config.text_llm_provider
    image_provider = state.course_state.config.image_search_provider
    course_title = state.course_state.title
    
    # Get config values with defaults
    use_vision_ranking = getattr(state.course_state.config, 'use_vision_ranking', True)
    k_images = getattr(state.course_state.config, 'num_images_to_fetch', num_images_to_fetch) if use_vision_ranking else 1
    vision_provider = getattr(state.course_state.config, 'vision_llm_provider', vision_provider)
    
    # Create LLM for query generation
    model_name = resolve_text_model_name(llm_provider)
    llm_kwargs = {"temperature": 0.3}  # Slightly creative for better queries
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=llm_provider, **llm_kwargs)
    
    # Create query generation chain
    query_chain = image_query_prompt | llm | StrOutputParser()
    
    # Get the image search function from factory
    search_images = create_image_search(image_provider)
    
    # Interactive formats that need images
    interactive_formats = ["paragraphs", "accordion", "tabs", "carousel", "flip", "timeline", "conversation"]
    
    images_added = 0
    
    # Process each HTML element
    for element in state.html_elements:
        # Check if this is an interactive format
        if element.type in interactive_formats:
            # Process each block in the content array
            if isinstance(element.content, list):
                for block in element.content:
                    # Ensure it's a ParagraphBlock (has title, icon, elements)
                    if isinstance(block, (dict, ParagraphBlock)):
                        # Convert dict to ParagraphBlock if needed
                        if isinstance(block, dict):
                            block_obj = ParagraphBlock(**block)
                        else:
                            block_obj = block
                        
                        # Extract first paragraph content for context
                        content_preview = ""
                        for elem in block_obj.elements:
                            if elem.type == "p":
                                content_preview = elem.content[:200]  # First 200 chars
                                break
                        
                        # Generate image query using LLM
                        try:
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
                                        "style": best_score.style,
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
                            
                            images_added += 1
                            
                        except Exception as e:
                            # Fail loudly as specified
                            print(f"❌ Error generating image for block '{block_obj.title}': {str(e)}")
                            raise
    
    print(f"✓ Generated {images_added} images for Module {state.module_idx+1}, "
          f"Submodule {state.submodule_idx+1}, Section {state.section_idx+1}")
    
    # Return the completed section info
    return {
        "completed_sections": [{
            "module_idx": state.module_idx,
            "submodule_idx": state.submodule_idx,
            "section_idx": state.section_idx,
            "html_elements": state.html_elements
        }]
    }


def reduce_images(state: ImageGenerationState) -> dict:
    """Fan-in: Aggregate all generated images back into the course state."""
    print(f"📦 Reducing {len(state.completed_sections)} sections with images")
    
    # Update course state with all generated images
    for section_info in state.completed_sections:
        m_idx = section_info["module_idx"]
        sm_idx = section_info["submodule_idx"]
        s_idx = section_info["section_idx"]
        
        section = state.course_state.modules[m_idx].submodules[sm_idx].sections[s_idx]
        section.html = section_info["html_elements"]
    
    print(f"✅ All {state.total_sections} section images generated successfully!")
    
    return {"course_state": state.course_state}


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
    num_images_to_fetch: int = DEFAULT_NUM_IMAGES_TO_FETCH,
    vision_provider: str = DEFAULT_VISION_PROVIDER,
) -> CourseState:
    """
    Main function to generate images for all section HTML blocks using LangGraph Send pattern.
    
    If use_vision_ranking is True, fetches K images for each block, ranks them using 
    vision LLM (Pixtral), and selects the best one based on rubric scoring.
    If False, fetches only 1 image and uses it directly (faster, no vision LLM calls).
    
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
    print(f"🚀 Starting parallel image generation with provider={image_provider}, "
          f"max_retries={max_retries}, initial_delay={initial_delay}s, backoff_factor={backoff_factor}")
    
    if use_vision_ranking:
        print(f"   📸 Fetching {num_images_to_fetch} images per block, ranking with {vision_provider}")
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
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    return result["course_state"]


if __name__ == "__main__":
    import json
    from main.state import CourseConfig
    
    INPUT_FILE = "/Users/inaki/Documents/Personal/course-generator-agent/output/Chess_masterclass_20251211_172927.json"
    # Generate OUTPUT_FILE from INPUT_FILE with _images suffix
    base, ext = os.path.splitext(INPUT_FILE)
    OUTPUT_FILE = f"{base}_images{ext}"
    
    # Hardcoded defaults - change these to override config values
    DEFAULT_LLM_PROVIDER = "mistral"
    DEFAULT_LANGUAGE = "English"
    DEFAULT_IMAGE_PROVIDER = "bing"  # Options: bing | freepik | ddg
    USE_VISION_RANKING = True  # Set to True to rank images with Pixtral, False to pick first result
    NUM_IMAGES_TO_FETCH = 5  # Number of images to fetch for ranking (only used if USE_VISION_RANKING=True)
    VISION_PROVIDER = "pixtral"  # Vision LLM for ranking
    
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
                                            score_str = f"score: {score}/10" if score is not None else "no ranking"
                                            print(f"   {query_count}. [{block_obj.title}] → \"{block_obj.image['query']}\" ({score_str})")
    
    print(f"\n✨ Total queries generated: {query_count}")
    print("="*60)
