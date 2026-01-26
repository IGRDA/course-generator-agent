"""
Video Search Agent.

Searches for video recommendations for course modules using:
1. LLM generates an optimal YouTube search query based on module content
2. YouTube search via yt-dlp retrieves video metadata
3. Results are stored per module in CourseVideos structure
"""

import json
import logging
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from main.state import (
    CourseState,
    VideoReference,
    ModuleVideos,
    CourseVideos,
    ModuleVideoEmbed,
    Module,
)
from LLMs.text2text import create_text_llm, resolve_text_model_name
from tools.videosearch import create_video_search
from .prompts import video_query_prompt, video_multi_query_prompt

logger = logging.getLogger(__name__)

# Minimum view count threshold for quality filtering
MIN_VIDEO_VIEWS = 5000


def _extract_module_topics(module: Module) -> str:
    """
    Extract topic list from module content for LLM context.
    
    Args:
        module: Module with submodules and sections
        
    Returns:
        Formatted string of key topics (submodule and section titles)
    """
    topics = []
    
    for submodule in module.submodules:
        topics.append(submodule.title)
        for section in submodule.sections:
            topics.append(f"  - {section.title}")
    
    # Limit to avoid overly long prompts
    return "\n".join(topics[:20])


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _generate_video_queries(
    module: Module,
    course_title: str,
    language: str,
    llm,
    num_videos: int,
) -> dict:
    """
    Generate multiple video search queries using LLM.
    
    Generates 1 general query + (num_videos - 1) specific queries for concrete concepts.
    
    Args:
        module: Module to generate queries for
        course_title: Course title for context
        language: Target language for queries
        llm: LLM instance for query generation
        num_videos: Total number of videos needed (determines specific query count)
        
    Returns:
        Dict with 'general_query' (str) and 'specific_queries' (list[str])
    """
    module_topics = _extract_module_topics(module)
    num_specific = max(1, num_videos - 1)  # At least 1 specific query
    
    chain = video_multi_query_prompt | llm | StrOutputParser()
    
    raw_response = chain.invoke({
        "course_title": course_title,
        "module_title": module.title,
        "module_description": module.description or "",
        "key_topics": module_topics,
        "language": language,
        "num_specific_queries": num_specific,
    })
    
    # Parse JSON response
    try:
        clean_response = _strip_markdown_fences(raw_response)
        data = json.loads(clean_response)
        
        general_query = data.get("general_query", "").strip()
        specific_queries = [q.strip() for q in data.get("specific_queries", []) if q.strip()]
        
        # Ensure we have valid data
        if not general_query:
            raise ValueError("Empty general_query in response")
        
        # Limit specific queries to what we need
        specific_queries = specific_queries[:num_specific]
        
        return {
            "general_query": general_query,
            "specific_queries": specific_queries,
        }
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse multi-query response: {e}. Falling back to single query.")
        # Fallback: use raw response as single general query
        fallback_query = raw_response.strip().replace('"', '').replace("'", "").strip()
        if not fallback_query or len(fallback_query) > 100:
            fallback_query = f"{module.title} tutorial {language}"
        return {
            "general_query": fallback_query,
            "specific_queries": [],
        }


def _convert_result_to_video(result: dict) -> VideoReference | None:
    """
    Convert a search result dict to VideoReference, filtering by view count.
    
    Args:
        result: Raw search result from video search provider
        
    Returns:
        VideoReference if valid, None if filtered out
    """
    # Skip error results
    if "error" in result:
        logger.warning(f"Video search error: {result.get('error')}")
        return None
    
    # Filter by minimum view count
    views = result.get("views", 0)
    if views < MIN_VIDEO_VIEWS:
        logger.debug(f"Skipping video with {views} views (min: {MIN_VIDEO_VIEWS})")
        return None
    
    return VideoReference(
        title=result.get("title", ""),
        url=result.get("url", ""),
        duration=result.get("duration", 0),
        published_at=result.get("published_at", 0),
        thumbnail=result.get("thumbnail", ""),
        channel=result.get("channel", ""),
        views=views,
        likes=result.get("likes", 0),
    )


def _select_videos_from_queries(
    general_results: list[dict],
    specific_results: dict[str, list[dict]],
    num_videos: int,
) -> list[VideoReference]:
    """
    Select videos from query results with deduplication.
    
    Selection order: general first, then specific queries in order.
    
    Args:
        general_results: Results from general query search
        specific_results: Dict mapping specific query -> results list
        num_videos: Target number of videos to select
        
    Returns:
        List of VideoReference, deduplicated by URL
    """
    videos: list[VideoReference] = []
    used_urls: set[str] = set()
    
    def add_first_valid_video(results: list[dict]) -> bool:
        """Add first valid video from results that isn't already used. Returns True if added."""
        for result in results:
            video = _convert_result_to_video(result)
            if video and video.url and video.url not in used_urls:
                videos.append(video)
                used_urls.add(video.url)
                return True
        return False
    
    # 1. FIRST: Add 1 video from general query
    add_first_valid_video(general_results)
    
    # 2. THEN: Add 1 video from each specific query (in order)
    for query, results in specific_results.items():
        if len(videos) >= num_videos:
            break
        if add_first_valid_video(results):
            logger.debug(f"Added video from specific query: {query}")
    
    # 3. FILL: If we still need more, add remaining from general query
    if len(videos) < num_videos:
        for result in general_results:
            if len(videos) >= num_videos:
                break
            video = _convert_result_to_video(result)
            if video and video.url and video.url not in used_urls:
                videos.append(video)
                used_urls.add(video.url)
    
    return videos


def generate_module_videos(
    module: Module,
    course_title: str,
    language: str,
    provider: str = "mistral",
    num_videos: int = 3,
    video_search_provider: str = "youtube",
) -> ModuleVideos:
    """
    Generate video recommendations for a single module using multi-query approach.
    
    Uses LLM to generate multiple search queries (1 general + N-1 specific),
    then searches YouTube for each query and selects videos with deduplication.
    Video order: general first, then specific queries.
    
    Args:
        module: Module to generate videos for
        course_title: Course title for context
        language: Course language (for query generation)
        provider: LLM provider for query generation
        num_videos: Number of videos to fetch
        video_search_provider: Video search provider (youtube | bing)
        
    Returns:
        ModuleVideos with video recommendations
    """
    # Create LLM
    model_name = resolve_text_model_name(provider)
    llm_kwargs = {"temperature": 0.3}
    if model_name:
        llm_kwargs["model_name"] = model_name
    llm = create_text_llm(provider=provider, **llm_kwargs)
    
    # 1. Generate multiple queries via LLM (single call)
    queries_data = _generate_video_queries(
        module=module,
        course_title=course_title,
        language=language,
        llm=llm,
        num_videos=num_videos,
    )
    
    general_query = queries_data["general_query"]
    specific_queries = queries_data["specific_queries"]
    
    print(f"      🔍 General: {general_query}")
    for i, sq in enumerate(specific_queries, 1):
        print(f"      🔍 Specific {i}: {sq}")
    
    # 2. Search for videos per query
    search_videos = create_video_search(video_search_provider)
    
    # Search general query (fetch extra to account for filtering)
    try:
        general_results = search_videos(general_query, max_results=num_videos * 3)
    except Exception as e:
        logger.error(f"General video search failed: {e}")
        general_results = []
    
    # Search each specific query
    specific_results: dict[str, list[dict]] = {}
    for sq in specific_queries:
        try:
            specific_results[sq] = search_videos(sq, max_results=3)
        except Exception as e:
            logger.error(f"Specific video search failed for '{sq}': {e}")
            specific_results[sq] = []
    
    # 3. Select videos with deduplication (general first, then specific)
    videos = _select_videos_from_queries(
        general_results=general_results,
        specific_results=specific_results,
        num_videos=num_videos,
    )
    
    # 4. Store all queries (pipe-separated)
    all_queries = [general_query] + specific_queries
    combined_query = " | ".join(all_queries)
    
    return ModuleVideos(
        module_index=module.index,
        module_title=module.title,
        query=combined_query,
        videos=videos,
    )


def _create_module_video_embed(module_videos: ModuleVideos) -> ModuleVideoEmbed:
    """
    Create ModuleVideoEmbed from ModuleVideos for module embedding.
    
    Args:
        module_videos: ModuleVideos with video recommendations
        
    Returns:
        ModuleVideoEmbed for embedding in module
    """
    return ModuleVideoEmbed(
        type="video",
        query=module_videos.query,
        content=module_videos.videos,
    )


def generate_course_videos(
    state: CourseState,
    provider: str | None = None,
    videos_per_module: int | None = None,
    video_search_provider: str | None = None,
    embed_in_modules: bool = True,
) -> CourseVideos:
    """
    Generate video recommendations for entire course.
    
    Processes modules sequentially, generating video recommendations for each.
    Also embeds video data directly in each Module.
    
    Args:
        state: CourseState with modules
        provider: LLM provider (defaults to state.config.text_llm_provider)
        videos_per_module: Videos per module (defaults to state.config.videos_per_module)
        video_search_provider: Video search provider (defaults to state.config.video_search_provider)
        embed_in_modules: Whether to embed videos in each module (default: True)
        
    Returns:
        CourseVideos with per-module video recommendations
    """
    provider = provider or state.config.text_llm_provider
    videos_per_module = videos_per_module or state.config.videos_per_module
    video_search_provider = video_search_provider or state.config.video_search_provider
    
    print(f"🎬 Generating video recommendations for {len(state.modules)} modules...")
    print(f"   Target: {videos_per_module} videos per module")
    print(f"   Provider: {provider}")
    print(f"   Search: {video_search_provider}")
    
    module_videos_list: list[ModuleVideos] = []
    
    for idx, module in enumerate(state.modules):
        print(f"\n   🎥 Module {idx + 1}/{len(state.modules)}: {module.title}")
        
        module_videos = generate_module_videos(
            module=module,
            course_title=state.title,
            language=state.config.language,
            provider=provider,
            num_videos=videos_per_module,
            video_search_provider=video_search_provider,
        )
        
        module_videos_list.append(module_videos)
        
        # Embed in module
        if embed_in_modules and module_videos.videos:
            module.video = _create_module_video_embed(module_videos)
        
        print(f"      ✓ Found {len(module_videos.videos)} videos")
    
    course_videos = CourseVideos(modules=module_videos_list)
    
    total_videos = sum(len(m.videos) for m in module_videos_list)
    print(f"\n✅ Video generation complete!")
    print(f"   Total videos: {total_videos}")
    
    return course_videos


def generate_videos_node(
    state: CourseState,
    config: Optional[RunnableConfig] = None,
) -> CourseState:
    """
    LangGraph node for video generation.
    
    Generates video recommendations for all modules and stores in state.
    Only runs if state.config.generate_videos is True.
    
    Args:
        state: CourseState with modules
        config: LangGraph runtime config
        
    Returns:
        Updated CourseState with videos
    """
    if not state.config.generate_videos:
        print("🎬 Video generation disabled, skipping...")
        return state
    
    course_videos = generate_course_videos(state)
    state.videos = course_videos
    
    return state

