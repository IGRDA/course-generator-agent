"""
Video Generator Agent.

Generates video recommendations for course modules using:
1. LLM generates an optimal YouTube search query based on module content
2. YouTube search via yt-dlp retrieves video metadata
3. Results are stored per module in CourseVideos structure
"""

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
from .prompts import video_query_prompt

logger = logging.getLogger(__name__)


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


def generate_module_videos(
    module: Module,
    course_title: str,
    language: str,
    provider: str = "mistral",
    num_videos: int = 3,
    video_search_provider: str = "youtube",
) -> ModuleVideos:
    """
    Generate video recommendations for a single module.
    
    Uses LLM to generate an optimal search query, then searches YouTube
    for relevant educational videos.
    
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
    
    # Extract module topics
    module_topics = _extract_module_topics(module)
    
    # Generate search query using LLM
    chain = video_query_prompt | llm | StrOutputParser()
    
    query = chain.invoke({
        "course_title": course_title,
        "module_title": module.title,
        "module_description": module.description or "",
        "key_topics": module_topics,
        "language": language,
    })
    
    # Clean query
    query = query.strip().replace('"', '').replace("'", "").strip()
    
    print(f"      🔍 Query: {query}")
    
    # Search for videos
    search_videos = create_video_search(video_search_provider)
    
    try:
        results = search_videos(query, max_results=num_videos)
    except Exception as e:
        logger.error(f"Video search failed: {e}")
        results = []
    
    # Convert results to VideoReference objects
    videos: list[VideoReference] = []
    for result in results:
        # Skip error results
        if "error" in result:
            logger.warning(f"Video search error: {result.get('error')}")
            continue
        
        video = VideoReference(
            title=result.get("title", ""),
            url=result.get("url", ""),
            duration=result.get("duration", 0),
            published_at=result.get("published_at", 0),
            thumbnail=result.get("thumbnail", ""),
            channel=result.get("channel", ""),
            views=result.get("views", 0),
            likes=result.get("likes", 0),
        )
        videos.append(video)
    
    return ModuleVideos(
        module_index=module.index,
        module_title=module.title,
        query=query,
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

