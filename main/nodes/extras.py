"""
Extra nodes for bibliography and podcast generation.
"""

import json
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig

from main.state import CourseState
from agents.bibliography_generator.agent import generate_course_bibliography
from agents.podcast_generator.agent import (
    generate_conversation,
    get_tts_language,
    TTSEngineType,
)
from .utils import get_output_manager


def generate_bibliography_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate book bibliography for the course (optional, controlled by config).
    
    This node creates book recommendations for each module using Open Library API.
    Only runs if generate_bibliography is enabled in config.
    
    Args:
        state: CourseState with populated course structure.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Updated CourseState with bibliography if enabled, unchanged otherwise.
    """
    if not state.config.generate_bibliography:
        print("📚 Bibliography generation disabled, skipping...")
        return state
    
    print("📚 Generating course bibliography...")
    
    bibliography = generate_course_bibliography(state)
    state.bibliography = bibliography
    
    print("Bibliography generation completed!")
    
    # Save step snapshot if OutputManager is available
    output_mgr = get_output_manager(config)
    if output_mgr:
        output_mgr.save_step("bibliography", state)
    
    return state


def generate_podcasts_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate podcasts for all modules in the course.
    
    This node:
    1. Converts the CourseState to course_data dict format
    2. Iterates through all modules
    3. Generates conversation for each module using LLM
    4. Synthesizes audio using configured TTS engine (Edge TTS by default)
    5. Saves conversation JSON and MP3 files to output folder
    
    Args:
        state: CourseState with populated section theories.
        config: LangGraph RunnableConfig for accessing OutputManager.
        
    Returns:
        Unchanged CourseState (podcasts are saved as files).
    """
    print("Generating podcasts for all modules...")
    
    # Get output manager for saving files
    output_mgr = get_output_manager(config)
    if not output_mgr:
        print("⚠️ Warning: No OutputManager found, skipping podcast generation")
        return state
    
    # Convert state to course_data dict format expected by podcast generator
    course_data = state.model_dump()
    
    # Get podcast configuration
    tts_engine: TTSEngineType = state.config.podcast_tts_engine
    target_words = state.config.podcast_target_words
    speaker_map = state.config.podcast_speaker_map
    provider = state.config.text_llm_provider
    
    # Get TTS language from course language
    tts_language = get_tts_language(state.config.language)
    
    # Setup podcast output directory
    podcast_dir = Path(output_mgr.get_run_folder()) / "podcast"
    podcast_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the path to background music
    project_root = Path(__file__).parent.parent.parent
    music_path = project_root / "tools" / "podcast" / "background_music.mp3"
    music_path_str = str(music_path) if music_path.exists() else None
    
    num_modules = len(state.modules)
    print(f"   Found {num_modules} modules to process")
    print(f"   TTS Engine: {tts_engine}")
    print(f"   Language: {tts_language}")
    print(f"   Target words per podcast: {target_words}")
    
    # Generate podcast for each module
    for module_idx in range(num_modules):
        module = state.modules[module_idx]
        print(f"\n🎙️ Generating podcast for Module {module_idx + 1}: {module.title}")
        
        # Generate conversation
        print(f"   Generating conversation...")
        conversation = generate_conversation(
            course_data=course_data,
            module_idx=module_idx,
            provider=provider,
            target_words=target_words,
        )
        
        # Save conversation JSON
        conv_filename = f"module_{module_idx + 1}_conversation.json"
        conv_path = podcast_dir / conv_filename
        with open(conv_path, "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Conversation saved: {conv_path.name}")
        
        # Generate audio
        audio_filename = f"module_{module_idx + 1}.mp3"
        audio_path = podcast_dir / audio_filename
        
        print(f"   🔊 Synthesizing audio with {tts_engine.upper()} TTS...")
        
        if tts_engine == "edge":
            from tools.podcast.tts_engine import generate_podcast_edge
            
            generate_podcast_edge(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                title=f"Module {module_idx + 1}: {module.title}",
                artist="Adinhub",
                album=state.title,
                track_number=module_idx + 1,
                music_path=music_path_str,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        else:
            from tools.podcast.tts_engine import generate_podcast
            
            generate_podcast(
                conversation=conversation,
                output_path=str(audio_path),
                language=tts_language,
                speaker_map=speaker_map,
                language_code=tts_language,
                title=f"Module {module_idx + 1}: {module.title}",
                artist="Adinhub",
                album=state.title,
                track_number=module_idx + 1,
                music_path=music_path_str,
                intro_duration_ms=10000,
                outro_duration_ms=10000,
                intro_fade_ms=5000,
                outro_fade_ms=5000,
            )
        
        print(f"   ✅ Audio saved: {audio_path.name}")
    
    print(f"\n✅ All {num_modules} podcasts generated successfully!")
    print(f"   Output folder: {podcast_dir}")
    
    return state

