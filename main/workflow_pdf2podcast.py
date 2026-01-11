"""
PDF to Podcast Workflow.

Generates a course from a PDF syllabus and creates podcasts for all modules.
Skips activities, HTML generation, and image generation for faster execution.

Pipeline:
1. Extract course structure from PDF
2. Generate section theories
3. Calculate metadata
4. Generate podcasts for all modules
"""

import json
from pathlib import Path
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from main.state import CourseState, CourseConfig
from main.output_manager import OutputManager
from main.workflow_pdf import (
    generate_index_from_pdf_node,
    generate_theories_node,
    calculate_metadata_node,
    _get_output_manager,
)
from agents.podcast_generator.agent import (
    generate_conversation,
    get_tts_language,
    TTSEngineType,
)


def generate_podcasts_node(state: CourseState, config: Optional[RunnableConfig] = None) -> CourseState:
    """Generate podcasts for all modules in the course.
    
    This node:
    1. Converts the CourseState to course_data dict format
    2. Iterates through all modules
    3. Generates conversation for each module using LLM
    4. Synthesizes audio using configured TTS engine (Edge TTS by default)
    5. Saves conversation JSON and MP3 files to output folder
    """
    print("Generating podcasts for all modules...")
    
    # Get output manager for saving files
    output_mgr = _get_output_manager(config)
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
    project_root = Path(__file__).parent.parent
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


def build_pdf2podcast_graph():
    """Build and return the PDF to podcast generation graph.
    
    This graph has a streamlined pipeline:
    1. generate_index_from_pdf - Extracts course structure from PDF syllabus
    2. generate_theories - Populates section content for podcast dialogue
    3. calculate_metadata - Sets IDs and indexes for proper course.json structure
    4. generate_podcasts - Generates podcast audio for ALL modules
    
    Skipped steps (not needed for podcasts):
    - generate_activities
    - generate_html
    - generate_images
    """
    graph = StateGraph(CourseState)
    
    # Add nodes for PDF to podcast pipeline
    graph.add_node("generate_index_from_pdf", generate_index_from_pdf_node)
    graph.add_node("generate_theories", generate_theories_node)
    graph.add_node("calculate_metadata", calculate_metadata_node)
    graph.add_node("generate_podcasts", generate_podcasts_node)
    
    # Add edges for sequential execution
    graph.add_edge(START, "generate_index_from_pdf")
    graph.add_edge("generate_index_from_pdf", "generate_theories")
    graph.add_edge("generate_theories", "calculate_metadata")
    graph.add_edge("calculate_metadata", "generate_podcasts")
    graph.add_edge("generate_podcasts", END)
    
    return graph.compile()


if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate podcasts from a PDF syllabus (no activities, HTML, or images)")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF syllabus file")
    parser.add_argument("--total-pages", type=int, default=50, help="Total pages for the course (default: 50)")
    parser.add_argument("--target-words", type=int, default=600, help="Target word count per podcast (default: 600)")
    parser.add_argument("--tts-engine", type=str, choices=["edge", "coqui"], default="edge", help="TTS engine (default: edge)")
    parser.add_argument("--language", type=str, default="Español", help="Language for content generation (default: Español)")
    parser.add_argument("--provider", type=str, default="mistral", help="LLM provider (default: mistral)")
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        exit(1)
    
    # Build the graph
    app = build_pdf2podcast_graph()
    
    # Create initial CourseState with PDF to podcast config
    course_config = CourseConfig(
        pdf_syllabus_path=str(pdf_path),
        text_llm_provider=args.provider,
        web_search_provider="ddg",
        total_pages=args.total_pages,
        words_per_page=400,
        language=args.language,
        max_retries=8,
        concurrency=10,
        use_reflection=True,
        num_reflection_queries=5,
        # Research configuration
        enable_research=True,
        research_max_queries=5,
        research_max_results_per_query=5,
        # Podcast configuration
        podcast_target_words=args.target_words,
        podcast_tts_engine=args.tts_engine,
        podcast_speaker_map={'host': 'es-ES-AlvaroNeural', 'guest': 'es-ES-XimenaNeural'},
        target_audience=None,
    )
    
    initial_state = CourseState(
        config=course_config,
        title="",  # Will be extracted from PDF
        modules=[]
    )
    
    # Create OutputManager - title will be updated after PDF extraction
    output_mgr = OutputManager(title=pdf_path.stem.replace(" ", "_"))
    print(f"📁 Output folder: {output_mgr.get_run_folder()}")
    print(f"📄 PDF Source: {pdf_path}")
    
    # Run the graph
    result = app.invoke(
        initial_state,
        config={
            "run_name": f"PDF2Podcast: {pdf_path.stem}",
            "configurable": {"output_manager": output_mgr}
        }
    )
    
    # Print the final course state
    print("\nWorkflow completed successfully!")
    
    # Extract the final state
    final_state = result if isinstance(result, CourseState) else CourseState.model_validate(result)
    
    # Save final outputs
    output_mgr.save_final(final_state)
    output_mgr.save_modules(final_state)
    
    # Print summary
    total_sections = sum(len(s.sections) for m in final_state.modules for s in m.submodules)
    print(f"\n📊 Course Summary:")
    print(f"   Title: {final_state.title}")
    print(f"   Modules: {len(final_state.modules)}")
    print(f"   Total Sections: {total_sections}")
    print(f"   Language: {final_state.language}")
    print(f"   TTS Engine: {final_state.config.podcast_tts_engine}")
    print(f"   Source PDF: {final_state.config.pdf_syllabus_path}")
    print(f"\n✅ All outputs saved to: {output_mgr.get_run_folder()}")

