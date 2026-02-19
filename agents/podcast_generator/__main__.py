"""CLI entry point for the Podcast Generator Agent.

Usage:
    python -m agents.podcast_generator <course.json> --module <N> [options]

Examples:
    python -m agents.podcast_generator output/Quantum_Theory_20260103_133822/course.json --module 1
    python -m agents.podcast_generator course.json --module 2 --provider openai --skip-tts
    python -m agents.podcast_generator course.json --module 1 --output-dir ./podcasts
"""

import argparse
import sys
from pathlib import Path

from .agent import generate_module_podcast


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate podcast conversation and audio from course module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m agents.podcast_generator output/Course/course.json --module 1
    python -m agents.podcast_generator course.json --module 2 --provider openai
    python -m agents.podcast_generator course.json --module 1 --skip-tts
        """,
    )
    parser.add_argument(
        "course_path",
        type=str,
        help="Path to the course.json file",
    )
    parser.add_argument(
        "--module", "-m",
        type=int,
        required=True,
        help="Module index (1-based, e.g., --module 1 for first module)",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default=None,
        help="LLM provider override (openai, deepseek, gemini, etc.)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Custom output directory (default: podcast/ in course folder)",
    )
    parser.add_argument(
        "--target-words", "-w",
        type=int,
        default=600,
        help="Target word count for conversation (default: 600)",
    )
    parser.add_argument(
        "--tts-engine", "-t",
        type=str,
        choices=["edge", "coqui", "elevenlabs", "chatterbox"],
        default="edge",
        help="TTS engine to use for audio synthesis (default: edge)",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Only generate conversation JSON, skip audio synthesis",
    )
    
    args = parser.parse_args()
    
    # Validate course path
    course_path = Path(args.course_path)
    if not course_path.exists():
        print(f"❌ Error: Course file not found: {course_path}", file=sys.stderr)
        return 1
    
    # Convert 1-based to 0-based index
    module_idx = args.module - 1
    if module_idx < 0:
        print(f"❌ Error: Module index must be >= 1", file=sys.stderr)
        return 1
    
    try:
        result = generate_module_podcast(
            course_path=str(course_path),
            module_idx=module_idx,
            output_dir=args.output_dir,
            provider=args.provider,
            target_words=args.target_words,
            skip_tts=args.skip_tts,
            tts_engine=args.tts_engine,
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print("📻 Podcast Generation Complete!")
        print("=" * 50)
        print(f"   Course: {result['course_title']}")
        print(f"   Module: {result['module_title']}")
        print(f"   Language: {result['language']}")
        print(f"   Conversation: {result['conversation_path']}")
        if 'audio_path' in result:
            print(f"   Audio: {result['audio_path']}")
        print(f"   Messages: {len(result['conversation'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

