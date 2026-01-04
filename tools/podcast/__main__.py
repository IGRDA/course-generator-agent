#!/usr/bin/env python3
"""
Podcast TTS Tool - Command Line Interface

Minimal example demonstrating 2-voice podcast generation using Coqui TTS.

Usage:
    python -m tools.podcast                           # Run with default English voices
    python -m tools.podcast --language es             # Spanish
    python -m tools.podcast --language multilingual   # Multilingual XTTS model
    python -m tools.podcast --output my_podcast.mp3   # Custom output path
    python -m tools.podcast --list-speakers en        # List available speakers
    python -m tools.podcast --music                   # Add background music (intro/outro)
    python -m tools.podcast --title "Episode 1"      # Set metadata
"""

import argparse
import sys
from pathlib import Path


# Sample conversation for testing
SAMPLE_CONVERSATION_EN = [
    {
        "role": "host",
        "content": "Welcome to our podcast about artificial intelligence! I'm your host, and today we have a special guest joining us."
    },
    {
        "role": "guest", 
        "content": "Thank you for having me! I'm excited to discuss the latest developments in AI technology."
    },
    {
        "role": "host",
        "content": "Let's start with the basics. Can you explain what machine learning is in simple terms?"
    },
    {
        "role": "guest",
        "content": "Of course! Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed. Instead of writing rules, we show the computer examples and it learns patterns."
    },
    {
        "role": "host",
        "content": "That's a great explanation. And how does this relate to deep learning?"
    },
    {
        "role": "guest",
        "content": "Deep learning is a subset of machine learning that uses neural networks with many layers. These deep networks can learn very complex patterns and are behind many recent breakthroughs like image recognition and natural language processing."
    },
]

SAMPLE_CONVERSATION_ES = [
    {
        "role": "host",
        "content": "¡Bienvenidos a nuestro podcast sobre inteligencia artificial! Soy su anfitrión y hoy tenemos un invitado especial."
    },
    {
        "role": "guest",
        "content": "¡Gracias por invitarme! Estoy emocionado de hablar sobre los últimos avances en tecnología de IA."
    },
    {
        "role": "host",
        "content": "Comencemos con lo básico. ¿Puedes explicar qué es el aprendizaje automático en términos simples?"
    },
    {
        "role": "guest",
        "content": "¡Por supuesto! El aprendizaje automático es un tipo de inteligencia artificial que permite a las computadoras aprender de los datos sin ser programadas explícitamente."
    },
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a podcast using Coqui TTS with multiple speakers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tools.podcast
    python -m tools.podcast --language multilingual --output podcast.mp3
    python -m tools.podcast --list-speakers en
    python -m tools.podcast --device cuda  # Use GPU acceleration
        """
    )
    
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        choices=["en", "es", "multilingual", "en_fast"],
        help="TTS language/model to use (default: en). Use 'en_fast' for faster but less natural VCTK model."
    )
    
    parser.add_argument(
        "--language-code",
        type=str,
        default=None,
        help="Language code for multilingual model (e.g., 'es', 'en', 'fr'). Only used with multilingual model."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="podcast_sample.mp3",
        help="Output audio file path (default: podcast_sample.mp3)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run TTS on (default: cpu)"
    )
    
    parser.add_argument(
        "--silence", "-s",
        type=int,
        default=500,
        help="Silence duration between messages in ms (default: 500)"
    )
    
    parser.add_argument(
        "--list-speakers",
        type=str,
        metavar="LANGUAGE",
        help="List available speakers for a language and exit"
    )
    
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List available languages and exit"
    )
    
    parser.add_argument(
        "--host-speaker",
        type=str,
        default=None,
        help="Speaker ID for the host role"
    )
    
    parser.add_argument(
        "--guest-speaker",
        type=str,
        default=None,
        help="Speaker ID for the guest role"
    )
    
    # Metadata arguments
    metadata_group = parser.add_argument_group("Metadata", "ID3 metadata for the output MP3")
    metadata_group.add_argument(
        "--title",
        type=str,
        default="Module",
        help="Track title (default: Module)"
    )
    metadata_group.add_argument(
        "--artist",
        type=str,
        default="Adinhub",
        help="Artist name (default: Adinhub)"
    )
    metadata_group.add_argument(
        "--album",
        type=str,
        default="Course",
        help="Album name (default: Course)"
    )
    metadata_group.add_argument(
        "--track-number",
        type=int,
        default=None,
        help="Track number (optional)"
    )
    
    # Background music arguments
    music_group = parser.add_argument_group("Background Music", "Intro/outro music settings")
    music_group.add_argument(
        "--music",
        type=str,
        nargs="?",
        const="default",
        default=None,
        metavar="PATH",
        help="Path to background music file. Use --music without a path to use default music."
    )
    music_group.add_argument(
        "--intro-duration",
        type=int,
        default=5000,
        metavar="MS",
        help="Intro music duration in ms (default: 5000)"
    )
    music_group.add_argument(
        "--outro-duration",
        type=int,
        default=5000,
        metavar="MS",
        help="Outro music duration in ms (default: 5000)"
    )
    music_group.add_argument(
        "--intro-fade",
        type=int,
        default=3000,
        metavar="MS",
        help="Intro fade-in duration in ms (default: 3000)"
    )
    music_group.add_argument(
        "--outro-fade",
        type=int,
        default=3000,
        metavar="MS",
        help="Outro fade-out duration in ms (default: 3000)"
    )
    music_group.add_argument(
        "--music-volume",
        type=int,
        default=-6,
        metavar="DB",
        help="Music volume adjustment in dB (default: -6)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Import here to avoid loading TTS when just showing help
    from .tts_engine import generate_podcast, list_available_languages, list_speakers
    from .models import get_language_config
    from .audio_utils import get_default_music_path
    
    # Handle list commands
    if args.list_languages:
        print("Available languages:")
        for lang in list_available_languages():
            config = get_language_config(lang)
            print(f"  {lang}: {config.model_name}")
        return 0
    
    if args.list_speakers:
        try:
            speakers = list_speakers(args.list_speakers)
            config = get_language_config(args.list_speakers)
            print(f"Available speakers for {args.list_speakers} ({config.model_name}):")
            for speaker in speakers:
                print(f"  - {speaker}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0
    
    # Select sample conversation based on language
    if args.language == "es" or (args.language == "multilingual" and args.language_code == "es"):
        conversation = SAMPLE_CONVERSATION_ES
    else:
        conversation = SAMPLE_CONVERSATION_EN
    
    # Build speaker map if custom speakers specified
    speaker_map = None
    if args.host_speaker or args.guest_speaker:
        config = get_language_config(args.language)
        speaker_map = config.default_speaker_map.copy()
        if args.host_speaker:
            speaker_map["host"] = args.host_speaker
        if args.guest_speaker:
            speaker_map["guest"] = args.guest_speaker
    
    # Determine language code (for multilingual/XTTS models)
    # The config already has a default language_code, but CLI arg can override it
    language_code = args.language_code  # Will be None if not specified, engine uses config default
    
    # Resolve music path
    music_path = None
    if args.music:
        if args.music == "default":
            music_path = get_default_music_path()
            if not music_path:
                print("⚠️  No default background music found. Skipping music.", file=sys.stderr)
        else:
            music_path = args.music
            if not Path(music_path).exists():
                print(f"❌ Music file not found: {music_path}", file=sys.stderr)
                return 1
    
    print(f"🎙️  Podcast TTS Generator")
    print(f"   Language: {args.language}")
    if language_code:
        print(f"   Language code: {language_code}")
    print(f"   Output: {args.output}")
    print(f"   Device: {args.device}")
    print(f"   Messages: {len(conversation)}")
    print(f"   Metadata: {args.title} by {args.artist} ({args.album})")
    if music_path:
        print(f"   Music: {music_path}")
        print(f"   Intro: {args.intro_duration}ms (fade: {args.intro_fade}ms)")
        print(f"   Outro: {args.outro_duration}ms (fade: {args.outro_fade}ms)")
    print()
    
    def progress_callback(current, total):
        print(f"   Synthesizing message {current}/{total}...", flush=True)
    
    try:
        output_path = generate_podcast(
            conversation=conversation,
            output_path=args.output,
            language=args.language,
            speaker_map=speaker_map,
            language_code=language_code,
            silence_duration_ms=args.silence,
            device=args.device,
            progress_callback=progress_callback,
            # Metadata
            title=args.title,
            artist=args.artist,
            album=args.album,
            track_number=args.track_number,
            # Background music
            music_path=music_path,
            intro_duration_ms=args.intro_duration,
            outro_duration_ms=args.outro_duration,
            intro_fade_ms=args.intro_fade,
            outro_fade_ms=args.outro_fade,
            music_volume_db=args.music_volume,
        )
        
        print()
        print(f"✅ Podcast generated successfully: {output_path}")
        
        # Show file size
        size = Path(output_path).stat().st_size
        if size > 1024 * 1024:
            print(f"   Size: {size / (1024 * 1024):.2f} MB")
        else:
            print(f"   Size: {size / 1024:.2f} KB")
        
        # Show metadata info
        print(f"   Title: {args.title}")
        print(f"   Artist: {args.artist}")
        print(f"   Album: {args.album}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

