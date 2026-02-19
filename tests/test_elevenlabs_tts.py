#!/usr/bin/env python3
"""
Minimal test for the ElevenLabs TTS engine.

Run directly:
    source env.secrets && python tests/test_elevenlabs_tts.py

Tests:
  1. Single message with a male voice (Adam)
  2. Single message with a female voice (Rachel)
  3. A two-message conversation (host + guest)
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_api_key():
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        print("ERROR: ELEVENLABS_API_KEY not set. Run: source env.secrets")
        sys.exit(1)
    print(f"API key found (ends with ...{key[-4:]})")


def test_single_male_voice(output_dir: Path):
    """Synthesize a short sentence with the male voice (Adam)."""
    from tools.podcast.elevenlabs.client import ElevenLabsTTSEngine
    from tools.podcast.models import Message

    print("\n--- Test 1: Male voice (Adam) ---")
    engine = ElevenLabsTTSEngine(language="en")

    msg = Message(role="host", content="Hello, welcome to the show.")
    out_path = str(output_dir / "test_male.mp3")
    engine.synthesize_message(msg, out_path)

    size = os.path.getsize(out_path)
    print(f"  Output: {out_path}")
    print(f"  Size:   {size:,} bytes")
    assert size > 0, "Male voice file is empty!"
    print("  PASS")
    return out_path


def test_single_female_voice(output_dir: Path):
    """Synthesize a short sentence with the female voice (Rachel)."""
    from tools.podcast.elevenlabs.client import ElevenLabsTTSEngine
    from tools.podcast.models import Message

    print("\n--- Test 2: Female voice (Rachel) ---")
    engine = ElevenLabsTTSEngine(language="en")

    msg = Message(role="guest", content="Thanks for having me today.")
    out_path = str(output_dir / "test_female.mp3")
    engine.synthesize_message(msg, out_path)

    size = os.path.getsize(out_path)
    print(f"  Output: {out_path}")
    print(f"  Size:   {size:,} bytes")
    assert size > 0, "Female voice file is empty!"
    print("  PASS")
    return out_path


def test_conversation(output_dir: Path):
    """Synthesize a minimal 2-message conversation."""
    from tools.podcast.elevenlabs.client import ElevenLabsTTSEngine
    from tools.podcast.models import Conversation, Message

    print("\n--- Test 3: Two-message conversation ---")
    engine = ElevenLabsTTSEngine(language="en")

    conv = Conversation(messages=[
        Message(role="host", content="What is machine learning?"),
        Message(role="guest", content="It is a branch of artificial intelligence."),
    ])
    out_path = str(output_dir / "test_conversation.mp3")
    engine.synthesize_conversation(conv, out_path, silence_duration_ms=400)

    size = os.path.getsize(out_path)
    print(f"  Output: {out_path}")
    print(f"  Size:   {size:,} bytes")
    assert size > 0, "Conversation file is empty!"
    print("  PASS")
    return out_path


def test_factory():
    """Verify the engine can be created through the factory."""
    from tools.podcast.factory import create_tts_engine

    print("\n--- Test 4: Factory instantiation ---")
    engine = create_tts_engine("elevenlabs", language="en")
    print(f"  Engine type: {type(engine).__name__}")
    print(f"  Model ID:    {engine.model_id}")
    assert type(engine).__name__ == "ElevenLabsTTSEngine"
    print("  PASS")


def main():
    check_api_key()

    output_dir = Path(tempfile.mkdtemp(prefix="elevenlabs_test_"))
    print(f"Output directory: {output_dir}")

    try:
        test_single_male_voice(output_dir)
        test_single_female_voice(output_dir)
        test_conversation(output_dir)
        test_factory()
        print("\n=== ALL TESTS PASSED ===")
        print(f"Audio files saved in: {output_dir}")
    except Exception as e:
        print(f"\n!!! TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
