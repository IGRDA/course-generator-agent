"""
Podcast Generator Agent.

Generates two-speaker educational dialogue conversations from course content
and synthesizes them into podcast audio using TTS.
"""

from .agent import (
    generate_conversation,
    extract_module_context,
    get_tts_language,
    LANGUAGE_MAP,
)

__all__ = [
    "generate_conversation",
    "extract_module_context",
    "get_tts_language",
    "LANGUAGE_MAP",
]

