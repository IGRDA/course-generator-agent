"""
Podcast TTS Tool

Generate multi-speaker podcast audio from structured conversations using Coqui TTS.
"""

from .models import (
    Message,
    Conversation,
    LanguageConfig,
    LANGUAGE_CONFIGS,
    get_language_config,
)
from .tts_engine import (
    TTSEngine,
    generate_podcast,
    list_available_languages,
    list_speakers,
)
from .audio_utils import (
    add_metadata,
    add_background_music,
    get_default_music_path,
)

__all__ = [
    # Models
    "Message",
    "Conversation",
    "LanguageConfig",
    "LANGUAGE_CONFIGS",
    "get_language_config",
    # TTS Engine
    "TTSEngine",
    "generate_podcast",
    "list_available_languages",
    "list_speakers",
    # Audio Utilities
    "add_metadata",
    "add_background_music",
    "get_default_music_path",
]

