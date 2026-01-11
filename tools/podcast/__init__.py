"""
Podcast TTS Tool

Generate multi-speaker podcast audio from structured conversations.
Supports both Coqui TTS (offline) and Edge TTS (SSML support).
"""

from .models import (
    Message,
    Conversation,
    LanguageConfig,
    LANGUAGE_CONFIGS,
    get_language_config,
)
from .base_engine import BaseTTSEngine
from .tts_engine import (
    TTSEngine,
    CoquiTTSEngine,
    generate_podcast,
    generate_podcast_edge,
    list_available_languages,
    list_speakers,
)
from .edge_engine import EdgeTTSEngine, EDGE_VOICE_MAP, EDGE_VOICES
from .factory import create_tts_engine, get_engine_info, list_engines, EngineType
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
    # Base Engine
    "BaseTTSEngine",
    # Coqui TTS Engine
    "TTSEngine",
    "CoquiTTSEngine",
    "generate_podcast",
    "list_available_languages",
    "list_speakers",
    # Edge TTS Engine
    "EdgeTTSEngine",
    "EDGE_VOICE_MAP",
    "EDGE_VOICES",
    "generate_podcast_edge",
    # Factory
    "create_tts_engine",
    "get_engine_info",
    "list_engines",
    "EngineType",
    # Audio Utilities
    "add_metadata",
    "add_background_music",
    "get_default_music_path",
]
