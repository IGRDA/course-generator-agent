"""
Podcast TTS Tool

Generate multi-speaker podcast audio from structured conversations.
Supports Edge TTS, Coqui TTS (offline), and Chatterbox TTS.
"""

from .models import (
    Message,
    Conversation,
    LanguageConfig,
    LANGUAGE_CONFIGS,
    get_language_config,
)
from .base_engine import BaseTTSEngine
from .edge import (
    EdgeTTSEngine,
    EDGE_VOICE_MAP,
    EDGE_VOICES,
    generate_podcast_edge,
)
from .coqui import (
    TTSEngine,
    CoquiTTSEngine,
    generate_podcast,
    list_available_languages,
    list_speakers,
)
from .chatterbox import ChatterboxEngine, generate_podcast_chatterbox
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
    # Edge TTS Engine
    "EdgeTTSEngine",
    "EDGE_VOICE_MAP",
    "EDGE_VOICES",
    "generate_podcast_edge",
    # Coqui TTS Engine
    "TTSEngine",
    "CoquiTTSEngine",
    "generate_podcast",
    "list_available_languages",
    "list_speakers",
    # Chatterbox TTS Engine
    "ChatterboxEngine",
    "generate_podcast_chatterbox",
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
