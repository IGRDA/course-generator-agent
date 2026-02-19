"""
TTS Engine Factory.

Provides a unified interface for creating TTS engines by name.
"""

from typing import Optional, Literal

from .base_engine import BaseTTSEngine


# Available engine types
EngineType = Literal["edge", "coqui", "chatterbox", "elevenlabs"]


def create_tts_engine(
    engine: EngineType = "edge",
    language: str = "en",
    speaker_map: Optional[dict[str, str]] = None,
    **kwargs,
) -> BaseTTSEngine:
    """Create a TTS engine by name.
    
    Args:
        engine: Engine type ("edge" or "coqui")
        language: Language code (e.g., "en", "es")
        speaker_map: Optional mapping of role names to speaker/voice IDs
        **kwargs: Additional engine-specific arguments
            - For Coqui: device (str) - "cpu" or "cuda"
    
    Returns:
        TTS engine instance
        
    Raises:
        ValueError: If engine type is unknown
        
    Example:
        >>> engine = create_tts_engine("edge", language="es")
        >>> engine.supports_ssml
        True
        
        >>> engine = create_tts_engine("coqui", language="en", device="cuda")
        >>> engine.supports_ssml
        False
    """
    if engine == "edge":
        from .edge.client import EdgeTTSEngine
        return EdgeTTSEngine(
            language=language,
            speaker_map=speaker_map,
        )
    elif engine == "coqui":
        from .coqui.client import CoquiTTSEngine
        return CoquiTTSEngine(
            language=language,
            speaker_map=speaker_map,
            device=kwargs.get("device", "cpu"),
        )
    elif engine == "chatterbox":
        from .chatterbox.client import ChatterboxEngine
        return ChatterboxEngine(
            language=language,
            speaker_map=speaker_map,
            device=kwargs.get("device", "cuda"),
            exaggeration=kwargs.get("exaggeration", 0.5),
            cfg_weight=kwargs.get("cfg_weight", 0.5),
        )
    elif engine == "elevenlabs":
        from .elevenlabs.client import ElevenLabsTTSEngine
        return ElevenLabsTTSEngine(
            language=language,
            speaker_map=speaker_map,
            model_id=kwargs.get("model_id", "eleven_multilingual_v2"),
            api_key=kwargs.get("api_key"),
        )
    else:
        available = ["edge", "coqui", "chatterbox", "elevenlabs"]
        raise ValueError(f"Unknown engine '{engine}'. Available: {available}")


def get_engine_info(engine: EngineType) -> dict:
    """Get information about a TTS engine.
    
    Args:
        engine: Engine type
        
    Returns:
        Dict with engine capabilities and info
    """
    info = {
        "edge": {
            "name": "Edge TTS",
            "description": "Microsoft Edge neural TTS - fast, high quality",
            "requires_internet": True,
            "languages": ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
        },
        "coqui": {
            "name": "Coqui TTS",
            "description": "Open-source TTS with multi-speaker support - works offline",
            "requires_internet": False,
            "languages": ["en", "es", "multilingual"],
        },
        "chatterbox": {
            "name": "Chatterbox TTS",
            "description": "Resemble AI's zero-shot TTS with voice cloning - 23 languages",
            "requires_internet": False,
            "languages": [
                "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
                "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
                "sw", "tr", "zh",
            ],
        },
        "elevenlabs": {
            "name": "ElevenLabs TTS",
            "description": "ElevenLabs cloud API - ultra-realistic voices, 29+ languages",
            "requires_internet": True,
            "languages": [
                "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
                "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
                "tr", "zh",
            ],
        },
    }
    
    if engine not in info:
        raise ValueError(f"Unknown engine '{engine}'")
    
    return info[engine]


def list_engines() -> list[dict]:
    """List all available TTS engines with their info.
    
    Returns:
        List of engine info dictionaries
    """
    return [
        {"engine": "edge", **get_engine_info("edge")},
        {"engine": "coqui", **get_engine_info("coqui")},
        {"engine": "chatterbox", **get_engine_info("chatterbox")},
        {"engine": "elevenlabs", **get_engine_info("elevenlabs")},
    ]

