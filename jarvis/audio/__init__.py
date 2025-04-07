"""
Audio processing module for Jarvis.
"""

# Re-export the TranscriptionMode enum and other important classes
from .audio_handler import (
    TranscriptionMode,
    WhisperXModelSize,
    record_microphone,
    transcribe_audio,
    visualize_audio,
    clear_transcription_cache
)