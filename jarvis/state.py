"""
State management classes for the Jarvis application.

This module contains classes that manage application state, separate from UI rendering.
"""

from dataclasses import dataclass, field
import datetime


@dataclass
class RecordingState:
    """
    State for tracking audio recording status.
    Includes recording status, duration, and volume level.
    """

    is_recording: bool = False
    duration: float = 0.0
    volume_level: float = 0.0
    transcribing: bool = False
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)

    def update(
        self,
        is_recording: bool,
        duration: float = 0.0,
        volume_level: float = 0.0,
        transcribing: bool = False,
    ):
        """
        Update the recording state with new values.

        Args:
            is_recording: Whether recording is active
            duration: Current recording duration in seconds
            volume_level: Audio volume level (0.0-1.0)
            transcribing: Whether transcription is in progress
        """
        self.is_recording = is_recording
        self.duration = duration
        self.volume_level = min(max(volume_level, 0.0), 1.0)  # Clamp between 0 and 1
        self.transcribing = transcribing
        self.last_updated = datetime.datetime.now()  # Update timestamp
