#!/usr/bin/env python3
"""
Test file for the audio_handler module.
This test ensures the proper import paths are working.
"""
import unittest
import sys
import os

# Add project root to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from jarvis package
from jarvis.audio.audio_handler import (
    TranscriptionMode,
    WhisperXModelSize,
    transcribe_audio
)

class TestAudioHandler(unittest.TestCase):
    """Basic tests for the audio handler module"""

    def test_import(self):
        """Test that the audio_handler module can be imported correctly"""
        self.assertIsNotNone(TranscriptionMode)
        self.assertIsNotNone(WhisperXModelSize)
        self.assertIsNotNone(transcribe_audio)

    def test_enum_values(self):
        """Test that the enums have expected values"""
        self.assertEqual(WhisperXModelSize.BASE.value, "base")
        self.assertEqual(WhisperXModelSize.TINY.value, "tiny")

if __name__ == "__main__":
    unittest.main()
