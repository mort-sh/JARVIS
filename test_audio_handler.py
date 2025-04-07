#!/usr/bin/env python3
"""
Test script for audio_handler integration.
"""

import os
import tempfile
from jarvis.audio import (
    TranscriptionMode,
    WhisperXModelSize,
    record_microphone,
    transcribe_audio,
    visualize_audio,
    clear_transcription_cache
)

def main():
    """
    Test recording and transcription.
    """
    print("Recording test:")
    print("Say something and press Enter to stop recording...")
    
    # Create a temporary file for the recording
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, "jarvis_test_recording.wav")
    
    # Record audio
    record_microphone(temp_file, visual=True)
    
    print(f"\nRecorded audio saved to {temp_file}")
    print("Transcribing...")
    
    # Transcribe audio using WhisperX
    try:
        transcription = transcribe_audio(
            temp_file,
            transcription_mode=TranscriptionMode.WHISPERX,
            whisperx_model_size=WhisperXModelSize.BASE,
            whisperx_language="en"
        )
        
        print("\nTranscription result:")
        print(f"'{transcription}'")
    except Exception as e:
        print(f"Transcription failed: {e}")
    
    # Clean up
    try:
        os.remove(temp_file)
        print(f"\nRemoved temporary file {temp_file}")
    except:
        pass
    
    # Clear model cache
    clear_transcription_cache()
    print("Model cache cleared")

if __name__ == "__main__":
    main()