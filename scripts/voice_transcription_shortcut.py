#!/usr/bin/env python3
"""
Quick keyboard-controlled voice transcription script using jarvis.audio.audio_handler
This script records voice while holding right-shift and automatically
types the transcription when the key is released.
"""
import os
import time
import tempfile
import threading
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
import keyboard
from typing import Optional, List
from jarvis.audio.audio_handler import (
    transcribe_audio,
    TranscriptionMode,
    WhisperXModelSize
)

# Configuration
TEMP_DIR = tempfile.gettempdir()
OUTPUT_FILENAME = os.path.join(TEMP_DIR, "voice_recording.wav")
HOTKEY = "right shift"  # The key to hold for recording
MODEL = TranscriptionMode.WHISPERX  # WHISPERX for local, OPENAI if you have API key
MODEL_SIZE = WhisperXModelSize.BASE  # Smaller = faster, larger = more accurate
SAMPLE_RATE = 16000
CHANNELS = 1

class AudioRecorder:
    """Records audio only while a key is pressed"""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.recording = False
        self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for the InputStream"""
        if status:
            print(f"Audio status: {status}")
        if self.recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        """Start the audio recording stream"""
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=self.audio_callback
            )
            self.recording = True
            self.stream.start()
            print("Recording started...")
        except Exception as e:
            print(f"Error starting recording: {e}")

    def stop_recording(self) -> bool:
        """Stop recording and save the audio file"""
        if not self.stream:
            return False

        self.recording = False

        try:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            self.stream = None

            # Save audio to file
            if self.audio_queue.empty():
                print("No audio recorded")
                return False

            with sf.SoundFile(OUTPUT_FILENAME, mode='w',
                              samplerate=SAMPLE_RATE,
                              channels=CHANNELS) as file:
                while not self.audio_queue.empty():
                    file.write(self.audio_queue.get())

            print("Recording saved")
            return True

        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False

def transcribe() -> Optional[str]:
    """Transcribe the recorded audio"""
    try:
        print("Transcribing...")

        # Transcribe the audio
        if MODEL == TranscriptionMode.OPENAI:
            # For OpenAI (requires API key in OPENAI_API_KEY environment variable)
            transcript = transcribe_audio(
                audio_filename=OUTPUT_FILENAME,
                transcription_mode=MODEL
            )
        else:
            # For WhisperX (local)
            transcript = transcribe_audio(
                audio_filename=OUTPUT_FILENAME,
                transcription_mode=MODEL,
                whisperx_model_size=MODEL_SIZE,
                whisperx_language="en"  # Optional language hint
            )

        return transcript

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def type_text(text: str):
    """Type out the text using keyboard module with a smooth, subtle animation"""
    # Add a small delay to ensure focus is in the right place
    time.sleep(0.5)

    # For a very subtle, natural typing effect
    # Calculate a variable typing speed that feels human but fast
    for char in text:
        keyboard.write(char)
        # Very brief random delay between keystrokes
        # Fast enough to seem superhuman but with subtle variation
        delay = np.random.normal(0.01, 0.005)  # Mean: 10ms, StdDev: 5ms
        delay = max(0.005, min(0.02, delay))  # Clamp between 5-20ms

        # Slightly longer pauses after punctuation for natural rhythm
        if char in ".!?,:;":
            delay *= 1.5

        time.sleep(delay)

def main():
    print("Voice Transcription Shortcut")
    print(f"Hold {HOTKEY} to record, release to transcribe and type")
    print("Type 'exit' and press Enter in this terminal to quit")

    recorder = AudioRecorder()
    running = True

    # Create a thread for handling the recording and typing
    def processing_thread():
        while running:
            try:
                # Wait for key press
                keyboard.wait(HOTKEY, trigger_on_release=False)
                print(f"{HOTKEY} pressed...")

                # Start recording
                recorder.start_recording()
                start_time = time.time()

                # Keep recording while key is held
                while keyboard.is_pressed(HOTKEY):
                    elapsed = time.time() - start_time
                    if int(elapsed) % 2 == 0:  # Show elapsed time every 2 seconds
                        print(f"Recording... {int(elapsed)}s", end="\r")
                    time.sleep(0.1)

                print("\nKey released, stopping recording...")
                # Stop recording when key is released
                if recorder.stop_recording():
                    transcript = transcribe()

                    if transcript:
                        print(f"Transcript: {transcript}")
                        # Type the transcript
                        type_text(transcript)
                        print("Transcription typed!")
                    else:
                        print("No transcription available")

                print(f"\nHold {HOTKEY} to record again, release to transcribe and type")
                print("Type 'exit' in this terminal to quit")

            except Exception as e:
                if running:  # Only print error if we're still supposed to be running
                    print(f"Error in processing thread: {e}")
                    time.sleep(1)  # Prevent rapid error loops

    # Start the processing thread
    thread = threading.Thread(target=processing_thread, daemon=True)
    thread.start()

    # Main thread handles terminal input for exit command
    try:
        while running:
            command = input()
            if command.lower().strip() == "exit":
                print("Exiting program...")
                running = False
                break

    finally:
        # Give thread a moment to clean up
        running = False
        time.sleep(0.5)

        # Clean up temp file
        if os.path.exists(OUTPUT_FILENAME):
            try:
                os.remove(OUTPUT_FILENAME)
                print("Temporary files cleaned up")
            except:
                pass

if __name__ == "__main__":
    main()
