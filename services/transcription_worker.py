"""
Handles keyboard hooking, audio recording, transcription, and simulated typing.

New functionality (swapped keybinds):
- When just Right Shift is held (without Control), the program records audio,
  transcribes it, and simulates typing the transcription (commands are not processed).
- When Control + Right Shift is held, the transcription is emitted via a signal for
  normal command processing.
"""

import time
from typing import List

import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from PyQt5.QtCore import QObject, pyqtSignal


class TranscriptionWorker(QObject):
    transcriptionReady = pyqtSignal(str)

    def __init__(self, parent: QObject = None) -> None:
        super().__init__(parent)
        self.recording: bool = False
        self.fs: int = 44100
        self.start_time: float = 0.0
        self.audio_frames: List[np.ndarray] = []
        self.stream = None
        # Flag to record if Control was held at the time recording started.
        self.ctrl_active: bool = False

        print("Loading Whisper model (tiny.en). Please wait...")
        self.model = whisper.load_model("tiny.en")
        print("Whisper model loaded.")
        self._running = True

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status) -> None:
        if status:
            print(f"Audio callback status: {status}")
        self.audio_frames.append(indata.copy())

    def start_recording(self) -> None:
        if not self.recording:
            print("Recording started...")
            self.audio_frames = []
            try:
                # Capture whether Control is held when recording starts.
                self.ctrl_active = keyboard.is_pressed("ctrl")
                self.stream = sd.InputStream(
                    samplerate=self.fs,
                    channels=1,
                    dtype="float32",
                    callback=self.audio_callback,
                )
                self.stream.start()
                self.start_time = time.time()
                self.recording = True
            except Exception as e:
                print(f"Failed to start recording: {e}")

    def stop_recording(self) -> None:
        if self.recording and self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.recording = False
            duration = time.time() - self.start_time
            print(f"Recording stopped after {duration:.2f} seconds.")

            if duration > 0.5 and self.audio_frames:
                audio_data = np.concatenate(self.audio_frames, axis=0)
                filename = "temp.wav"
                sf.write(filename, audio_data, self.fs, subtype="PCM_32")
                transcription = self.transcribe_and_send(filename)
                if transcription:
                    if self.ctrl_active:
                        # Control was held: emit transcription for normal command processing.
                        print("Emitting transcription for command processing (Control held)...")
                        self.transcriptionReady.emit(transcription)
                    else:
                        # No Control held: simulate typing the transcription (bypassing command processing).
                        print("Simulating typing of transcription (Right Shift only)...")
                        keyboard.write(transcription, delay=0.01)
            else:
                print("Recording too short or no frames. No transcription performed.")

    def transcribe_and_send(self, filename: str) -> str:
        print("Transcribing audio...")
        try:
            result = self.model.transcribe(filename, fp16=False)
            transcription = result.get("text", "").strip()
            print("Transcription:\n", transcription, "\n")
            return transcription
        except Exception as e:
            print(f"Transcription failed: {e}")
            return ""

    def run_keyboard_hook(self) -> None:
        # Start recording when Right Shift is pressed and stop when released.
        keyboard.on_press_key("right shift", lambda _: self.start_recording())
        keyboard.on_release_key("right shift", lambda _: self.stop_recording())

        print(
            "Press & hold Right Shift to record; release to transcribe.\n"
            " - Hold only Right Shift to simulate typing (commands bypassed).\n"
            " - Hold Control + Right Shift to emit transcription for command processing."
        )
        while self._running:
            time.sleep(0.1)
