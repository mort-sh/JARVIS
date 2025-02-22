"""
Handles keyboard hooking, audio recording, transcription, and simulated typing.

New functionality (swapped keybinds):
- When just Right Shift is held (without Control), the transcription is emitted via a signal for normal command processing.
- When Control + Right Shift is held, the program records audio, transcribes it, and simulates typing the transcription (commands are not processed).
"""

import time
from typing import List

import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import logging
from PyQt5.QtCore import QObject, pyqtSignal
from rich.console import Console
from rich.panel import Panel
from rich import box
console = Console()


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

        console.print(Panel("[blue]Loading Whisper model (tiny.en). Please wait...[/blue]", box=box.ROUNDED))
        self.model = whisper.load_model("tiny.en")
        console.print(Panel("[green]Whisper model loaded.[/green]", box=box.ROUNDED))
        self._running = True

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status) -> None:
        if status:
            console.print(Panel(f"[yellow]Audio callback status: {status}", box=box.ROUNDED))
        self.audio_frames.append(indata.copy())

    def start_recording(self) -> None:
        if not self.recording:
            console.print(Panel("[blue]Recording started...[/blue]", box=box.ROUNDED))
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
                logging.error("Failed to start recording: %s", e)

    def stop_recording(self) -> None:
        if self.recording and self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.recording = False
            duration = time.time() - self.start_time
            console.print(Panel(f"[blue]Recording stopped after {duration:.2f} seconds.[/blue]", box=box.ROUNDED))

            if duration > 0.5 and self.audio_frames:
                audio_data = np.concatenate(self.audio_frames, axis=0)
                filename = "temp.wav"
                sf.write(filename, audio_data, self.fs, subtype="PCM_32")
                transcription = self.transcribe_and_send(filename)
                if transcription:
                    if self.ctrl_active:
                        # Control was held: simulate typing of transcription (bypassing command processing).
                        console.print(Panel("[cyan]Simulating typing of transcription (Control + Right Shift held)...[/cyan]", box=box.ROUNDED))
                        keyboard.write(transcription, delay=0.01)
                    else:
                        # No Control held: emit transcription for normal command processing.
                        console.print(Panel("[cyan]Emitting transcription for command processing (Right Shift only)...[/cyan]", box=box.ROUNDED))
                        self.transcriptionReady.emit(transcription)
            else:
                console.print(Panel("[yellow]Recording too short or no frames. No transcription performed.[/yellow]", box=box.ROUNDED))

    def transcribe_and_send(self, filename: str) -> str:
        console.print(Panel("[blue]Transcribing audio...[/blue]", box=box.ROUNDED))
        try:
            result = self.model.transcribe(filename, fp16=False)
            transcription = result.get("text", "").strip()
            console.print(Panel(f"[green]Transcription: {transcription}", box=box.ROUNDED))
            return transcription
        except Exception as e:
            console.print(Panel(f"[red bold]Transcription failed: {e}", box=box.ROUNDED))
            return ""

    def run_keyboard_hook(self) -> None:
        # Start recording when Right Shift is pressed and stop when released.
        keyboard.on_press_key("right shift", lambda _: self.start_recording())
        keyboard.on_release_key("right shift", lambda _: self.stop_recording())

        console.print(Panel("[blue]Press & hold Right Shift to record; release to transcribe.\n - Hold only Right Shift to simulate typing (commands bypassed).\n - Hold Control + Right Shift to emit transcription for command processing.[/blue]", box=box.ROUNDED))
        while self._running:
            time.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        keyboard.unhook_all()
