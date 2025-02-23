"""
Handles keyboard hooking, audio recording, transcription, and simulated typing.

Hotkey functionality:
- Right Shift: Record audio and emit transcription for command processing
- Control + Right Shift: Record audio and simulate typing (bypass commands)
"""

from typing import Dict, List
import time
import warnings

import keyboard
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import logging
from PyQt5.QtCore import QObject, pyqtSignal
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich import box
from datetime import datetime
from ui.print_handler import advanced_console as console, RecordingState

# Configure logging to only show warnings and above
logging.basicConfig(level=logging.WARNING)

# Suppress specific loggers
logging.getLogger("whisper").setLevel(logging.WARNING)
logging.getLogger("sounddevice").setLevel(logging.WARNING)

# Suppress the torch.load FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")


class TranscriptionWorker(QObject):
    transcriptionReady = pyqtSignal(str)

    # Centralized hotkey configuration
    HOTKEY_DEFINITIONS: Dict[str, Dict[str, any]] = {
        'right shift': {
            'description': 'Record and emit transcription for command processing',
            'requires_ctrl': False
        },
        'ctrl+right shift': {
            'description': 'Record and simulate typing (bypass commands)',
            'requires_ctrl': True
        }
    }

    def __init__(self, parent: QObject = None) -> None:
        super().__init__(parent)
        self.recording: bool = False
        self.fs: int = 44100
        self.start_time: float = 0.0
        self.audio_frames: List[np.ndarray] = []
        self.stream = None
        self.ctrl_pressed: bool = False  # Track ctrl state from start of recording
        
        # Create recording state
        self.recording_state = RecordingState()
        
        # Create live display
        self.live = None
        
        console.log("[blue]Loading Whisper model (tiny.en). Please wait...[/blue]")
        self.model = whisper.load_model("tiny.en", device="cpu")  # Explicitly set device and use weights_only
        console.log("[green]Whisper model loaded.[/green]")
        self._running = True
        
    def calculate_volume_level(self) -> float:
        """Calculate the current volume level from recent audio frames."""
        if not self.audio_frames:
            return 0.0
        
        # Get the most recent frame
        recent_frame = self.audio_frames[-1]
        # Calculate RMS value and normalize
        rms = np.sqrt(np.mean(np.square(recent_frame)))
        # Convert to a 0-1 range with some scaling for better visualization
        normalized = min(rms * 5, 1.0)  # Scale up by 5x but cap at 1.0
        return normalized
        
    def update_recording_state(self) -> None:
        """Update the recording state with current values."""
        if self.recording:
            duration = time.time() - self.start_time
            volume = self.calculate_volume_level()
        else:
            duration = 0.0
            volume = 0.0
            
        self.recording_state.update(
            is_recording=self.recording,
            duration=duration,
            volume_level=volume
        )

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: dict, status) -> None:
        if status:
            console.log(f"[yellow]Audio callback status: {status}")
        self.audio_frames.append(indata.copy())

    def start_recording(self, with_ctrl: bool = False) -> None:
        if not self.recording:
            self.audio_frames = []
            try:
                self.stream = sd.InputStream(
                    samplerate=self.fs,
                    channels=1,
                    dtype="float32",
                    callback=self.audio_callback,
                )
                self.stream.start()
                self.start_time = time.time()
                self.recording = True
                self.ctrl_pressed = with_ctrl  # Store ctrl state at start
                
                # Update recording state immediately
                self.update_recording_state()
                
            except Exception as e:
                console.log(f"[red]Failed to start recording: {e}[/red]")

    def stop_recording(self) -> None:
        if self.recording and self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.recording = False
            duration = time.time() - self.start_time
            
            # Update recording state one final time
            self.update_recording_state()

            if duration > 0.5 and self.audio_frames:
                # Set transcribing state
                self.recording_state.update(
                    is_recording=False,
                    duration=duration,
                    volume_level=0.0,
                    transcribing=True
                )
                
                audio_data = np.concatenate(self.audio_frames, axis=0)
                filename = "temp.wav"
                sf.write(filename, audio_data, self.fs, subtype="PCM_32")
                transcription = self.transcribe_and_send(filename)
                
                if transcription:
                    if self.ctrl_pressed:
                        # Control was held at start: simulate typing of transcription
                        keyboard.write(transcription, delay=0.01)
                    else:
                        # No Control at start: emit transcription for command processing
                        self.transcriptionReady.emit(transcription)
            else:
                self.recording_state.update(
                    is_recording=False,
                    duration=0.0,
                    volume_level=0.0,
                    transcribing=False
                )

    def transcribe_and_send(self, filename: str) -> str:
        try:
            result = self.model.transcribe(filename, fp16=False)
            transcription = result.get("text", "").strip()
            
            # Reset recording state after transcription
            self.recording_state.update(
                is_recording=False,
                duration=0.0,
                volume_level=0.0,
                transcribing=False
            )
            
            return transcription
        except Exception as e:
            console.log(f"[red bold]Transcription failed: {e}")
            
            # Reset recording state on error
            self.recording_state.update(
                is_recording=False,
                duration=0.0,
                volume_level=0.0,
                transcribing=False
            )
            
            return ""

    def get_status_panel(self) -> Panel:
        """Get the current recording status panel."""
        # Create status content with Rich styling
        status_icon = "🔴" if self.recording else "⚪"
        status_text = "Recording" if self.recording else "Ready"
        if self.recording_state.transcribing:
            status_text = "Transcribing..."
            status_icon = "💭"
        
        duration = time.time() - self.start_time if self.recording else 0.0
        volume = self.calculate_volume_level() if self.recording else 0.0
        
        # Use Rich's Text for styling
        content = Text()
        content.append(f"  {status_icon} {status_text}\n\n")
        content.append(f"  Duration: {duration:.1f}s\n\n")
        content.append(f"  Volume: {volume*100:.0f}%\n")
        
        return Panel(
            content,
            title="Recording Status",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def print_registered_hotkeys(self) -> None:
        """Display registered hotkeys and initial recording status panel."""
        from rich.console import Group
        
        # Print hotkeys panel
        tree = Tree("Hotkeys")
        for hotkey, config in self.HOTKEY_DEFINITIONS.items():
            desc_start = 30
            padding = "･" * (desc_start - len(hotkey))
            hotkey_text = f"[cyan]{hotkey}[/cyan][dim]{padding}{config['description']}[/dim]"
            tree.add(hotkey_text)
        
        # Create initial display
        hotkeys_panel = Panel(tree, title="Registered Hotkeys", box=box.ROUNDED, expand=False)
        console.print(hotkeys_panel)
        print()  # Add a blank line between panels
        
        # Start live display and render initial status
        console.start_live()
        console.live_render(self.get_status_panel())

    def run_keyboard_hook(self) -> None:
        def on_right_shift_press(e):
            # Check if ctrl is pressed when right shift is pressed
            ctrl_pressed = keyboard.is_pressed('ctrl')
            self.start_recording(with_ctrl=ctrl_pressed)

        def on_right_shift_release(e):
            self.stop_recording()

        # Register both hotkey combinations
        keyboard.on_press_key("right shift", on_right_shift_press)
        keyboard.on_release_key("right shift", on_right_shift_release)

        # Display registered hotkeys and start live status display
        self.print_registered_hotkeys()
        
        # Main update loop
        while self._running:
            if self.recording or self.recording_state.transcribing:
                self.update_recording_state()
                console.live_render(self.get_status_panel())
            time.sleep(0.1)  # Update every 100ms

    def stop(self) -> None:
        self._running = False
        console.stop_live()
        keyboard.unhook_all()
