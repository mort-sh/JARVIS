"""
Handles keyboard hooking, audio recording, transcription, and simulated typing.

Hotkey functionality:
- Right Shift: Record audio and emit transcription for command processing
- Control + Right Shift: Record audio and simulate typing (bypass commands)
"""

from typing import Dict, List, Optional
import time
import warnings
import os
import logging
import tempfile
from pathlib import Path

import keyboard
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from rich.panel import Panel
from rich.tree import Tree
from rich.text import Text
from rich import box
from datetime import datetime

from jarvis.ui.print_handler import advanced_console as console, RecordingState

# Import audio_handler
from jarvis.audio import (
    TranscriptionMode,
    WhisperXModelSize,
    AudioRecorder,       # Import the class
    RecordingConfig,     # Import config class
    transcribe_audio,
    visualize_audio,
    clear_transcription_cache
)

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Suppress specific loggers
logging.getLogger("sounddevice").setLevel(logging.ERROR)


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

    def __init__(self, parent: QObject = None, controller=None) -> None:
        super().__init__(parent)
        self.recording: bool = False
        self.start_time: float = 0.0
        self.ctrl_pressed: bool = False  # Track ctrl state from start of recording
        self.controller = controller
        self.temp_file_path: Optional[Path] = None # Store path object

        # Create recording state
        self.recording_state = RecordingState()

        # Create live display
        self.live = None

        # Configure audio handler
        self.transcription_mode = TranscriptionMode.WHISPERX
        self.model_size = WhisperXModelSize.BASE

        # Initialize AudioRecorder instance (config will be updated before recording)
        self.audio_recorder: Optional[AudioRecorder] = None

        console.log("[blue]Audio handler initialized.[/blue]")
        self._running = True

    def _get_temp_filepath(self) -> Path:
        """Generates a unique temporary file path for recording."""
        # Ensure a unique filename each time to avoid potential conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Path(tempfile.gettempdir()) / f"jarvis_recording_{timestamp}.wav"

    def update_recording_state(self) -> None:
        """Update the recording state with current values."""
        if self.recording:
            duration = time.time() - self.start_time
            # Volume level estimation needs access to recorder's RMS values
            # For now, keep placeholder or fetch from recorder if implemented
            volume = 0.5
            # Example: Fetching RMS if recorder exposes it
            # if self.audio_recorder:
            #     with self.audio_recorder._rms_values['lock']:
            #         current = self.audio_recorder._rms_values['current']
            #         min_rms = self.audio_recorder._rms_values['min']
            #         max_rms = self.audio_recorder._rms_values['max']
            #     bar_str = visualize_audio(current, min_rms, max_rms)
            #     # Convert bar to a simple level for now
            #     volume = bar_str.count('█') / len(bar_str) if len(bar_str) > 0 else 0.0
        else:
            duration = 0.0
            volume = 0.0

        self.recording_state.update(
            is_recording=self.recording,
            duration=duration,
            volume_level=volume,
            transcribing=self.recording_state.transcribing # Preserve transcribing state
        )

    def start_recording(self, with_ctrl: bool = False) -> None:
        """Start recording audio using the non-blocking stream."""
        if not self.recording:
            try:
                self.temp_file_path = self._get_temp_filepath()
                log_msg = f"[cyan]Starting recording to {self.temp_file_path}...[/cyan]"
                console.log(log_msg)

                # Ensure AudioRecorder is initialized with the correct config
                # We create a new instance each time to ensure clean state and correct filename
                recording_config = RecordingConfig(
                    output_filename=str(self.temp_file_path), # Pass as string
                    # Add other config options like sample_rate, device_name if needed
                    visual=False # Visual mode uses blocking prompt
                )
                self.audio_recorder = AudioRecorder(recording_config)

                # Set recording state *before* starting stream
                self.recording = True
                self.start_time = time.time()
                self.ctrl_pressed = with_ctrl
                self.recording_state.update(is_recording=True, transcribing=False)
                self.update_recording_state()

                # Start the non-blocking stream
                self.audio_recorder.start_stream()

            except Exception as e:
                console.log(f"[red]Failed to start recording: {e}[/red]")
                self.recording = False
                self.audio_recorder = None # Clear recorder on error
                self.recording_state.update(is_recording=False, transcribing=False)

    def stop_recording(self) -> None:
        """Stop the non-blocking stream, save audio, and process it."""
        if self.recording:
            self.recording = False # Set state immediately
            duration = time.time() - self.start_time
            console.log(f"[cyan]Stopping recording. Duration: {duration:.2f}s[/cyan]")

            # Update recording state (show not recording, but maybe transcribing soon)
            self.recording_state.update(is_recording=False)
            self.update_recording_state()

            if not self.audio_recorder:
                 console.log("[yellow]Audio recorder not initialized. Cannot stop.[/yellow]")
                 return

            # Stop the stream and save the file
            saved_filename = self.audio_recorder.stop_stream_and_save()
            self.audio_recorder = None # Clear recorder instance after stopping

            if saved_filename and duration > 0.5:
                # Set transcribing state before starting transcription
                self.recording_state.update(transcribing=True)
                self.update_recording_state()
                console.live_render(self.get_status_panel()) # Update UI immediately

                # Transcribe the saved file
                transcription = self.transcribe_and_send(saved_filename)

                if transcription:
                    if self.ctrl_pressed:
                        # Control was held at start: simulate typing
                        console.log(f"[magenta]Simulating typing: '{transcription}'[/magenta]")
                        keyboard.write(transcription, delay=0.01)
                    else:
                        # No Control at start: emit for command processing
                        console.log(f"[green]Emitting transcription for processing: '{transcription}'[/green]")
                        self.transcriptionReady.emit(transcription)

                    # Also emit on the controller if available (for potential logging/history)
                    if hasattr(self, 'controller') and hasattr(self.controller, 'transcription_result'):
                        self.controller.transcription_result.emit(transcription)
                else:
                     console.log("[yellow]Transcription resulted in empty text.[/yellow]")

            elif not saved_filename:
                 console.log("[yellow]Recording failed to save. No transcription.[/yellow]")
            else: # Duration too short
                 console.log("[yellow]Recording too short, discarding.[/yellow]")
                 # Clean up the short audio file if it exists
                 if self.temp_file_path and self.temp_file_path.exists():
                     try:
                         os.remove(self.temp_file_path)
                         console.log(f"[grey]Cleaned up short recording: {self.temp_file_path}[/grey]")
                     except Exception as e:
                         console.log(f"[yellow]Could not remove short recording file: {e}[/yellow]")

            # Reset recording and transcribing state finally
            self.recording_state.update(is_recording=False, transcribing=False)
            self.temp_file_path = None # Clear temp file path
            self.update_recording_state() # Update UI one last time for this cycle

    def transcribe_and_send(self, filename: str) -> str:
        """Transcribe audio file using audio_handler."""
        try:
            # UI state is already set to transcribing in stop_recording
            console.log(f"[blue]Transcribing audio from {filename}...[/blue]")

            # Perform transcription
            transcription = transcribe_audio(
                filename,
                transcription_mode=self.transcription_mode,
                whisperx_model_size=self.model_size,
                whisperx_language="en"  # Force English
            )

            # Clean up temp file after successful transcription
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    console.log(f"[grey]Cleaned up temporary file: {filename}[/grey]")
            except Exception as e:
                console.log(f"[yellow]Failed to delete temporary file {filename}: {e}[/yellow]")

            # Reset transcribing state (is_recording is already False)
            self.recording_state.update(transcribing=False)
            self.update_recording_state()

            if transcription:
                console.log(f"[green]Transcription successful.[/green]") # Don't log the full text here
            else:
                 console.log("[yellow]Transcription returned empty.[/yellow]")

            return transcription or "" # Return empty string if None

        except Exception as e:
            console.log(f"[red bold]Transcription failed: {e}[/red bold]")

            # Clean up temp file even on error
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    console.log(f"[grey]Cleaned up temporary file after error: {filename}[/grey]")
            except Exception as cleanup_e:
                console.log(f"[yellow]Failed to delete temporary file {filename} after error: {cleanup_e}[/yellow]")

            # Reset transcribing state on error
            self.recording_state.update(transcribing=False)
            self.update_recording_state()

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

        # Use Rich's Text for styling
        content = Text()
        content.append(f"  {status_icon} {status_text}\n\n")
        content.append(f"  Duration: {duration:.1f}s\n\n")

        # Volume visualization is basic since audio_handler visualize_audio isn't directly usable here
        bar = "█" * 10 if self.recording else "░" * 10
        content.append(f"  Volume: {bar}\n")

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
        """Stop the worker and clean up resources."""
        self._running = False
        console.stop_live()
        keyboard.unhook_all()
        clear_transcription_cache()  # Clean up audio_handler models
