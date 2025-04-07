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
    record_microphone,
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
        self.temp_file = None
        
        # Create recording state
        self.recording_state = RecordingState()
        
        # Create live display
        self.live = None
        
        # Configure audio handler
        self.transcription_mode = TranscriptionMode.WHISPERX
        self.model_size = WhisperXModelSize.BASE
        
        console.log("[blue]Audio handler initialized.[/blue]")
        self._running = True
        
    def update_recording_state(self) -> None:
        """Update the recording state with current values."""
        if self.recording:
            duration = time.time() - self.start_time
            # Volume level estimation is handled differently in audio_handler
            volume = 0.5  # Default placeholder value
        else:
            duration = 0.0
            volume = 0.0
            
        self.recording_state.update(
            is_recording=self.recording,
            duration=duration,
            volume_level=volume,
            transcribing=False
        )

    def start_recording(self, with_ctrl: bool = False) -> None:
        """Start recording audio."""
        if not self.recording:
            try:
                # Create a temporary file
                temp_dir = tempfile.gettempdir()
                self.temp_file = os.path.join(temp_dir, "jarvis_recording.wav")
                
                # Set recording state
                self.recording = True
                self.start_time = time.time()
                self.ctrl_pressed = with_ctrl  # Store ctrl state at start
                
                # Update recording state immediately
                self.update_recording_state()
                
                # Start recording in a non-blocking way
                # We don't actually start the recording here since audio_handler
                # uses a blocking approach. Instead, we'll just set the state.
                
            except Exception as e:
                console.log(f"[red]Failed to start recording: {e}[/red]")
                self.recording = False

    def stop_recording(self) -> None:
        """Stop recording and process the audio."""
        if self.recording:
            self.recording = False
            duration = time.time() - self.start_time
            
            # Update recording state one final time
            self.update_recording_state()

            if duration > 0.5:
                # Set transcribing state
                self.recording_state.update(
                    is_recording=False,
                    duration=duration,
                    volume_level=0.0,
                    transcribing=True
                )
                
                # Now actually record the audio (blocking)
                # This is not ideal, but required for now due to how audio_handler works
                try:
                    console.log(f"[blue]Recording audio to {self.temp_file}...[/blue]")
                    # Use non-visual mode since we can't do both keyboard hooking and visual recording
                    record_microphone(self.temp_file, visual=False)
                    transcription = self.transcribe_and_send(self.temp_file)
                    
                    if transcription:
                        if self.ctrl_pressed:
                            # Control was held at start: simulate typing of transcription
                            keyboard.write(transcription, delay=0.01)
                        else:
                            # No Control at start: emit transcription for command processing
                            self.transcriptionReady.emit(transcription)
                        
                        # Also emit on the controller if available
                        if hasattr(self, 'controller') and hasattr(self.controller, 'transcription_result'):
                            self.controller.transcription_result.emit(transcription)
                except Exception as e:
                    console.log(f"[red]Recording or transcription failed: {e}[/red]")
            
            # Reset recording state after transcription
            self.recording_state.update(
                is_recording=False,
                duration=0.0,
                volume_level=0.0,
                transcribing=False
            )

    def transcribe_and_send(self, filename: str) -> str:
        """Transcribe audio file using audio_handler."""
        try:
            # Update UI
            console.log("[blue]Transcribing audio...[/blue]")
            self.recording_state.update(
                is_recording=False,
                duration=0.0,
                volume_level=0.0,
                transcribing=True
            )
            
            # Perform transcription with WhisperX
            transcription = transcribe_audio(
                filename,
                transcription_mode=self.transcription_mode,
                whisperx_model_size=self.model_size,
                whisperx_language="en"  # Force English
            )
            
            # Clean up temp file
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                console.log(f"[yellow]Failed to delete temporary file: {e}[/yellow]")
            
            # Reset recording state after transcription
            self.recording_state.update(
                is_recording=False,
                duration=0.0,
                volume_level=0.0,
                transcribing=False
            )
            
            if transcription:
                console.log(f"[green]Transcription successful: '{transcription}'[/green]")
            
            return transcription
            
        except Exception as e:
            console.log(f"[red bold]Transcription failed: {e}[/red bold]")
            
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