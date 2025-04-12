"""
Handles keyboard hooking, audio recording, transcription, and simulated typing.

Hotkey functionality:
- Right Shift: Record audio and emit transcription for command processing
- Control + Right Shift: Record audio and simulate typing (bypass commands)
"""

from datetime import datetime
import logging
import os
from pathlib import Path
import tempfile
import threading
import time

import keyboard
from PyQt6.QtCore import QObject, pyqtSignal
from rich import box
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from jarvis.audio import (
    AudioRecorder,
    RecordingConfig,
    TranscriptionMode,
    WhisperXModelSize,
    transcribe_audio,
)
from jarvis.state import RecordingState
from jarvis.ui.print_handler import advanced_console as console

# Configure logging first
log = logging.getLogger(__name__)


class TranscriptionWorker(QObject):
    transcriptionReady = pyqtSignal(str)

    # Centralized hotkey configuration
    HOTKEY_DEFINITIONS: dict[str, dict[str, any]] = {
        "right shift": {
            "description": "Record and emit transcription for command processing",
            "requires_ctrl": False,
        },
        "ctrl+right shift": {
            "description": "Record and simulate typing (bypass commands)",
            "requires_ctrl": True,
        },
    }

    def __init__(self, parent: QObject = None, controller=None) -> None:
        super().__init__(parent)
        self.recording: bool = False
        self.start_time: float = 0.0
        self.ctrl_pressed: bool = False  # Track ctrl state from start of recording
        self.controller = controller
        self.temp_file_path: Path | None = None
        self.recording_state = RecordingState()
        self.audio_recorder: AudioRecorder | None = None
        self._running = True
        self.record_thread: threading.Thread | None = None

        # Configure transcription settings
        self.transcription_mode = TranscriptionMode.WHISPERX
        self.model_size = WhisperXModelSize.BASE
        log.info(
            f"Configured transcription: mode={self.transcription_mode.name}, model_size={self.model_size.value}"
        )

        console.log("[blue]TranscriptionWorker initialized.[/blue]")

    def _get_temp_filepath(self) -> Path:
        """Generates a unique temporary file path for recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return Path(tempfile.gettempdir()) / f"jarvis_recording_{timestamp}.wav"

    def update_recording_state(self) -> None:
        """Update the recording state with current values."""
        duration = time.time() - self.start_time if self.recording else 0.0
        volume = 0.0  # Placeholder - volume calculation needs implementation

        # Ensure the update call matches the RecordingState definition
        self.recording_state.update(
            is_recording=self.recording,
            duration=duration,
            volume_level=volume,
            transcribing=self.recording_state.transcribing,
        )
        # Update the UI if it's running using the correct method
        if console.live and console.live.is_started:
            console.update_live(self.get_status_panel())

    def start_recording(self, with_ctrl: bool = False) -> None:
        """Start recording audio in a non-blocking way."""
        if self.recording:
            log.info("Already recording. Ignoring start request.")  # Changed level
            return

        try:
            self.temp_file_path = self._get_temp_filepath()
            log.info(f"Starting recording to {self.temp_file_path}")

            recording_config = RecordingConfig(
                output_filename=str(self.temp_file_path), visual=False
            )
            self.audio_recorder = AudioRecorder(recording_config)

            # Set state *before* starting thread
            self.recording = True
            self.start_time = time.time()
            self.ctrl_pressed = with_ctrl
            self.recording_state.update(is_recording=True, transcribing=False)
            self.update_recording_state()  # Update UI

            # Run the blocking record method in a separate thread
            self.record_thread = threading.Thread(target=self._record_thread_target)
            self.record_thread.daemon = True
            self.record_thread.start()
            log.info("Recording thread started.")

        except Exception as e:
            log.exception(f"Failed to start recording: {e}")
            self.recording = False
            self.audio_recorder = None
            self.recording_state.update(is_recording=False, transcribing=False)
            self.update_recording_state()  # Update UI

    def _record_thread_target(self):
        """Target function for the recording thread."""
        if self.audio_recorder:
            try:
                # This is a blocking call, will run until stopped externally
                # or by internal logic (like pressing Enter in visual mode)
                # Since visual=False, it relies on stop_recording being called.
                self.audio_recorder.record()
                log.debug("AudioRecorder.record() method finished.")
            except Exception as e:
                log.exception(f"Error within recording thread: {e}")
            finally:
                log.debug("Recording thread target finished.")
                # State is reset in stop_recording, no need to reset here unless error
        else:
            log.error("Recording thread started but audio_recorder was None.")

    def stop_recording(self) -> None:
        """Stop the recording stream, save audio, and process it."""
        if not self.recording:
            log.info("Not recording. Ignoring stop request.")  # Changed level
            return

        log.info("Stopping recording...")
        self.recording = False  # Set state immediately
        duration = time.time() - self.start_time
        log.info(f"Recording duration: {duration:.2f}s")

        # Update state to show not recording, but potentially transcribing
        self.recording_state.update(is_recording=False, transcribing=True)
        self.update_recording_state()  # Update UI

        saved_filename = None
        current_temp_file = self.temp_file_path  # Keep track for fallback

        if self.audio_recorder:
            try:
                # Use the combined stop and save method from AudioRecorder
                log.debug("Calling audio_recorder.stop_stream_and_save()")
                saved_filename = self.audio_recorder.stop_stream_and_save()
                log.debug(f"stop_stream_and_save returned: {saved_filename}")
            except Exception as e:
                log.exception(f"Error stopping/saving audio recorder: {e}")
            finally:
                self.audio_recorder = None  # Clear recorder instance
        else:
            log.info("stop_recording called but audio_recorder was already None.")  # Changed level

        # Fallback if saving failed but temp file might exist
        if not saved_filename and current_temp_file and current_temp_file.exists():
            log.info(
                f"Saving failed, attempting to use existing temp file: {current_temp_file}"
            )  # Changed level
            saved_filename = str(current_temp_file)

        if saved_filename and os.path.exists(saved_filename):
            file_size = os.path.getsize(saved_filename)
            log.info(f"Audio saved to: {saved_filename} (Size: {file_size} bytes)")
            if file_size == 0:
                log.info("Saved audio file is empty (0 bytes).")  # Changed level

            # Add a small delay before transcription to allow file handle release
            time.sleep(0.1)
            log.debug("Added 0.1s delay before transcription.")

            # Proceed to transcription
            try:
                transcription = self.transcribe_and_process(saved_filename)
                if transcription:
                    self.handle_transcription_result(transcription)
                else:
                    log.info("Transcription resulted in empty text.")  # Changed level
                    console.log("[yellow]Transcription resulted in empty text.[/yellow]")
            except Exception as trans_error:
                log.exception(f"Error during transcription process: {trans_error}")
                console.log(f"[red]Transcription error: {trans_error}[/red]")
        else:
            log.error("Recording failed to save or file doesn't exist. No transcription.")
            console.log("[yellow]Recording failed to save. No transcription.[/yellow]")

        # Final state update is handled within transcribe_and_process finally block
        self.temp_file_path = None
        log.info("stop_recording process complete.")

    def transcribe_and_process(self, filename: str) -> str | None:
        """Transcribe audio file and handle cleanup."""
        transcription = None
        try:
            log.info(f"Starting transcription for {filename}")
            # Ensure the state reflects transcribing (already set in stop_recording)
            # self.recording_state.update(is_recording=False, transcribing=True) # Redundant
            self.update_recording_state()  # Update UI

            transcription = transcribe_audio(
                filename,
                transcription_mode=self.transcription_mode,
                whisperx_model_size=self.model_size,
                whisperx_language="en",  # Force English for consistency
            )

            if transcription:
                log.info(f"Transcription successful: '{transcription}'")
            else:
                log.info("Transcription returned empty or None.")  # Changed level

            return transcription

        except Exception as e:
            log.exception(f"Transcription failed for {filename}: {e}")
            return None
        finally:
            # Clean up the temporary file regardless of success/failure
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    log.info(f"Cleaned up temporary file: {filename}")
            except Exception as e:
                log.error(f"Failed to delete temporary file {filename}: {e}")

            # Reset transcribing state after processing
            self.recording_state.update(is_recording=False, transcribing=False)
            self.update_recording_state()  # Update UI one last time

    def handle_transcription_result(self, transcription: str):
        """Process the transcription based on whether Ctrl was pressed."""
        if self.ctrl_pressed:
            log.info(f"Simulating typing transcription: '{transcription}'")
            console.log(f"[magenta]Simulating typing: '{transcription}'[/magenta]")
            try:
                keyboard.write(transcription, delay=0.01)
            except Exception as e:
                log.exception(f"Error simulating typing: {e}")
        else:
            log.info(f"Emitting transcription for processing: '{transcription}'")
            console.log(f"[green]Emitting transcription for processing: '{transcription}'[/green]")
            self.transcriptionReady.emit(transcription)

        # The logic above correctly handles emitting to the UI (via transcriptionReady)
        # or typing directly based on ctrl_pressed. No further emission needed here.

    def get_status_panel(self) -> Panel:
        """Get the current recording status panel."""
        status_icon = "⚪"
        status_text = "Ready"
        if self.recording_state.transcribing:
            status_icon = "💭"
            status_text = "Transcribing..."
        elif self.recording:
            status_icon = "🔴"
            status_text = "Recording"

        duration = self.recording_state.duration

        content = Text()
        content.append(f"  {status_icon} {status_text}\n\n")
        content.append(f"  Duration: {duration:.1f}s\n\n")
        # Basic volume bar placeholder
        bar = "█" * int(self.recording_state.volume_level * 10) + "░" * (
            10 - int(self.recording_state.volume_level * 10)
        )
        content.append(f"  Volume: {bar}\n")

        return Panel(content, title="Recording Status", box=box.ROUNDED, padding=(1, 2))

    def print_registered_hotkeys(self) -> None:
        """Display registered hotkeys and initial recording status panel."""
        tree = Tree("Hotkeys")
        for hotkey, config in self.HOTKEY_DEFINITIONS.items():
            desc_start = 30
            padding = "･" * (desc_start - len(hotkey))
            hotkey_text = f"[cyan]{hotkey}[/cyan][dim]{padding}{config['description']}[/dim]"
            tree.add(hotkey_text)

        hotkeys_panel = Panel(tree, title="Registered Hotkeys", box=box.ROUNDED, expand=False)
        console.print(hotkeys_panel)
        print()

        # Start live display if not already started
        if not console.live or not console.live.is_started:
            console.start_live()
        console.update_live(self.get_status_panel())

    def run_keyboard_hook(self) -> None:
        """Set up and run the keyboard listeners."""
        log.info("Setting up keyboard hooks...")

        # --- Event Handlers ---
        def on_right_shift_press(e):
            # Check if ctrl is pressed *at the moment* right shift is pressed
            is_ctrl_pressed = keyboard.is_pressed("ctrl")
            log.debug(f"Right shift pressed. Event: {e}, Ctrl pressed: {is_ctrl_pressed}")
            if not self.recording:
                self.start_recording(with_ctrl=is_ctrl_pressed)
            else:
                # This handles rapid presses - ignore if already recording
                log.info("Right shift pressed, but already recording. Ignoring.")  # Changed level

        def on_right_shift_release(e):
            log.debug(f"Right shift released event detected. Event: {e}")
            if self.recording:
                self.stop_recording()
            else:
                # This might happen if the press event was missed or recording stopped early
                log.info("Right shift released, but was not recording. Ignoring.")  # Changed level

        # --- End Event Handlers ---

        try:
            # Register hooks using keyboard library
            keyboard.on_press_key("right shift", on_right_shift_press, suppress=False)
            keyboard.on_release_key("right shift", on_right_shift_release, suppress=False)
            log.info("Keyboard hooks registered for Right Shift.")

            self.print_registered_hotkeys()

            # Keep the main thread alive while hooks are active
            log.info("Keyboard hook listener running. Press Ctrl+C in terminal to exit.")
            while self._running:
                # Update UI periodically only if live display is active
                if console.live and console.live.is_started:
                    self.update_recording_state()
                time.sleep(0.1)  # Check running flag periodically

        except ImportError:
            log.error("The 'keyboard' library is not installed. Hotkeys disabled.")
            log.error("Please install it using: uv pip install keyboard")
        except Exception as e:
            log.exception(f"An error occurred setting up or running keyboard hooks: {e}")
        finally:
            log.info("Unhooking all keyboard listeners.")
            try:
                keyboard.unhook_all()
            except Exception as e:
                log.error(f"Error unhooking keyboard listeners: {e}")
            # Stop live display if it was started
            if console.live and console.live.is_started:
                console.stop_live()

    def stop(self) -> None:
        """Stop the worker and clean up resources."""
        log.info("Stopping TranscriptionWorker...")
        self._running = False  # Signal the loop to stop

        # Ensure recording is stopped if it was active
        if self.recording:
            log.info(
                "Worker stopped while recording was active. Attempting cleanup."
            )  # Changed level
            try:
                # Directly call stop_recording which handles state and cleanup
                self.stop_recording()
            except Exception as e:
                log.error(f"Error stopping recording during shutdown: {e}")

        # Wait briefly for the hook loop to potentially exit
        time.sleep(0.2)

        # Unhook keyboard listeners (redundant if finally block runs, but safe)
        try:
            keyboard.unhook_all()
        except Exception as e:
            log.error(f"Error unhooking keyboard listeners during stop: {e}")

        # Stop the live display if it was started
        if console.live and console.live.is_started:
            try:
                console.stop_live()
            except Exception as e:
                log.error(f"Error stopping console live display: {e}")

        log.info("TranscriptionWorker stopped.")
