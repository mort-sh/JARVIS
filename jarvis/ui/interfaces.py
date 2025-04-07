"""
Defines Protocol classes (interfaces) for UI components.
"""

from typing import Protocol
from PyQt6.QtCore import pyqtSignal, QObject


class IUIController(Protocol):
    """
    Protocol defining the methods the UI calls to interact with backend services.
    """
    def process_user_input(self, text: str) -> None:
        """
        Process user input text and generate a response.
        
        Args:
            text: The text input from the user
        """
        ...
    
    def start_recording(self, ctrl_pressed: bool = False) -> None:
        """
        Start recording audio.
        
        Args:
            ctrl_pressed: Whether Ctrl key is pressed during recording
        """
        ...
    
    def stop_recording(self) -> None:
        """Stop recording audio and process the recording."""
        ...
    
    def increase_font(self) -> None:
        """Increase the UI font size."""
        ...
    
    def decrease_font(self) -> None:
        """Decrease the UI font size."""
        ...
    
    def clear_dialog(self) -> None:
        """Clear the conversation history."""
        ...
    
    def close_dialog(self) -> None:
        """Hide the dialog."""
        ...
    
    def shutdown(self) -> None:
        """Shutdown all threads and services."""
        ...
    
    def get_model_list(self) -> list[str]:
        """
        Get list of available AI models.
        
        Returns:
            List of model IDs
        """
        ...
    
    def set_current_model(self, model_id: str) -> None:
        """
        Set the current AI model.
        
        Args:
            model_id: The ID of the model to use
        """
        ...


class IUISignals(Protocol):
    """
    Protocol defining signals that the backend emits to update the UI.
    """
    update_assistant_message: pyqtSignal
    stream_assistant_chunk: pyqtSignal
    clear_assistant_message: pyqtSignal
    recording_state_changed: pyqtSignal
    transcription_result: pyqtSignal