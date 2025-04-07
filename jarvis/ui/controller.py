"""
UIController that mediates between UI and backend services.
"""

from PyQt6.QtCore import QObject, pyqtSignal

from jarvis.ai.openai_wrapper import OpenAIWrapper, GLOBAL_DEFAULT_MODEL
from jarvis.commands.command_library import CommandLibrary
from jarvis.services.transcription_worker import TranscriptionWorker
from jarvis.ui.interfaces import IUISignals


class UIController(QObject):
    """
    Controller that mediates interactions between the UI and backend services.
    Implements the IUIController protocol and provides IUISignals.
    """
    # Signals from IUISignals
    update_assistant_message = pyqtSignal(str)
    stream_assistant_chunk = pyqtSignal(str)
    clear_assistant_message = pyqtSignal()
    recording_state_changed = pyqtSignal(bool, float)
    transcription_result = pyqtSignal(str)
    
    def __init__(self, parent=None) -> None:
        """
        Initialize the UIController.
        
        Args:
            parent: Optional parent QObject
        """
        super().__init__(parent)
        
        # Initialize backend services
        try:
            self.ai_wrapper = OpenAIWrapper()
            self.command_library = CommandLibrary()
            self.transcription_worker = TranscriptionWorker(controller=self)
            
            # Connect transcription worker signals
            self.transcription_worker.transcriptionReady.connect(self.process_user_input)
            
            # Current font size and model
            self.font_size = 15
            self.current_model = GLOBAL_DEFAULT_MODEL
            
        except Exception as e:
            # In a real app, we'd handle this better
            print(f"Error initializing UIController: {e}")
            raise
    
    def process_user_input(self, text: str) -> None:
        """
        Process user input and emit signals with response.
        
        Args:
            text: User input text
        """
        response = self.command_library.process_text(text, self)
        
        # In case of streaming responses, response will be a placeholder
        # and actual content will come via stream_assistant_chunk signal
        if response and "query" not in text.lower():
            self.update_assistant_message.emit(response)
    
    def start_recording(self, ctrl_pressed: bool = False) -> None:
        """
        Start audio recording.
        
        Args:
            ctrl_pressed: Whether Ctrl key is pressed
        """
        self.transcription_worker.start_recording(with_ctrl=ctrl_pressed)
        
    def stop_recording(self) -> None:
        """Stop audio recording and process the recording."""
        self.transcription_worker.stop_recording()
    
    def increase_font(self) -> None:
        """Increase the font size and notify the UI."""
        self.font_size += 1
        # We'd normally emit a signal here, but for simplicity,
        # we'll let the UI handle font size changes
    
    def decrease_font(self) -> None:
        """Decrease the font size and notify the UI."""
        if self.font_size > 1:
            self.font_size -= 1
    
    def clear_dialog(self) -> None:
        """Clear the conversation history."""
        self.clear_assistant_message.emit()
    
    def close_dialog(self) -> None:
        """Close/hide the dialog."""
        # This will typically be handled by the UI directly, but included for completeness
        pass
    
    def shutdown(self) -> None:
        """Shutdown all worker threads and services."""
        self.transcription_worker.stop()
        self.command_library.shutdown_threads()
    
    def run_keyboard_hook(self) -> None:
        """Start the keyboard hook for recording shortcuts."""
        self.transcription_worker.run_keyboard_hook()
    
    def get_model_list(self) -> list[str]:
        """
        Get list of available AI models.
        
        Returns:
            List of model IDs
        """
        try:
            models = self.ai_wrapper.list_models()
            model_ids = []
            if models:
                for m in models:
                    if isinstance(m, dict) and "id" in m:
                        model_ids.append(m["id"])
                    elif hasattr(m, "id"):
                        model_ids.append(m.id)
            if not model_ids:
                model_ids = [GLOBAL_DEFAULT_MODEL]
                
            # Ensure default model is included
            if GLOBAL_DEFAULT_MODEL not in model_ids:
                model_ids.insert(0, GLOBAL_DEFAULT_MODEL)
                
            return model_ids
        except Exception:
            return [GLOBAL_DEFAULT_MODEL]
    
    def set_current_model(self, model_id: str) -> None:
        """
        Set the current AI model.
        
        Args:
            model_id: The ID of the model to use
        """
        self.current_model = model_id