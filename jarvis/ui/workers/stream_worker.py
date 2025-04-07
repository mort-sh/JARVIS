"""
Worker for streaming AI responses.
"""

from PyQt6.QtCore import QObject, pyqtSignal
import logging
from typing import List, Optional


class StreamWorker(QObject):
    # Signal to deliver partial text updates.
    partialResult = pyqtSignal(str)

    def __init__(
        self, 
        ai, 
        prompt: str, 
        messages: List[dict], 
        temperature: float, 
        max_tokens: int, 
        model: str, 
        controller: Optional[QObject] = None, 
        parent=None
    ) -> None:
        """
        Initialize the stream worker.
        
        Args:
            ai: The AI wrapper instance
            prompt: The user prompt
            messages: The conversation history
            temperature: The sampling temperature
            max_tokens: Maximum tokens to generate
            model: The model ID to use
            controller: Optional UIController to emit signals to
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.ai = ai
        self.prompt = prompt
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.controller = controller

    def run(self) -> None:
        """Run the streaming operation."""
        partial_tokens: List[str] = []

        def update_text(new_chunk: str) -> None:
            """
            Update the UI with new streaming content.
            
            Args:
                new_chunk: The new text chunk to add
            """
            partial_tokens.append(new_chunk)
            combined = "".join(partial_tokens)
            
            # Emit to both the traditional signal and the controller if available
            self.partialResult.emit(combined)
            
            if self.controller and hasattr(self.controller, 'stream_assistant_chunk'):
                self.controller.stream_assistant_chunk.emit(combined)

        result = self.ai.send_prompt(
            prompt=self.prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=self.messages,
            stream_callback=update_text,
        )
        
        # Optionally, signal completion or final result
        assistant_reply = result.get("assistant_reply", "")
        if self.controller and hasattr(self.controller, 'update_assistant_message') and assistant_reply:
            self.controller.update_assistant_message.emit(assistant_reply)