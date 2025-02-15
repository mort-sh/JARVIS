"""
Worker for streaming AI responses.
"""

from PyQt5.QtCore import QObject, pyqtSignal
from typing import List


class StreamWorker(QObject):
    # Signal to deliver partial text updates.
    partialResult = pyqtSignal(str)

    def __init__(self, ai, prompt: str, messages: List[dict], temperature: float, max_tokens: int, parent=None) -> None:
        super().__init__(parent)
        self.ai = ai
        self.prompt = prompt
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self) -> None:
        partial_tokens: List[str] = []

        def update_text(new_chunk: str) -> None:
            partial_tokens.append(new_chunk)
            combined = "".join(partial_tokens)
            self.partialResult.emit(combined)

        result = self.ai.send_prompt(
            prompt=self.prompt,
            model="gpt-4o",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=self.messages,
            stream_callback=update_text,
        )
        # Optionally, you can emit a final result here.
        assistant_reply = result.get("assistant_reply", "")
        # self.partialResult.emit(assistant_reply)
