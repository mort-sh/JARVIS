"""
CommandLibrary provides a mapping between natural language command phrases
and functions that perform those commands.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyperclip
from PyQt5.QtCore import QThread

from ai.openai_wrapper import OpenAIWrapper
from ui.workers.stream_worker import StreamWorker


class CommandLibrary:
    COMMAND_PHRASES: Dict[str, List[str]] = {
        "format": ["format as code", "wrap in code", "format"],
        "code": [
            "optimize",
            "update",
            "refactor",
            "fix",
            "implement",
            "write",
            "design",
            "python",
            "code",
        ],
        "exit": ["exit", "quit", "close"],
        "talk": ["talk", "speech", "say"],
        "query": [
            "tell",
            "show",
            "what",
            "who",
            "how",
            "should",
            "question",
            "explain",
        ],
    }

    PROMPTS: Dict[str, str] = {
        "general_question": (
            "Answer the following question clearly and concisely, providing accurate information "
            "without unnecessary elaboration. Assume plaintext (no markdown)."
        ),
        "code_completion": (
            "You are an expert software engineer. Write concise, efficient, and industry-standard code "
            "with a focus on performance, readability, and maintainability. Ensure best practices, "
            "modularity, and robust handling of edge cases in every solution."
        ),
    }

    def __init__(self) -> None:
        self.commands: Dict[str, Callable[[str, Any], str]] = {}
        self.active_threads: List[Tuple[QThread, Any]] = []
        try:
            self.ai = OpenAIWrapper()
        except ValueError as e:
            raise ValueError(f"Failed to initialize OpenAI wrapper: {e}")

        self._initialize_commands()

    def _initialize_commands(self) -> None:
        for command, phrases in self.COMMAND_PHRASES.items():
            method_name = f"command_{command}"
            method = getattr(self, method_name, None)
            if method and callable(method):
                for phrase in phrases:
                    self.commands[phrase.lower()] = method
            else:
                logging.warning(f"Method '{method_name}' not found for command '{command}'.")

    def process_text(self, text: str, dialog: Optional[Any] = None) -> str:
        """
        Determines which command to execute based on the input text.
        Scans the text from the beginning and executes only the first command found.
        """
        text_lower = text.lower()
        first_index = len(text_lower) + 1
        chosen_function = None

        # Loop over each registered command phrase.
        for phrase, function in self.commands.items():
            pattern = r'\b' + re.escape(phrase) + r'\b'
            match = re.search(pattern, text_lower)
            if match and match.start() < first_index:
                first_index = match.start()
                chosen_function = function

        # If a command was found, execute it.
        if chosen_function:
            return chosen_function(text, dialog)
        
        # Fallback to query command if no known command phrase is detected.
        return self.command_query(text, dialog)


    def command_copy(self, text: str, dialog: Optional[Any] = None) -> str:
        """
        Copies the text to the clipboard.
        """
        try:
            pyperclip.copy(text)
            return text
        except pyperclip.PyperclipException as e:
            logging.error(f"Failed to copy text: {e}")
            return ""

    def command_format(self, text: str, dialog: Optional[Any] = None) -> str:
        """
        Wraps clipboard text in markdown code blocks.
        """
        try:
            current_text = pyperclip.paste()
            wrapped_text = f"```\n{current_text}\n```"
            pyperclip.copy(wrapped_text)
            return wrapped_text
        except pyperclip.PyperclipException as e:
            logging.error(f"Clipboard error: {e}")
            return ""

    def command_code(self, text: str, dialog: Optional[Any] = None) -> str:
        """
        Requests a code generation from the AI.
        """
        try:
            current_text = pyperclip.paste()
            if len(current_text) > 10:
                current_text = f"```\n{current_text}\n```"
            else:
                current_text = ""
            code_response = self.ai.code(
                prompt_usr=f"{text}\n{current_text}",
                prompt_sys=self.PROMPTS["code_completion"],
                model="gpt-4o",
                temperature=0.1,
            )
            if code_response:
                pyperclip.copy(code_response)
                return f"```\n{code_response}\n```"
            else:
                return ""
        except pyperclip.PyperclipException as e:
            logging.error(f"Failed to process clipboard content: {e}")
            return ""

    def command_exit(self, text: str, dialog: Optional[Any] = None) -> str:
        """
        Exits the application.
        """
        import os

        os._exit(0)

    def command_talk(self, text: str, dialog: Optional[Any] = None) -> None:
        """
        Simulates typing the provided text by sending keystrokes using the keyboard module.
        """
        try:
            import keyboard
            # Remove the first word from the text
            text = ' '.join(text.split()[1:])
            # Simulate typing the text with a slight delay between keystrokes.
            keyboard.write(text, delay=0.01)
            return None
        except Exception as e:
            logging.error(f"Failed to simulate keyboard input: {e}")
            return None


    def command_query(self, text: str, dialog: Optional[Any] = None) -> str:
        """
        Handles general queries by starting a StreamWorker in a separate thread.
        """
        prompt = self._extract_prompt(text)
        if not prompt:
            return ""

        # Optionally include clipboard content.
        if "clipboard" in prompt.lower():
            prompt += f"\n```\n{pyperclip.paste()}\n```"

        messages = [
            {"role": "system", "content": self.PROMPTS["general_question"]},
            {"role": "user", "content": prompt},
        ]

        worker_thread = QThread()
        worker = StreamWorker(
            ai=self.ai,
            prompt=prompt,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        worker.moveToThread(worker_thread)

        worker.partialResult.connect(dialog.stream_assistant_update)
        worker_thread.started.connect(worker.run)
        worker_thread.finished.connect(worker_thread.deleteLater)
        worker_thread.finished.connect(
            lambda: self.active_threads.remove((worker_thread, worker))
        )

        worker_thread.start()
        self.active_threads.append((worker_thread, worker))

        # Return an immediate placeholder so the GUI remains responsive.
        return "(Streaming response...)"

    def _extract_prompt(self, text: str) -> str:
        """
        Strips known query phrases from the text.
        """
        text_lower = text.lower()
        for phrase in self.COMMAND_PHRASES["query"]:
            if phrase in text_lower:
                return text_lower.replace(phrase, "").strip()
        return text

    def shutdown_threads(self) -> None:
        for thread, worker in list(self.active_threads):
            if thread.isRunning():
                thread.quit()
                thread.wait()
        self.active_threads.clear()
