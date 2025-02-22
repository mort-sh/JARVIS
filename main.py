#!/usr/bin/env python3
"""
Main entry point for the application.
"""

import sys
import threading

import keyboard
from PyQt5.QtWidgets import QApplication
import logging
from rich import print  # Added for beautified logging

from ui.popup_dialog import PopupDialog
from services.transcription_worker import TranscriptionWorker


def main() -> None:
    app = QApplication(sys.argv)

    # Create the popup dialog; this also instantiates the CommandLibrary.
    dialog = PopupDialog()
    dialog.set_font_size(15)
    dialog.setWindowOpacity(0.95)

    # Print the list of registered commands.
    print("[blue]Registered commands:[/blue]")
    for command in sorted(dialog.cmd_library.commands.keys()):
        print(f"[magenta]- {command}[/magenta]")

    # Create the transcription worker.
    worker = TranscriptionWorker()
    worker.transcriptionReady.connect(dialog.custom_update)

    # Run the keyboard hook in a background thread.
    threading.Thread(target=worker.run_keyboard_hook, daemon=True).start()

    def increase_font_size() -> None:
        current_size = dialog.text_edit.font().pointSize()
        dialog.set_font_size(current_size + 1)

    def decrease_font_size() -> None:
        current_size = dialog.text_edit.font().pointSize()
        if current_size > 1:
            dialog.set_font_size(current_size - 1)

    # Register hotkeys for adjusting font size.
    keyboard.add_hotkey("=", increase_font_size)
    keyboard.add_hotkey("-", decrease_font_size)

    # # New hotkey: When RIGHT-CONTROL is released, simulate typing of the transcription.
    # def simulate_typing() -> None:
    #     if worker.last_transcription:
    #         print("Simulating typing of transcription...")
    #         keyboard.write(worker.last_transcription)
    #     else:
    #         print("No transcription available to type.")

    # keyboard.on_release_key("right ctrl", lambda _: simulate_typing())

    try:
        app.exec_()
    except KeyboardInterrupt:
        print("[red]Keyboard interrupt detected. Exiting application...[/red]")
    finally:
        worker.stop()
        dialog.cmd_library.shutdown_threads()
        sys.exit()


if __name__ == "__main__":
    main()
