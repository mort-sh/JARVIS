#!/usr/bin/env python3
"""
Main entry point for the application.
"""

import sys
import threading

import keyboard
from PyQt6.QtWidgets import QApplication
import logging
from rich.console import Console
from rich.panel import Panel
from rich import box
from datetime import datetime
from jarvis.ui.print_handler import advanced_console as console
console = Console()

from jarvis.ui.popup_dialog import PopupDialog
from jarvis.ui.controller import UIController


def main() -> None:
    app = QApplication(sys.argv)

    # Create the UI controller
    controller = UIController()
    
    # Create the popup dialog
    dialog = PopupDialog(controller)
    dialog.set_font_size(15)
    dialog.setWindowOpacity(0.95)

    # Print the list of registered commands
    controller.command_library.print_registered_commands()

    # Connect the transcription worker from the controller to ensure signals work correctly
    controller.transcription_worker.transcriptionReady.connect(dialog.custom_update)

    # Run the keyboard hook in a background thread
    threading.Thread(target=controller.run_keyboard_hook, daemon=True).start()

    def increase_font_size() -> None:
        current_size = dialog.text_edit.font().pointSize()
        dialog.set_font_size(current_size + 1)
        controller.font_size = current_size + 1

    def decrease_font_size() -> None:
        current_size = dialog.text_edit.font().pointSize()
        if current_size > 1:
            dialog.set_font_size(current_size - 1)
            controller.font_size = current_size - 1

    # Register hotkeys for adjusting font size
    keyboard.add_hotkey("=", increase_font_size)
    keyboard.add_hotkey("-", decrease_font_size)

    try:
        app.exec()
    except KeyboardInterrupt:
        console.log("[red]Keyboard interrupt detected. Exiting application...[/red]")
    finally:
        controller.shutdown()
        sys.exit()


if __name__ == "__main__":
    main()