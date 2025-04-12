#!/usr/bin/env python3
"""
Main entry point for the application.
"""

import sys
import threading
import argparse
import logging
import signal
import os

import keyboard
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from rich.console import Console
from rich.panel import Panel
from rich import box
from datetime import datetime
from jarvis.ui.print_handler import advanced_console as console
console = Console()

from jarvis.ui.popup_dialog import PopupDialog
from jarvis.ui.controller import UIController


def main() -> None:
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run Jarvis UI")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for detailed output."
    )
    args = parser.parse_args()

    # --- Configure Logging ---
    log_level = logging.DEBUG if args.debug else logging.INFO
    # Configure root logger - this might affect libraries as well
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True # Override any existing basicConfig by libraries
    )
    # Optionally, set specific loggers to different levels if needed
    # logging.getLogger("jarvis").setLevel(log_level) # Example
    # logging.getLogger("keyboard").setLevel(log_level) # Might reveal keyboard lib issues
    log = logging.getLogger(__name__) # Get logger for main module
    log.info(f"Logging level set to: {logging.getLevelName(log_level)}")

    # --- Application Setup ---
    app = QApplication(sys.argv)

    # Create the UI controller
    log.debug("Initializing UIController...")
    controller = UIController()
    log.debug("UIController initialized.")

    # Create the popup dialog
    log.debug("Creating PopupDialog...")
    dialog = PopupDialog(controller)
    dialog.set_font_size(15)
    dialog.setWindowOpacity(0.95)
    log.debug("PopupDialog created.")

    # Print the list of registered commands
    log.debug("Printing registered commands...")
    controller.command_library.print_registered_commands()

    # Connect the transcription worker from the controller to ensure signals work correctly
    log.debug("Connecting transcriptionReady signal...")
    controller.transcription_worker.transcriptionReady.connect(dialog.custom_update)

    # --- Signal Handler Setup ---
    # Set up proper signal handling for clean exit on Ctrl+C
    def signal_handler(sig, frame):
        log.info("SIGINT (Ctrl+C) received. Initiating shutdown...")
        console.print("[bold red]Ctrl+C pressed. Shutting down...[/bold red]")

        # Set a flag to stop all threads and event loops
        app.quit()  # Signal Qt app to quit

        # Unhook keyboard listeners immediately to prevent conflicts during shutdown
        try:
            keyboard.unhook_all()
            log.debug("Keyboard hooks removed.")
        except Exception as e:
            log.error(f"Error unhooking keyboard: {e}")

        # Trigger controller shutdown which will stop all worker threads
        controller.shutdown()
        log.info("Controller shutdown complete.")

        # Normal exit - no need for forced termination if threads are properly managed
        sys.exit(0)

    # Register the signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # Enable processing of signals in Qt by setting a timer
    # This creates a small window every 500ms for Python to process signals
    signal_timer = QTimer()
    signal_timer.timeout.connect(lambda: None)  # Empty slot
    signal_timer.start(500)  # Check every 500ms

    # Run the keyboard hook in a background thread
    log.info("Starting keyboard hook thread...")
    hook_thread = threading.Thread(target=controller.run_keyboard_hook, daemon=True)
    hook_thread.start()
    log.debug("Keyboard hook thread started.")

    def increase_font_size() -> None:
        log.debug("Increase font size hotkey pressed.")
        current_size = dialog.text_edit.font().pointSize()
        dialog.set_font_size(current_size + 1)
        controller.font_size = current_size + 1

    def decrease_font_size() -> None:
        log.debug("Decrease font size hotkey pressed.")
        current_size = dialog.text_edit.font().pointSize()
        if current_size > 1:
            dialog.set_font_size(current_size - 1)
            controller.font_size = current_size - 1

    # Register hotkeys for adjusting font size
    log.debug("Registering font size hotkeys...")
    try:
        keyboard.add_hotkey("=", increase_font_size)
        keyboard.add_hotkey("-", decrease_font_size)
        log.debug("Font size hotkeys registered.")
    except Exception as e:
        log.error(f"Failed to register font size hotkeys: {e}")

    try:
        log.info("Starting Qt application event loop...")
        exit_code = app.exec()
        log.info(f"Qt application event loop exited with code: {exit_code}")
    except KeyboardInterrupt:
        log.info("Keyboard interrupt detected. Exiting application...")
    except Exception as e:
        log.exception(f"Unexpected error in main Qt loop: {e}")
    finally:
        log.info("Shutting down controller...")
        controller.shutdown()

        # Join any remaining threads with a timeout
        if hook_thread.is_alive():
            log.info("Waiting for keyboard hook thread to terminate...")
            hook_thread.join(timeout=1.0)
            if hook_thread.is_alive():
                log.warning("Keyboard hook thread did not terminate in time.")

        log.info("Exiting system.")
        sys.exit(0)


if __name__ == "__main__":
    main()
