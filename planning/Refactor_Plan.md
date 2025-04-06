# JARVIS Refactoring and Optimization Plan

## 1. Goals

*   **Enhance Cross-Platform Compatibility:** Migrate from PyQt5 to PyQt6 for better support on modern macOS and Windows, leveraging Qt6 improvements. (Top Priority)
*   **Decouple UI (`PopupDialog`):** Refactor the UI component to reduce tight coupling with backend services (AI, Commands, Transcription), making it more modular and potentially reusable as a standalone library. (High Priority)
*   **Integrate Modern Agent Framework:** Replace the current AI interaction logic (`openai_wrapper`, `command_library`) with **PydanticAI** to leverage structured outputs, robust tool definitions, and a modern agentic workflow. (High Priority)
*   **Prioritize GUI:** Focus development efforts on the Qt GUI, removing or significantly refactoring the `rich`-based console handler (`print_handler.py`) to eliminate redundancy and simplify the UI architecture.
*   **Improve Maintainability:** Standardize logging, error handling, and code structure.
*   **Update Documentation:** Ensure `README.md` accurately reflects the project's current state after refactoring.

## 2. Prerequisites (User Action Required)

*   Add `rich` dependency to `pyproject.toml` (if not already done).
*   *(Cline Action in ACT Mode)* Add `PyQt6` dependency to `pyproject.toml`.
*   *(Cline Action in ACT Mode)* Add `pydantic-ai` dependency to `pyproject.toml`.

## 3. Phase 1: PyQt6 Migration

*   **Objective:** Replace PyQt5 with PyQt6 throughout the application for improved cross-platform support and future-proofing.
*   **Steps:**
    1.  **Add Dependency:** Add `PyQt6` to the `[project.dependencies]` section in `pyproject.toml`. Run `uv pip install -e .` to install it.
    2.  **Forward-Compatible Updates (Within PyQt5 Context - Optional but Recommended):**
        *   Search for and replace short-form enums with fully qualified names (e.g., `Qt.Checked` -> `QtCore.Qt.CheckState.Checked`, `Qt.AlignRight` -> `QtCore.Qt.AlignmentFlag.AlignRight`). Use `search_files` if needed to locate usages.
        *   Search for and replace `.exec_()` calls with `.exec()`.
    3.  **Import Migration:** Systematically change all `from PyQt5...` imports to `from PyQt6...` across all relevant files (`transcription_worker.py`, `popup_dialog.py`, `stream_worker.py`, potentially `main.py`).
    4.  **API Change Addressal:**
        *   Locate `QAction` imports and change them from `QtWidgets` to `QtGui`.
        *   Search for `QMouseEvent` handlers (`mousePressEvent`, `mouseMoveEvent`):
            *   Replace `.pos()` with `.position()`.
            *   Replace `.x()` with `.position().x()`.
            *   Replace `.y()` with `.position().y()`.
            *   Replace `.globalPos()` with `.globalPosition()`.
        *   Search for `Qt.MidButton` and replace with `QtCore.Qt.MouseButton.MiddleButton`.
        *   Search for `QDesktopWidget` usage (likely in `popup_dialog.py` for centering/positioning) and replace with `QScreen` methods (e.g., `self.screen().geometry()`, `QGuiApplication.primaryScreen().availableGeometry()`).
        *   Search for `QFontMetrics` usage and rename `.width()` calls to `.horizontalAdvance()`.
        *   Search for and remove High DPI attributes (`Qt.AA_EnableHighDpiScaling`, `Qt.AA_DisableHighDpiScaling`, `Qt.AA_UseHighDpiPixmaps`) usually set on the `QApplication` instance in `main.py`.
        *   Address any other specific deprecation warnings or errors encountered during testing.
    5.  **Basic Testing:** Run `main.py`. Verify the application launches, the popup dialog appears, basic interactions (moving the dialog, closing) work, and no immediate Qt-related errors occur.

## 4. Phase 2: Decouple UI (`PopupDialog`)

*   **Objective:** Isolate `PopupDialog` from direct dependencies on backend logic, enabling potential reuse and improving testability.
*   **Steps:**
    1.  **Define Interfaces (Protocols):** Create a new file (e.g., `jarvis/ui/interfaces.py`) defining `typing.Protocol` classes for the interactions:
        *   `IUIController`: Methods the UI calls (e.g., `process_user_input(text: str)`, `start_recording(ctrl_pressed: bool)`, `stop_recording()`, `increase_font()`, `decrease_font()`, `clear_dialog()`, `close_dialog()`).
        *   `IUISignaler`: Signals the backend needs to emit for the UI (e.g., `update_assistant_message = Signal(str)`, `stream_assistant_chunk = Signal(str)`, `clear_assistant_message = Signal()`, `recording_state_changed = Signal(bool, float)`, `transcription_result = Signal(str)`). Note: These might be defined on the Controller or specific worker interfaces depending on the final design.
    2.  **Implement UIController:** Create `jarvis/ui/controller.py` with a `UIController` class inheriting from `QObject` and potentially `IUISignaler`.
        *   The `__init__` method should accept instances of the backend services (initially `CommandLibrary`, `OpenAIWrapper`, `TranscriptionWorker`; later the PydanticAI `Agent`).
        *   Implement methods defined in `IUIController`. These methods will delegate calls to the appropriate backend service instances.
        *   Connect signals from backend services (e.g., `TranscriptionWorker.transcription_ready`, `StreamWorker.chunk_received`) to internal slots in the `UIController`. These slots will then emit the corresponding signals defined in `IUISignaler`.
    3.  **Refactor `PopupDialog`:**
        *   Modify `PopupDialog.__init__` to accept an instance of `IUIController` (dependency injection). Store it as `self.controller`.
        *   Remove direct imports of `CommandLibrary`, `OpenAIWrapper`, `TranscriptionWorker`, `StreamWorker`.
        *   Replace direct calls to backend services with calls to `self.controller` methods (e.g., replace `self.command_library.process_text(...)` with `self.controller.process_user_input(...)`).
        *   Connect UI element signals (buttons, text input `returnPressed`) to `self.controller` methods.
        *   Connect signals from `self.controller` (implementing `IUISignaler`) to the appropriate slots within `PopupDialog` (e.g., `controller.stream_assistant_chunk.connect(self.stream_assistant_update)`).
    4.  **Refactor Workers (`TranscriptionWorker`, `StreamWorker`):**
        *   Ensure they emit signals for relevant events (e.g., `transcription_ready`, `chunk_received`, `recording_state_changed`).
        *   Remove any direct references to `PopupDialog`. If they need to trigger UI updates, they should do so via signals connected to the `UIController`.
    5.  **Update `main.py`:**
        *   Instantiate backend services.
        *   Instantiate `UIController`, passing services to it.
        *   Instantiate `PopupDialog`, passing the `UIController` to it.
        *   Ensure the application event loop starts correctly.

## 5. Phase 3: Refactor/Remove `print_handler.py`

*   **Objective:** Eliminate the redundant `rich`-based console UI handler, focusing solely on the Qt GUI and standard logging.
*   **Steps:**
    1.  **Analyze Usage:** Use `search_files` with regex `from .*print_handler import|print_handler\.|AdvancedConsole` to find all usages of the module and its classes (`AdvancedConsole`, `LayoutManager`, `ThemePreset`, state classes).
    2.  **Replace Logging/Status:** If `AdvancedConsole` methods (`.log`, `.print`, `.show_status`) are used for developer feedback or status messages not intended for the main GUI, replace them with standard Python `logging` calls (see Phase 5).
    3.  **Migrate GUI Rendering:** If `rich` components (`Panel`, `Markdown`, `Table`, etc.) are used within `PopupDialog` or related UI code (e.g., `PopupDialog._refresh_markdown` might use `print_handler.print_markdown`):
        *   Refactor the rendering logic to use native Qt capabilities. `QTextEdit` supports rich text (HTML subset) and has basic Markdown support. Use `QTextEdit.setMarkdown()` or `QTextEdit.setHtml()`.
        *   For tables or complex layouts, consider using `QTableWidget` or custom Qt layouts with `QLabel` widgets.
        *   Remove the `rich` dependency from UI rendering code.
    4.  **Remove State Classes:** If the state classes (`RecordingState`, etc.) in `print_handler.py` are used, migrate their logic either into the relevant Qt components (`PopupDialog`, `TranscriptionWorker`) or the `UIController`.
    5.  **Delete File:** Once all dependencies are removed, delete `jarvis/ui/print_handler.py`.
    6.  **Update Dependencies:** Remove `rich` from `pyproject.toml` if it's confirmed to be no longer needed anywhere in the project.

## 6. Phase 4: Integrate PydanticAI Agent Framework

*   **Objective:** Replace the custom AI interaction logic (`openai_wrapper`, `command_library`) with a unified PydanticAI agent.
*   **Steps:**
    1.  **Add Dependency:** Add `pydantic-ai` to `pyproject.toml` and install via `uv pip install -e .`.
    2.  **Define Agent:** Create a new file (e.g., `jarvis/agent/jarvis_agent.py`).
        *   Define Pydantic models for tool arguments (e.g., `FormatCodeArgs(language: str = 'python')`, `CodeArgs(action: Literal['optimize', 'fix', ...], code: str)`, `QueryArgs(topic: str)`).
        *   Define a Pydantic model for the expected structured result if applicable (e.g., `CodeResult(modified_code: str, explanation: Optional[str] = None)`), or use `str` for simple text responses.
        *   Define a Pydantic model or dataclass for dependencies needed by tools (e.g., `ToolDependencies(clipboard: ClipboardService, keyboard: KeyboardService)`).
        *   Instantiate `pydantic_ai.Agent`, configuring it with the chosen LLM (from `settings.py`), the `deps_type`, and potentially `result_type`. Define a base system prompt.
    3.  **Implement Tools:**
        *   Decorate functions with `@agent.tool`. These functions will encapsulate the logic currently in `CommandLibrary` methods (`command_copy`, `command_format`, `command_code`, `command_talk`).
        *   Use the Pydantic argument models defined above as type hints for tool parameters.
        *   Use `RunContext[ToolDependencies]` to access dependencies like clipboard/keyboard services within tools.
        *   The `command_query` logic will likely become the default agent behavior (no specific tool needed, just process the prompt).
        *   The `command_exit` logic should be handled by the `UIController` based on user input, not as an agent tool.
    4.  **Refactor `UIController`:**
        *   Modify `UIController.__init__` to accept the PydanticAI `Agent` instance instead of `CommandLibrary` and `OpenAIWrapper`.
        *   Update `UIController.process_user_input` to call `agent.run(...)` or `agent.run_stream(...)` with the user text and necessary dependencies.
        *   Handle the `AgentRunResult` or streamed chunks, emitting signals (`IUISignaler`) for the `PopupDialog` to display. Manage potential tool calls and retries if needed (PydanticAI handles much of this).
    5.  **Refactor `StreamWorker`:** This worker might become redundant if `agent.run_stream` is handled directly within the `UIController` or a dedicated PydanticAI stream handler. Analyze if its specific Qt threading logic is still required or can be simplified/removed.
    6.  **Remove Old Code:** Delete `jarvis/ai/openai_wrapper.py` and `jarvis/commands/command_library.py`. Remove their imports and instantiation from `main.py` and `UIController`.
    7.  **Update `settings.py`:** Remove settings related to the old `openai_wrapper` if they are now handled by PydanticAI configuration (e.g., model selection logic might change).

## 7. Phase 5: Standardize Logging

*   **Objective:** Implement consistent application-wide logging using Python's standard `logging` module.
*   **Steps:**
    1.  **Configure Logging:** In `main.py` or a new `jarvis/logging_config.py`, set up `logging.basicConfig` or `logging.config.dictConfig`. Define a standard format (e.g., `%(asctime)s - %(name)s - %(levelname)s - %(message)s`) and configure handlers (e.g., `logging.StreamHandler` for console output during development, potentially `logging.FileHandler` for persistent logs). Set an appropriate default level (e.g., `logging.INFO`).
    2.  **Instantiate Loggers:** In each `.py` file, add `import logging` and `logger = logging.getLogger(__name__)` at the module level.
    3.  **Replace Print/Console Calls:** Search for `print()` statements used for debugging or status messages and replace them with `logger.debug()`, `logger.info()`, or `logger.warning()`.
    4.  **Log Errors:** In `try...except` blocks, especially around API calls (now within PydanticAI agent runs), file I/O, audio processing, and UI event handling, use `logger.error("Error description: %s", e)` or `logger.exception("Unhandled exception occurred")` to capture errors and stack traces.
    5.  **Remove `print_handler` Logging:** Ensure any logging previously done via `print_handler.py` is replaced with standard `logging` calls.

## 8. Phase 6: Update Documentation (`README.md`)

*   **Objective:** Ensure the README accurately reflects the refactored application.
*   **Steps:**
    1.  **Dependencies:** Update the installation section (`pyproject.toml` dependencies) to list `PyQt6` and `pydantic-ai`. Remove `PyQt5` if previously mentioned. Confirm `rich` is listed if still used (e.g., for tracebacks) or removed if not. Update `uv pip install` instructions if needed.
    2.  **Architecture:** Update the Project Structure section to reflect the removal of `openai_wrapper`, `command_library`, `print_handler` and the addition of `UIController`, `interfaces.py`, `jarvis_agent.py`. Briefly explain the role of PydanticAI.
    3.  **Tooling:** Update the Development Tools section. Remove mentions of Black, isort, MyPy if `ruff` is handling formatting, linting, and import sorting. Ensure `ruff` configuration is mentioned.
    4.  **Configuration:** Update the Configuration section if `settings.py` or `.env` usage changed due to PydanticAI integration.
    5.  **Usage:** Update command examples if the PydanticAI agent interaction model changed how users interact (though the goal is likely to keep the external UX similar).
    6.  **Clarifications:** Correct the "UV client" mention to clarify `uv` is the package manager. Ensure descriptions match the current implementation (e.g., GUI library is Qt6).

## 9. Phase 7: Testing (Optional but Recommended)

*   **Objective:** Add automated tests to verify functionality and prevent regressions.
*   **Steps:**
    1.  **Setup:** Ensure `pytest` is configured correctly in `pyproject.toml`. Create a `tests/` directory if it doesn't exist.
    2.  **Unit Tests:**
        *   Write tests for PydanticAI agent tools (mocking dependencies like clipboard/keyboard).
        *   Write tests for the `UIController` logic (mocking the Agent and UI signals/slots).
        *   Test any utility functions or configuration loading.
    3.  **Integration Tests:** (More complex)
        *   Consider tests that simulate user input to the `UIController` and verify that the correct PydanticAI `Agent` methods are called (using mocking/spying).
        *   Test the interaction between the `UIController` and `PopupDialog` via signals/slots if feasible using `pytest-qt`.
    4.  **CI:** Consider adding a GitHub Actions workflow to run `ruff check`, `ruff format --check`, and `pytest` on pushes/PRs.

## 10. Finalization (Post-Refactor)

*   **Memory Update:** Use the Memory MCP server (`create_entities`, `create_relations`, `add_observations`) to record key details about the refactored JARVIS project architecture, dependencies (PyQt6, PydanticAI), and design decisions (UI decoupling, agent integration).
*   **Commit Changes:** Use `autocommit.exe` via the `cli` MCP server to commit the refactored code and the updated `planning/Refactor_Plan.md`.
