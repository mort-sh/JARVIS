# print_handler.py
#
# A production-ready Rich-based printing library that supports text, 
# actions (function/method calls), and *any* arbitrary data (lists, dicts, JSON, etc.). 
#
import datetime
import json
from typing import Any, Dict, Optional, Union

import rich
import rich.pretty
from pydantic import BaseModel
from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table


# A global spinner for consistent animation across states
RUNNING_SPINNER = Spinner("dots")


class DisplayState(BaseModel):
    """
    Base class for a piece of data/content/action to be displayed.

    Attributes:
        identifier: Unique ID to track this display state (e.g., a message or event ID).
        source_name: The name/label of the 'source' (e.g., user, system, agent).
        first_timestamp: When this state was created or first recorded.
    """
    identifier: str
    source_name: str
    first_timestamp: datetime.datetime

    def format_timestamp(self) -> str:
        """Convert stored timestamp to a readable string."""
        local_timestamp = self.first_timestamp.astimezone()
        return local_timestamp.strftime("%I:%M:%S %p").lstrip("0").rjust(11)

    def render_panel(self) -> Panel:
        """
        Subclasses should override this to return a Panel or
        any Renderable representing the state's content.
        """
        raise NotImplementedError("Subclasses must implement render_panel().")


class ContentState(DisplayState):
    """
    A display state for text-based content (including markdown).

    Attributes:
        content: The text or markdown content to display.
    """
    content: str = ""

    @staticmethod
    def _convert_content_to_str(content: Any) -> str:
        """
        Helper to convert various possible content formats into a single string.
        Extend or customize based on your project's data structures.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            return content.get("content", content.get("text", ""))

        if isinstance(content, list):
            # Concatenate string pieces or dict-based text fields
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    part = item.get("content", item.get("text", ""))
                    if part:
                        parts.append(part)
            return "\n".join(parts)

        # Fallback: just string-cast
        return str(content)

    def update_content(self, new_content: Any) -> None:
        """Update the stored content."""
        self.content = self._convert_content_to_str(new_content)

    def render_panel(self) -> Panel:
        """Render the text or Markdown in a styled Panel."""
        return Panel(
            Markdown(self.content),
            title=f"[bold]Source: {self.source_name}[/]",
            subtitle=f"[italic]{self.format_timestamp()}[/]",
            title_align="left",
            subtitle_align="right",
            border_style="blue",
            box=box.ROUNDED,
            width=100,
            padding=(1, 2),
        )


class ActionState(DisplayState):
    """
    A display state to represent an 'action' call, such as a function or method invocation.

    Attributes:
        name: Name of the action/function (e.g., "download_file").
        args: Arguments/parameters passed to the action.
        result: The result or output from the action, if any.
        is_error: Whether the action ended in an error state.
        is_complete: Whether the action has finished processing.
    """
    name: str
    args: dict
    result: Optional[str] = None
    is_error: bool = False
    is_complete: bool = False

    def get_status_style(self) -> tuple[Union[str, Spinner], str, str]:
        """
        Determine the icon, text style, and border color based on current status.
        
        Returns:
            (icon_or_spinner, text_style, border_style).
        """
        if self.is_complete:
            if self.is_error:
                return "❌", "red", "red"
            return "✅", "green", "green3"
        return RUNNING_SPINNER, "yellow", "gray50"

    def render_panel(self, show_inputs: bool = True, show_outputs: bool = True) -> Panel:
        """Render the action call in a panel, optionally showing inputs/outputs."""
        icon, text_style, border_style = self.get_status_style()
        table = Table.grid(padding=0, expand=True)

        # Header row: an icon and the formatted action name
        header = Table.grid(padding=1)
        header.add_column(width=2)
        header.add_column()
        action_name = self.name.replace("_", " ").title()
        header.add_row(icon, f"[{text_style} bold]{action_name}[/]")
        table.add_row(header)

        # Optionally display inputs/outputs
        if show_inputs or show_outputs:
            details = Table.grid(padding=(0, 2))
            details.add_column(style="dim", width=9)
            details.add_column()

            if show_inputs and self.args:
                details.add_row(
                    "    Input:",
                    rich.pretty.Pretty(self.args, indent_size=2, expand_all=True),
                )

            if show_outputs and self.is_complete and self.result is not None:
                label = "Error" if self.is_error else "Output"
                style = "red" if self.is_error else "green3"
                details.add_row(
                    f"    {label}:",
                    f"[{style}]{self.result}[/]",
                )

            table.add_row(details)

        return Panel(
            table,
            title=f"[bold]Source: {self.source_name}[/]",
            subtitle=f"[italic]{self.format_timestamp()}[/]",
            title_align="left",
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            width=100,
            padding=(0, 1),
        )


class DataState(DisplayState):
    """
    A display state for arbitrary data structures: lists, dict, JSON strings, etc.

    The logic uses either:
      - Rich's `Syntax` for JSON formatting (if the content is valid JSON or
        can be converted to JSON).
      - Rich's `Pretty` if it's a Python object that doesn't serialize nicely to JSON.
    
    Attributes:
        data: Any Python object. If it can be JSON-serialized, it will be syntax-highlighted.
    """
    data: Any

    def _prepare_renderable(self) -> RenderableType:
        """
        Convert self.data into a Rich renderable. 
        If it's recognized as a dict/list, use JSON syntax highlight. Otherwise, fallback to Pretty.
        """
        # 1) If it's already a Python dict/list, attempt to convert to JSON string
        if isinstance(self.data, (dict, list)):
            try:
                as_json_str = json.dumps(self.data, indent=2)
                return Syntax(as_json_str, "json", theme="monokai", line_numbers=False)
            except (TypeError, ValueError):
                # If JSON serialization fails, fallback to pretty
                return rich.pretty.Pretty(self.data, expand_all=True)

        # 2) If it's a string, check if it's valid JSON
        if isinstance(self.data, str):
            trimmed = self.data.strip()
            if (trimmed.startswith("{") and trimmed.endswith("}")) or \
               (trimmed.startswith("[") and trimmed.endswith("]")):
                # Possibly JSON
                try:
                    parsed = json.loads(self.data)
                    as_json_str = json.dumps(parsed, indent=2)
                    return Syntax(as_json_str, "json", theme="monokai", line_numbers=False)
                except (TypeError, ValueError):
                    # Not valid JSON, just show as text
                    return rich.pretty.Pretty(self.data)
            else:
                # Just show the text
                return rich.pretty.Pretty(self.data)

        # 3) Fallback to Pretty for anything else
        return rich.pretty.Pretty(self.data, expand_all=True)

    def render_panel(self) -> Panel:
        """Render the data in a styled panel."""
        content_renderable = self._prepare_renderable()
        return Panel(
            content_renderable,
            title=f"[bold]Data from: {self.source_name}[/]",
            subtitle=f"[italic]{self.format_timestamp()}[/]",
            title_align="left",
            subtitle_align="right",
            border_style="magenta",
            box=box.ROUNDED,
            width=100,
            padding=(0, 1),
        )


class PrintHandler:
    """
    A generic print handler using Rich for terminal-based visualization of:
      - Text/Markdown content  (ContentState)
      - Action calls/results   (ActionState)
      - Arbitrary data         (DataState)
    
    All managed within a Rich Live console.

    Example usage:
        handler = PrintHandler()
        handler.start()

        # Content
        handler.on_content_update(
            identifier="msg_1",
            source_name="User",
            timestamp=datetime.datetime.now(),
            new_content="Hello, World!"
        )

        # Action call
        handler.on_action_call(
            identifier="action_1",
            source_name="System",
            timestamp=datetime.datetime.now(),
            action_name="process_data",
            args={"data": [1, 2, 3]}
        )

        # Action result
        handler.on_action_result(
            identifier="action_1",
            is_error=False,
            result="Process finished successfully."
        )

        # Arbitrary data
        handler.on_data_update(
            identifier="json_blob_1",
            source_name="External API",
            timestamp=datetime.datetime.now(),
            data={"response": {"status": "ok", "items": [1,2,3]}}
        )

        handler.stop()
    """

    def __init__(
        self,
        show_action_inputs: bool = True,
        show_action_outputs: bool = True,
    ):
        self.show_action_inputs = show_action_inputs
        self.show_action_outputs = show_action_outputs

        self.live: Optional[Live] = None
        self.console = Console()
        self.states: Dict[str, DisplayState] = {}

    def start(self) -> None:
        """Begin the live rendering process."""
        if not self.live:
            self.live = Live(
                console=self.console,
                vertical_overflow="visible",
                auto_refresh=True,
            )
        if not self.live.is_started:
            self.live.start()

    def stop(self) -> None:
        """Stop the live rendering process."""
        if self.live and self.live.is_started:
            self.live.stop()

    def update_display(self) -> None:
        """
        Render all states in a single panel with a table layout.
        This is automatically called at the end of each 'on_*' event method.
        """
        if not self.live or not self.live.is_started:
            return

        # Sort states by creation time so display is consistent
        sorted_states = sorted(self.states.values(), key=lambda s: s.first_timestamp)
        
        if not sorted_states:
            return

        # Create main table for all states
        table = Table(
            show_header=True,
            box=box.SIMPLE,
            expand=True,
            padding=(0, 1),
        )
        
        # Configure columns
        table.add_column("Time", style="dim", width=11)
        table.add_column("Source", style="bold")
        table.add_column("Status", width=3)
        table.add_column("Content", ratio=1)

        # Add each state as a row
        for state in sorted_states:
            time = state.format_timestamp()
            source = state.source_name
            
            if isinstance(state, ActionState):
                # For actions, show status icon and formatted content
                icon, text_style, _ = state.get_status_style()
                action_name = state.name.replace("_", " ").title()
                
                content = Table.grid(padding=0, expand=True)
                content.add_row(f"[{text_style} bold]{action_name}[/]")
                
                if self.show_action_inputs and state.args:
                    content.add_row(
                        Table.grid(padding=(0, 2))
                        .add_row(
                            "Input:",
                            rich.pretty.Pretty(state.args, indent_size=2, expand_all=True)
                        )
                    )
                
                if self.show_action_outputs and state.is_complete and state.result is not None:
                    label = "Error" if state.is_error else "Output"
                    style = "red" if state.is_error else "green3"
                    content.add_row(
                        Table.grid(padding=(0, 2))
                        .add_row(
                            f"{label}:",
                            f"[{style}]{state.result}[/]"
                        )
                    )
                
                table.add_row(time, source, icon, content)
                
            elif isinstance(state, ContentState):
                # For content, show markdown
                table.add_row(
                    time,
                    source,
                    "📝",
                    Markdown(state.content)
                )
                
            elif isinstance(state, DataState):
                # For data, show the formatted data
                table.add_row(
                    time,
                    source,
                    "📊",
                    state._prepare_renderable()
                )

        # Wrap table in a panel
        panel = Panel(
            table,
            title="[bold]Activity Log[/]",
            border_style="blue",
            box=box.ROUNDED,
            width=100,
            padding=(0, 1),
        )
        
        self.live.update(panel, refresh=True)

    # -------------------------
    # Content events
    # -------------------------
    def on_content_update(
        self,
        identifier: str,
        source_name: str,
        timestamp: datetime.datetime,
        new_content: Any,
    ) -> None:
        """
        For updating or creating a ContentState.
        If the identifier doesn't exist, a new ContentState is created.
        If it does exist, we update the existing state's content.
        """
        if identifier not in self.states:
            state = ContentState(
                identifier=identifier,
                source_name=source_name,
                first_timestamp=timestamp,
            )
            state.update_content(new_content)
            self.states[identifier] = state
        else:
            existing_state = self.states[identifier]
            if isinstance(existing_state, ContentState):
                existing_state.update_content(new_content)
            else:
                # If it exists but is not ContentState, replace or handle differently
                state = ContentState(
                    identifier=identifier,
                    source_name=source_name,
                    first_timestamp=timestamp,
                )
                state.update_content(new_content)
                self.states[identifier] = state

        self.update_display()

    # -------------------------
    # Action events
    # -------------------------
    def on_action_call(
        self,
        identifier: str,
        source_name: str,
        timestamp: datetime.datetime,
        action_name: str,
        args: dict,
    ) -> None:
        """
        For creating or updating an ActionState with new arguments.
        """
        if identifier not in self.states:
            self.states[identifier] = ActionState(
                identifier=identifier,
                source_name=source_name,
                first_timestamp=timestamp,
                name=action_name,
                args=args,
            )
        else:
            existing_state = self.states[identifier]
            if isinstance(existing_state, ActionState):
                existing_state.args = args
            else:
                # If it exists but not ActionState, overwrite or handle differently
                self.states[identifier] = ActionState(
                    identifier=identifier,
                    source_name=source_name,
                    first_timestamp=timestamp,
                    name=action_name,
                    args=args,
                )

        self.update_display()

    def on_action_result(
        self,
        identifier: str,
        is_error: bool,
        result: Optional[str],
    ) -> None:
        """
        Mark an existing ActionState as complete, storing its output or error info.
        """
        if identifier in self.states:
            state = self.states[identifier]
            if isinstance(state, ActionState):
                state.is_complete = True
                state.is_error = is_error
                state.result = result

        self.update_display()

    # -------------------------
    # Data events
    # -------------------------
    def on_data_update(
        self,
        identifier: str,
        source_name: str,
        timestamp: datetime.datetime,
        data: Any
    ) -> None:
        """
        For creating or updating a DataState with arbitrary data (lists, dict, JSON, etc.).
        """
        if identifier not in self.states:
            self.states[identifier] = DataState(
                identifier=identifier,
                source_name=source_name,
                first_timestamp=timestamp,
                data=data,
            )
        else:
            existing_state = self.states[identifier]
            if isinstance(existing_state, DataState):
                existing_state.data = data
            else:
                # If the existing state isn't DataState, replace it or handle differently
                self.states[identifier] = DataState(
                    identifier=identifier,
                    source_name=source_name,
                    first_timestamp=timestamp,
                    data=data,
                )

        self.update_display()
# At the very end of ui/print_handler.py, add:
print_handler_instance = PrintHandler()
print_handler_instance.start()
