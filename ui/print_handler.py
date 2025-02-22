"""
Rich-based terminal printing library with advanced formatting capabilities.

This module provides a comprehensive set of tools for terminal-based visualization
with support for themes, layouts, progress tracking, and various content types.

Features & Wrappers Implemented:
- Rich Print (advanced drop-in print replacement with markup support)
- Console API (auto-detects terminal capabilities, handles logging, etc.)
- Prompt/Input (stylized input method)
- Syntax Highlighting (render syntax-highlighted code blocks)
- Pretty Printing (automatic formatting for Python objects)
- Inspect (introspect objects in a formatted manner)
- Logging (enhanced logging with timestamps, rich tracebacks, etc.)
- Traceback Rendering (colorful, concise Python tracebacks)
- Tables (construct elegant tables with flexible styling)
- Progress Bars (animated progress bars for concurrent tasks)
- Status/Spinner (context-managed status displays with spinners)
- Tree (render hierarchical data with guide lines)
- Columns (arrange output in multiple, balanced columns)
- Markdown Rendering (render Markdown directly to the terminal)
- Panels (enclose content within stylish panels)
- Live Updates (dynamically update parts of your terminal display)
- Layout Management (build structured multi-panel UIs via Layout)
- JSON Rendering (pretty-print and colorize JSON data)
- Rules (draw horizontal lines with optional titles)
- Emoji Support (insert emojis by name, e.g. :smiley:)
- Exporting Capabilities (capture console output to text, HTML, SVG)
- Alternate Screen (use a separate screen for full-screen apps)
- Overflow Control (fold or crop text that exceeds terminal width)
- Capturing Output (programmatically capture output for testing)
- Low-Level Output (minimal formatting for raw output)
- Paging (pipe long outputs into a pager)
- Styling/Themes (custom or preset styles for a consistent look)
"""

import datetime
import json
import logging
from typing import Any, Dict, List, Optional, Union

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, ConsoleOptions, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.status import Status
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree
from rich import traceback

# Enable Rich's nicer traceback by default
traceback.install()


# --------------------------------------------------------------------------
# THEME & STYLES
# --------------------------------------------------------------------------
class ThemePreset:
    """
    Predefined or custom color/style theme for the Rich console.
    Allows consistent styling across all output.
    """

    def __init__(self, name: str = "default", styles: Optional[Dict[str, Style]] = None):
        self.name = name
        self.styles = styles or {
            "info": Style(color="cyan"),
            "warning": Style(color="yellow", bold=True),
            "error": Style(color="red", bold=True),
            "success": Style(color="green", bold=True),
            "content": Style(color="blue"),
            "action": Style(color="magenta"),
            "data": Style(color="bright_magenta"),
        }

    @staticmethod
    def create_default() -> "ThemePreset":
        """Return a basic default theme preset."""
        return ThemePreset()

    def get_rich_theme(self) -> Theme:
        """Convert the preset styles into a Rich Theme object."""
        theme_styles = {}
        for style_name, style_obj in self.styles.items():
            theme_styles[style_name] = style_obj
        return Theme(theme_styles)


# --------------------------------------------------------------------------
# LAYOUT MANAGEMENT
# --------------------------------------------------------------------------
class LayoutManager:
    """
    Simplifies usage of Rich Layout features.
    It can manage multiple named panels and update them dynamically.
    """

    def __init__(self, console: Console):
        self.console = console
        self.layout = Layout(name="root")
        self.layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )

    def update_dimensions(self) -> None:
        """
        Dynamically adjust Layout sizes based on current terminal width/height.
        You could call this before drawing if you need responsive resizing.
        """
        self.layout.size = self.console.size

    def get_layout(self) -> Layout:
        """
        Return the root layout object, so you can customize splits.
        """
        return self.layout

    def render(self) -> RenderableType:
        """
        Return the entire layout as a Renderable, ready to be printed or live-updated.
        """
        return self.layout


# --------------------------------------------------------------------------
# DISPLAY STATES & MODELS (OPTIONAL)
# --------------------------------------------------------------------------
class BaseState:
    """
    Optional: base class for advanced display states, storing metadata
    about messages or data to be rendered in the terminal. Subclasses
    define how they're rendered.
    """

    def __init__(
        self,
        identifier: str,
        source_name: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.identifier = identifier
        self.source_name = source_name
        self.first_timestamp = datetime.datetime.now()
        self.tags = tags or []
        self.metadata = metadata or {}

    def format_timestamp(self) -> str:
        """Return a formatted timestamp for display."""
        local_time = self.first_timestamp.astimezone()
        return local_time.strftime("%Y-%m-%d %H:%M:%S")

    def render(self) -> RenderableType:
        """Subclasses should override this to provide their own logic."""
        return Text(f"BaseState: {self.identifier}")


class ContentState(BaseState):
    """
    State for storing text/Markdown content.
    """

    def __init__(
        self,
        identifier: str,
        source_name: str,
        content: str,
        format_type: str = "markdown",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(identifier, source_name, tags, metadata)
        self.content = content
        self.format_type = format_type

    def render(self) -> RenderableType:
        if self.format_type.lower() == "markdown":
            content_renderable = Markdown(self.content)
        else:
            content_renderable = Text(self.content)

        panel_title = f"[bold]Source: {self.source_name}[/]"
        panel_subtitle = f"[italic]{self.format_timestamp()}[/]"

        return Panel(
            content_renderable,
            title=panel_title,
            subtitle=panel_subtitle,
            expand=True,
            border_style="content",
            box=box.ROUNDED,
        )


class ActionState(BaseState):
    """
    State for representing an ongoing or completed action with optional progress.
    """

    def __init__(
        self,
        identifier: str,
        source_name: str,
        name: str,
        args: Dict[str, Any],
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(identifier, source_name, tags, metadata)
        self.name = name
        self.args = args
        self.result: Optional[str] = None
        self.is_error: bool = False
        self.is_complete: bool = False
        self.progress: float = 0.0

    def render(self) -> RenderableType:
        if not self.is_complete:
            # Show progress bar
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            task_id = progress.add_task(
                f"{self.name.title()}", total=100.0, start=False
            )
            progress.update(task_id, completed=self.progress * 100, start=True)
            return Panel(
                progress,
                title=f"[bold]Action: {self.name}[/]",
                subtitle=f"[italic]{self.format_timestamp()}[/]",
                border_style="action",
                box=box.ROUNDED,
            )
        else:
            # Completed or error
            status_icon = "❌" if self.is_error else "✅"
            style_name = "error" if self.is_error else "success"
            msg = Text.assemble(
                (f"{status_icon} {self.name.title()}\n", style_name),
            )

            if self.result:
                msg.append(Text(f"\nResult: {self.result}", style=style_name))

            return Panel(
                msg,
                title=f"[bold]Action: {self.name}[/]",
                subtitle=f"[italic]{self.format_timestamp()}[/]",
                border_style=style_name,
                box=box.ROUNDED,
            )


class DataState(BaseState):
    """
    State for structured data (JSON, dict, table, etc.).
    """

    def __init__(
        self,
        identifier: str,
        source_name: str,
        data: Any,
        render_mode: str = "auto",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(identifier, source_name, tags, metadata)
        self.data = data
        self.render_mode = render_mode

    def render(self) -> RenderableType:
        if self.render_mode == "json":
            # Attempt to dump as JSON
            try:
                json_str = json.dumps(self.data, indent=2)
                content_renderable = Syntax(json_str, "json", theme="monokai", word_wrap=True)
            except (TypeError, ValueError):
                content_renderable = Pretty(self.data, expand_all=True)
        elif self.render_mode == "table" and isinstance(self.data, list):
            table = Table(box=box.HEAVY_HEAD, expand=True)
            if self.data:
                # Use keys from the first element as columns
                columns = list(self.data[0].keys())
                for col in columns:
                    table.add_column(col, style="data")

                for row in self.data:
                    row_str_values = [str(row.get(col, "")) for col in columns]
                    table.add_row(*row_str_values)
            else:
                table.add_row("No data available")
            content_renderable = table
        elif self.render_mode == "pretty":
            content_renderable = Pretty(self.data, expand_all=True)
        else:
            # Auto fallback
            try:
                json_str = json.dumps(self.data, indent=2)
                content_renderable = Syntax(json_str, "json", theme="monokai", word_wrap=True)
            except (TypeError, ValueError):
                content_renderable = Pretty(self.data, expand_all=True)

        panel_title = f"[bold]Data from: {self.source_name}[/]"
        panel_subtitle = f"[italic]{self.format_timestamp()}[/]"

        return Panel(
            content_renderable,
            title=panel_title,
            subtitle=panel_subtitle,
            border_style="data",
            box=box.ROUNDED,
            expand=True,
        )


# --------------------------------------------------------------------------
# ADVANCED CONSOLE CLASS WITH WRAPPERS
# --------------------------------------------------------------------------
class AdvancedConsole:
    """
    A rich-powered console wrapper that provides convenience methods
    for printing, logging, syntax highlighting, tables, progress,
    spinners, layouts, etc. Includes robust theming and live-updates.
    """

    def __init__(
        self,
        theme: Optional[ThemePreset] = None,
        auto_live_refresh: bool = True,
        log_level: int = logging.INFO,
    ):
        # Setup theme
        self.theme_preset = theme or ThemePreset.create_default()
        rich_theme = self.theme_preset.get_rich_theme()

        # Configure console
        self.console = Console(
            theme=rich_theme,
            log_time=True,
            log_path=False,
            record=True,  # allows capturing/exporting output
        )

        # Configure logging to use RichHandler
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)],
        )
        self.logger = logging.getLogger("AdvancedConsole")

        # Optional live-updating
        self.live: Optional[Live] = None
        self.auto_live_refresh = auto_live_refresh

        # A LayoutManager for advanced usage
        self.layout_manager = LayoutManager(self.console)

        # Store states (optional usage)
        self.states: Dict[str, BaseState] = {}

    # ------------------------------
    # Live management
    # ------------------------------
    def start_live(self) -> None:
        """Begin live-rendering mode."""
        if not self.live:
            self.live = Live(auto_refresh=self.auto_live_refresh, console=self.console)
        if not self.live.is_started:
            self.live.start()

    def stop_live(self) -> None:
        """Stop live-rendering mode."""
        if self.live and self.live.is_started:
            self.live.stop()

    def update_live(self, renderable: Optional[RenderableType] = None) -> None:
        """Update the live display with a new renderable or layout."""
        if not self.live or not self.live.is_started:
            return
        if renderable:
            self.live.update(renderable)
        else:
            # Default usage: re-render the layout manager
            self.layout_manager.update_dimensions()
            self.live.update(self.layout_manager.render())

    # ------------------------------
    # Basic Print (Rich Print)
    # ------------------------------
    def print(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        style: Union[str, Style, None] = None,
        justify: Optional[str] = None,
        overflow: Optional[str] = None,
        no_wrap: Optional[bool] = None,
        emoji: Optional[bool] = None,
        markup: Optional[bool] = True,
        highlight: Optional[bool] = False,
        width: Optional[int] = None,
        crop: bool = True,
        soft_wrap: Optional[bool] = None,
        new_line_start: bool = False,
    ) -> None:
        """
        An enhanced drop-in replacement for Python's built-in print.
        Supports markup, styling, auto-highlighting, and additional formatting options.
        
        Args:
            objects (positional args): Objects to log to the terminal.
            sep (str, optional): String to write between print data. Defaults to " ".
            end (str, optional): String to write at end of print data. Defaults to "\n".
            style (Union[str, Style], optional): A style to apply to output. Defaults to None.
            justify (str, optional): Justify method ("default", "left", "right", "center", or "full"). Defaults to None.
            overflow (str, optional): Overflow method ("ignore", "crop", "fold", or "ellipsis"). Defaults to None.
            no_wrap (Optional[bool], optional): Disable word wrapping. Defaults to None.
            emoji (Optional[bool], optional): Enable emoji codes. Defaults to None.
            markup (Optional[bool], optional): Enable markup. Defaults to True.
            highlight (Optional[bool], optional): Enable automatic highlighting. Defaults to False.
            width (Optional[int], optional): Width of output. Defaults to None.
            crop (bool, optional): Crop output to terminal width. Defaults to True.
            soft_wrap (Optional[bool], optional): Enable soft wrap mode. Defaults to None.
            new_line_start (bool, optional): Insert a new line at the start if output spans multiple lines. Defaults to False.
        """
        self.console.print(
            *objects,
            sep=sep,
            end=end,
            style=style,
            justify=justify,
            overflow=overflow,
            no_wrap=no_wrap,
            emoji=emoji,
            markup=markup,
            highlight=highlight,
            width=width,
            crop=crop,
            soft_wrap=soft_wrap,
            new_line_start=new_line_start,
        )

    # ------------------------------
    # Prompt/Input
    # ------------------------------
    def prompt(self, message: str, console_style: str = "info") -> str:
        """
        Prompt user for input with Rich formatting.
        Styles the prompt message using the provided console_style.
        """
        styled_message = f"[{console_style}]{message}[/{console_style}]"
        return Prompt.ask(styled_message)

    # ------------------------------
    # Syntax Highlighting
    # ------------------------------
    def show_syntax(
        self,
        code: str,
        lexer_name: str = "python",
        theme: str = "monokai",
        line_numbers: bool = True,
        word_wrap: bool = True,
    ) -> None:
        """Render syntax-highlighted code."""
        syntax = Syntax(code, lexer_name, theme=theme, line_numbers=line_numbers, word_wrap=word_wrap)
        self.print(syntax)

    # ------------------------------
    # Pretty Printing
    # ------------------------------
    def pretty_print(self, obj: Any, expand_all: bool = False) -> None:
        """Use Rich's Pretty to format Python objects in a more readable way."""
        self.print(Pretty(obj, expand_all=expand_all))

    # ------------------------------
    # Inspect
    # ------------------------------
    def inspect_object(self, obj: Any, methods: bool = True, private: bool = False, dunder: bool = False) -> None:
        """
        Display an object's attributes, optionally including methods, private, or dunder attributes.
        """
        self.console.inspect(obj, methods=methods, private=private, dunder=dunder)

    # ------------------------------
    # Logging
    # ------------------------------
    def log(self, message: str, level: int = logging.INFO) -> None:
        """Log a message using Python logging integrated with Rich."""
        self.logger.log(level, message)

    # ------------------------------
    # Traceback Rendering
    # ------------------------------
    def install_traceback(self, show_locals: bool = True, width: Optional[int] = None) -> None:
        """
        Install Rich-based traceback handler globally.
        Optionally show locals and set a maximum width.
        """
        traceback.install(
            console=self.console,
            show_locals=show_locals,
            width=width or self.console.width,
        )

    # ------------------------------
    # Tables
    # ------------------------------
    def print_table(
        self,
        data: List[Dict[str, Any]],
        title: str = "Data Table",
        columns_style: str = "data",
    ) -> None:
        """
        Construct a table from a list of dicts and print it.
        """
        if not data:
            self.print("[warning]No data to display[/warning]")
            return

        table = Table(title=title, box=box.SIMPLE_HEAD, expand=True)
        first_item = data[0]
        for col in first_item.keys():
            table.add_column(col, style=columns_style)

        for row in data:
            row_values = [str(row.get(col, "")) for col in first_item.keys()]
            table.add_row(*row_values)

        self.print(table)

    # ------------------------------
    # Progress Bars
    # ------------------------------
    def track_progress(
        self,
        tasks: Dict[str, float],
        description: str = "Processing",
        transient: bool = False,
    ) -> None:
        """
        Show multiple tasks in a progress bar set. The tasks dict
        maps `task_name` -> `completion_percentage`.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=transient,
            console=self.console,
        ) as progress:
            task_map = {}
            for name in tasks.keys():
                task_map[name] = progress.add_task(f"{description}: {name}", total=100.0)

            # Simulate updating progress
            for name, pct in tasks.items():
                progress.update(task_map[name], completed=pct * 100)

    # ------------------------------
    # Status/Spinner
    # ------------------------------
    def show_status(self, message: str, spinner: str = "dots", delay: float = 2.0) -> None:
        """
        Show a temporary status spinner with a message for some operation.
        delay is just to simulate a long-running process in this demo.
        """
        import time

        with Status(message, spinner=spinner, console=self.console) as status:
            time.sleep(delay)

    # ------------------------------
    # Tree
    # ------------------------------
    def print_tree(self, data: Dict[str, Any], title: str = "Tree") -> None:
        """
        Render hierarchical dictionary data as a tree structure.
        """
        tree = Tree(f"[bold]{title}[/bold]", guide_style="dim")
        self._build_tree(data, tree)
        self.print(tree)

    def _build_tree(self, data: Any, parent) -> None:
        """Recursively build tree nodes from nested dict/list structures."""
        if isinstance(data, dict):
            for key, value in data.items():
                branch = parent.add(f"[bold]{key}[/bold]")
                self._build_tree(value, branch)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                branch = parent.add(f"[bold][{idx}][/bold]")
                self._build_tree(item, branch)
        else:
            parent.add(str(data))

    # ------------------------------
    # Columns
    # ------------------------------
    def print_columns(self, items: List[str], width: Optional[int] = None) -> None:
        """
        Arrange multiple items in balanced columns.
        """
        renderables = [Text(item) for item in items]
        self.print(Columns(renderables, width=width))

    # ------------------------------
    # Markdown Rendering
    # ------------------------------
    def print_markdown(self, md_text: str, code_theme: str = "monokai") -> None:
        """
        Directly render markdown-formatted text to the console.
        """
        md = Markdown(md_text, code_theme=code_theme)
        self.print(md)

    # ------------------------------
    # Panels
    # ------------------------------
    def print_panel(self, content: Any, title: str = "", style_name: str = "info") -> None:
        """
        Enclose content in a Rich Panel with optional title and style.
        """
        panel_style = style_name if style_name in self.theme_preset.styles else "info"
        panel = Panel(
            Text(str(content)),
            title=title,
            border_style=panel_style,
            box=box.ROUNDED,
            expand=False,
        )
        self.print(panel)

    # ------------------------------
    # Live Updates
    # ------------------------------
    def live_render(self, renderable: RenderableType) -> None:
        """
        Dynamically update the live console with the given renderable.
        This method is especially useful if you manage the live context manually.
        """
        if self.live and self.live.is_started:
            self.live.update(renderable)
        else:
            self.print(renderable)

    # ------------------------------
    # Layout Management
    # ------------------------------
    def get_layout_manager(self) -> LayoutManager:
        """Expose the LayoutManager for custom usage."""
        return self.layout_manager

    # ------------------------------
    # JSON Rendering
    # ------------------------------
    def print_json_data(self, data: Any, highlight: bool = True) -> None:
        """
        Pretty-print JSON data with optional syntax highlighting.
        """
        try:
            json_str = json.dumps(data, indent=2)
            if highlight:
                self.print(Syntax(json_str, "json", theme="monokai", word_wrap=True))
            else:
                self.print(json_str)
        except (TypeError, ValueError):
            self.print("[error]Invalid JSON data[/error]")

    # ------------------------------
    # Rules (Horizontal Dividers)
    # ------------------------------
    def draw_rule(self, title: str = "", style_name: str = "info") -> None:
        """
        Draw a horizontal rule across the console, optionally with a title.
        """
        if style_name not in self.theme_preset.styles:
            style_name = "info"
        self.print(Rule(title=title, style=style_name))

    # ------------------------------
    # Emoji Support
    # ------------------------------
    def print_emoji(self, emoji_name: str) -> None:
        """
        Print an emoji by its colon-delimited name (e.g., :smiley:).
        """
        self.print(Text.from_markup(emoji_name))

    # ------------------------------
    # Exporting Capabilities
    # ------------------------------
    def export_text(self) -> str:
        """
        Return all console output captured so far as plain text.
        """
        return self.console.export_text()

    def export_html(self, clear: bool = False) -> str:
        """
        Return console output as HTML.
        """
        return self.console.export_html(clear=clear)

    def export_svg(self, clear: bool = False) -> str:
        """
        Return console output as an SVG document.
        """
        return self.console.export_svg(clear=clear)

    # ------------------------------
    # Alternate Screen
    # ------------------------------
    def use_alternate_screen(self, renderable: RenderableType, delay: float = 2.0) -> None:
        """
        Temporarily switch to an alternate screen, display something,
        then revert back to main screen.
        """
        import time
        with self.console.screen():
            self.print(renderable)
            time.sleep(delay)

    # ------------------------------
    # Overflow Control
    # ------------------------------
    def safe_print(self, text_str: str, max_width: Optional[int] = None) -> None:
        """
        Print text, wrapping or truncating if it exceeds the given max_width.
        If max_width is None, uses console width. 
        """
        terminal_width = max_width or self.console.width
        wrapped = Text(text_str).wrap(terminal_width, expand=False)
        self.print(wrapped)

    # ------------------------------
    # Capturing Output
    # ------------------------------
    def capture_output(self) -> str:
        """
        Capture what’s printed to the console within a context. Example usage:
            with console.capture_capture_output() as captured:
                console.print("Hello")
                out = captured.get()
        """
        return self.console.export_text()

    # Context manager style capture:
    def capture_capture_output(self):
        """
        Return a context manager that captures console output within its scope.
        Example:
            with advanced_console.capture_capture_output() as capture:
                advanced_console.print("Inside capture")
            result = capture.get()
        """
        return self.console.capture()

    # ------------------------------
    # Low-Level Output
    # ------------------------------
    def raw_out(self, text_str: str) -> None:
        """
        Minimal formatting for raw output to stdout (bypasses markup).
        """
        self.console.out(text_str)

    # ------------------------------
    # Paging (long output)
    # ------------------------------
    def page_output(self, content: str) -> None:
        """
        Render long text with a pager so users can scroll comfortably.
        """
        from rich.pager import Pager
        Pager(self.console, content)

    # ------------------------------
    # Styling / Themes
    # ------------------------------
    def set_theme(self, theme_preset: ThemePreset) -> None:
        """
        Switch the console to a new theme dynamically.
        """
        self.console.set_theme(theme_preset.get_rich_theme())

    # ----------------------------------------------------------------------
    # State-based APIs (OPTIONAL, for demonstration of advanced usage)
    # ----------------------------------------------------------------------
    def set_state(self, identifier: str, state: BaseState) -> None:
        """Register or update a display state by identifier."""
        self.states[identifier] = state
        self.refresh_states()

    def get_state(self, identifier: str) -> Optional[BaseState]:
        """Retrieve a previously stored state."""
        return self.states.get(identifier)

    def remove_state(self, identifier: str) -> None:
        """Remove a stored state."""
        if identifier in self.states:
            del self.states[identifier]
        self.refresh_states()

    def refresh_states(self) -> None:
        """
        Redraw all states in chronological order, if live mode is active.
        Otherwise, just print them once.
        """
        sorted_states = sorted(self.states.values(), key=lambda s: s.first_timestamp)
        if self.live and self.live.is_started:
            # We'll group them in a vertical layout using a Group.
            group = Group(*[st.render() for st in sorted_states])
            self.update_live(group)
        else:
            # Print them normally (one-time)
            for st in sorted_states:
                self.print(st.render())


# ------------------------------------------------------------------------------
# EXAMPLE GLOBAL INSTANCE (similar to print_handler in old code)
# ------------------------------------------------------------------------------
advanced_console = AdvancedConsole()
