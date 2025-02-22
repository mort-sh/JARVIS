"""
Rich-based terminal printing library with advanced formatting capabilities.

This module provides a comprehensive set of tools for terminal-based visualization
with support for themes, layouts, progress tracking, and various content types.
"""

import datetime
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import rich
import rich.pretty
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console, ConsoleOptions, Group, RenderableType
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.spinner import Spinner
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich_printer")

class DisplayMode(Enum):
    """Available display modes for content rendering."""
    FULL = "full"  # Full detail with all formatting
    COMPACT = "compact"  # Minimal formatting
    RAW = "raw"  # No formatting, just content

@dataclass
class ThemePreset:
    """Predefined color and style combinations."""
    name: str
    styles: Dict[str, Style]
    
    @classmethod
    def create_default(cls) -> 'ThemePreset':
        """Create the default theme preset."""
        return cls(
            name="default",
            styles={
                "content": Style(color="blue"),
                "action": Style(color="green"),
                "data": Style(color="magenta"),
                "error": Style(color="red"),
                "success": Style(color="green"),
                "warning": Style(color="yellow"),
                "info": Style(color="cyan"),
            }
        )

class LayoutManager:
    """Manages terminal layout and formatting constraints."""
    
    def __init__(self, console: Console):
        self.console = console
        self._update_dimensions()
        
    def _update_dimensions(self) -> None:
        """Update stored terminal dimensions."""
        self.width = min(100, self.console.width)
        self.height = self.console.height
        
    def get_content_width(self, padding: int = 2) -> int:
        """Calculate available content width accounting for padding."""
        return max(20, self.width - (padding * 2))

class BaseRenderer(ABC):
    """Abstract base class for content renderers."""
    
    def __init__(self, theme: ThemePreset, layout: LayoutManager):
        self.theme = theme
        self.layout = layout
        
    @abstractmethod
    def render(self, content: Any) -> RenderableType:
        """Render content into a Rich renderable."""
        pass
    
    def create_panel(
        self,
        content: RenderableType,
        title: str,
        border_style: Union[str, Style],
        subtitle: Optional[str] = None,
    ) -> Panel:
        """Create a styled panel with the given content."""
        return Panel(
            content,
            title=title,
            subtitle=subtitle,
            title_align="left",
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            width=self.layout.width,
            padding=(1, 2),
        )

class MarkdownRenderer(BaseRenderer):
    """Renders markdown content with syntax highlighting."""
    
    def render(self, content: str) -> RenderableType:
        return Markdown(
            content,
            code_theme="monokai",
            inline_code_theme="monokai",
        )

class JsonRenderer(BaseRenderer):
    """Renders JSON data with syntax highlighting."""
    
    def render(self, content: Any) -> RenderableType:
        try:
            if isinstance(content, str):
                # Try to parse if it's a string
                content = json.loads(content)
            json_str = json.dumps(content, indent=2)
            return Syntax(
                json_str,
                "json",
                theme="monokai",
                line_numbers=False,
                word_wrap=True,
            )
        except (json.JSONDecodeError, TypeError):
            return rich.pretty.Pretty(content, expand_all=True)

class TableRenderer(BaseRenderer):
    """Renders tabular data with customizable styling."""
    
    def render(self, data: List[Dict[str, Any]]) -> RenderableType:
        if not data:
            return Text("No data")
            
        table = Table(
            show_header=True,
            header_style="bold",
            box=box.ROUNDED,
            expand=True,
        )
        
        # Add columns based on first row
        columns = list(data[0].keys())
        for col in columns:
            table.add_column(col)
            
        # Add rows
        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in columns])
            
        return table

class DisplayState(BaseModel):
    """Base model for display states with enhanced metadata."""
    
    identifier: str
    source_name: str
    first_timestamp: datetime.datetime
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def format_timestamp(self) -> str:
        """Format timestamp for display."""
        local_time = self.first_timestamp.astimezone()
        return local_time.strftime("%I:%M:%S %p").lstrip("0").rjust(11)
    
    @abstractmethod
    def render(self, renderer: BaseRenderer) -> RenderableType:
        """Render the state's content."""
        pass

class ContentState(DisplayState):
    """State for markdown/text content with enhanced formatting."""
    
    content: str = ""
    format_type: str = "markdown"  # or "plain"
    
    def render(self, renderer: MarkdownRenderer) -> RenderableType:
        if self.format_type == "markdown":
            content = renderer.render(self.content)
        else:
            content = Text(self.content)
            
        return renderer.create_panel(
            content,
            f"[bold]Source: {self.source_name}[/]",
            renderer.theme.styles["content"],
            f"[italic]{self.format_timestamp()}[/]"
        )

class ActionState(DisplayState):
    """State for action/operation tracking with progress support."""
    
    name: str
    args: dict
    result: Optional[str] = None
    is_error: bool = False
    is_complete: bool = False
    progress: float = 0.0  # 0-1 for progress tracking
    
    def render(self, renderer: BaseRenderer) -> RenderableType:
        # Create progress bar if incomplete
        if not self.is_complete:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                auto_refresh=False,
            )
            task = progress.add_task(
                self.name.replace("_", " ").title(),
                total=100,
            )
            progress.update(task, completed=int(self.progress * 100))
            content = progress
        else:
            # Show completion status
            status = "❌" if self.is_error else "✅"
            style = "error" if self.is_error else "success"
            content = Text(
                f"{status} {self.name.replace('_', ' ').title()}",
                style=renderer.theme.styles[style],
            )
            
            # Add result if present
            if self.result:
                content = Group(
                    content,
                    Text("\n" + self.result, style=renderer.theme.styles[style]),
                )
        
        return renderer.create_panel(
            content,
            f"[bold]Source: {self.source_name}[/]",
            renderer.theme.styles["action"],
            f"[italic]{self.format_timestamp()}[/]"
        )

class DataState(DisplayState):
    """State for structured data with flexible rendering options."""
    
    data: Any
    render_mode: str = "auto"  # auto, json, table, or pretty
    
    def render(self, renderer: JsonRenderer) -> RenderableType:
        if self.render_mode == "table" and isinstance(self.data, list):
            table_renderer = TableRenderer(renderer.theme, renderer.layout)
            content = table_renderer.render(self.data)
        else:
            content = renderer.render(self.data)
            
        return renderer.create_panel(
            content,
            f"[bold]Data from: {self.source_name}[/]",
            renderer.theme.styles["data"],
            f"[italic]{self.format_timestamp()}[/]"
        )

class PrintHandler:
    """
    Advanced terminal printing handler with rich formatting support.
    
    Features:
    - Theme customization
    - Multiple display modes
    - Progress tracking
    - Layout management
    - Logging integration
    - Custom renderers
    """
    
    def __init__(
        self,
        theme: Optional[ThemePreset] = None,
        display_mode: DisplayMode = DisplayMode.FULL,
        console: Optional[Console] = None,
    ):
        self.theme = theme or ThemePreset.create_default()
        self.display_mode = display_mode
        self.console = console or Console()
        self.layout = LayoutManager(self.console)
        
        # Initialize renderers
        self.markdown_renderer = MarkdownRenderer(self.theme, self.layout)
        self.json_renderer = JsonRenderer(self.theme, self.layout)
        
        self.live: Optional[Live] = None
        self.states: Dict[str, DisplayState] = {}
        
    def start(self) -> None:
        """Start live display updates."""
        if not self.live:
            self.live = Live(
                console=self.console,
                auto_refresh=True,
                vertical_overflow="visible",
            )
        if not self.live.is_started:
            self.live.start()
            
    def stop(self) -> None:
        """Stop live display updates."""
        if self.live and self.live.is_started:
            self.live.stop()
            
    def clear(self) -> None:
        """Clear all display states."""
        self.states.clear()
        self.update_display()
        
    def update_display(self) -> None:
        """Update the live display with current states."""
        if not self.live or not self.live.is_started:
            return
            
        # Sort states by timestamp
        sorted_states = sorted(
            self.states.values(),
            key=lambda s: s.first_timestamp
        )
        
        if not sorted_states:
            return
            
        # Create main table
        table = Table(
            show_header=True,
            box=box.SIMPLE,
            expand=True,
            padding=(0, 1),
        )
        
        # Add columns based on display mode
        if self.display_mode == DisplayMode.FULL:
            table.add_column("Time", style="dim", width=11)
            table.add_column("Source", style="bold")
            table.add_column("Type", width=8)
            table.add_column("Content", ratio=1)
        else:
            table.add_column("Content", ratio=1)
            
        # Add rows for each state
        for state in sorted_states:
            if self.display_mode == DisplayMode.FULL:
                if isinstance(state, ContentState):
                    renderer = self.markdown_renderer
                    type_icon = "📝"
                elif isinstance(state, ActionState):
                    renderer = self.markdown_renderer
                    type_icon = "⚡"
                else:  # DataState
                    renderer = self.json_renderer
                    type_icon = "📊"
                    
                table.add_row(
                    state.format_timestamp(),
                    state.source_name,
                    type_icon,
                    state.render(renderer),
                )
            else:
                # Compact mode - just show content
                if isinstance(state, ContentState):
                    content = Text(state.content)
                elif isinstance(state, ActionState):
                    status = "❌" if state.is_error else "✅"
                    content = Text(f"{status} {state.name}")
                else:
                    content = Text(str(state.data))
                table.add_row(content)
                
        # Create main panel
        panel = Panel(
            table,
            title="[bold]Activity Log[/]",
            border_style=self.theme.styles["info"],
            box=box.ROUNDED,
            padding=(0, 1),
        )
        
        self.live.update(panel)
        
    def on_content_update(
        self,
        identifier: str,
        source_name: str,
        content: Any,
        format_type: str = "markdown",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Add or update content with enhanced metadata."""
        state = ContentState(
            identifier=identifier,
            source_name=source_name,
            first_timestamp=datetime.datetime.now(),
            content=str(content),
            format_type=format_type,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.states[identifier] = state
        self.update_display()
        
    def on_action_start(
        self,
        identifier: str,
        source_name: str,
        action_name: str,
        args: dict,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Start tracking a new action."""
        state = ActionState(
            identifier=identifier,
            source_name=source_name,
            first_timestamp=datetime.datetime.now(),
            name=action_name,
            args=args,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.states[identifier] = state
        self.update_display()
        
    def on_action_progress(
        self,
        identifier: str,
        progress: float,
    ) -> None:
        """Update action progress (0-1)."""
        if identifier in self.states:
            state = self.states[identifier]
            if isinstance(state, ActionState):
                state.progress = max(0.0, min(1.0, progress))
                self.update_display()
                
    def on_action_complete(
        self,
        identifier: str,
        result: Optional[str] = None,
        is_error: bool = False,
    ) -> None:
        """Mark action as complete with optional result."""
        if identifier in self.states:
            state = self.states[identifier]
            if isinstance(state, ActionState):
                state.is_complete = True
                state.is_error = is_error
                state.result = result
                self.update_display()
                
    def on_data_update(
        self,
        identifier: str,
        source_name: str,
        data: Any,
        render_mode: str = "auto",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Add or update data state."""
        state = DataState(
            identifier=identifier,
            source_name=source_name,
            first_timestamp=datetime.datetime.now(),
            data=data,
            render_mode=render_mode,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.states[identifier] = state
        self.update_display()

# Create global instance
print_handler = PrintHandler()
print_handler.start()
