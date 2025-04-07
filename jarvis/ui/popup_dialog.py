"""
A customizable popup dialog for displaying conversation history.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QColor, QTextCursor, QAction
from PyQt6.QtWidgets import (
    QDialog,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QApplication,
)
from jarvis.ui.print_handler import advanced_console as console
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich import box
console = Console()

from jarvis.ui.interfaces import IUIController


class PopupDialog(QDialog):
    def __init__(self, controller: IUIController, parent=None) -> None:
        """
        Initialize the popup dialog.
        
        Args:
            controller: The UI controller that handles business logic
            parent: The parent widget
        """
        super().__init__(parent)
        self.controller = controller
        self.conversation_history = []  # List of tuples (sender, message)
        self.oldPos = None
        self.initUI()
        
        # Connect controller signals
        if hasattr(self.controller, 'update_assistant_message'):
            self.controller.update_assistant_message.connect(
                lambda text: self.append_message(text, sender="Assistant")
            )
        
        if hasattr(self.controller, 'stream_assistant_chunk'):
            self.controller.stream_assistant_chunk.connect(
                self.stream_assistant_update
            )
            
        if hasattr(self.controller, 'clear_assistant_message'):
            self.controller.clear_assistant_message.connect(
                self.handle_clear
            )
            
        if hasattr(self.controller, 'transcription_result'):
            self.controller.transcription_result.connect(
                self.custom_update
            )

    def initUI(self) -> None:
        # Frameless and translucent
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.resize(1000, 700)
        self.right_align()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(27, 27, 0, 15)  # Remove right margin
        layout.setSpacing(10)

        clear_button = QPushButton("Clear")
        clear_button.setFixedSize(60, 30)
        clear_button.clicked.connect(self.handle_clear)
        clear_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #00dd00;
                font-size: 20px;
                border: none;
                border-radius: 15px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #00dd00;
                color: #000000;
            }
            """
        )
        close_button = QPushButton("×")
        close_button.setFixedSize(30, 30)
        close_button.clicked.connect(self.handle_close)
        close_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #00dd00;
                font-size: 20px;
                border: none;
                border-radius: 15px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #00dd00;
                color: #000000;
            }
            """
        )

        button_layout = QHBoxLayout()
        button_layout.addWidget(clear_button)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)
        
        self.model_combo = QComboBox(self)
        self.model_combo.setMinimumWidth(300)
        self.model_combo.setStyleSheet(
            """
            QComboBox {
                background-color: rgba(19, 19, 20, 0.9);
                color: #00dd00;
                border: 1px solid #00dd00;
                border-radius: 5px;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(19, 19, 20, 0.9);
                color: #00dd00;
                selection-background-color: rgba(0, 221, 0, 0.3);
            }
            """
        )
        
        # Get models from controller
        model_ids = self.controller.get_model_list()
        self.model_combo.addItems(model_ids)
        self.current_model = model_ids[0]
        self.model_combo.setCurrentIndex(0)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
    
        self.text_edit = QTextEdit(self)
        self.text_edit.setAcceptRichText(True)
        self.text_edit.setAutoFormatting(QTextEdit.AutoFormattingFlag.AutoAll)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("JetBrains Mono", 13))
        self.text_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.text_edit.setStyleSheet(
            """
            QTextEdit {
                background-color: rgba(19,19,20, 0.9);
                color: #00dd00;
                border: none;
                border-radius: 10px;
                padding: 15px;
                selection-background-color: rgba(0, 221, 0, 0.3);
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(19,19,20, 0.7);
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #666666;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #888888;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )
        layout.addWidget(self.text_edit)

        self.setStyleSheet(
            """
            PopupDialog {
                background: transparent;
                border: 1px solid #00dd00;
                border-radius: 15px;
            }
            """
        )

    def right_align(self) -> None:
        """Align the dialog to the right side of the screen at full height."""
        screen = QApplication.primaryScreen().geometry()
        self.resize(1000, screen.height())
        self.move(screen.width() - self.width(), 0)

    def center(self) -> None:
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def left(self) -> None:
        """Move the dialog to the left side of the screen, nearly full height."""
        screen = QApplication.primaryScreen().geometry()
        self.resize(screen.width() // 2, int(screen.height() * 0.9))
        self.move(0, 0)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(27, 27, 27, 235))
        painter.setPen(QColor("#00dd00"))
        painter.drawRoundedRect(self.rect(), 15, 15)
        painter.setBrush(QColor(0, 0, 0, 30))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, 1, 1), 15, 15)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.oldPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event) -> None:
        if self.oldPos:
            delta = event.globalPosition().toPoint() - self.oldPos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event) -> None:
        self.oldPos = None

    def handle_close(self) -> None:
        self.controller.close_dialog()
        self.hide()

    def handle_clear(self) -> None:
        self.controller.clear_dialog()
        self.conversation_history = []
        self.text_edit.clear()

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)

    def set_font_size(self, size: int) -> None:
        if size < 1:
            console.log("[red]Font size must be a positive integer.[/red]")
            return
        current_font = self.text_edit.font()
        current_font.setPointSize(size)
        self.text_edit.setFont(current_font)

    def on_model_changed(self, model_id: str) -> None:
        """Handle model selection change."""
        self.current_model = model_id
        self.controller.set_current_model(model_id)

    def custom_update(self, text: str) -> None:
        """
        Handle user input and generate a response.
        
        Args:
            text: The user's input text
        """
        # Append the user's message
        self.append_message(text, sender="User")
        
        # Delegate processing to the controller
        self.controller.process_user_input(text)
        
        if not self.isVisible():
            self.show()
        self.activateWindow()
        self.setWindowTitle("AI")
        self.text_edit.setFocus()
        self.raise_()
        self.repaint()
        self.setVisible(True)

    def append_message(self, message: str, sender: str = "Assistant") -> None:
        """
        Add a message to the conversation and refresh the display.
        
        Args:
            message: The message text
            sender: The sender name
        """
        self.conversation_history.append((sender, message))
        self._refresh_markdown()
        self._scroll_to_bottom()

    def stream_assistant_update(self, partial_content: str) -> None:
        """
        Update the latest assistant message with streaming content.
        
        Args:
            partial_content: The partial message content
        """
        if not self.conversation_history or self.conversation_history[-1][0] != "Assistant":
            self.conversation_history.append(("Assistant", partial_content))
        else:
            sender, _ = self.conversation_history[-1]
            self.conversation_history[-1] = (sender, partial_content)
        self._refresh_markdown()
        self._scroll_to_bottom()

    def _refresh_markdown(self) -> None:
        """Rebuild the entire conversation as Markdown."""
        markdown_output = ""
        for sender, message in self.conversation_history:
            markdown_output += f"**{sender}**:\n\n{message}\n\n---\n\n"
        self.text_edit.setMarkdown(markdown_output)

    def _scroll_to_bottom(self) -> None:
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.text_edit.ensureCursorVisible()