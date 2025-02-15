"""
A customizable popup dialog for displaying conversation history.
"""

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont, QPainter, QColor, QTextCursor
from PyQt5.QtWidgets import (
    QDialog,
    QTextEdit,
    QDesktopWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
)

from commands.command_library import CommandLibrary


class PopupDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.conversation_history = []  # List of tuples (sender, message)
        self.oldPos = None
        self.initUI()

    def initUI(self) -> None:
        self.cmd_library = CommandLibrary()

        # Frameless and translucent
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.resize(1000, 700)
        self.center()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(27, 27, 27, 15)
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
        # Populate the model_combo with available models
        try:
            from ai.openai_wrapper import OpenAIWrapper
            ai_wrapper = OpenAIWrapper()
            models = ai_wrapper.list_models()
            model_ids = []
            if models:
                for m in models:
                    if isinstance(m, dict) and "id" in m:
                        model_ids.append(m["id"])
                    elif hasattr(m, "id"):
                        model_ids.append(m.id)
            if not model_ids:
                model_ids = ["GPT-40"]
        except Exception as e:
            model_ids = ["GPT-40"]
        # Ensure default model is GPT-40
        if "GPT-40" not in model_ids:
            model_ids.insert(0, "GPT-40")
        self.model_combo.addItems(model_ids)
        self.model_combo.setCurrentIndex(self.model_combo.findText("GPT-40"))
        self.current_model = self.model_combo.currentText()
        self.model_combo.currentTextChanged.connect(lambda text: setattr(self, "current_model", text))
    
        self.text_edit = QTextEdit(self)
        self.text_edit.setAcceptRichText(True)
        self.text_edit.setAutoFormatting(QTextEdit.AutoAll)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("JetBrains Mono", 13))
        self.text_edit.setFocusPolicy(Qt.StrongFocus)
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

    def center(self) -> None:
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def left(self) -> None:
        """Move the dialog to the left side of the screen, nearly full height."""
        screen = QDesktopWidget().screenGeometry()
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
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event) -> None:
        if self.oldPos:
            delta = event.globalPos() - self.oldPos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()

    def mouseReleaseEvent(self, event) -> None:
        self.oldPos = None

    def handle_close(self) -> None:
        self.hide()

    def handle_clear(self) -> None:
        self.conversation_history = []
        self.text_edit.clear()

    def closeEvent(self, event) -> None:
        event.ignore()
        self.hide()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)

    def set_font_size(self, size: int) -> None:
        if size < 1:
            print("Font size must be a positive integer.")
            return
        current_font = self.text_edit.font()
        current_font.setPointSize(size)
        self.text_edit.setFont(current_font)

    def custom_update(self, text: str) -> None:
        # Append the user's message.
        self.append_message(text, sender="User")

        # Process the text (this may call stream_assistant_update if streaming).
        response = self.cmd_library.process_text(text, self)

        if response:
            if "query" not in text.lower():
                self.append_message(response, sender="Assistant")

            if not self.isVisible():
                self.show()
            self.activateWindow()
            self.setWindowTitle("AI")
            self.text_edit.setFocus()
            self.raise_()
            self.repaint()
            self.setVisible(True)

    def append_message(self, message: str, sender: str = "Assistant") -> None:
        """Add a message to the conversation and refresh the display."""
        self.conversation_history.append((sender, message))
        self._refresh_markdown()
        self._scroll_to_bottom()

    def stream_assistant_update(self, partial_content: str) -> None:
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
        self.text_edit.moveCursor(QTextCursor.End)
        self.text_edit.ensureCursorVisible()


