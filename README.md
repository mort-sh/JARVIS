# Jarvis Desktop Assistant

## Overview

A powerful desktop assistant that combines voice transcription, AI-powered command processing, and a modern PyQt5-based UI. The application provides seamless integration with OpenAI's APIs for transcription, code generation, and conversational AI, all accessible through an elegant floating interface.

## Features

- **Voice Commands**: Record and transcribe voice using OpenAI's Whisper model
  - Hold Right Shift to record, release to transcribe
  - Hold Ctrl + Right Shift to transcribe and auto-type the result
  
- **AI Command Processing**:
  - Code generation and refactoring with GPT models
  - Natural language queries with streaming responses
  - Code formatting and clipboard integration
  
- **Modern UI**:
  - Floating, translucent dialog that stays on top
  - Adjustable font size (+ and - keys)
  - Markdown rendering for conversation history
  - Model selection dropdown
  
- **Keyboard Shortcuts**:
  - Right Shift: Start/stop voice recording
  - Ctrl + Right Shift: Record and auto-type
  - Plus (+): Increase font size
  - Minus (-): Decrease font size
  - Escape: Hide dialog

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd jarvis
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Unix/MacOS
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install PyQt5 keyboard numpy sounddevice soundfile whisper openai python-dotenv pyperclip
   ```

4. Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Starting the Application

```bash
python main.py
```

### Command Types

1. **Code Commands**:
   - "optimize [code]" - Improve code efficiency
   - "refactor [code]" - Restructure code
   - "implement [description]" - Generate new code
   - "fix [code]" - Debug and correct code issues

2. **Format Commands**:
   - "format as code" - Wrap clipboard content in code blocks
   - "wrap in code" - Format text as code

3. **Query Commands**:
   - "explain [topic]" - Get detailed explanations
   - "how to [task]" - Get step-by-step instructions
   - "what is [concept]" - Get definitions and explanations

4. **System Commands**:
   - "exit" or "quit" - Close the application
   - "talk [text]" - Simulate typing the text

### Voice Recording

1. Hold Right Shift to start recording
2. Release Right Shift to stop and process the recording
3. For direct typing of transcription:
   - Hold Ctrl + Right Shift while recording
   - Release to auto-type the transcribed text

## Configuration

The application can be configured through `config/settings.py`:

```python
# OpenAI API Configuration
USE_OFFICIAL_OPENAI = False  # Toggle between official/custom client

# Model Defaults
GLOBAL_DEFAULT_MODEL = "gpt-4o"
GLOBAL_DEFAULT_AUDIO_MODEL = "whisper-1"
GLOBAL_DEFAULT_IMAGE_MODEL = "dall-e-3"
GLOBAL_DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
```

## Project Structure

```
jarvis/
├── ai/
│   ├── __init__.py
│   └── openai_wrapper.py    # OpenAI API integration
├── commands/
│   ├── __init__.py
│   └── command_library.py   # Command processing system
├── config/
│   ├── __init__.py
│   └── settings.py          # Application configuration
├── services/
│   ├── __init__.py
│   └── transcription_worker.py  # Audio processing
├── ui/
│   ├── __init__.py
│   ├── popup_dialog.py      # Main UI component
│   └── workers/
│       ├── __init__.py
│       └── stream_worker.py  # Streaming response handler
├── main.py                  # Application entry point
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Follow the code style:
   - Use type hints for all function parameters and returns
   - Include docstrings for classes and methods
   - Follow PEP 8 guidelines
   - Add error handling with appropriate logging
4. Test your changes thoroughly
5. Submit a pull request with a clear description of your changes

## Troubleshooting

1. **Audio Recording Issues**:
   - Ensure your microphone is properly connected
   - Check system permissions for microphone access
   - Verify sounddevice and soundfile are properly installed

2. **OpenAI API Issues**:
   - Verify your API key is correctly set in .env
   - Check your API quota and limits
   - Ensure internet connectivity

3. **UI Issues**:
   - For display problems, ensure PyQt5 is properly installed
   - For font issues, verify JetBrains Mono font is available

## License

MIT License. See LICENSE file for details.

## Acknowledgments

- OpenAI for their powerful API and models
- PyQt5 for the UI framework
- The Whisper team for the speech recognition model
