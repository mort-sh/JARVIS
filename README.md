# My App

## Overview

A desktop tool integrating OpenAI APIs for transcription, code generation, and conversational AI with a PyQt5-based UI.

## Features

- Audio transcription using OpenAI Whisper.
- Code generation and refactoring with GPT models.
- Real-time conversational AI.
- Customizable UI with adjustable font size and frameless design.

## File Structure

- **main.py**: Entry point for the application.
- **ai**: OpenAI API wrapper.
- **commands**: Command processing library.
- **services**: Audio transcription and keyboard hooks.
- **ui**: PyQt5-based user interface.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your OpenAI API key to `.env`:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## License

MIT License.
