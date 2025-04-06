"""
Settings for the application.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    USE_OFFICIAL_OPENAI = False

    GLOBAL_DEFAULT_MODEL = "gpt-4o"
    GLOBAL_DEFAULT_AUDIO_MODEL = "whisper-1"
    GLOBAL_DEFAULT_IMAGE_MODEL = "dall-e-3"
    GLOBAL_DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError(
            "API key not found. Please set the 'OPENAI_API_KEY' environment variable."
        )
