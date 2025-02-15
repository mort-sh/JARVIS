"""
A convenient wrapper around various OpenAI API endpoints,
using either the official package or a custom client.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

# Toggle if needed
USE_OFFICIAL_OPENAI = False

# Default model settings
GLOBAL_DEFAULT_MODEL = "gpt-4o"
GLOBAL_DEFAULT_AUDIO_MODEL = "whisper-1"
GLOBAL_DEFAULT_IMAGE_MODEL = "dall-e-3"
GLOBAL_DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

if USE_OFFICIAL_OPENAI:
    import openai
else:
    # Replace with your custom class or library.
    from openai import OpenAI


class OpenAIWrapper:
    """
    Wrapper around OpenAI API endpoints.
    """

    def __init__(self) -> None:
        load_dotenv()
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set the 'OPENAI_API_KEY' environment variable."
            )

        if USE_OFFICIAL_OPENAI:
            openai.api_key = self.api_key
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = GLOBAL_DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """
        Generates a simple chat completion.
        """
        try:
            if USE_OFFICIAL_OPENAI:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"An error occurred in 'complete': {e}")
            return None

    def send_prompt(
        self,
        prompt: Optional[str] = None,
        model: str = GLOBAL_DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 400,
        messages: Optional[List[Dict[str, str]]] = None,
        role: str = "user",
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a prompt to the ChatCompletion API with streaming enabled.
        """
        try:
            if messages is None:
                messages = [{"role": role, "content": prompt}]
            logger.debug(f"Sending chat completion with messages: {messages}")

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            assistant_reply = ""
            for chunk in response:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content is not None:
                    assistant_reply += content
                    if stream_callback:
                        stream_callback(content)

            return {"assistant_reply": assistant_reply, "response": response}
        except Exception as e:
            logger.error(f"Exception in send_prompt: {e}", exc_info=True)
            return {"assistant_reply": "", "response": None}

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        model: str = GLOBAL_DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Optional[str]:
        """
        Generates a chat completion that follows a specific JSON schema.
        """
        code_schema = {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        }

        try:
            if USE_OFFICIAL_OPENAI:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "code_response",
                            "strict": True,
                            "schema": code_schema,
                        },
                    },
                )

            response_content = response.choices[0].message.content
            parsed_json = json.loads(response_content)
            return parsed_json["code"]
        except json.JSONDecodeError:
            logger.error("Failed to parse response content as JSON.")
            return None
        except KeyError:
            logger.error("The key 'code' was not found in the response content.")
            return None
        except Exception as e:
            logger.error(f"An error occurred in 'structured_output': {e}")
            return None

    def code(
        self,
        prompt_usr: str,
        prompt_sys: Optional[str] = None,
        model: str = GLOBAL_DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Optional[str]:
        """
        Generates code strictly adhering to a JSON schema.
        """
        if prompt_sys is None:
            prompt_sys = (
                "You are a helpful assistant proficient in generating clean, "
                "efficient, and well-documented code. Respond only with the code "
                "and any necessary comments, without additional explanations."
            )

        messages = [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_usr},
        ]

        return self.structured_output(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )

    def list_models(self) -> Optional[List[Any]]:
        """
        Lists all available models.
        """
        try:
            if USE_OFFICIAL_OPENAI:
                response = openai.Model.list()
                return response["data"]
            else:
                response = self.client.models.list()
                return response.data
        except Exception as e:
            logger.error(f"An error occurred in 'list_models': {e}")
            return None

    def retrieve_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific model by its ID.
        """
        try:
            if USE_OFFICIAL_OPENAI:
                response = openai.Model.retrieve(model_id)
            else:
                response = self.client.models.retrieve(model=model_id)
            return response
        except Exception as e:
            logger.error(f"An error occurred in 'retrieve_model': {e}")
            return None

    def transcribe_audio(
        self, file_path: str, model: str = GLOBAL_DEFAULT_AUDIO_MODEL
    ) -> Optional[str]:
        """
        Transcribes audio using an OpenAI Whisper model.
        """
        try:
            if USE_OFFICIAL_OPENAI:
                with open(file_path, "rb") as audio_file:
                    response = openai.Audio.transcribe(model=model, file=audio_file)
                return response["text"]
            else:
                with open(file_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=model, file=audio_file
                    )
                return response.text
        except Exception as e:
            logger.error(f"An error occurred in 'transcribe_audio': {e}")
            return None

    def generate_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        model: str = GLOBAL_DEFAULT_IMAGE_MODEL,
    ) -> Optional[List[str]]:
        """
        Generates images using DALLE or similar OpenAI image models.
        """
        try:
            if USE_OFFICIAL_OPENAI:
                response = openai.Image.create(prompt=prompt, n=n, size=size)
                return [data["url"] for data in response["data"]]
            else:
                response = self.client.images.generate(
                    model=model, prompt=prompt, n=n, size=size
                )
                return [image.url for image in response.data]
        except Exception as e:
            logger.error(f"An error occurred in 'generate_image': {e}")
            return None


# Quick Usage Demonstration
if __name__ == "__main__":
    wrapper = OpenAIWrapper()

    # Simple chat completion
    chat_response = wrapper.complete(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=0.7,
    )
    print("\n--- Chat Completion Response ---\n", chat_response)

    # Code generation
    code_response = wrapper.code(
        prompt_usr="Write a Python function to greet a user by name.",
        temperature=0.3,
    )
    print("\n--- Generated Code ---\n", code_response)
