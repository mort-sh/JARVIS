"""
A robust, production-ready wrapper around various OpenAI API endpoints.
It supports both the official OpenAI Python client and a custom client,
toggled via the USE_OFFICIAL_OPENAI constant.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

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
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    if USE_OFFICIAL_OPENAI:
        import openai
    else:
        # Replace with your custom class or library.
        from openai import OpenAI
except ImportError:
    logger.exception("Failed to import the required OpenAI library.")
    raise

class OpenAIWrapper:
    """
    A wrapper around various OpenAI API endpoints. Can use the official
    openai library or a custom client, depending on the USE_OFFICIAL_OPENAI flag.
    """

    def __init__(self) -> None:
        """
        Initializes the OpenAIWrapper, loading the API key from the environment.
        
        Raises:
            ValueError: If the 'OPENAI_API_KEY' environment variable is not set.
        """
        load_dotenv()  # Load environment variables from a .env file if present
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set the 'OPENAI_API_KEY' environment variable."
            )

        if USE_OFFICIAL_OPENAI:
            openai.api_key = self.api_key
            self.client = None
            logger.debug("Using official OpenAI client.")
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.debug("Using custom OpenAI client.")

    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = GLOBAL_DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """
        Generates a chat completion given a list of messages and parameters.

        Args:
            messages (List[Dict[str, str]]): A list of message dicts, each having 'role' and 'content'.
            model (str): The model name to use for completion.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens to generate.

        Returns:
            Optional[str]: The assistant's response text if successful, None otherwise.
        """
        try:
            if not messages:
                raise ValueError("'messages' must contain at least one message.")

            if USE_OFFICIAL_OPENAI:
                import openai
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
        except Exception as exc:
            logger.error("Error in 'generate_chat_completion': %s", exc, exc_info=True)
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
        Sends a prompt to the ChatCompletion API, optionally streaming the response.

        Args:
            prompt (Optional[str]): The user prompt if messages are not directly provided.
            model (str): The model to use for the completion.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens to generate in the response.
            messages (Optional[List[Dict[str, str]]]): Pre-structured conversation messages.
            role (str): Role to assign to the prompt if building a messages list internally.
            stream_callback (Optional[Callable[[str], None]]): 
                If provided, will be called with chunks of text as they are generated (streaming).

        Returns:
            Dict[str, Any]: A dictionary with 'assistant_reply' and 'response' keys.
        """
        try:
            # Validate or build messages
            if messages is None:
                if not prompt:
                    raise ValueError("Either 'messages' or 'prompt' must be provided.")
                messages = [{"role": role, "content": prompt}]
            
            if not messages:
                raise ValueError("'messages' must contain at least one message.")
            
            logger.debug("Sending chat completion with messages: %s", messages)

            if USE_OFFICIAL_OPENAI:
                import openai
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )

            assistant_reply = ""
            for chunk in response:
                # In streaming responses, chunk.choices[0].delta.content
                # contains incremental additions to the text.
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    assistant_reply += content
                    if stream_callback:
                        stream_callback(content)

            return {"assistant_reply": assistant_reply, "response": response}
        except Exception as exc:
            logger.error("Exception in send_prompt: %s", exc, exc_info=True)
            return {"assistant_reply": "", "response": None}

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        model: str = GLOBAL_DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> Optional[str]:
        """
        Generates a chat completion following a specific JSON schema that includes a 'code' field.

        Args:
            messages (List[Dict[str, str]]): The conversation messages.
            model (str): Model to use.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens to generate.

        Returns:
            Optional[str]: The 'code' field from the JSON if parsing is successful, None otherwise.
        """
        code_schema = {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        }

        try:
            if not messages:
                raise ValueError("'messages' must contain at least one message.")

            if USE_OFFICIAL_OPENAI:
                import openai
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response_content = response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
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
            logger.error("Failed to parse response content as valid JSON.")
            return None
        except KeyError:
            logger.error("The key 'code' was not found in the response content.")
            return None
        except Exception as exc:
            logger.error("Error in 'structured_output': %s", exc, exc_info=True)
            return None

    def generate_code(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        model: str = GLOBAL_DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        Generates code strictly adhering to a JSON schema. Returns only the code snippet.

        Args:
            user_prompt (str): The user instruction regarding the code to generate.
            system_prompt (Optional[str]): Optional system-level instructions for context.
            model (str): Model to use.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens in the generated code.

        Returns:
            Optional[str]: The extracted 'code' from the JSON-based completion.
        """
        if not user_prompt:
            logger.warning("Empty user_prompt passed to generate_code.")
            return None

        if not system_prompt:
            system_prompt = (
                "You are a helpful assistant proficient in generating clean, "
                "efficient, and well-documented code. Respond only with the code "
                "and any necessary comments, without additional explanations."
            )

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})

        return self.structured_output(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def list_models(self) -> Optional[List[Any]]:
        """
        Lists all available models.

        Returns:
            Optional[List[Any]]: A list of model data dicts if successful, None otherwise.
        """
        try:
            if USE_OFFICIAL_OPENAI:
                import openai
                response = openai.Model.list()
                return response["data"]
            else:
                response = self.client.models.list()
                return response.data
        except Exception as exc:
            logger.error("Error in 'list_models': %s", exc, exc_info=True)
            return None

    def retrieve_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific model by its ID.

        Args:
            model_id (str): The unique ID of the model to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The model data if successful, None otherwise.
        """
        if not model_id:
            logger.warning("Empty model_id passed to retrieve_model.")
            return None

        try:
            if USE_OFFICIAL_OPENAI:
                import openai
                response = openai.Model.retrieve(model_id)
            else:
                response = self.client.models.retrieve(model=model_id)
            return response
        except Exception as exc:
            logger.error("Error in 'retrieve_model': %s", exc, exc_info=True)
            return None

    def transcribe_audio(
        self,
        file_path: str,
        model: str = GLOBAL_DEFAULT_AUDIO_MODEL,
    ) -> Optional[str]:
        """
        Transcribes audio using an OpenAI Whisper model.

        Args:
            file_path (str): The path to the audio file to transcribe.
            model (str): The audio model name.

        Returns:
            Optional[str]: The transcribed text if successful, None otherwise.
        """
        if not os.path.exists(file_path):
            logger.error("File not found for audio transcription: %s", file_path)
            return None

        try:
            if USE_OFFICIAL_OPENAI:
                import openai
                with open(file_path, "rb") as audio_file:
                    response = openai.Audio.transcribe(model=model, file=audio_file)
                return response["text"]
            else:
                with open(file_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=model,
                        file=audio_file,
                    )
                return response.text
        except Exception as exc:
            logger.error("Error in 'transcribe_audio': %s", exc, exc_info=True)
            return None

    def generate_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        model: str = GLOBAL_DEFAULT_IMAGE_MODEL,
    ) -> Optional[List[str]]:
        """
        Generates images using an OpenAI image model (e.g., DALL·E).

        Args:
            prompt (str): The text prompt describing the desired images.
            n (int): Number of images to generate.
            size (str): Size of the generated images, e.g. "512x512" or "1024x1024".
            model (str): The image model to use.

        Returns:
            Optional[List[str]]: A list of URLs for the generated images if successful, None otherwise.
        """
        if not prompt:
            logger.warning("Empty prompt passed to generate_image.")
            return None
        if n < 1:
            logger.warning("'n' must be at least 1 for generate_image.")
            return None

        try:
            if USE_OFFICIAL_OPENAI:
                import openai
                response = openai.Image.create(
                    prompt=prompt, 
                    n=n, 
                    size=size
                )
                return [data["url"] for data in response["data"]]
            else:
                response = self.client.images.generate(
                    model=model,
                    prompt=prompt,
                    n=n,
                    size=size,
                )
                return [image.url for image in response.data]
        except Exception as exc:
            logger.error("Error in 'generate_image': %s", exc, exc_info=True)
            return None


# Quick Usage Demonstration
if __name__ == "__main__":
    try:
        wrapper = OpenAIWrapper()

        # Simple chat completion
        chat_response = wrapper.generate_chat_completion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            temperature=0.7,
        )
        print("\n--- Chat Completion Response ---\n", chat_response)

        # Code generation
        code_response = wrapper.generate_code(
            user_prompt="Write a Python function to greet a user by name.",
            temperature=0.3,
        )
        print("\n--- Generated Code ---\n", code_response)

    except ValueError as ve:
        logger.error("Initialization error: %s", ve)
