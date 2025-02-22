"""
A robust, production-ready wrapper around various OpenAI API endpoints.
It supports both the official OpenAI Python client and a custom client,
toggled via the USE_OFFICIAL_OPENAI constant.
"""

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import BadRequestError, APIError, RateLimitError

# Toggle if needed
USE_OFFICIAL_OPENAI = False

# Default model settings
GLOBAL_DEFAULT_MODEL = "gpt-4o"  # Latest GPT-4 model
GLOBAL_DEFAULT_AUDIO_MODEL = "whisper-1"
GLOBAL_DEFAULT_IMAGE_MODEL = "dall-e-3"
GLOBAL_DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"

# Model type constants
MODEL_TYPE_CHAT = "chat"
MODEL_TYPE_AUDIO = "audio"
MODEL_TYPE_IMAGE = "image"
MODEL_TYPE_EMBEDDING = "embedding"

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
        Initializes the OpenAIWrapper, loading the API key from the environment and setting up model caching.

        Raises:
            ValueError: If the 'OPENAI_API_KEY' environment variable is not set.
        """
        load_dotenv()  # Load environment variables from a .env file if present
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Please set the 'OPENAI_API_KEY' environment variable."
            )

        # Initialize model caching
        self._model_cache = {
            "models": None,  # List of all models
            "model_info": {},  # Cache for individual model details
            "last_update": None,
            "cache_duration": 3600,  # Cache duration in seconds (1 hour)
        }

        if USE_OFFICIAL_OPENAI:
            openai.api_key = self.api_key
            self.client = None
            logger.debug("Using official OpenAI client.")
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.debug("Using custom OpenAI client.")

        # Initialize model capabilities mapping
        self._model_capabilities = {
            MODEL_TYPE_CHAT: ["gpt-4", "gpt-3.5"],
            MODEL_TYPE_AUDIO: ["whisper"],
            MODEL_TYPE_IMAGE: ["dall-e"],
            MODEL_TYPE_EMBEDDING: ["text-embedding", "ada"],
        }

        # Initialize cache with models
        self._update_model_cache()

    def _update_model_cache(self) -> None:
        """
        Updates the model cache with fresh data from the API.
        """
        try:
            models = self.list_models()
            if models:
                self._model_cache["models"] = models
                self._model_cache["last_update"] = time.time()
                logger.debug("Model cache updated successfully")
        except Exception as e:
            logger.error("Failed to update model cache: %s", str(e))

    def _get_cached_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets model info from cache or fetches it if not cached.

        Args:
            model_id (str): The model ID to get info for

        Returns:
            Optional[Dict[str, Any]]: The model info if available
        """
        # Check if cache needs refresh
        if (
            not self._model_cache["last_update"]
            or time.time() - self._model_cache["last_update"] > self._model_cache["cache_duration"]
        ):
            self._update_model_cache()

        # Check if model info is in cache
        if model_id not in self._model_cache["model_info"]:
            try:
                model_info = self.retrieve_model(model_id)
                if model_info:
                    self._model_cache["model_info"][model_id] = model_info
                    logger.debug("Added model %s to cache", model_id)
                return model_info
            except Exception as e:
                logger.warning("Failed to get info for model %s: %s", model_id, str(e))
                return None

        return self._model_cache["model_info"].get(model_id)

    def _is_chat_model(self, model: str) -> bool:
        """
        Determines if a model supports the chat completions endpoint.
        Uses cached model info to avoid repeated API calls.

        Args:
            model (str): The model ID to check

        Returns:
            bool: True if the model supports chat completions, False otherwise
        """
        try:
            # First check if model exists in cache
            model_info = self._get_cached_model_info(model)
            if not model_info:
                return False

            # Check if model ID contains indicators of non-chat models
            non_chat_indicators = [
                "realtime-preview",  # Realtime preview models use completions
                "-preview",  # Preview models generally use completions
                "instruct",  # Instruct models use completions
            ]

            return not any(indicator in model.lower() for indicator in non_chat_indicators)
        except Exception as e:
            logger.warning("Error checking model type: %s. Assuming not chat model.", str(e))
            return False

    def get_valid_model(self, model_type: str = MODEL_TYPE_CHAT) -> str:
        """
        Returns a valid model ID from the list of available models based on the model type.
        Uses cached model list to avoid repeated API calls.

        Args:
            model_type (str): The type of model to retrieve (chat, audio, image, embedding)

        Returns:
            str: A valid model ID for the specified type

        Raises:
            ValueError: If no valid model is found for the specified type
        """
        try:
            # If a specific model is requested via environment variable, use it
            env_model = os.getenv("OPENAI_MODEL")
            if env_model:
                logger.info(
                    "Using model specified in OPENAI_MODEL environment variable: %s", env_model
                )
                return env_model

            # Check if cache needs refresh
            if (
                not self._model_cache["models"]
                or not self._model_cache["last_update"]
                or time.time() - self._model_cache["last_update"]
                > self._model_cache["cache_duration"]
            ):
                self._update_model_cache()

            if not self._model_cache["models"]:
                raise ValueError("Failed to retrieve available models")

            # Filter models based on type
            valid_models = []
            type_prefixes = self._model_capabilities.get(model_type, [])

            for model in self._model_cache["models"]:
                model_id = model.id if hasattr(model, "id") else model.get("id")
                if not model_id:
                    continue

                # Check if model matches any prefix for the specified type
                if any(model_id.startswith(prefix) for prefix in type_prefixes):
                    # For chat models, ensure it's compatible with chat completions
                    if model_type == MODEL_TYPE_CHAT and not self._is_chat_model(model_id):
                        continue
                    valid_models.append(model_id)

            if not valid_models:
                raise ValueError(f"No valid models found for type: {model_type}")

            # Sort by version and return the latest
            sorted_models = sorted(valid_models, reverse=True)
            return sorted_models[0]

        except Exception as e:
            logger.error("Error in get_valid_model: %s", str(e))
            # Fall back to default models based on type
            defaults = {
                MODEL_TYPE_CHAT: GLOBAL_DEFAULT_MODEL,
                MODEL_TYPE_AUDIO: GLOBAL_DEFAULT_AUDIO_MODEL,
                MODEL_TYPE_IMAGE: GLOBAL_DEFAULT_IMAGE_MODEL,
                MODEL_TYPE_EMBEDDING: GLOBAL_DEFAULT_EMBEDDING_MODEL,
            }
            return defaults.get(model_type, GLOBAL_DEFAULT_MODEL)

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
        Sends a prompt to either the ChatCompletion or Completion API, depending on model compatibility.
        Supports streaming responses from both endpoints.

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
            # Determine model type and get appropriate model
            if model == GLOBAL_DEFAULT_MODEL:
                model = self.get_valid_model(MODEL_TYPE_CHAT)

            # Validate or build messages/prompt
            if messages is None:
                if not prompt:
                    raise ValueError("Either 'messages' or 'prompt' must be provided.")
                messages = [{"role": role, "content": prompt}]
                prompt = messages[0]["content"]  # For non-chat models

            if not messages:
                raise ValueError("'messages' must contain at least one message.")

            # Check if model supports chat completions
            is_chat_model = self._is_chat_model(model)
            logger.debug(
                "Using model %s with %s endpoint",
                model,
                "chat completions" if is_chat_model else "completions",
            )

            if is_chat_model:
                # Use chat completions endpoint
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
            else:
                # Use completions endpoint
                if USE_OFFICIAL_OPENAI:
                    import openai

                    response = openai.Completion.create(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                else:
                    response = self.client.completions.create(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )

            assistant_reply = ""
            for chunk in response:
                if is_chat_model:
                    # Chat completions format
                    content = getattr(chunk.choices[0].delta, "content", None)
                else:
                    # Regular completions format
                    content = getattr(chunk.choices[0], "text", None)

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
            if model == GLOBAL_DEFAULT_MODEL:
                model = self.get_valid_model()
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
        Lists all available models with enhanced error handling and retries.

        Returns:
            Optional[List[Any]]: A list of model data if successful, None otherwise.

        Note:
            Implements exponential backoff for rate limits and handles various API errors.
        """
        max_retries = 3
        base_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                if USE_OFFICIAL_OPENAI:
                    import openai

                    response = openai.Model.list()
                    return response["data"]
                else:
                    response = self.client.models.list()
                    return response.data

            except RateLimitError as rate_err:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Rate limit hit, attempt {attempt + 1}/{max_retries}. "
                    f"Waiting {delay} seconds..."
                )
                time.sleep(delay)
                if attempt == max_retries - 1:
                    logger.error("Rate limit persisted after all retries")
                    return None

            except BadRequestError as bad_err:
                logger.error("Invalid request when listing models: %s", str(bad_err))
                return None

            except APIError as api_err:
                logger.error("API error when listing models: %s", str(api_err))
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                    continue
                return None

            except Exception as exc:
                logger.error("Unexpected error in list_models: %s", exc, exc_info=True)
                return None

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

                response = openai.Image.create(prompt=prompt, n=n, size=size)
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
