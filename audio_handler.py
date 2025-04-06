import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
import math
import os
import wave
import json
import warnings
import logging
import requests
import zipfile
import shutil
from pathlib import Path
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

from prompt_toolkit.shortcuts import prompt

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Optional Imports & Setup ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    log.debug("OpenAI library not found. OpenAI transcription mode will be unavailable.")

try:
    import torch
    import whisperx # Includes faster_whisper
except ImportError:
    torch = None
    whisperx = None
    log.debug("PyTorch or WhisperX library not found. WhisperX transcription mode will be unavailable.")

try:
    from vosk import Model as VoskModel, KaldiRecognizer
except ImportError:
    VoskModel = None
    KaldiRecognizer = None
    log.debug("Vosk library not found. Vosk transcription mode will be unavailable.")

# --- Enums ---
class TranscriptionMode(Enum):
    OPENAI = auto()
    WHISPERX = auto()
    VOSK = auto()

class WhisperXModelSize(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    # LARGE = "large" # Original whisper large
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"

class ComputeType(Enum):
    FLOAT16 = "float16"
    INT8 = "int8"
    # Add other types if needed, e.g., FLOAT32

class VoskModelType(Enum):
    # Format: (model_name, download_url, expected_folder_name)
    # URLs from https://alphacephei.com/vosk/models
    SMALL_EN_US = ("small-en-us", "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", "vosk-model-small-en-us-0.15")
    # Add more models as needed, e.g.:
    # MEDIUM_EN_US = ("medium-en-us", "URL_HERE", "vosk-model-en-us-0.22")
    # LARGE_EN_US = ("large-en-us", "URL_HERE", "vosk-model-en-us-0.42-gigaspeech")

# --- Constants ---
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
VISUALIZER_WIDTH = 20
VISUALIZER_THRESHOLD = 0.05
REFRESH_INTERVAL = 0.1
DEFAULT_VOSK_CACHE_DIR = Path.home() / ".cache" / "aider_voice_utils" / "vosk_models"

# --- Configuration Dataclasses ---
@dataclass
class RecordingConfig:
    output_filename: str
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    device_name: Optional[str] = None
    visual: bool = True

@dataclass
class OpenAITranscriptionConfig:
    api_key: Optional[str] = None
    prompt_text: Optional[str] = None
    language: Optional[str] = None

@dataclass
class WhisperXTranscriptionConfig:
    model_size: WhisperXModelSize = WhisperXModelSize.BASE
    language: Optional[str] = None # Specify language code (e.g., "en")

@dataclass
class VoskTranscriptionConfig:
    model_type: VoskModelType = VoskModelType.SMALL_EN_US
    model_cache_dir: Path = DEFAULT_VOSK_CACHE_DIR

# --- Model Cache ---
class ModelCache:
    _instance = None
    _cache: Dict[str, Any] = {}
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._cache = {}
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, model: Any):
        with self._lock:
            # Clear old models before adding a new one to manage memory
            self._clear_internal()
            self._cache[key] = model
            log.info(f"Model cached with key: {key}")

    def clear(self):
        with self._lock:
            self._clear_internal()

    def _clear_internal(self):
        """Internal clear method without acquiring the lock again."""
        if not self._cache:
            log.info("Model cache is already empty.")
            return

        log.info("Clearing model cache...")
        keys = list(self._cache.keys())
        for key in keys:
            model = self._cache.pop(key, None)
            if model:
                # Attempt GPU memory release if it's a torch model
                if torch and hasattr(model, 'device') and 'cuda' in str(model.device):
                     log.debug(f"Attempting to release GPU memory for model {key}...")
                     del model
                     try:
                         torch.cuda.empty_cache()
                     except Exception as e:
                         log.warning(f"Error during torch.cuda.empty_cache(): {e}")
                else:
                     del model # Standard deletion for non-torch or CPU models
        import gc
        gc.collect()
        log.info("Model cache cleared.")

# --- Vosk Model Downloader ---
def _download_and_extract_vosk_model(model_type: VoskModelType, cache_dir: Path) -> Path:
    """Downloads and extracts a Vosk model if not already present."""
    model_name, model_url, expected_folder = model_type.value
    model_dir = cache_dir / expected_folder
    zip_path = cache_dir / f"{expected_folder}.zip"

    if model_dir.exists() and model_dir.is_dir():
        log.info(f"Vosk model '{model_name}' found in cache: {model_dir}")
        return model_dir

    log.info(f"Vosk model '{model_name}' not found in cache. Attempting download...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        log.info(f"Downloading from {model_url} to {zip_path}...")
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            chunk_size = 8192
            downloaded_size = 0
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Basic progress indication
                    progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                    print(f"\rDownloading... {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({progress:.1f}%)", end="")
        print("\nDownload complete.")

        # Extract
        log.info(f"Extracting {zip_path} to {cache_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        log.info("Extraction complete.")

        # Verify expected folder exists
        if not model_dir.exists() or not model_dir.is_dir():
             raise FileNotFoundError(f"Extracted model folder '{expected_folder}' not found after download.")

        # Clean up zip file
        zip_path.unlink()
        log.info(f"Cleaned up {zip_path}.")

        return model_dir

    except requests.exceptions.RequestException as e:
        log.error(f"Failed to download Vosk model: {e}")
        if zip_path.exists(): zip_path.unlink(missing_ok=True) # Cleanup partial download
        raise
    except (zipfile.BadZipFile, OSError) as e:
        log.error(f"Failed to extract Vosk model: {e}")
        if zip_path.exists(): zip_path.unlink(missing_ok=True) # Cleanup zip
        if model_dir.exists(): shutil.rmtree(model_dir) # Cleanup partial extraction
        raise
    except Exception as e:
         log.error(f"An unexpected error occurred during Vosk model setup: {e}")
         if zip_path.exists(): zip_path.unlink(missing_ok=True)
         if model_dir.exists(): shutil.rmtree(model_dir)
         raise

# --- Audio Recorder ---
class AudioRecorder:
    def __init__(self, config: RecordingConfig):
        self.config = config
        self._audio_queue = queue.Queue()
        self._rms_values = {'min': 1e5, 'max': 0.0, 'current': 0.0, 'lock': threading.Lock()}
        self._stream = None
        self._device_id = self._get_device_id()
        self._validate_sample_rate()

    def _get_device_id(self) -> Optional[int]:
        """Finds the device ID for a given device name."""
        if self.config.device_name is None:
            log.info("Using default audio input device.")
            return None
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if self.config.device_name.lower() in device.get("name", "").lower() and device.get("max_input_channels", 0) > 0:
                    log.info(f"Found matching device: '{device.get('name', '')}' (ID: {i})")
                    return i
            # Try partial match if exact not found (already done in original code)
            for i, device in enumerate(devices):
                 if self.config.device_name.lower() in device.get("name", "").lower() and device.get("max_input_channels", 0) > 0:
                     log.warning(f"Exact device '{self.config.device_name}' not found. Using partial match: '{device.get('name', '')}' (ID: {i})")
                     return i
            raise ValueError(f"Input device '{self.config.device_name}' not found.")
        except Exception as e:
            log.error(f"Error querying audio devices: {e}")
            raise

    def _validate_sample_rate(self):
        """Checks if sample rate is supported, falls back if necessary."""
        try:
            sd.check_input_settings(samplerate=self.config.sample_rate, device=self._device_id, channels=self.config.channels)
            log.info(f"Using sample rate: {self.config.sample_rate} Hz")
        except sd.PortAudioError:
             log.warning(f"Sample rate {self.config.sample_rate} Hz not directly supported by device. Falling back.")
             try:
                 dev_info = sd.query_devices(self._device_id, 'input')
                 self.config.sample_rate = int(dev_info.get('default_samplerate', DEFAULT_SAMPLE_RATE))
                 log.info(f"Using device default sample rate: {self.config.sample_rate} Hz")
             except Exception:
                 self.config.sample_rate = DEFAULT_SAMPLE_RATE # Final fallback
                 log.warning(f"Using fallback sample rate: {self.config.sample_rate} Hz")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback function for the sounddevice InputStream."""
        if status:
            log.warning(f"Audio callback status: {status}")

        current_rms = np.sqrt(np.mean(indata**2)) if indata.size > 0 else 0.0

        with self._rms_values['lock']:
            self._rms_values['max'] = max(self._rms_values['max'], current_rms)
            if current_rms > 1e-6:
                 self._rms_values['min'] = min(self._rms_values['min'], current_rms)
            self._rms_values['current'] = current_rms

        self._audio_queue.put(indata.copy())

    def _visualizer_prompt_func(self, start_time: float) -> str:
        """Generates the dynamic prompt string for the visualizer."""
        with self._rms_values['lock']:
            current = self._rms_values['current']
            min_rms = self._rms_values['min']
            max_rms = self._rms_values['max']

        bar = visualize_audio(current, min_rms, max_rms) # Use the standalone function
        elapsed_time = time.time() - start_time
        return f"Recording... Press ENTER to stop. [{elapsed_time:.1f}s] {bar}"

    def record(self):
        """Starts recording audio."""
        log.info(f"Starting recording (Device ID: {self._device_id if self._device_id is not None else 'Default'})...")
        start_time = time.time()
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                device=self._device_id,
                callback=self._audio_callback
            )
            self._stream.start()

            if self.config.visual:
                prompt(lambda: self._visualizer_prompt_func(start_time), refresh_interval=REFRESH_INTERVAL)
            else:
                input("Recording... Press ENTER to stop.")

        except sd.PortAudioError as pae:
            log.error(f"PortAudio Error: {pae}. Is an input device available and configured?")
            raise
        except Exception as e:
            log.error(f"An unexpected error occurred during recording setup: {e}")
            self._stop_stream() # Attempt cleanup
            raise
        finally:
            self._stop_stream()
            log.info("Recording stopped.")
            self._save_audio()

    def _stop_stream(self):
        """Stops and closes the audio stream safely."""
        if self._stream:
            try:
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
                log.debug("Audio stream stopped and closed.")
            except Exception as e:
                log.error(f"Error stopping/closing audio stream: {e}")
            finally:
                self._stream = None

    def _save_audio(self):
        """Saves the recorded audio from the queue to a file."""
        if self._audio_queue.empty():
            log.warning("No audio data recorded.")
            return

        log.info(f"Saving audio to {self.config.output_filename}...")
        try:
            with sf.SoundFile(self.config.output_filename, mode='w', samplerate=self.config.sample_rate, channels=self.config.channels) as file:
                while not self._audio_queue.empty():
                    file.write(self._audio_queue.get())
            log.info("Audio saved successfully.")
        except Exception as e:
            log.error(f"Failed to save audio file: {e}")
            raise

# --- Transcription Services ---
class TranscriptionService(ABC):
    """Abstract base class for transcription services."""
    def __init__(self, model_cache: ModelCache):
        self.model_cache = model_cache

    @abstractmethod
    def transcribe(self, audio_filename: str, **kwargs) -> str:
        """Transcribes the given audio file."""
        pass

    def _load_audio_info(self, audio_filename: str) -> Tuple[float, int, int]:
        """Helper to get audio duration, channels, sample rate."""
        try:
            info = sf.info(audio_filename)
            return info.duration, info.channels, info.samplerate
        except Exception as e:
            log.error(f"Could not read audio file info: {e}")
            raise ValueError(f"Could not read audio file info for {audio_filename}") from e


class OpenAITranscriber(TranscriptionService):
    """Transcription using OpenAI API."""
    def __init__(self, config: OpenAITranscriptionConfig, model_cache: ModelCache):
        super().__init__(model_cache)
        if OpenAI is None:
            raise ImportError("OpenAI library not installed. Please install with 'pip install openai'.")
        self.config = config
        self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=self.api_key)

    def transcribe(self, audio_filename: str, **kwargs) -> str:
        log.info(f"Transcribing {audio_filename} using OpenAI Whisper API...")
        start_time = time.time()
        try:
            with open(audio_filename, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1", # Currently hardcoded, could be configurable
                    file=audio_file,
                    prompt=kwargs.get('prompt_text', self.config.prompt_text),
                    language=kwargs.get('language', self.config.language)
                )
            duration = time.time() - start_time
            log.info(f"OpenAI transcription successful ({duration:.2f}s).")
            return transcription.text
        except Exception as e:
            log.error(f"Error during OpenAI transcription: {e}")
            raise

class WhisperXTranscriber(TranscriptionService):
    """Transcription using WhisperX (faster-whisper backend)."""
    def __init__(self, config: WhisperXTranscriptionConfig, model_cache: ModelCache):
        super().__init__(model_cache)
        if whisperx is None or torch is None:
            raise ImportError("WhisperX or PyTorch not installed. Please install with 'pip install whisperx torch'.")
        self.config = config
        self.device, self.compute_type = self._get_device_and_compute_type()
        self.model_key = f"whisperx_{self.config.model_size.value}_{self.device}_{self.compute_type.value}_{self.config.language or 'auto'}"

    def _get_device_and_compute_type(self) -> Tuple[str, ComputeType]:
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        # Default to float16 on GPU for speed, int8 on CPU
        compute_type_enum = ComputeType.FLOAT16 if device == "cuda" else ComputeType.INT8
        log.info(f"WhisperX - CUDA available: {cuda_available}, Using device: {device}, Compute type: {compute_type_enum.value}")
        return device, compute_type_enum

    def _load_model(self) -> Any:
        """Loads or retrieves the WhisperX model from cache."""
        model = self.model_cache.get(self.model_key)
        if model:
            log.info(f"Using cached WhisperX model: {self.model_key}")
            return model

        log.info(f"Loading WhisperX model: {self.model_key}...")
        load_start_time = time.time()
        # Suppress specific warnings during model load
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*pytorch_lightning.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*pyannote.audio.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.*")
            try:
                 model = whisperx.load_model(
                      self.config.model_size.value,
                      self.device,
                      compute_type=self.compute_type.value,
                      language=self.config.language
                 )
                 load_duration = time.time() - load_start_time
                 log.info(f"WhisperX model loaded ({load_duration:.2f}s). Caching...")
                 self.model_cache.set(self.model_key, model)
                 return model
            except Exception as load_error:
                 log.error(f"Error loading WhisperX model: {load_error}")
                 raise

    def transcribe(self, audio_filename: str, **kwargs) -> str:
        model = self._load_model()
        log.info(f"Transcribing {audio_filename} locally using WhisperX...")
        start_time = time.time()
        try:
            audio = whisperx.load_audio(audio_filename)
            duration, _, _ = self._load_audio_info(audio_filename)

            # Handle language detection warning suppression implicitly by passing language to load_model
            # If language is None, whisperx handles detection internally.
            if not self.config.language and duration < 30:
                 log.warning("Audio is shorter than 30s, language detection may be less accurate if language not specified.")

            # Batch size could be made configurable
            result = model.transcribe(audio, batch_size=16)

            transcribe_duration = time.time() - start_time
            log.info(f"WhisperX transcription successful ({transcribe_duration:.2f}s).")
            return " ".join([segment['text'].strip() for segment in result["segments"]])
        except Exception as e:
            log.error(f"Error during WhisperX transcription: {e}")
            raise

class VoskTranscriber(TranscriptionService):
    """Transcription using Vosk."""
    def __init__(self, config: VoskTranscriptionConfig, model_cache: ModelCache):
        super().__init__(model_cache)
        if VoskModel is None or KaldiRecognizer is None:
            raise ImportError("Vosk library not installed. Please install with 'pip install vosk'.")
        self.config = config
        self.model_path = _download_and_extract_vosk_model(self.config.model_type, self.config.model_cache_dir)
        self.model_key = f"vosk_{self.config.model_type.value[0]}" # Use model name as part of key

    def _load_model(self) -> Any:
        """Loads or retrieves the Vosk model from cache."""
        model = self.model_cache.get(self.model_key)
        if model:
            log.info(f"Using cached Vosk model: {self.model_path}")
            return model

        log.info(f"Loading Vosk model from: {self.model_path}...")
        load_start_time = time.time()
        try:
            model = VoskModel(str(self.model_path)) # Vosk expects string path
            load_duration = time.time() - load_start_time
            log.info(f"Vosk model loaded ({load_duration:.2f}s). Caching...")
            self.model_cache.set(self.model_key, model)
            return model
        except Exception as load_error:
             log.error(f"Error loading Vosk model: {load_error}")
             raise

    def transcribe(self, audio_filename: str, **kwargs) -> str:
        model = self._load_model()
        log.info(f"Transcribing {audio_filename} locally using Vosk...")
        start_time = time.time()
        try:
            wf = wave.open(audio_filename, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                 wf.close()
                 raise ValueError("Vosk requires WAV format mono PCM audio. Please convert the audio file.")

            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(False) # Set to True if word timings are needed

            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res_json = json.loads(rec.Result())
                    results.append(res_json.get("text", ""))

            final_res_json = json.loads(rec.FinalResult())
            results.append(final_res_json.get("text", ""))
            wf.close()

            transcribe_duration = time.time() - start_time
            log.info(f"Vosk transcription successful ({transcribe_duration:.2f}s).")
            return " ".join(filter(None, results)).strip()
        except Exception as e:
            log.error(f"Error during Vosk transcription: {e}")
            raise

# --- Transcription Factory ---
class TranscriptionFactory:
    @staticmethod
    def create(mode: TranscriptionMode, cache: ModelCache, **kwargs) -> TranscriptionService:
        if mode == TranscriptionMode.OPENAI:
            config = OpenAITranscriptionConfig(**{k.replace('openai_', ''): v for k, v in kwargs.items() if k.startswith('openai_')})
            return OpenAITranscriber(config, cache)
        elif mode == TranscriptionMode.WHISPERX:
            config = WhisperXTranscriptionConfig(**{k.replace('whisperx_', ''): v for k, v in kwargs.items() if k.startswith('whisperx_')})
            return WhisperXTranscriber(config, cache)
        elif mode == TranscriptionMode.VOSK:
            config = VoskTranscriptionConfig(**{k.replace('vosk_', ''): v for k, v in kwargs.items() if k.startswith('vosk_')})
            return VoskTranscriber(config, cache)
        else:
            raise ValueError(f"Unsupported transcription mode: {mode}")

# --- Public API Functions (using the new structure) ---

def visualize_audio(current_rms: float, min_rms: float, max_rms: float, width: int = VISUALIZER_WIDTH, threshold: float = VISUALIZER_THRESHOLD) -> str:
    """
    Generates a text-based visualization bar for audio RMS level. (Standalone function)
    """
    if max_rms <= min_rms or max_rms < 1e-6:
        percentage = 0.0
    else:
        percentage = max(0.0, min(1.0, (current_rms - min_rms) / (max_rms - min_rms)))

    if math.isnan(percentage) or percentage < threshold:
        filled_count = 0
    else:
        filled_count = int(percentage * width)

    filled_count = max(0, min(width, filled_count))
    empty_count = width - filled_count
    bar = "█" * filled_count + "░" * empty_count
    return bar

def record_microphone(
    output_filename: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    device_name: Optional[str] = None,
    visual: bool = True
) -> None:
    """
    Records audio from the microphone and saves it to a WAV file.
    """
    config = RecordingConfig(
        output_filename=output_filename,
        sample_rate=sample_rate,
        channels=channels,
        device_name=device_name,
        visual=visual
    )
    recorder = AudioRecorder(config)
    recorder.record() # Handles start, stop, save

def transcribe_audio(
    audio_filename: str,
    transcription_mode: TranscriptionMode = TranscriptionMode.OPENAI,
    **kwargs # Pass mode-specific configs here, e.g., openai_api_key, whisperx_model_size
) -> str:
    """
    Transcribes an audio file using the specified mode. Delegates to the factory.
    """
    if not Path(audio_filename).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_filename}")

    cache = ModelCache() # Get singleton instance
    try:
        # Pass all kwargs; factory will filter relevant ones
        transcriber = TranscriptionFactory.create(transcription_mode, cache, **kwargs)
        # Pass relevant kwargs again for the actual transcription call if needed by specific implementation
        # (though most config is handled at init)
        return transcriber.transcribe(audio_filename, **kwargs)
    except (ImportError, ValueError, FileNotFoundError) as e:
         log.error(f"Transcription setup failed: {e}")
         raise
    except Exception as e:
         log.error(f"Unexpected error during transcription: {e}")
         raise

def clear_transcription_cache():
     """Public function to clear the model cache."""
     cache = ModelCache()
     cache.clear()

# --- Example Usage (if run directly) ---
if __name__ == "__main__":
    log.info("--- Aider Voice Utils - Example Usage ---")

    # --- Configuration ---
    TEST_OUTPUT_FILENAME = "test_recording.wav"
    MIC_DEVICE_NAME = None # Optional: Specify microphone name

    # --- Choose Transcription Mode ---
    # SELECTED_MODE = TranscriptionMode.OPENAI
    SELECTED_MODE = TranscriptionMode.WHISPERX
    # SELECTED_MODE = TranscriptionMode.VOSK

    # --- Mode Specific Config ---
    transcription_kwargs = {}
    if SELECTED_MODE == TranscriptionMode.OPENAI:
        # openai_api_key is read from env var by default if not passed
        # transcription_kwargs['openai_api_key'] = "YOUR_KEY_HERE" # Optional override
        transcription_kwargs['openai_language'] = "en"
        log.info("Mode: OpenAI")
    elif SELECTED_MODE == TranscriptionMode.WHISPERX:
        transcription_kwargs['whisperx_model_size'] = WhisperXModelSize.BASE
        transcription_kwargs['whisperx_language'] = "en" # Optional: helps speed/accuracy
        log.info(f"Mode: WhisperX (Model: {transcription_kwargs['whisperx_model_size'].value}, Lang: {transcription_kwargs['whisperx_language'] or 'auto'})")
    elif SELECTED_MODE == TranscriptionMode.VOSK:
        # Model will be downloaded automatically to ~/.cache/aider_voice_utils/vosk_models if not present
        transcription_kwargs['vosk_model_type'] = VoskModelType.SMALL_EN_US
        # transcription_kwargs['vosk_model_cache_dir'] = Path("./my_vosk_cache") # Optional override
        log.info(f"Mode: Vosk (Model: {transcription_kwargs['vosk_model_type'].value[0]})")

    # --- Recording ---
    try:
        log.info("--- Starting Microphone Recording ---")
        record_microphone(TEST_OUTPUT_FILENAME, device_name=MIC_DEVICE_NAME, visual=True)

        # --- Transcription ---
        log.info("--- Starting Transcription ---")
        if not Path(TEST_OUTPUT_FILENAME).exists():
            log.warning(f"Skipping transcription: Recording file {TEST_OUTPUT_FILENAME} not found.")
        else:
            transcript = None
            try:
                transcript = transcribe_audio(
                    TEST_OUTPUT_FILENAME,
                    transcription_mode=SELECTED_MODE,
                    **transcription_kwargs
                )

                if transcript is not None:
                    print("\n--- Transcript ---") # Use print for final user output
                    print(transcript)
                    print("------------------")
                else:
                     log.warning("Transcription was skipped or failed.")

            except (ImportError, FileNotFoundError, ValueError) as e:
                 log.error(f"Transcription failed: {e}")
                 if SELECTED_MODE == TranscriptionMode.VOSK and isinstance(e, FileNotFoundError):
                      log.info("Ensure the Vosk model is downloaded or the path is correct.")
                 elif SELECTED_MODE == TranscriptionMode.WHISPERX and isinstance(e, ImportError):
                      log.info("Install WhisperX dependencies: pip install whisperx torch")
                 elif SELECTED_MODE == TranscriptionMode.VOSK and isinstance(e, ImportError):
                      log.info("Install Vosk dependencies: pip install vosk")
            except Exception as e:
                 log.exception(f"Transcription failed with unexpected error.") # Use log.exception for stack trace

    except FileNotFoundError as fnf:
        log.error(f"Error: {fnf}")
    except ValueError as ve:
        log.error(f"Error: {ve}")
    except Exception as e:
        log.exception(f"An error occurred in the example usage.") # Use log.exception for stack trace
    finally:
        # Clean up the test file and cached models
        if Path(TEST_OUTPUT_FILENAME).exists():
            try:
                os.remove(TEST_OUTPUT_FILENAME)
                log.info(f"Cleaned up {TEST_OUTPUT_FILENAME}")
            except Exception as cleanup_e:
                 log.error(f"Error cleaning up {TEST_OUTPUT_FILENAME}: {cleanup_e}")
        clear_transcription_cache() # Clear models from memory at the end
