"""
Speech-to-Text (STT) — Audio transcription using OpenAI Whisper.

Supports local whisper model, file uploads, and language detection.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

try:
    import whisper
    import numpy as np
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SpeechToText:
    """
    Speech-to-text transcription using OpenAI Whisper.

    Supports:
      • File path transcription
      • Raw audio data (numpy array)
      • Language detection
      • Multiple model sizes (tiny → large)

    Parameters
    ----------
    model_size : str
        Whisper model size: "tiny", "base", "small", "medium", "large".
    device : str | None
        Compute device ("cpu", "cuda", "mps"). Auto-detected if None.
    language : str | None
        Force a specific language code (e.g., "en", "zh"). None = auto-detect.
    """

    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ) -> None:
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "whisper not installed. Run: pip install openai-whisper"
            )

        self._model_size = model_size
        self._language = language
        self._model = None

        # Auto-detect device
        if device:
            self._device = device
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            self._device = "cuda"
        elif TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        logger.info(
            "SpeechToText initialised: model=%s, device=%s, language=%s",
            model_size,
            self._device,
            language or "auto",
        )

    @property
    def model(self):
        """Lazy-load the whisper model."""
        if self._model is None:
            logger.info("Loading whisper model '%s' on %s...", self._model_size, self._device)
            self._model = whisper.load_model(self._model_size, device=self._device)
            logger.info("Whisper model loaded successfully")
        return self._model

    # ── Transcription ───────────────────────────────────────────────

    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Transcribe an audio file.

        Parameters
        ----------
        file_path : str | Path
            Path to audio file.
        language : str | None
            Override language for this transcription.
        initial_prompt : str | None
            Optional prompt to guide transcription style.

        Returns
        -------
        dict with keys: text, language, segments, duration
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {suffix}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        lang = language or self._language
        logger.info("Transcribing: %s (lang=%s)", file_path.name, lang or "auto")

        # Run whisper in a thread to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()

        def _transcribe():
            options: dict[str, Any] = {}
            if lang:
                options["language"] = lang
            if initial_prompt:
                options["initial_prompt"] = initial_prompt

            result = self.model.transcribe(str(file_path), **options)
            return result

        result = await loop.run_in_executor(None, _transcribe)

        # Extract duration from segments
        duration = 0.0
        if result.get("segments"):
            duration = result["segments"][-1].get("end", 0.0)

        output = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
            "duration": duration,
        }

        logger.info(
            "Transcribed %.1fs audio → '%s' (lang=%s)",
            duration,
            output["text"][:50],
            output["language"],
        )
        return output

    async def transcribe_bytes(
        self,
        audio_data: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Transcribe raw audio bytes (writes to temp file first)."""
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            return await self.transcribe_file(tmp_path, language=language)
        finally:
            os.unlink(tmp_path)

    async def transcribe_numpy(
        self,
        audio_array: Any,  # np.ndarray
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """Transcribe a numpy audio array directly."""
        import asyncio
        loop = asyncio.get_event_loop()

        lang = language or self._language

        def _transcribe():
            # Whisper expects float32 audio normalized to [-1, 1]
            if hasattr(audio_array, "dtype"):
                audio = audio_array.astype(np.float32)
                if audio.max() > 1.0:
                    audio = audio / 32768.0 if audio.dtype == np.int16 else audio
            else:
                audio = np.array(audio_array, dtype=np.float32)

            # Resample to 16kHz if needed (whisper expects 16kHz)
            # Simple downsampling placeholder — use proper resampling in production

            options: dict[str, Any] = {}
            if lang:
                options["language"] = lang

            # Use the audio directly if whisper supports it
            result = self.model.transcribe(audio, **options)
            return result

        result = await loop.run_in_executor(None, _transcribe)

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
            "duration": result["segments"][-1]["end"] if result.get("segments") else 0.0,
        }

    def detect_language(self, file_path: Union[str, Path]) -> str:
        """Detect the language of an audio file."""
        audio = whisper.load_audio(str(file_path))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected = max(probs, key=probs.get)
        logger.info("Detected language: %s (%.1f%%)", detected, probs[detected] * 100)
        return detected

    def __repr__(self) -> str:
        return f"SpeechToText(model={self._model_size!r}, device={self._device!r})"
