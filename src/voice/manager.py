"""
Voice Manager — Unified STT + TTS interface.

Provides a high-level API for voice input/output, handling:
  • Audio format conversion
  • STT → LLM → TTS pipeline coordination
  • File management for audio artifacts
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Union

from .stt import SpeechToText
from .tts import TextToSpeech

logger = logging.getLogger(__name__)


class VoiceManager:
    """
    Unified voice I/O manager.

    Coordinates speech-to-text and text-to-speech for the JARVIS assistant.

    Parameters
    ----------
    stt_model : str
        Whisper model size for STT.
    tts_engine : str
        TTS engine ("edge" or "pyttsx3").
    tts_voice : str | None
        Voice preset for TTS.
    audio_dir : Path | str | None
        Directory to store audio artifacts. None = temp dir only.
    """

    def __init__(
        self,
        stt_model: str = "base",
        tts_engine: str = "edge",
        tts_voice: Optional[str] = None,
        audio_dir: Optional[Path | str] = None,
        stt_language: Optional[str] = None,
    ) -> None:
        self._audio_dir = Path(audio_dir) if audio_dir else Path(tempfile.gettempdir()) / "jarvis_audio"
        self._audio_dir.mkdir(parents=True, exist_ok=True)

        # Initialise STT
        try:
            self.stt = SpeechToText(
                model_size=stt_model,
                language=stt_language,
            )
            self._stt_available = True
        except ImportError as e:
            logger.warning("STT unavailable: %s", e)
            self.stt = None  # type: ignore[assignment]
            self._stt_available = False

        # Initialise TTS
        try:
            self.tts = TextToSpeech(
                engine=tts_engine,
                voice=tts_voice,
            )
            self._tts_available = True
        except ImportError as e:
            logger.warning("TTS unavailable: %s", e)
            self.tts = None  # type: ignore[assignment]
            self._tts_available = False

        logger.info(
            "VoiceManager: stt=%s, tts=%s, audio_dir=%s",
            self._stt_available,
            self._tts_available,
            self._audio_dir,
        )

    # ── Speech-to-Text ──────────────────────────────────────────────

    async def listen(
        self,
        source: Union[str, Path, bytes],
        filename: Optional[str] = None,
        language: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Transcribe audio input.

        Parameters
        ----------
        source : str | Path | bytes
            Audio file path or raw audio bytes.
        filename : str | None
            Filename hint (used when source is bytes).
        language : str | None
            Override language detection.

        Returns
        -------
        dict with: text, language, segments, duration
        """
        if not self._stt_available or self.stt is None:
            raise RuntimeError("Speech-to-text not available. Install: pip install openai-whisper")

        if isinstance(source, bytes):
            fname = filename or "recording.wav"
            return await self.stt.transcribe_bytes(
                source, filename=fname, language=language
            )
        else:
            return await self.stt.transcribe_file(source, language=language)

    # ── Text-to-Speech ──────────────────────────────────────────────

    async def speak(
        self,
        text: str,
        output_path: Optional[Path | str] = None,
    ) -> Path:
        """
        Convert text to speech and save to file.

        Returns Path to the generated audio file.
        """
        if not self._tts_available or self.tts is None:
            raise RuntimeError("Text-to-speech not available. Install: pip install edge-tts")

        if output_path is None:
            output_path = self._audio_dir / f"tts_{hash(text) & 0xFFFFFF:06x}.mp3"

        return await self.tts.synthesize(text, output_path=output_path)

    async def speak_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream TTS audio chunks for real-time playback."""
        if not self._tts_available or self.tts is None:
            raise RuntimeError("Text-to-speech not available")
        async for chunk in self.tts.synthesize_stream(text):
            yield chunk

    # ── Voice Pipeline ──────────────────────────────────────────────

    async def voice_to_text(
        self,
        audio_source: Union[str, Path, bytes],
        filename: Optional[str] = None,
    ) -> str:
        """
        Convenience: transcribe audio and return just the text string.
        """
        result = await self.listen(audio_source, filename=filename)
        return result["text"]

    async def text_to_audio(
        self,
        text: str,
        save: bool = True,
    ) -> Path:
        """
        Convenience: synthesize text to audio, optionally saving to disk.
        """
        output_path = None if not save else None  # auto-generate path
        return await self.speak(text, output_path=output_path)

    # ── Audio Utilities ─────────────────────────────────────────────

    @staticmethod
    def get_audio_info(file_path: Union[str, Path]) -> dict[str, Any]:
        """Get basic info about an audio file."""
        path = Path(file_path)
        stat = path.stat()
        return {
            "path": str(path),
            "format": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 1),
        }

    def list_audio_files(self) -> list[Path]:
        """List all audio files in the audio directory."""
        audio_exts = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".webm"}
        return sorted(
            p for p in self._audio_dir.iterdir()
            if p.suffix.lower() in audio_exts
        )

    def cleanup_audio(self, max_age_hours: float = 24.0) -> int:
        """Delete audio files older than max_age_hours. Returns count deleted."""
        import time
        cutoff = time.time() - (max_age_hours * 3600)
        deleted = 0
        for p in self._audio_dir.iterdir():
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                deleted += 1
        if deleted:
            logger.info("Cleaned up %d old audio files", deleted)
        return deleted

    # ── Status ──────────────────────────────────────────────────────

    @property
    def is_available(self) -> dict[str, bool]:
        return {
            "stt": self._stt_available,
            "tts": self._tts_available,
        }

    def __repr__(self) -> str:
        return (
            f"VoiceManager(stt={'✓' if self._stt_available else '✗'}, "
            f"tts={'✓' if self._tts_available else '✗'})"
        )
