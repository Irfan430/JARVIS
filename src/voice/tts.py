"""
Text-to-Speech (TTS) — Voice synthesis.

Supports two engines:
  1. pyttsx3 — offline, free, fast
  2. edge-tts — online, free, high quality Microsoft voices
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


# ── Edge TTS voice presets ──────────────────────────────────────────

EDGE_VOICES: dict[str, str] = {
    "en-us-male": "en-US-GuyNeural",
    "en-us-female": "en-US-JennyNeural",
    "en-us-calm": "en-US-AriaNeural",
    "en-gb-male": "en-GB-RyanNeural",
    "en-gb-female": "en-GB-SoniaNeural",
    "zh-male": "zh-CN-YunxiNeural",
    "zh-female": "zh-CN-XiaoxiaoNeural",
    "ja-male": "ja-JP-KeitaNeural",
    "ja-female": "ja-JP-NanamiNeural",
    "ko-male": "ko-KR-InJoonNeural",
    "ko-female": "ko-KR-SunHiNeural",
}


class TextToSpeech:
    """
    Text-to-speech with offline (pyttsx3) and online (edge-tts) backends.

    Parameters
    ----------
    engine : str
        TTS engine: "pyttsx3" (offline) or "edge" (edge-tts, high quality).
    voice : str | None
        Voice name or preset key. For edge-tts, use presets like "en-us-male".
    rate : int
        Speech rate (words per minute). Default: 200.
    volume : float
        Volume 0.0 – 1.0. Default: 1.0.
    """

    def __init__(
        self,
        engine: str = "edge",
        voice: Optional[str] = None,
        rate: int = 200,
        volume: float = 1.0,
    ) -> None:
        self._engine_name = engine
        self._rate = rate
        self._volume = volume
        self._pyttsx3_engine = None

        # Resolve voice
        if engine == "edge" and voice and voice in EDGE_VOICES:
            self._voice = EDGE_VOICES[voice]
        else:
            self._voice = voice or (
                "en-US-GuyNeural" if engine == "edge" else ""
            )

        if engine == "pyttsx3":
            if not PYTTSX3_AVAILABLE:
                raise ImportError(
                    "pyttsx3 not installed. Run: pip install pyttsx3"
                )
            self._pyttsx3_init()
        elif engine == "edge":
            if not EDGE_TTS_AVAILABLE:
                raise ImportError(
                    "edge-tts not installed. Run: pip install edge-tts"
                )

        logger.info(
            "TTS initialised: engine=%s, voice=%s, rate=%d",
            engine,
            self._voice,
            rate,
        )

    def _pyttsx3_init(self) -> None:
        """Initialise pyttsx3 engine."""
        self._pyttsx3_engine = pyttsx3.init()
        self._pyttsx3_engine.setProperty("rate", self._rate)
        self._pyttsx3_engine.setProperty("volume", self._volume)
        if self._voice:
            self._pyttsx3_engine.setProperty("voice", self._voice)

    # ── Synthesis ───────────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        output_path: Union[str, Path, None] = None,
        output_format: str = "mp3",
    ) -> Path:
        """
        Convert text to speech and save to file.

        Parameters
        ----------
        text : str
            Text to synthesize.
        output_path : str | Path | None
            Where to save the audio. Auto-generated temp file if None.
        output_format : str
            Audio format ("mp3", "wav", "ogg"). Edge-tts supports mp3/ogg.

        Returns
        -------
        Path to the generated audio file.
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        if output_path is None:
            suffix = f".{output_format}"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            output_path = Path(tmp.name)
            tmp.close()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._engine_name == "edge":
            await self._edge_synthesize(text, output_path, output_format)
        elif self._engine_name == "pyttsx3":
            await self._pyttsx3_synthesize(text, output_path)
        else:
            raise ValueError(f"Unknown engine: {self._engine_name}")

        logger.info("TTS output: %s (%.1fKB)", output_path, output_path.stat().st_size / 1024)
        return output_path

    async def _edge_synthesize(
        self, text: str, output_path: Path, output_format: str
    ) -> None:
        """Synthesize using edge-tts."""
        communicate = edge_tts.Communicate(
            text,
            self._voice,
            rate=f"{self._rate - 200:+d}%",  # edge-tts uses relative rate
            volume=f"{int(self._volume * 100)}%",
        )
        await communicate.save(str(output_path))

    async def _pyttsx3_synthesize(self, text: str, output_path: Path) -> None:
        """Synthesize using pyttsx3 (runs in thread executor)."""
        loop = asyncio.get_event_loop()

        def _do_synth():
            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            if self._voice:
                engine.setProperty("voice", self._voice)

            # pyttsx3 saves to wav
            wav_path = str(output_path.with_suffix(".wav"))
            engine.save_to_file(text, wav_path)
            engine.runAndWait()
            return Path(wav_path)

        wav_path = await loop.run_in_executor(None, _do_synth)

        # If a non-wav format was requested, we'd convert here
        # For now, copy the wav to the target path
        if output_path.suffix != ".wav":
            import shutil
            shutil.copy2(str(wav_path), str(output_path))
            wav_path.unlink(missing_ok=True)
        else:
            # Rename if needed
            if wav_path != output_path:
                import shutil
                shutil.move(str(wav_path), str(output_path))

    async def synthesize_stream(
        self,
        text: str,
        output_format: str = "mp3",
    ):
        """
        Yield audio chunks for streaming playback.
        Only supported with edge-tts.
        """
        if self._engine_name != "edge":
            raise NotImplementedError("Streaming only supported with edge-tts")

        communicate = edge_tts.Communicate(
            text,
            self._voice,
            rate=f"{self._rate - 200:+d}%",
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    # ── Utilities ───────────────────────────────────────────────────

    @staticmethod
    def list_edge_voices() -> dict[str, str]:
        """Return available edge-tts voice presets."""
        return dict(EDGE_VOICES)

    async def list_available_voices(self) -> list[dict[str, str]]:
        """List all available edge-tts voices (requires internet)."""
        if not EDGE_TTS_AVAILABLE:
            return []
        voices = await edge_tts.list_voices()
        return [
            {"id": v["ShortName"], "name": v["FriendlyName"], "locale": v["Locale"]}
            for v in voices
        ]

    def __repr__(self) -> str:
        return f"TextToSpeech(engine={self._engine_name!r}, voice={self._voice!r})"
