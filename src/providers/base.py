"""
Abstract Base LLM Provider.

Defines the interface every provider must implement, plus shared
data structures for chat messages, responses, and streaming.
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

logger = __import__("logging").getLogger(__name__)


# ── Data Structures ─────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class StreamingChunk:
    """A single chunk from a streaming response."""

    delta: str
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, int]] = None
    chunk_index: int = 0


@dataclass
class ChatResponse:
    """Complete (non-streaming) chat response."""

    content: str
    model: str = ""
    finish_reason: str = "stop"
    usage: Optional[dict[str, int]] = None  # {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
    latency_ms: float = 0.0
    raw: Optional[dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return 0


# ── Abstract Provider ───────────────────────────────────────────────

class BaseLLMProvider(abc.ABC):
    """
    Abstract base for all LLM providers.

    Subclasses must implement `chat()` and optionally `stream()`.
    """

    name: str = "base"

    def __init__(self, api_key: str = "", **kwargs: Any) -> None:
        self.api_key = api_key
        self.config = kwargs
        self._request_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0.0

    # ── Core Interface ──────────────────────────────────────────────

    @abc.abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Send a chat completion request.

        Parameters
        ----------
        messages : list
            Conversation history.
        model : str | None
            Model override (provider default if None).
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum tokens to generate.
        """
        ...

    async def stream(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Stream a chat completion. Default falls back to non-streaming.
        Providers that support streaming should override this.
        """
        response = await self.chat(messages, model, temperature, max_tokens, **kwargs)
        yield StreamingChunk(delta=response.content, finish_reason="stop")

    # ── Token Counting (approximate) ────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return max(1, len(text) // 4)

    @staticmethod
    def estimate_messages_tokens(messages: list[ChatMessage | dict[str, str]]) -> int:
        """Estimate total tokens for a list of messages."""
        total = 0
        for m in messages:
            if isinstance(m, ChatMessage):
                total += BaseLLMProvider.estimate_tokens(m.content)
            elif isinstance(m, dict):
                total += BaseLLMProvider.estimate_tokens(m.get("content", ""))
            total += 4  # Overhead per message (role, separators)
        return total

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": (
                self._total_latency_ms / self._request_count
                if self._request_count > 0
                else 0
            ),
        }

    def _record_request(self, tokens: int, latency_ms: float) -> None:
        self._request_count += 1
        self._total_tokens += tokens
        self._total_latency_ms += latency_ms

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def normalize_messages(
        messages: list[ChatMessage | dict[str, str]],
    ) -> list[dict[str, str]]:
        """Convert mixed message types to plain dicts."""
        result: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, ChatMessage):
                result.append(m.to_dict())
            elif isinstance(m, dict):
                result.append(m)
            else:
                raise TypeError(f"Unsupported message type: {type(m)}")
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
