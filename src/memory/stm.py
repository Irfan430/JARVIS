"""
Short-Term Memory (STM) — In-memory sliding window for conversation context.

Maintains a per-session sliding window of the last N messages,
auto-pruning old entries and providing context for LLM calls.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryMessage:
    """A single message stored in short-term memory."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible dict."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class ShortTermMemory:
    """
    Sliding-window short-term memory for a single session.

    Parameters
    ----------
    max_messages : int
        Maximum number of messages to retain (default 50).
    max_token_estimate : int
        Rough character-based token budget (default 120_000 chars ≈ 30k tokens).
    """

    def __init__(
        self,
        max_messages: int = 50,
        max_token_estimate: int = 120_000,
    ) -> None:
        self._window: deque[MemoryMessage] = deque(maxlen=max_messages)
        self._max_messages = max_messages
        self._max_token_estimate = max_token_estimate
        self._total_chars: int = 0
        self._session_id: str = "default"
        logger.debug(
            "STM created: max_messages=%d, char_budget=%d",
            max_messages,
            max_token_estimate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryMessage:
        """
        Append a message to the sliding window.

        If the window is full, the oldest entry is silently dropped.
        Returns the created MemoryMessage.
        """
        msg = MemoryMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._window.append(msg)
        self._total_chars += len(content)
        logger.debug(
            "STM add [%s]: %d chars (total: %d)",
            role,
            len(content),
            self._total_chars,
        )
        return msg

    def get_context(
        self,
        max_messages: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Return messages formatted for an LLM chat call.

        Parameters
        ----------
        max_messages : int | None
            If set, return at most this many recent messages.
        system_prompt : str | None
            If provided, prepend as a system message.
        """
        messages = list(self._window)
        if max_messages is not None:
            messages = messages[-max_messages:]

        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        result.extend(m.to_dict() for m in messages)
        return result

    def get_recent_text(self, n: int = 5) -> str:
        """Return a plain-text summary of the last *n* messages (for embedding)."""
        msgs = list(self._window)[-n:]
        lines = [f"[{m.role}]: {m.content}" for m in msgs]
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all messages and reset counters."""
        self._window.clear()
        self._total_chars = 0
        logger.debug("STM cleared for session %s", self._session_id)

    @property
    def message_count(self) -> int:
        return len(self._window)

    @property
    def estimated_tokens(self) -> int:
        """Rough estimate: ~4 chars per token."""
        return self._total_chars // 4

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all messages to a list of dicts."""
        return [m.to_dict() for m in self._window]

    @classmethod
    def from_list(
        cls,
        messages: list[dict[str, Any]],
        max_messages: int = 50,
        max_token_estimate: int = 120_000,
    ) -> "ShortTermMemory":
        """Reconstruct STM from serialized messages."""
        stm = cls(max_messages=max_messages, max_token_estimate=max_token_estimate)
        for m in messages:
            stm.add_message(
                role=m["role"],
                content=m["content"],
                metadata=m.get("metadata"),
            )
        return stm

    def __len__(self) -> int:
        return len(self._window)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(session={self._session_id!r}, "
            f"msgs={len(self._window)}/{self._max_messages})"
        )
