"""
Memory Manager — Combines Short-Term and Long-Term memory.

Provides a unified interface for the orchestrator to:
  • Store and retrieve conversation context (STM)
  • Persist important facts for future sessions (LTM)
  • Auto-extract and save notable information
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from .ltm import LongTermMemory, LTMEntry
from .stm import ShortTermMemory, MemoryMessage

logger = logging.getLogger(__name__)

# Patterns that suggest an important fact worth persisting
_FACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:my name is|i'm|i am|call me)\s+(.+?)[\.\!\?\n]", re.I),
    re.compile(r"(?:i (?:like|love|prefer|enjoy|hate|dislike))\s+(.+?)[\.\!\?\n]", re.I),
    re.compile(r"(?:i live in|my address is|i'm from)\s+(.+?)[\.\!\?\n]", re.I),
    re.compile(r"(?:my (?:email|phone|number) is)\s+(.+?)[\.\!\?\n]", re.I),
    re.compile(r"(?:remember that|don't forget|note that)\s+(.+?)[\.\!\?\n]", re.I),
    re.compile(r"(?:important|crucial|vital)[:\s]+(.+?)[\.\!\?\n]", re.I),
]


class MemoryManager:
    """
    Unified memory layer combining STM + LTM.

    Parameters
    ----------
    stm_max_messages : int
        Sliding window size for short-term memory.
    ltm_db_path : Path | str | None
        Path to the LTM SQLite database. None uses default.
    auto_save_facts : bool
        Whether to automatically extract and persist important facts.
    """

    def __init__(
        self,
        stm_max_messages: int = 50,
        ltm_db_path: Optional[Path | str] = None,
        auto_save_facts: bool = True,
    ) -> None:
        self.stm = ShortTermMemory(max_messages=stm_max_messages)
        self.ltm = LongTermMemory(db_path=ltm_db_path) if ltm_db_path is not False else LongTermMemory()
        self._auto_save = auto_save_facts
        self._fact_buffer: list[dict[str, Any]] = []
        logger.info("MemoryManager created (auto_save=%s)", auto_save_facts)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(self, session_id: Optional[str] = None) -> None:
        """Start a new session: clear STM, keep LTM."""
        self.stm.clear()
        if session_id:
            self.stm.session_id = session_id
        logger.info("New session: %s", self.stm.session_id)

    def end_session(self) -> None:
        """Flush any pending facts and archive session summary."""
        self._flush_fact_buffer()
        if self.stm.message_count > 0:
            summary = self.stm.get_recent_text(n=10)
            self.ltm.add_fact(
                content=f"[Session {self.stm.session_id}] {summary}",
                category="summary",
                importance=0.3,
                source="session_archive",
            )
        logger.info("Session ended: %s", self.stm.session_id)

    # ------------------------------------------------------------------
    # Conversation interface (wraps STM)
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> MemoryMessage:
        """Add a message to STM and optionally extract facts."""
        msg = self.stm.add_message(role, content, metadata)

        if self._auto_save and role == "user":
            self._try_extract_facts(content)

        return msg

    def get_context(
        self,
        max_messages: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get conversation context for the LLM.

        Injects relevant LTM facts as system context if available.
        """
        # Build enriched system prompt with LTM context
        ltm_context = self._build_ltm_context(system_prompt)
        return self.stm.get_context(max_messages=max_messages, system_prompt=ltm_context)

    def search_memory(self, query: str, limit: int = 5) -> list[LTMEntry]:
        """Search long-term memory."""
        return self.ltm.search(query, limit=limit)

    def save_fact(
        self,
        content: str,
        category: str = "fact",
        keywords: Optional[list[str]] = None,
        importance: float = 0.7,
        source: str = "manual",
    ) -> int:
        """Explicitly save a fact to LTM."""
        return self.ltm.add_fact(
            content=content,
            category=category,
            keywords=keywords,
            importance=importance,
            source=source,
        )

    def get_user_context(self) -> str:
        """Get a text summary of known user facts for system prompt injection."""
        facts = self.ltm.get_all(category="fact", limit=20)
        if not facts:
            return ""
        lines = [f"- {f.content}" for f in facts]
        return "Known user information:\n" + "\n".join(lines)

    def clear_all(self) -> None:
        """Clear everything (use with caution)."""
        self.stm.clear()
        logger.warning("All memory cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_extract_facts(self, text: str) -> None:
        """Run fact-extraction patterns on user text."""
        for pattern in _FACT_PATTERNS:
            match = pattern.search(text)
            if match:
                fact_text = match.group(0).strip()
                self._fact_buffer.append({
                    "content": fact_text,
                    "category": "fact",
                    "importance": 0.7,
                    "source": "auto_extract",
                })
                logger.debug("Extracted fact: %s", fact_text[:60])

        # Flush buffer periodically (every 5 facts)
        if len(self._fact_buffer) >= 5:
            self._flush_fact_buffer()

    def _flush_fact_buffer(self) -> None:
        """Persist all buffered facts to LTM."""
        for fact in self._fact_buffer:
            self.ltm.add_fact(**fact)
        if self._fact_buffer:
            logger.info("Flushed %d facts to LTM", len(self._fact_buffer))
        self._fact_buffer.clear()

    def _build_ltm_context(self, existing_prompt: Optional[str] = None) -> str:
        """Build a system prompt enriched with LTM facts."""
        parts: list[str] = []
        if existing_prompt:
            parts.append(existing_prompt)

        user_context = self.get_user_context()
        if user_context:
            parts.append(user_context)

        # Recent summaries for continuity
        summaries = self.ltm.get_all(category="summary", limit=3)
        if summaries:
            summary_lines = [f"- {s.content[:200]}" for s in summaries]
            parts.append("Recent session context:\n" + "\n".join(summary_lines))

        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        return {
            "stm_messages": self.stm.message_count,
            "stm_estimated_tokens": self.stm.estimated_tokens,
            "ltm_total_entries": self.ltm.count(),
            "ltm_facts": self.ltm.count("fact"),
            "ltm_preferences": self.ltm.count("preference"),
            "ltm_summaries": self.ltm.count("summary"),
            "session_id": self.stm.session_id,
            "pending_facts": len(self._fact_buffer),
        }

    def __repr__(self) -> str:
        return (
            f"MemoryManager(stm={self.stm!r}, ltm_entries={self.ltm.count()})"
        )
