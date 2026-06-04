"""
Memory store for conversation history and knowledge.
"""

import json
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: MemoryType = MemoryType.CONVERSATION
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: Optional[datetime] = None
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "access_count": self.access_count,
        }


class MemoryStore:
    """
    In-memory conversation store with optional persistence.
    """

    def __init__(self, max_entries: int = 1000, persist_path: Optional[Path] = None):
        self.max_entries = max_entries
        self.persist_path = persist_path
        self._entries: List[MemoryEntry] = []
        self._index: Dict[str, MemoryEntry] = {}

    def add(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.CONVERSATION,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Add a new memory entry."""
        entry = MemoryEntry(
            type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
        )

        self._entries.append(entry)
        self._index[entry.id] = entry

        # Evict oldest low-importance entries if over limit
        if len(self._entries) > self.max_entries:
            self._evict()

        return entry

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        entry = self._index.get(entry_id)
        if entry:
            entry.accessed_at = datetime.now()
            entry.access_count += 1
        return entry

    def search(self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries by content."""
        results = []
        query_lower = query.lower()

        for entry in reversed(self._entries):
            if query_lower in entry.content.lower():
                if memory_type is None or entry.type == memory_type:
                    results.append(entry)
                    if len(results) >= limit:
                        break

        return results

    def get_recent(self, limit: int = 20, memory_type: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """Get recent memory entries."""
        entries = self._entries
        if memory_type:
            entries = [e for e in entries if e.type == memory_type]
        return entries[-limit:]

    def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        entry = self._index.pop(entry_id, None)
        if entry:
            self._entries.remove(entry)
            return True
        return False

    def clear(self) -> int:
        """Clear all memory entries. Returns count of cleared entries."""
        count = len(self._entries)
        self._entries.clear()
        self._index.clear()
        return count

    def _evict(self) -> None:
        """Evict the least important, oldest entry."""
        if not self._entries:
            return
        # Sort by importance then access time
        min_entry = min(self._entries, key=lambda e: (e.importance, e.access_count))
        self.delete(min_entry.id)

    @property
    def size(self) -> int:
        return len(self._entries)

    def get_conversation_history(self, limit: int = 50) -> List[Dict[str, str]]:
        """Get conversation history formatted for LLM context."""
        entries = self.get_recent(limit=limit, memory_type=MemoryType.CONVERSATION)
        return [{"role": e.metadata.get("role", "user"), "content": e.content} for e in entries]

    def save(self) -> None:
        """Persist memory to disk."""
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.to_dict() for e in self._entries]
        self.persist_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Load memory from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        data = json.loads(self.persist_path.read_text())
        for item in data:
            entry = MemoryEntry(
                id=item["id"],
                type=MemoryType(item["type"]),
                content=item["content"],
                metadata=item.get("metadata", {}),
                importance=item.get("importance", 0.5),
                created_at=datetime.fromisoformat(item["created_at"]),
            )
            self._entries.append(entry)
            self._index[entry.id] = entry
