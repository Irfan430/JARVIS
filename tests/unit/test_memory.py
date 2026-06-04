"""
Tests for the Memory system.
"""

import pytest
from src.memory.store import MemoryStore, MemoryEntry, MemoryType


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_creation_defaults(self):
        entry = MemoryEntry()
        assert entry.id is not None
        assert entry.type == MemoryType.CONVERSATION
        assert entry.content == ""
        assert entry.importance == 0.5
        assert entry.access_count == 0

    def test_creation_with_values(self):
        entry = MemoryEntry(
            content="Hello, JARVIS!",
            memory_type=MemoryType.FACT,
            importance=0.8,
        )
        assert entry.content == "Hello, JARVIS!"
        assert entry.type == MemoryType.FACT
        assert entry.importance == 0.8

    def test_to_dict(self):
        entry = MemoryEntry(content="test", memory_type=MemoryType.PREFERENCE)
        d = entry.to_dict()
        assert d["content"] == "test"
        assert d["type"] == "preference"
        assert "id" in d
        assert "created_at" in d


class TestMemoryStore:
    """Tests for MemoryStore."""

    def test_creation(self):
        store = MemoryStore()
        assert store.size == 0

    def test_add_entry(self):
        store = MemoryStore()
        entry = store.add("Hello", memory_type=MemoryType.CONVERSATION)
        assert store.size == 1
        assert entry.content == "Hello"

    def test_get_entry(self):
        store = MemoryStore()
        entry = store.add("Test memory")
        retrieved = store.get(entry.id)
        assert retrieved is entry
        assert retrieved.access_count == 1

    def test_get_nonexistent(self):
        store = MemoryStore()
        assert store.get("nonexistent") is None

    def test_search(self):
        store = MemoryStore()
        store.add("Python is a programming language")
        store.add("JavaScript is also popular")
        store.add("Python tutorials are great")

        results = store.search("Python")
        assert len(results) == 2

    def test_search_with_type_filter(self):
        store = MemoryStore()
        store.add("Python is great", memory_type=MemoryType.FACT)
        store.add("I like Python", memory_type=MemoryType.CONVERSATION)

        results = store.search("Python", memory_type=MemoryType.FACT)
        assert len(results) == 1

    def test_search_limit(self):
        store = MemoryStore()
        for i in range(20):
            store.add(f"Memory {i}")

        results = store.search("Memory", limit=5)
        assert len(results) == 5

    def test_get_recent(self):
        store = MemoryStore()
        for i in range(10):
            store.add(f"Entry {i}")

        recent = store.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[-1].content == "Entry 9"

    def test_delete(self):
        store = MemoryStore()
        entry = store.add("To be deleted")
        assert store.size == 1

        result = store.delete(entry.id)
        assert result is True
        assert store.size == 0

    def test_delete_nonexistent(self):
        store = MemoryStore()
        result = store.delete("nonexistent")
        assert result is False

    def test_clear(self):
        store = MemoryStore()
        store.add("Memory 1")
        store.add("Memory 2")
        count = store.clear()
        assert count == 2
        assert store.size == 0

    def test_max_entries_eviction(self):
        store = MemoryStore(max_entries=3)
        e1 = store.add("First", importance=0.1)
        e2 = store.add("Second", importance=0.9)
        e3 = store.add("Third", importance=0.5)
        store.add("Fourth", importance=0.8)

        assert store.size == 3
        # The least important entry should be evicted
        assert store.get(e1.id) is None  # importance 0.1 was evicted

    def test_conversation_history(self):
        store = MemoryStore()
        store.add("Hello", metadata={"role": "user"})
        store.add("Hi there!", metadata={"role": "assistant"})
        store.add("How are you?", metadata={"role": "user"})

        history = store.get_conversation_history()
        assert len(history) == 3
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_save_and_load(self, tmp_path):
        persist_path = tmp_path / "memory.json"
        store = MemoryStore(persist_path=persist_path)
        store.add("Persistent memory 1", importance=0.9)
        store.add("Persistent memory 2", importance=0.7)
        store.save()

        # Load into new store
        store2 = MemoryStore(persist_path=persist_path)
        store2.load()
        assert store2.size == 2

    def test_unique_ids(self):
        store = MemoryStore()
        e1 = store.add("First")
        e2 = store.add("Second")
        assert e1.id != e2.id

    def test_memory_types(self):
        store = MemoryStore()
        for mt in MemoryType:
            store.add(f"Memory of type {mt.value}", memory_type=mt)
        assert store.size == len(MemoryType)
