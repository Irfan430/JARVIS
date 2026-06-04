"""
Long-Term Memory (LTM) — SQLite-backed persistent storage.

Stores user facts, preferences, and conversation summaries that persist
across sessions. Supports keyword search and full-text retrieval.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("~/.jarvis/memory/ltm.db").expanduser()


@dataclass
class LTMEntry:
    """A single long-term memory entry."""

    id: Optional[int] = None
    category: str = "fact"  # "fact" | "preference" | "summary" | "note"
    content: str = ""
    keywords: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0.0 – 1.0
    source: str = ""
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "keywords": self.keywords,
            "importance": self.importance,
            "source": self.source,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


class LongTermMemory:
    """
    SQLite-backed long-term memory.

    Parameters
    ----------
    db_path : Path | str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: Path | str = _DEFAULT_DB) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()
        logger.info("LTM initialised: %s", self._db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                category    TEXT    NOT NULL DEFAULT 'fact',
                content     TEXT    NOT NULL,
                keywords    TEXT    NOT NULL DEFAULT '[]',
                importance  REAL    NOT NULL DEFAULT 0.5,
                source      TEXT    NOT NULL DEFAULT '',
                created_at  REAL    NOT NULL,
                accessed_at REAL    NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata    TEXT    NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_entries_category
                ON entries(category);
            CREATE INDEX IF NOT EXISTS idx_entries_importance
                ON entries(importance);

            -- FTS5 virtual table for full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
                USING fts5(content, keywords, content=entries, content_rowid=id);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "fact",
        keywords: Optional[list[str]] = None,
        importance: float = 0.5,
        source: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """
        Insert a new memory entry and return its row ID.
        Automatically extracts keywords from content if none provided.
        """
        if keywords is None:
            keywords = self._extract_keywords(content)

        now = time.time()
        cur = self._conn.execute(
            """
            INSERT INTO entries
                (category, content, keywords, importance, source,
                 created_at, accessed_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                category,
                content,
                json.dumps(keywords),
                importance,
                source,
                now,
                now,
                json.dumps(metadata or {}),
            ),
        )
        row_id = cur.lastrowid

        # Update FTS index
        self._conn.execute(
            "INSERT INTO entries_fts(rowid, content, keywords) VALUES (?, ?, ?)",
            (row_id, content, json.dumps(keywords)),
        )
        self._conn.commit()
        logger.info("LTM add [%s] id=%d: %s", category, row_id, content[:80])
        return row_id  # type: ignore[return-value]

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> list[LTMEntry]:
        """
        Full-text keyword search across stored memories.
        Returns results ordered by relevance then importance.
        """
        # Use FTS5 for relevance ranking
        sql = """
            SELECT e.*, rank
            FROM entries_fts fts
            JOIN entries e ON e.id = fts.rowid
            WHERE entries_fts MATCH ?
        """
        params: list[Any] = [query]

        if category:
            sql += " AND e.category = ?"
            params.append(category)
        if min_importance > 0:
            sql += " AND e.importance >= ?"
            params.append(min_importance)

        sql += " ORDER BY rank, e.importance DESC LIMIT ?"
        params.append(limit)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS query is malformed
            logger.warning("FTS search failed, falling back to LIKE")
            rows = self._fallback_search(query, category, limit, min_importance)

        entries = [self._row_to_entry(r) for r in rows]
        # Update access times
        for e in entries:
            self._touch_entry(e.id)  # type: ignore[arg-type]

        logger.debug("LTM search '%s': %d results", query, len(entries))
        return entries

    def _fallback_search(
        self,
        query: str,
        category: Optional[str],
        limit: int,
        min_importance: float,
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM entries WHERE (content LIKE ? OR keywords LIKE ?)"
        like_q = f"%{query}%"
        params: list[Any] = [like_q, like_q]
        if category:
            sql += " AND category = ?"
            params.append(category)
        if min_importance > 0:
            sql += " AND importance >= ?"
            params.append(min_importance)
        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(limit)
        return self._conn.execute(sql, params).fetchall()

    def get_all(
        self,
        category: Optional[str] = None,
        limit: int = 100,
        order_by: str = "created_at DESC",
    ) -> list[LTMEntry]:
        """Retrieve all entries, optionally filtered by category."""
        # Whitelist order_by to prevent injection
        allowed_orders = {
            "created_at DESC", "created_at ASC",
            "importance DESC", "importance ASC",
            "accessed_at DESC", "accessed_at ASC",
        }
        if order_by not in allowed_orders:
            order_by = "created_at DESC"

        sql = f"SELECT * FROM entries"  # noqa: S608
        params: list[Any] = []
        if category:
            sql += " WHERE category = ?"
            params.append(category)
        sql += f" ORDER BY {order_by} LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_by_id(self, entry_id: int) -> Optional[LTMEntry]:
        """Fetch a single entry by ID."""
        row = self._conn.execute(
            "SELECT * FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def update(self, entry_id: int, **fields: Any) -> bool:
        """Update specific fields of an entry. Returns True if updated."""
        allowed = {"content", "category", "keywords", "importance", "source", "metadata"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False

        if "keywords" in updates and isinstance(updates["keywords"], list):
            updates["keywords"] = json.dumps(updates["keywords"])
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            updates["metadata"] = json.dumps(updates["metadata"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [entry_id]
        self._conn.execute(f"UPDATE entries SET {set_clause} WHERE id = ?", values)
        self._conn.commit()
        return self._conn.total_changes > 0

    def delete(self, entry_id: int) -> bool:
        """Delete an entry by ID."""
        self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self._conn.execute(
            "DELETE FROM entries_fts WHERE rowid = ?", (entry_id,)
        )
        self._conn.commit()
        return True

    def count(self, category: Optional[str] = None) -> int:
        """Count entries, optionally filtered."""
        if category:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE category = ?", (category,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _touch_entry(self, entry_id: int) -> None:
        """Update access time and count."""
        self._conn.execute(
            "UPDATE entries SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            (time.time(), entry_id),
        )
        self._conn.commit()

    @staticmethod
    def _extract_keywords(text: str, max_kw: int = 8) -> list[str]:
        """Simple keyword extraction: lowercase, remove stopwords, take top N."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "don", "now", "and", "but", "or", "if", "this", "that", "it", "i",
        }
        import re
        words = re.findall(r"[a-z0-9]{3,}", text.lower())
        seen: set[str] = set()
        keywords: list[str] = []
        for w in words:
            if w not in stopwords and w not in seen:
                seen.add(w)
                keywords.append(w)
                if len(keywords) >= max_kw:
                    break
        return keywords

    def _row_to_entry(self, row: sqlite3.Row) -> LTMEntry:
        return LTMEntry(
            id=row["id"],
            category=row["category"],
            content=row["content"],
            keywords=json.loads(row["keywords"]),
            importance=row["importance"],
            source=row["source"],
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug("LTM connection closed")

    def __enter__(self) -> "LongTermMemory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"LongTermMemory(db={self._db_path!r}, entries={self.count()})"
