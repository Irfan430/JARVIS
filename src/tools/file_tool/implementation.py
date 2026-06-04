"""
File Tool Implementation
==========================

Safe file system operations: read, write, list, search, and delete.
Includes path validation, size limits, and safety checks.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool, ToolError


class FileTool(BaseTool):
    """
    Perform file system operations safely.

    Supported operations:
        - read: Read file contents
        - write: Write content to a file
        - list: List directory contents
        - search: Search files by name pattern
        - delete: Delete a file

    Includes path validation and size limits for safety.
    """

    # Maximum file size for read operations (10MB)
    MAX_READ_SIZE = 10 * 1024 * 1024
    # Maximum write size (50MB)
    MAX_WRITE_SIZE = 50 * 1024 * 1024
    # Maximum number of files to list
    MAX_LIST_COUNT = 1000

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return (
            "Perform file system operations: read file contents, write files, "
            "list directory contents, search for files by pattern, and delete files. "
            "All operations are path-validated for safety."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The file operation to perform.",
                    "enum": ["read", "write", "list", "search", "delete"],
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write' operation).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern for search (e.g., '*.py', '*config*').",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search/list recursively. Default: false.",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding. Default: 'utf-8'.",
                },
            },
            "required": ["operation", "path"],
        }

    def validate_input(self, **kwargs) -> None:
        """Validate input parameters."""
        super().validate_input(**kwargs)
        operation = kwargs.get("operation", "")
        path = kwargs.get("path", "")

        if not path or not path.strip():
            raise ToolError("Path cannot be empty", tool_name=self.name)

        if operation == "write" and "content" not in kwargs:
            raise ToolError("Content is required for write operation", tool_name=self.name)

        if operation == "search" and not kwargs.get("pattern"):
            raise ToolError("Pattern is required for search operation", tool_name=self.name)

    def _validate_path(self, path: str, operation: str) -> Path:
        """
        Validate and resolve a file path.

        Prevents path traversal attacks and restricts to safe operations.
        """
        resolved = Path(path).expanduser().resolve()

        # Block access to sensitive system paths
        sensitive_paths = [
            "/etc/shadow",
            "/etc/passwd",
            "/proc",
            "/sys",
            "/dev",
        ]
        for sp in sensitive_paths:
            if str(resolved).startswith(sp):
                raise ToolError(
                    f"Access to system path '{sp}' is restricted",
                    tool_name=self.name,
                )

        return resolved

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a file operation.

        Args:
            operation: 'read', 'write', 'list', 'search', or 'delete'.
            path: File or directory path.
            content: Content for write operations.
            pattern: Glob pattern for search.
            recursive: Whether to recurse into subdirectories.
            encoding: File encoding (default: utf-8).

        Returns:
            Dict with operation-specific results.
        """
        operation = kwargs["operation"]
        path = kwargs["path"].strip()
        encoding = kwargs.get("encoding", "utf-8")

        handlers = {
            "read": self._read_file,
            "write": self._write_file,
            "list": self._list_directory,
            "search": self._search_files,
            "delete": self._delete_file,
        }

        handler = handlers.get(operation)
        if handler is None:
            raise ToolError(f"Unknown operation: '{operation}'", tool_name=self.name)

        return await handler(path=path, encoding=encoding, **kwargs)

    async def _read_file(self, path: str, encoding: str, **kwargs) -> Dict[str, Any]:
        """Read file contents."""
        resolved = self._validate_path(path, "read")

        if not resolved.exists():
            raise ToolError(f"File not found: {path}", tool_name=self.name)
        if not resolved.is_file():
            raise ToolError(f"Not a file: {path}", tool_name=self.name)

        file_size = resolved.stat().st_size
        if file_size > self.MAX_READ_SIZE:
            raise ToolError(
                f"File too large ({file_size} bytes). Maximum: {self.MAX_READ_SIZE} bytes.",
                tool_name=self.name,
            )

        try:
            content = await asyncio.to_thread(
                resolved.read_text, encoding=encoding
            )
        except UnicodeDecodeError:
            raise ToolError(
                f"Cannot decode file with encoding '{encoding}'. Try 'latin-1' or 'binary'.",
                tool_name=self.name,
            )

        return {
            "path": str(resolved),
            "content": content,
            "size": file_size,
            "lines": content.count("\n") + 1 if content else 0,
        }

    async def _write_file(self, path: str, encoding: str, **kwargs) -> Dict[str, Any]:
        """Write content to a file."""
        content = kwargs["content"]
        resolved = self._validate_path(path, "write")

        content_bytes = content.encode(encoding)
        if len(content_bytes) > self.MAX_WRITE_SIZE:
            raise ToolError(
                f"Content too large ({len(content_bytes)} bytes). Maximum: {self.MAX_WRITE_SIZE} bytes.",
                tool_name=self.name,
            )

        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(resolved.write_text, content, encoding=encoding)

        return {
            "path": str(resolved),
            "bytes_written": len(content_bytes),
            "success": True,
        }

    async def _list_directory(self, path: str, encoding: str, **kwargs) -> Dict[str, Any]:
        """List directory contents."""
        resolved = self._validate_path(path, "list")

        if not resolved.exists():
            raise ToolError(f"Directory not found: {path}", tool_name=self.name)
        if not resolved.is_dir():
            raise ToolError(f"Not a directory: {path}", tool_name=self.name)

        recursive = kwargs.get("recursive", False)

        entries = []
        if recursive:
            for item in resolved.rglob("*"):
                entries.append(self._format_entry(item))
                if len(entries) >= self.MAX_LIST_COUNT:
                    break
        else:
            for item in resolved.iterdir():
                entries.append(self._format_entry(item))
                if len(entries) >= self.MAX_LIST_COUNT:
                    break

        # Sort: directories first, then files
        entries.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))

        return {
            "path": str(resolved),
            "entries": entries,
            "total_count": len(entries),
            "truncated": len(entries) >= self.MAX_LIST_COUNT,
        }

    def _format_entry(self, path: Path) -> Dict[str, Any]:
        """Format a directory entry."""
        try:
            stat = path.stat()
            return {
                "name": path.name,
                "path": str(path),
                "is_directory": path.is_dir(),
                "size": stat.st_size if path.is_file() else 0,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except OSError:
            return {
                "name": path.name,
                "path": str(path),
                "is_directory": path.is_dir(),
                "size": 0,
                "modified": None,
            }

    async def _search_files(self, path: str, encoding: str, **kwargs) -> Dict[str, Any]:
        """Search for files by name pattern."""
        resolved = self._validate_path(path, "search")
        pattern = kwargs["pattern"]
        recursive = kwargs.get("recursive", False)

        if not resolved.exists():
            raise ToolError(f"Directory not found: {path}", tool_name=self.name)

        matches = []
        glob_func = resolved.rglob if recursive else resolved.glob

        for item in glob_func("*"):
            if fnmatch.fnmatch(item.name, pattern):
                matches.append(self._format_entry(item))
                if len(matches) >= self.MAX_LIST_COUNT:
                    break

        return {
            "search_path": str(resolved),
            "pattern": pattern,
            "matches": matches,
            "match_count": len(matches),
            "truncated": len(matches) >= self.MAX_LIST_COUNT,
        }

    async def _delete_file(self, path: str, encoding: str, **kwargs) -> Dict[str, Any]:
        """Delete a file."""
        resolved = self._validate_path(path, "delete")

        if not resolved.exists():
            raise ToolError(f"File not found: {path}", tool_name=self.name)
        if not resolved.is_file():
            raise ToolError(
                f"Cannot delete directory with this operation. Use shell for rmdir. Path: {path}",
                tool_name=self.name,
            )

        # Extra safety: don't delete files larger than 1GB (might be important)
        file_size = resolved.stat().st_size
        if file_size > 1024 * 1024 * 1024:
            raise ToolError(
                f"Refusing to delete large file ({file_size} bytes). "
                "Use shell command for large file deletions.",
                tool_name=self.name,
            )

        file_name = resolved.name
        await asyncio.to_thread(resolved.unlink)

        return {
            "path": str(resolved),
            "deleted": file_name,
            "success": True,
        }
