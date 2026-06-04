"""
Base tool class and registry for all JARVIS tools.
"""

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime


class ToolError(Exception):
    """Raised when a tool execution fails."""
    pass


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    tool_name: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tool_name": self.tool_name,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseTool(ABC):
    """Abstract base class for all JARVIS tools."""

    name: str = "base_tool"
    description: str = "Base tool"
    category: str = "general"
    required_params: List[str] = []
    optional_params: List[str] = []

    def __init__(self):
        self._enabled = True
        self._call_count = 0

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    def validate_params(self, **kwargs) -> None:
        """Validate required parameters are present."""
        missing = [p for p in self.required_params if p not in kwargs]
        if missing:
            raise ToolError(f"Missing required parameters: {', '.join(missing)}")

    async def safe_execute(self, **kwargs) -> ToolResult:
        """Execute with error handling and timing."""
        import time
        start = time.monotonic()
        try:
            self.validate_params(**kwargs)
            result = await self.execute(**kwargs)
            result.execution_time = time.monotonic() - start
            result.tool_name = self.name
            self._call_count += 1
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
                execution_time=time.monotonic() - start,
            )

    def get_definition(self) -> Dict[str, Any]:
        """Get tool definition for LLM function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "required_params": self.required_params,
            "optional_params": self.optional_params,
        }

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def call_count(self) -> int:
        return self._call_count


class ToolRegistry:
    """Registry for managing and dispatching tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Get definitions of all registered tools."""
        return [t.get_definition() for t in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
                tool_name=tool_name,
            )
        if not tool.enabled:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' is disabled",
                tool_name=tool_name,
            )
        return await tool.safe_execute(**kwargs)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
