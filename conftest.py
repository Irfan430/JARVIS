"""
Pytest configuration and shared fixtures.
"""

import sys
import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_query():
    """A sample user query for testing."""
    return "What is the weather in Tokyo?"


@pytest.fixture
def calculator_query():
    """A calculator query for testing."""
    return "Calculate 2 + 2"


@pytest.fixture
def search_query():
    """A search query for testing."""
    return "Search for Python tutorials"


@pytest.fixture
def shell_query():
    """A shell command query for testing."""
    return "Run command ls -la"


@pytest.fixture
def code_query():
    """A code generation query for testing."""
    return "Write a Python function to sort a list"


@pytest.fixture
def file_query():
    """A file operation query for testing."""
    return "Read file /tmp/test.txt"


@pytest.fixture
def browser_query():
    """A browser query for testing."""
    return "Open https://example.com"


@pytest.fixture
def planner_config():
    """Default planner configuration."""
    from src.planner.agent import PlannerConfig
    return PlannerConfig(max_steps=10, temperature=0.3)


@pytest.fixture
def security_config():
    """Default security configuration."""
    from src.security.guard import RateLimiter, InputValidator, SecurityGuard
    return SecurityGuard(
        rate_limiter=RateLimiter(max_requests=30, window_seconds=60),
        validator=InputValidator(max_length=4096),
    )


@pytest.fixture
def tool_registry():
    """A fresh tool registry with default tools."""
    from src.tools.base import ToolRegistry
    from src.tools.calculator_tool import CalculatorTool
    from src.tools.shell_tool import ShellTool
    from src.tools.search_tool import SearchTool
    from src.tools.file_tool import FileTool
    from src.tools.python_tool import PythonTool
    from src.tools.browser_tool import BrowserTool
    from src.tools.api_tool import APITool

    registry = ToolRegistry()
    for tool_cls in [CalculatorTool, ShellTool, SearchTool, FileTool, PythonTool, BrowserTool, APITool]:
        registry.register(tool_cls())
    return registry


@pytest.fixture
def memory_store():
    """A fresh memory store."""
    from src.memory.store import MemoryStore
    return MemoryStore(max_entries=100)
