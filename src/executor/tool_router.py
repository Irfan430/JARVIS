"""
Tool Router — Registry, dispatch, and validation for JARVIS tool calls.

Every tool is a coroutine that accepts a dict of parameters and returns a
ToolResult. The router handles registration, lookup, input validation, and
fallback logic when a tool is unavailable or errors out.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Coroutine, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Metadata about a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)
    handler: Callable[..., Awaitable[ToolResult]] | None = None
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function schema for LLM tool-use."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """Successful tool execution result."""

    tool_name: str
    output: Any
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return True


@dataclass
class ToolError:
    """Failed tool execution result."""

    tool_name: str
    error: str
    error_type: str = "ToolError"
    elapsed_ms: float = 0.0
    traceback: str = ""
    retryable: bool = False

    @property
    def success(self) -> bool:
        return False


# Type alias for the union
ToolOutcome = ToolResult | ToolError

# Handler signature: async (params: dict) -> ToolResult
ToolHandler = Callable[..., Awaitable[ToolResult]]

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ToolRouter:
    """
    Central registry that maps tool names → handlers and dispatches calls.

    Supports:
      - Registration with metadata / schema
      - Lookup by name or tag
      - Input validation (required params)
      - Fallback handlers when a tool is missing or errors out
      - Concurrency limits via a semaphore
    """

    def __init__(self, *, max_concurrent: int = 10) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._fallbacks: dict[str, ToolHandler] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        logger.info("ToolRouter created (max_concurrent=%d)", max_concurrent)

    # -- Registration -------------------------------------------------------

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        required_params: list[str] | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
    ) -> ToolDefinition:
        """Register a tool handler."""
        if name in self._tools:
            logger.warning("Overwriting existing tool '%s'", name)

        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
            required_params=required_params or [],
            handler=handler,
            tags=tags or [],
            enabled=enabled,
        )
        self._tools[name] = definition
        logger.info("Registered tool '%s' (tags=%s, enabled=%s)", name, tags, enabled)
        return definition

    def register_function(self, fn: ToolHandler) -> ToolDefinition:
        """
        Auto-register a decorated coroutine.  Infers name, description,
        required params from the function signature and docstring.
        """
        name = getattr(fn, "_tool_name", fn.__name__)
        description = getattr(fn, "_tool_description", fn.__doc__ or "")
        sig = inspect.signature(fn)
        required = [
            p.name
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        return self.register(
            name,
            fn,
            description=description.strip(),
            required_params=required,
        )

    def register_fallback(self, tool_name: str, handler: ToolHandler) -> None:
        """Register a fallback handler for a specific tool."""
        self._fallbacks[tool_name] = handler
        logger.info("Registered fallback for tool '%s'", tool_name)

    # -- Lookup -------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self, *, enabled_only: bool = True) -> list[ToolDefinition]:
        """Return all registered tools."""
        if enabled_only:
            return [t for t in self._tools.values() if t.enabled]
        return list(self._tools.values())

    def list_by_tag(self, tag: str) -> list[ToolDefinition]:
        """Return tools matching a given tag."""
        return [t for t in self._tools.values() if tag in t.tags and t.enabled]

    def get_schemas(self, *, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Return OpenAI-style tool schemas for all registered tools."""
        return [t.to_schema() for t in self.list_tools(enabled_only=enabled_only)]

    # -- Validation ---------------------------------------------------------

    def validate_params(self, tool_name: str, params: dict[str, Any]) -> list[str]:
        """
        Validate that required parameters are present.
        Returns a list of error messages (empty == valid).
        """
        definition = self._tools.get(tool_name)
        if definition is None:
            return [f"Tool '{tool_name}' is not registered"]
        if not definition.enabled:
            return [f"Tool '{tool_name}' is disabled"]
        errors: list[str] = []
        for rp in definition.required_params:
            if rp not in params or params[rp] is None:
                errors.append(f"Missing required parameter '{rp}' for tool '{tool_name}'")
        return errors

    # -- Dispatch -----------------------------------------------------------

    async def dispatch(
        self, tool_name: str, params: dict[str, Any], *, timeout: float = 60.0
    ) -> ToolOutcome:
        """
        Validate inputs and call the tool handler with concurrency control.

        Returns ToolResult on success or ToolError on failure.
        """
        start = time.monotonic()

        # Look up definition
        definition = self._tools.get(tool_name)
        if definition is None:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Tool '%s' not found", tool_name)
            return ToolError(
                tool_name=tool_name,
                error=f"Tool '{tool_name}' is not registered",
                error_type="LookupError",
                elapsed_ms=elapsed,
            )

        if not definition.enabled:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Tool '%s' is disabled", tool_name)
            return ToolError(
                tool_name=tool_name,
                error=f"Tool '{tool_name}' is disabled",
                error_type="DisabledError",
                elapsed_ms=elapsed,
            )

        # Validate
        errors = self.validate_params(tool_name, params)
        if errors:
            elapsed = (time.monotonic() - start) * 1000
            return ToolError(
                tool_name=tool_name,
                error="; ".join(errors),
                error_type="ValidationError",
                elapsed_ms=elapsed,
                retryable=False,
            )

        # Execute
        handler = definition.handler
        if handler is None:
            elapsed = (time.monotonic() - start) * 1000
            return ToolError(
                tool_name=tool_name,
                error=f"Tool '{tool_name}' has no handler",
                error_type="ConfigError",
                elapsed_ms=elapsed,
            )

        try:
            async with self._semaphore:
                result = await asyncio.wait_for(handler(params), timeout=timeout)
            elapsed = (time.monotonic() - start) * 1000
            logger.info(
                "Tool '%s' executed in %.1fms", tool_name, elapsed
            )
            if isinstance(result, ToolResult):
                result.elapsed_ms = elapsed
                return result
            # If handler returns a raw value, wrap it
            return ToolResult(
                tool_name=tool_name, output=result, elapsed_ms=elapsed
            )
        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Tool '%s' timed out after %.1fs", tool_name, timeout)
            return ToolError(
                tool_name=tool_name,
                error=f"Tool '{tool_name}' timed out after {timeout}s",
                error_type="TimeoutError",
                elapsed_ms=elapsed,
                retryable=True,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            tb = traceback.format_exc()
            logger.error("Tool '%s' raised %s: %s", tool_name, type(exc).__name__, exc)

            # Try fallback
            fallback = self._fallbacks.get(tool_name)
            if fallback:
                logger.info("Attempting fallback for tool '%s'", tool_name)
                try:
                    async with self._semaphore:
                        fb_result = await asyncio.wait_for(
                            fallback(params), timeout=timeout
                        )
                    fb_elapsed = (time.monotonic() - start) * 1000
                    if isinstance(fb_result, ToolResult):
                        fb_result.elapsed_ms = fb_elapsed
                        fb_result.metadata["fallback"] = True
                        return fb_result
                    return ToolResult(
                        tool_name=tool_name,
                        output=fb_result,
                        elapsed_ms=fb_elapsed,
                        metadata={"fallback": True},
                    )
                except Exception:
                    logger.error("Fallback for '%s' also failed", tool_name)

            return ToolError(
                tool_name=tool_name,
                error=str(exc),
                error_type=type(exc).__name__,
                elapsed_ms=elapsed,
                traceback=tb,
                retryable=isinstance(exc, (ConnectionError, TimeoutError, OSError)),
            )

    async def dispatch_many(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        *,
        timeout: float = 60.0,
    ) -> list[ToolOutcome]:
        """Dispatch multiple tool calls concurrently."""
        tasks = [self.dispatch(name, params, timeout=timeout) for name, params in calls]
        return list(await asyncio.gather(*tasks))
