"""
Shell Tool Implementation
==========================

Execute shell commands with timeout support, stdout/stderr capture,
and safety checks to block dangerous commands.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool, ToolError


# Dangerous commands that should be blocked or require confirmation
BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    ":(){:|:&};:",  # fork bomb
    "dd if=/dev/zero of=/dev/sd",
    "dd if=/dev/random of=/dev/sd",
    "mv / ",
    "chmod -R 777 /",
    "chown -R ",
    "> /dev/sda",
}

# Patterns that indicate destructive operations
DESTRUCTIVE_PATTERNS = [
    r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*)?\s+/",  # rm -rf /
    r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*)?\s+\*",  # rm -rf * (in root context)
    r">\s*/dev/sd[a-z]",  # writing to disk devices
    r"mkfs\.",  # formatting filesystems
    r"shutdown",
    r"reboot",
    r"init\s+[06]",  # init 0 (halt) or init 6 (reboot)
    r"halt",
    r"poweroff",
]


class ShellTool(BaseTool):
    """
    Execute shell commands in a controlled environment.

    Features:
        - Timeout support (configurable, default 30s)
        - stdout and stderr capture
        - Return code checking
        - Working directory specification
        - Safety checks for dangerous commands
    """

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command and return the output. Supports timeout, "
            "working directory, and captures both stdout and stderr. Dangerous "
            "commands (rm -rf /, mkfs, etc.) are blocked for safety."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (1-300). Default: 30.",
                    "minimum": 1,
                    "maximum": 300,
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for command execution.",
                },
                "env": {
                    "type": "object",
                    "description": "Additional environment variables as key-value pairs.",
                },
                "allow_dangerous": {
                    "type": "boolean",
                    "description": "Override safety checks. Use with extreme caution. Default: false.",
                },
            },
            "required": ["command"],
        }

    def validate_input(self, **kwargs) -> None:
        """Validate command and check for dangerous operations."""
        super().validate_input(**kwargs)
        command = kwargs.get("command", "")
        if not command or not command.strip():
            raise ToolError("Command cannot be empty", tool_name=self.name)
        if len(command) > 10000:
            raise ToolError(
                f"Command too long ({len(command)} chars). Maximum: 10000 characters.",
                tool_name=self.name,
            )

        # Safety check
        if not kwargs.get("allow_dangerous", False):
            self._check_safety(command)

    def _check_safety(self, command: str) -> None:
        """Check if a command is potentially dangerous."""
        normalized = command.strip().lower()

        # Check blocked commands
        for blocked in BLOCKED_COMMANDS:
            if blocked in normalized:
                raise ToolError(
                    f"Blocked dangerous command detected: '{blocked}'. "
                    "Use allow_dangerous=true to override (not recommended).",
                    tool_name=self.name,
                )

        # Check destructive patterns
        for pattern in DESTRUCTIVE_PATTERNS:
            if re.search(pattern, normalized):
                raise ToolError(
                    f"Potentially destructive command detected matching pattern: '{pattern}'. "
                    "Use allow_dangerous=true to override (not recommended).",
                    tool_name=self.name,
                )

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: The shell command to run.
            timeout: Timeout in seconds (default: 30).
            cwd: Working directory.
            env: Additional environment variables.
            allow_dangerous: Override safety checks.

        Returns:
            Dict with stdout, stderr, return_code, and timed_out flag.
        """
        command = kwargs["command"].strip()
        timeout = kwargs.get("timeout", 30)
        cwd = kwargs.get("cwd")
        env = kwargs.get("env")
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=5,
                    )
                except (asyncio.TimeoutError, ProcessLookupError):
                    stdout_bytes = b""
                    stderr_bytes = b"Process killed due to timeout"

            # Decode output (handle encoding errors gracefully)
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

            # Truncate very long outputs
            max_output = 100_000  # 100KB
            if len(stdout) > max_output:
                stdout = stdout[:max_output] + f"\n... (truncated, total {len(stdout)} chars)"
            if len(stderr) > max_output:
                stderr = stderr[:max_output] + f"\n... (truncated, total {len(stderr)} chars)"

            return {
                "stdout": stdout,
                "stderr": stderr,
                "return_code": proc.returncode,
                "timed_out": timed_out,
                "success": proc.returncode == 0 and not timed_out,
            }

        except OSError as e:
            raise ToolError(
                f"Failed to execute command: {e}",
                tool_name=self.name,
            )
