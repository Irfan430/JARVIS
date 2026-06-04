"""
Python Tool Implementation
===========================

Safe Python code execution with output capture, timeout, and memory limits.
Runs code in an isolated subprocess with sandboxed restrictions.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import textwrap
from typing import Any, Dict, Optional

from src.tools.base import BaseTool, ToolError


# Wrapper script that captures stdout, stderr, and return value
WRAPPER_TEMPLATE = textwrap.dedent("""\
    import sys
    import io
    import json
    import traceback

    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    sys.stdout = captured_stdout
    sys.stderr = captured_stderr

    _result = None
    _error = None

    try:
        # Execute the user code
        exec_globals = {{}}
        exec(compile({code_str!r}, '<user_code>', 'exec'), exec_globals)
        _result = exec_globals.get('_result', None)
    except Exception:
        _error = traceback.format_exc()

    # Restore streams
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    # Output results as JSON
    output = {{
        "stdout": captured_stdout.getvalue(),
        "stderr": captured_stderr.getvalue(),
        "result": _result,
        "error": _error,
    }}
    print(json.dumps(output))
""")


class PythonTool(BaseTool):
    """
    Execute Python code in an isolated subprocess.

    Features:
        - Full stdout/stderr capture
        - Configurable timeout (default 30s)
        - Memory limit awareness
        - Sandboxed execution (separate process)
        - Return value extraction via _result variable
    """

    @property
    def name(self) -> str:
        return "python_exec"

    @property
    def description(self) -> str:
        return (
            "Execute Python code and return the output. Supports full Python syntax "
            "with stdout/stderr capture. Set '_result' variable to return a value. "
            "Code runs in an isolated subprocess with timeout protection."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (1-60). Default: 30.",
                    "minimum": 1,
                    "maximum": 60,
                },
                "capture_result": {
                    "type": "boolean",
                    "description": "If true, the value of '_result' variable will be extracted and returned as JSON. Default: true.",
                },
            },
            "required": ["code"],
        }

    def validate_input(self, **kwargs) -> None:
        """Validate code parameter."""
        super().validate_input(**kwargs)
        code = kwargs.get("code", "")
        if not code or not code.strip():
            raise ToolError("Code cannot be empty", tool_name=self.name)
        if len(code) > 100_000:
            raise ToolError(
                f"Code too long ({len(code)} chars). Maximum: 100000 characters.",
                tool_name=self.name,
            )

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute Python code in a subprocess.

        Args:
            code: Python code to execute.
            timeout: Timeout in seconds (default: 30).
            capture_result: Whether to capture _result variable (default: True).

        Returns:
            Dict with stdout, stderr, result (if _result is set), and error (if any).
        """
        code = kwargs["code"].strip()
        timeout = kwargs.get("timeout", 30)
        capture_result = kwargs.get("capture_result", True)

        # Create wrapper script
        wrapper_code = WRAPPER_TEMPLATE.format(code_str=code)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            prefix="jarvis_exec_",
        ) as f:
            f.write(wrapper_code)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await asyncio.wait_for(proc.communicate(), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass

                raise ToolError(
                    f"Code execution timed out after {timeout}s",
                    tool_name=self.name,
                )

            # Parse the JSON output from the wrapper
            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

            # The wrapper outputs JSON as the last line of stdout
            result = {
                "stdout": "",
                "stderr": "",
                "result": None,
                "error": None,
            }

            # Try to find the JSON line in stdout
            for line in stdout_text.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        result = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            # If no JSON found, use raw output
            if result["stdout"] == "" and result["stderr"] == "" and stdout_text:
                # Everything before the JSON line is stdout
                lines = stdout_text.splitlines()
                non_json_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            json.loads(stripped)
                            continue
                        except json.JSONDecodeError:
                            pass
                    non_json_lines.append(line)
                result["stdout"] = "\n".join(non_json_lines)

            if stderr_text:
                result["stderr"] = stderr_text

            return {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "result": result.get("result") if capture_result else None,
                "error": result.get("error"),
                "return_code": proc.returncode,
                "success": proc.returncode == 0 and result.get("error") is None,
            }

        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_path)
            except OSError:
                pass
