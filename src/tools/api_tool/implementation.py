"""
API Tool Implementation
========================

Generic HTTP API client supporting GET, POST, PUT, DELETE, and PATCH.
Features custom headers, authentication, timeout, and JSON response parsing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Union

from src.tools.base import BaseTool, ToolError


class APITool(BaseTool):
    """
    Generic HTTP API client.

    Makes HTTP requests to REST APIs with support for:
        - All common HTTP methods (GET, POST, PUT, DELETE, PATCH)
        - Custom headers and authentication
        - JSON request/response bodies
        - Timeout configuration
        - Response metadata (status, headers, timing)
    """

    @property
    def name(self) -> str:
        return "api_client"

    @property
    def description(self) -> str:
        return (
            "Make HTTP API requests (GET, POST, PUT, DELETE, PATCH). "
            "Supports custom headers, authentication tokens, JSON bodies, "
            "and returns structured response data."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The API endpoint URL.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method. Default: 'GET'.",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                },
                "headers": {
                    "type": "object",
                    "description": "Custom HTTP headers as key-value pairs.",
                },
                "body": {
                    "description": "Request body. For JSON, provide a dict. For raw, provide a string.",
                },
                "auth_token": {
                    "type": "string",
                    "description": "Bearer token for authorization.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds (1-120). Default: 30.",
                    "minimum": 1,
                    "maximum": 120,
                },
                "params": {
                    "type": "object",
                    "description": "URL query parameters as key-value pairs.",
                },
            },
            "required": ["url"],
        }

    def validate_input(self, **kwargs) -> None:
        """Validate request parameters."""
        super().validate_input(**kwargs)
        url = kwargs.get("url", "")
        if not url or not url.strip():
            raise ToolError("URL cannot be empty", tool_name=self.name)

        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ToolError(
                f"Unsupported URL scheme: '{parsed.scheme}'. Only http/https allowed.",
                tool_name=self.name,
            )

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP API request.

        Args:
            url: The API endpoint URL.
            method: HTTP method (default: GET).
            headers: Custom HTTP headers.
            body: Request body (dict for JSON, string for raw).
            auth_token: Bearer token for Authorization header.
            timeout: Request timeout in seconds (default: 30).
            params: URL query parameters.

        Returns:
            Dict with status_code, headers, body (parsed JSON or text), and metadata.
        """
        import httpx

        url = kwargs["url"].strip()
        method = kwargs.get("method", "GET").upper()
        custom_headers = kwargs.get("headers", {})
        body = kwargs.get("body")
        auth_token = kwargs.get("auth_token")
        timeout = kwargs.get("timeout", 30)
        params = kwargs.get("params")

        headers = {**custom_headers}

        # Add auth token
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # Prepare body
        content = None
        if body is not None:
            if isinstance(body, dict):
                content = json.dumps(body).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            elif isinstance(body, str):
                content = body.encode("utf-8")
                headers.setdefault("Content-Type", "text/plain")
            else:
                content = str(body).encode("utf-8")
                headers.setdefault("Content-Type", "text/plain")

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
                max_redirects=10,
            ) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=content,
                    params=params,
                )

            # Parse response body
            response_body = self._parse_response(response)

            return {
                "url": str(response.url),
                "method": method,
                "status_code": response.status_code,
                "status_text": response.reason_phrase,
                "headers": dict(response.headers),
                "body": response_body,
                "content_length": len(response.content),
                "encoding": response.encoding,
            }

        except httpx.TimeoutException:
            raise ToolError(
                f"Request timed out after {timeout}s: {method} {url}",
                tool_name=self.name,
            )
        except httpx.TooManyRedirects:
            raise ToolError(
                f"Too many redirects for: {url}",
                tool_name=self.name,
            )
        except httpx.RequestError as e:
            raise ToolError(
                f"Request failed: {type(e).__name__}: {e}",
                tool_name=self.name,
                details={"url": url, "method": method},
            )

    def _parse_response(self, response: httpx.Response) -> Any:
        """
        Parse the response body.

        Tries JSON first, falls back to text.
        """
        content_type = response.headers.get("content-type", "")

        # Try JSON parsing
        if "json" in content_type:
            try:
                return response.json()
            except (json.JSONDecodeError, ValueError):
                pass

        # Try JSON anyway (some servers don't set content-type correctly)
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            pass

        # Fall back to text
        return response.text
