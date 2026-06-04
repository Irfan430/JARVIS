"""
Browser Tool Implementation
=============================

Web content fetching and extraction using httpx.
Supports HTML and text extraction from web pages.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urljoin

from src.tools.base import BaseTool, ToolError


class BrowserTool(BaseTool):
    """
    Fetch and extract content from web pages.

    Uses httpx for HTTP requests with support for:
        - HTML content retrieval
        - Plain text extraction
        - Response metadata (headers, status code, size)
        - Custom headers and user agents
    """

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return (
            "Fetch content from a web URL. Can return raw HTML or extract "
            "plain text. Uses httpx for reliable HTTP requests with "
            "configurable headers and timeout."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "extract_mode": {
                    "type": "string",
                    "description": "Content extraction mode: 'html' (raw HTML), 'text' (plain text). Default: 'text'.",
                    "enum": ["html", "text"],
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length to return (chars). Default: 100000.",
                    "minimum": 100,
                    "maximum": 500000,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds (1-120). Default: 30.",
                    "minimum": 1,
                    "maximum": 120,
                },
                "headers": {
                    "type": "object",
                    "description": "Custom HTTP headers as key-value pairs.",
                },
            },
            "required": ["url"],
        }

    def validate_input(self, **kwargs) -> None:
        """Validate URL and parameters."""
        super().validate_input(**kwargs)
        url = kwargs.get("url", "")
        if not url or not url.strip():
            raise ToolError("URL cannot be empty", tool_name=self.name)

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ToolError(
                f"Unsupported URL scheme: '{parsed.scheme}'. Only http/https allowed.",
                tool_name=self.name,
            )

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch content from a web URL.

        Args:
            url: The URL to fetch.
            extract_mode: 'html' or 'text' (default: 'text').
            max_length: Maximum content length (default: 100000).
            timeout: Request timeout in seconds (default: 30).
            headers: Custom HTTP headers.

        Returns:
            Dict with content, headers, status code, and metadata.
        """
        import httpx

        url = kwargs["url"].strip()
        extract_mode = kwargs.get("extract_mode", "text")
        max_length = kwargs.get("max_length", 100_000)
        timeout = kwargs.get("timeout", 30)
        custom_headers = kwargs.get("headers", {})

        headers = {
            "User-Agent": self.DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            **custom_headers,
        }

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
                max_redirects=10,
                verify=True,
            ) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            # Extract content based on mode
            if extract_mode == "html":
                content = response.text[:max_length]
            else:
                content = self._extract_text(response.text)
                content = content[:max_length]

            # Parse response headers
            response_headers = dict(response.headers)

            return {
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": content_type,
                "content": content,
                "content_length": len(content),
                "encoding": response.encoding,
                "headers": response_headers,
                "redirected": str(response.url) != url,
            }

        except httpx.TimeoutException:
            raise ToolError(
                f"Request timed out after {timeout}s: {url}",
                tool_name=self.name,
            )
        except httpx.HTTPStatusError as e:
            raise ToolError(
                f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                tool_name=self.name,
                details={"url": url, "status_code": e.response.status_code},
            )
        except httpx.RequestError as e:
            raise ToolError(
                f"Request failed: {type(e).__name__}: {e}",
                tool_name=self.name,
                details={"url": url},
            )

    def _extract_text(self, html: str) -> str:
        """
        Extract plain text from HTML.

        A simple but effective HTML-to-text converter that strips tags
        and normalizes whitespace.
        """
        # Remove script and style elements with content
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML comments
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # Replace block elements with newlines
        block_tags = r"(div|p|br|h[1-6]|li|tr|th|td|blockquote|pre|section|article|header|footer|nav|aside)"
        html = re.sub(f"<{block_tags}[^>]*/?>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(f"</{block_tags}>", "\n", html, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        html = re.sub(r"<[^>]+>", "", html)

        # Decode common HTML entities
        entities = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&apos;": "'",
            "&nbsp;": " ",
            "&#39;": "'",
            "&#x27;": "'",
        }
        for entity, char in entities.items():
            html = html.replace(entity, char)

        # Normalize whitespace
        lines = [line.strip() for line in html.splitlines()]
        # Remove empty lines and collapse multiple spaces
        lines = [re.sub(r"\s+", " ", line) for line in lines if line]

        return "\n".join(lines)
