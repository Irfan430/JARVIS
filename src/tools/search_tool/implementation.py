"""
Search Tool Implementation
==========================

Web search using DuckDuckGo via the duckduckgo_search library.
Returns titles, URLs, and snippets for search results.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.tools.base import BaseTool, ToolError


class SearchTool(BaseTool):
    """
    Search the web using DuckDuckGo.

    This tool performs web searches and returns structured results
    including titles, URLs, and content snippets.

    Features:
        - Configurable result count (1-50)
        - Region-specific searches
        - Safe search options
        - Time range filtering
    """

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web using DuckDuckGo. Returns a list of search results "
            "with titles, URLs, and content snippets. Use this to find information "
            "about any topic on the internet."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-50). Default: 10.",
                    "minimum": 1,
                    "maximum": 50,
                },
                "region": {
                    "type": "string",
                    "description": "Region code for localized results (e.g., 'wt-wt' for worldwide, 'us-en' for US English). Default: 'wt-wt'.",
                },
                "safesearch": {
                    "type": "string",
                    "description": "Safe search level: 'on', 'moderate', or 'off'. Default: 'moderate'.",
                    "enum": ["on", "moderate", "off"],
                },
                "time_range": {
                    "type": "string",
                    "description": "Time range filter: 'd' (day), 'w' (week), 'm' (month), 'y' (year). Default: None (no filter).",
                    "enum": ["d", "w", "m", "y"],
                },
            },
            "required": ["query"],
        }

    def validate_input(self, **kwargs) -> None:
        """Custom validation for search parameters."""
        super().validate_input(**kwargs)
        query = kwargs.get("query", "")
        if not query or not query.strip():
            raise ToolError("Search query cannot be empty", tool_name=self.name)
        if len(query) > 1000:
            raise ToolError(
                f"Query too long ({len(query)} chars). Maximum: 1000 characters.",
                tool_name=self.name,
            )

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Perform a web search.

        Args:
            query: Search query string.
            max_results: Maximum results to return (default: 10).
            region: Region code (default: 'wt-wt').
            safesearch: Safe search level (default: 'moderate').
            time_range: Time range filter (default: None).

        Returns:
            Dict with 'results' list and metadata.
        """
        from duckduckgo_search import DDGS

        query = kwargs["query"].strip()
        max_results = min(kwargs.get("max_results", 10), 50)
        region = kwargs.get("region", "wt-wt")
        safesearch = kwargs.get("safesearch", "moderate")
        time_range = kwargs.get("time_range")

        try:
            # Run the blocking search in a thread pool
            results = await asyncio.to_thread(
                self._perform_search,
                query=query,
                max_results=max_results,
                region=region,
                safesearch=safesearch,
                time_range=time_range,
            )

            return {
                "query": query,
                "result_count": len(results),
                "results": results,
            }

        except Exception as e:
            raise ToolError(
                f"Search failed: {type(e).__name__}: {e}",
                tool_name=self.name,
                details={"query": query},
            )

    def _perform_search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
        time_range: Optional[str],
    ) -> List[Dict[str, str]]:
        """Perform the actual DuckDuckGo search (blocking)."""
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            search_kwargs = {
                "keywords": query,
                "region": region,
                "safesearch": safesearch,
                "max_results": max_results,
            }
            if time_range:
                search_kwargs["timelimit"] = time_range

            raw_results = list(ddgs.text(**search_kwargs))

        # Normalize results
        results = []
        for i, r in enumerate(raw_results, 1):
            results.append({
                "rank": i,
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })

        return results
