"""
OpenAI LLM Provider — GPT-4o, GPT-5, and other OpenAI models.

Uses the official `openai` async SDK. Supports streaming.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Optional

from .base import BaseLLMProvider, ChatMessage, ChatResponse, StreamingChunk

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[misc,assignment]


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API provider.

    Parameters
    ----------
    api_key : str
        OpenAI API key (or set OPENAI_API_KEY env var).
    default_model : str
        Default model to use.
    base_url : str | None
        Custom base URL (for proxies / Azure).
    """

    name = "openai"

    DEFAULT_MODELS = {
        "fast": "gpt-4o-mini",
        "balanced": "gpt-4o",
        "powerful": "gpt-4o",
        "default": "gpt-4o",
    }

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "gpt-4o",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        super().__init__(api_key=api_key, **kwargs)
        self.default_model = default_model
        self._client = AsyncOpenAI(
            api_key=api_key or None,
            base_url=base_url,
        )
        logger.info("OpenAI provider initialised (model=%s)", default_model)

    async def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request to OpenAI."""
        model = model or self.default_model
        norm_msgs = self.normalize_messages(messages)
        start = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=norm_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ("stream",)},
            )
        except Exception as e:
            logger.error("OpenAI chat error: %s", e)
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        usage_dict = None
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        result = ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason or "stop",
            usage=usage_dict,
            latency_ms=latency_ms,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )

        self._record_request(result.total_tokens, latency_ms)
        logger.info(
            "OpenAI chat: %d tokens in %.0fms",
            result.total_tokens,
            latency_ms,
        )
        return result

    async def stream(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        """Stream a chat completion from OpenAI."""
        model = model or self.default_model
        norm_msgs = self.normalize_messages(messages)
        chunk_index = 0
        start = time.perf_counter()

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=norm_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            full_content = ""
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason
                text = delta.content if delta and delta.content else ""
                full_content += text

                usage_info = None
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_info = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

                yield StreamingChunk(
                    delta=text,
                    finish_reason=finish,
                    usage=usage_info,
                    chunk_index=chunk_index,
                )
                chunk_index += 1

            latency_ms = (time.perf_counter() - start) * 1000
            self._record_request(self.estimate_tokens(full_content), latency_ms)

        except Exception as e:
            logger.error("OpenAI stream error: %s", e)
            raise
