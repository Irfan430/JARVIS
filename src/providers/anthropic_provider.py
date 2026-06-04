"""
Anthropic LLM Provider — Claude models.

Uses the official `anthropic` async SDK. Supports streaming.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Optional

from .base import BaseLLMProvider, ChatMessage, ChatResponse, StreamingChunk

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.

    Parameters
    ----------
    api_key : str
        Anthropic API key (or set ANTHROPIC_API_KEY env var).
    default_model : str
        Default Claude model.
    """

    name = "anthropic"

    DEFAULT_MODELS = {
        "fast": "claude-sonnet-4-20250514",
        "balanced": "claude-sonnet-4-20250514",
        "powerful": "claude-opus-4-20250514",
        "default": "claude-sonnet-4-20250514",
    }

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "claude-sonnet-4-20250514",
        **kwargs: Any,
    ) -> None:
        if anthropic is None:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        super().__init__(api_key=api_key, **kwargs)
        self.default_model = default_model
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or None,
        )
        logger.info("Anthropic provider initialised (model=%s)", default_model)

    # ── Message conversion ──────────────────────────────────────────

    @staticmethod
    def _convert_messages(
        messages: list[ChatMessage | dict[str, str]],
    ) -> tuple[Optional[str], list[dict[str, str]]]:
        """
        Convert messages to Anthropic format.

        Anthropic requires a separate `system` param, so extract it.
        Returns (system_prompt, messages).
        """
        system_prompt: Optional[str] = None
        converted: list[dict[str, str]] = []

        for m in messages:
            if isinstance(m, ChatMessage):
                role = m.role
                content = m.content
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                continue

            if role == "system":
                system_prompt = content
            else:
                # Anthropic only supports "user" and "assistant" roles
                if role not in ("user", "assistant"):
                    role = "user"
                converted.append({"role": role, "content": content})

        return system_prompt, converted

    # ── Core Interface ──────────────────────────────────────────────

    async def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion to Anthropic."""
        model = model or self.default_model
        system_prompt, anthropic_msgs = self._convert_messages(messages)
        start = time.perf_counter()

        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": anthropic_msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if system_prompt:
                create_kwargs["system"] = system_prompt

            response = await self._client.messages.create(**create_kwargs)
        except Exception as e:
            logger.error("Anthropic chat error: %s", e)
            raise

        latency_ms = (time.perf_counter() - start) * 1000

        # Extract text content from response blocks
        content_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                content_text += block.text

        usage_dict = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        finish_reason = "stop"
        if response.stop_reason:
            finish_reason = response.stop_reason

        result = ChatResponse(
            content=content_text,
            model=response.model,
            finish_reason=finish_reason,
            usage=usage_dict,
            latency_ms=latency_ms,
        )

        self._record_request(result.total_tokens, latency_ms)
        logger.info(
            "Anthropic chat: %d tokens in %.0fms",
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
        """Stream a chat completion from Anthropic."""
        model = model or self.default_model
        system_prompt, anthropic_msgs = self._convert_messages(messages)
        chunk_index = 0
        start = time.perf_counter()

        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": anthropic_msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if system_prompt:
                create_kwargs["system"] = system_prompt

            full_content = ""
            async with self._client.messages.stream(**create_kwargs) as stream:
                async for text in stream.text_stream:
                    full_content += text
                    yield StreamingChunk(
                        delta=text,
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1

                # Get final message for usage stats
                final_msg = await stream.get_final_message()
                usage_info = {
                    "prompt_tokens": final_msg.usage.input_tokens,
                    "completion_tokens": final_msg.usage.output_tokens,
                    "total_tokens": final_msg.usage.input_tokens + final_msg.usage.output_tokens,
                }

            latency_ms = (time.perf_counter() - start) * 1000
            self._record_request(self.estimate_tokens(full_content), latency_ms)

            # Final chunk with usage
            yield StreamingChunk(
                delta="",
                finish_reason="stop",
                usage=usage_info,
                chunk_index=chunk_index,
            )

        except Exception as e:
            logger.error("Anthropic stream error: %s", e)
            raise
