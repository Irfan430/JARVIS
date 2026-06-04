"""
Provider Manager — Multi-provider orchestration with fallback and load balancing.

Manages multiple LLM providers with:
  • Automatic fallback on failure
  • Weighted round-robin load balancing
  • Per-provider rate limiting
  • Provider selection by tier (fast / balanced / powerful)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from .base import BaseLLMProvider, ChatMessage, ChatResponse, StreamingChunk

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Simple token-bucket rate limiter."""

    max_requests: int = 30
    window_seconds: float = 60.0
    _timestamps: list[float] = field(default_factory=list)

    def acquire(self) -> bool:
        """Try to acquire a rate limit slot. Returns False if exceeded."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self.max_requests:
            return False
        self._timestamps.append(now)
        return True

    @property
    def wait_time(self) -> float:
        """Seconds until the next slot is available."""
        if not self._timestamps:
            return 0.0
        oldest = self._timestamps[0]
        wait = self.window_seconds - (time.monotonic() - oldest)
        return max(0.0, wait)


@dataclass
class ProviderEntry:
    """Registered provider with metadata."""

    provider: BaseLLMProvider
    weight: float = 1.0  # For load balancing
    enabled: bool = True
    tier: str = "balanced"  # "fast" | "balanced" | "powerful"
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    consecutive_failures: int = 0
    max_failures: int = 5  # Disable after N consecutive failures


class ProviderManager:
    """
    Manages multiple LLM providers with fallback and load balancing.

    Usage::

        manager = ProviderManager()
        manager.register(OpenAIProvider(api_key="sk-..."), tier="powerful")
        manager.register(MiMoProvider(api_key="..."), tier="fast", weight=2.0)
        response = await manager.chat(messages, tier="balanced")
    """

    def __init__(self, fallback_enabled: bool = True) -> None:
        self._providers: dict[str, ProviderEntry] = {}
        self._tier_order: dict[str, list[str]] = {
            "fast": ["fast", "balanced", "powerful"],
            "balanced": ["balanced", "fast", "powerful"],
            "powerful": ["powerful", "balanced", "fast"],
        }
        self._fallback_enabled = fallback_enabled
        self._rr_index: dict[str, int] = defaultdict(int)  # Round-robin per tier
        logger.info("ProviderManager created (fallback=%s)", fallback_enabled)

    # ── Registration ────────────────────────────────────────────────

    def register(
        self,
        provider: BaseLLMProvider,
        tier: str = "balanced",
        weight: float = 1.0,
        rate_limit: int = 30,
        rate_window: float = 60.0,
    ) -> None:
        """Register a provider."""
        entry = ProviderEntry(
            provider=provider,
            weight=weight,
            tier=tier,
            rate_limiter=RateLimiter(max_requests=rate_limit, window_seconds=rate_window),
        )
        self._providers[provider.name] = entry
        logger.info(
            "Registered provider: %s (tier=%s, weight=%.1f)",
            provider.name,
            tier,
            weight,
        )

    def unregister(self, name: str) -> None:
        """Remove a provider."""
        self._providers.pop(name, None)
        logger.info("Unregistered provider: %s", name)

    def enable(self, name: str) -> None:
        if name in self._providers:
            self._providers[name].enabled = True
            self._providers[name].consecutive_failures = 0

    def disable(self, name: str) -> None:
        if name in self._providers:
            self._providers[name].enabled = False

    # ── Provider Selection ──────────────────────────────────────────

    def _get_candidates(
        self, tier: Optional[str] = None, model: Optional[str] = None
    ) -> list[ProviderEntry]:
        """Get enabled providers ordered by tier preference and weight."""
        tier = tier or "balanced"
        tier_prefs = self._tier_order.get(tier, ["balanced", "fast", "powerful"])

        candidates: list[ProviderEntry] = []
        for preferred_tier in tier_prefs:
            for name, entry in self._providers.items():
                if (
                    entry.enabled
                    and entry.tier == preferred_tier
                    and entry.consecutive_failures < entry.max_failures
                ):
                    candidates.append(entry)

        # Sort by weight descending (higher weight = prefer)
        candidates.sort(key=lambda e: e.weight, reverse=True)
        return candidates

    def _select_provider(
        self, tier: Optional[str] = None
    ) -> Optional[ProviderEntry]:
        """Select a provider using weighted round-robin with rate limiting."""
        candidates = self._get_candidates(tier)
        if not candidates:
            return None

        # Try weighted round-robin
        tier_key = tier or "balanced"
        idx = self._rr_index[tier_key]

        for attempt in range(len(candidates)):
            candidate = candidates[(idx + attempt) % len(candidates)]
            if candidate.rate_limiter.acquire():
                self._rr_index[tier_key] = (idx + attempt + 1) % len(candidates)
                return candidate

        # All rate-limited; try without rate limit check (best effort)
        logger.warning("All providers rate-limited, selecting best available")
        return candidates[0]

    # ── Chat ────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tier: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Send a chat request with automatic fallback.

        Parameters
        ----------
        tier : str | None
            Preferred provider tier ("fast" | "balanced" | "powerful").
        provider : str | None
            Force a specific provider by name.
        """
        # Direct provider selection
        if provider and provider in self._providers:
            entry = self._providers[provider]
            if not entry.enabled:
                raise RuntimeError(f"Provider '{provider}' is disabled")
            try:
                response = await entry.provider.chat(
                    messages, model, temperature, max_tokens, **kwargs
                )
                entry.consecutive_failures = 0
                return response
            except Exception as e:
                entry.consecutive_failures += 1
                logger.error("Provider %s failed: %s", provider, e)
                if not self._fallback_enabled:
                    raise
                # Fall through to fallback

        # Fallback chain
        candidates = self._get_candidates(tier)
        last_error: Optional[Exception] = None

        for entry in candidates:
            try:
                response = await entry.provider.chat(
                    messages, model, temperature, max_tokens, **kwargs
                )
                entry.consecutive_failures = 0
                return response
            except Exception as e:
                entry.consecutive_failures += 1
                last_error = e
                logger.warning(
                    "Provider %s failed (%d/%d): %s",
                    entry.provider.name,
                    entry.consecutive_failures,
                    entry.max_failures,
                    e,
                )
                continue

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        )

    async def stream(
        self,
        messages: list[ChatMessage | dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tier: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamingChunk]:
        """Stream with fallback support."""
        candidates = self._get_candidates(tier)

        for entry in candidates:
            try:
                async for chunk in entry.provider.stream(
                    messages, model, temperature, max_tokens, **kwargs
                ):
                    yield chunk
                entry.consecutive_failures = 0
                return
            except Exception as e:
                entry.consecutive_failures += 1
                logger.warning("Stream failed on %s: %s", entry.provider.name, e)
                continue

        raise RuntimeError("All providers failed for streaming")

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return stats for all registered providers."""
        return {
            name: {
                "tier": entry.tier,
                "enabled": entry.enabled,
                "weight": entry.weight,
                "consecutive_failures": entry.consecutive_failures,
                **entry.provider.stats(),
            }
            for name, entry in self._providers.items()
        }

    def __repr__(self) -> str:
        providers = ", ".join(
            f"{n}({e.tier})" for n, e in self._providers.items()
        )
        return f"ProviderManager([{providers}])"
