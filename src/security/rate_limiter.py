"""
JARVIS Rate Limiter
Token bucket algorithm with per-user limits and configurable cooldowns.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter implementation.
    
    Tokens are refilled at a steady rate up to the max capacity.
    Each request consumes one token.
    """
    capacity: int           # Max tokens in the bucket
    refill_rate: float      # Tokens per second
    tokens: float = field(init=False)  # Current token count
    last_refill: float = field(init=False)  # Last refill timestamp

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        Returns True if successful, False if not enough tokens.
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self) -> float:
        """Get the time in seconds until the next token is available."""
        self._refill()
        if self.tokens >= 1:
            return 0.0
        return (1 - self.tokens) / self.refill_rate

    @property
    def available(self) -> float:
        """Get current available tokens."""
        self._refill()
        return self.tokens


class RateLimiter:
    """
    Per-user rate limiter using token buckets.
    Supports configurable limits, windows, and cooldowns.
    """

    def __init__(
        self,
        max_requests: int = 30,
        window_seconds: int = 60,
        cooldown_seconds: int = 30,
        enabled: bool = True,
    ):
        """
        Args:
            max_requests: Maximum requests allowed per window.
            window_seconds: Time window in seconds.
            cooldown_seconds: Cooldown period after hitting the limit.
            enabled: Whether rate limiting is active.
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        
        # Per-user buckets and cooldown tracking
        self._buckets: Dict[int, TokenBucket] = {}
        self._cooldowns: Dict[int, float] = {}
        self._blocked_users: Dict[int, float] = {}
        
        logger.info(
            f"RateLimiter initialized — "
            f"max={max_requests}/{window_seconds}s, "
            f"cooldown={cooldown_seconds}s, "
            f"enabled={enabled}"
        )

    def _get_bucket(self, user_id: int) -> TokenBucket:
        """Get or create a token bucket for a user."""
        if user_id not in self._buckets:
            refill_rate = self.max_requests / self.window_seconds
            self._buckets[user_id] = TokenBucket(
                capacity=self.max_requests,
                refill_rate=refill_rate,
            )
        return self._buckets[user_id]

    def is_rate_limited(self, user_id: int) -> bool:
        """
        Check if a user is currently rate limited.
        Returns True if they should be blocked.
        """
        if not self.enabled:
            return False

        # Check cooldown period
        if user_id in self._cooldowns:
            cooldown_end = self._cooldowns[user_id]
            if time.time() < cooldown_end:
                return True
            else:
                # Cooldown expired, allow access again
                del self._cooldowns[user_id]
                if user_id in self._blocked_users:
                    del self._blocked_users[user_id]

        # Check token bucket
        bucket = self._get_bucket(user_id)
        if not bucket.consume(1):
            # No tokens left — start cooldown
            wait_time = bucket.get_wait_time()
            self._cooldowns[user_id] = time.time() + self.cooldown_seconds
            self._blocked_users[user_id] = time.time()
            
            logger.warning(
                f"User {user_id} rate limited — "
                f"waiting {wait_time:.1f}s, cooldown {self.cooldown_seconds}s"
            )
            return True

        return False

    def get_remaining(self, user_id: int) -> float:
        """Get remaining requests for a user."""
        if not self.enabled:
            return float('inf')
        
        bucket = self._get_bucket(user_id)
        return bucket.available

    def get_reset_time(self, user_id: int) -> float:
        """Get time in seconds until rate limit resets."""
        if not self.enabled:
            return 0.0
        
        if user_id in self._cooldowns:
            remaining = self._cooldowns[user_id] - time.time()
            return max(0.0, remaining)
        
        bucket = self._get_bucket(user_id)
        return bucket.get_wait_time()

    def get_user_status(self, user_id: int) -> Dict:
        """Get comprehensive rate limit status for a user."""
        bucket = self._get_bucket(user_id)
        in_cooldown = user_id in self._cooldowns
        cooldown_remaining = 0.0
        if in_cooldown:
            cooldown_remaining = max(0.0, self._cooldowns[user_id] - time.time())
        
        return {
            "user_id": user_id,
            "is_limited": self.is_rate_limited(user_id) if self.enabled else False,
            "remaining": bucket.available if self.enabled else -1,
            "capacity": self.max_requests,
            "in_cooldown": in_cooldown,
            "cooldown_remaining": cooldown_remaining,
            "reset_in": self.get_reset_time(user_id),
        }

    def reset_user(self, user_id: int):
        """Reset a user's rate limit state."""
        if user_id in self._buckets:
            del self._buckets[user_id]
        if user_id in self._cooldowns:
            del self._cooldowns[user_id]
        if user_id in self._blocked_users:
            del self._blocked_users[user_id]
        logger.info(f"Reset rate limit state for user {user_id}")

    def cleanup(self) -> int:
        """Clean up expired cooldowns and old buckets. Returns count removed."""
        now = time.time()
        cleaned = 0
        
        # Clean expired cooldowns
        expired = [uid for uid, end in self._cooldowns.items() if now >= end]
        for uid in expired:
            del self._cooldowns[uid]
            cleaned += 1
        
        # Clean blocked users that are no longer in cooldown
        stale = [uid for uid in self._blocked_users if uid not in self._cooldowns]
        for uid in stale:
            del self._blocked_users[uid]
            cleaned += 1
        
        return cleaned

    def set_enabled(self, enabled: bool):
        """Enable or disable rate limiting."""
        self.enabled = enabled
        logger.info(f"Rate limiting {'enabled' if enabled else 'disabled'}")

    def update_config(
        self,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
        cooldown_seconds: Optional[int] = None,
    ):
        """Update rate limiting configuration at runtime."""
        if max_requests is not None:
            self.max_requests = max_requests
        if window_seconds is not None:
            self.window_seconds = window_seconds
        if cooldown_seconds is not None:
            self.cooldown_seconds = cooldown_seconds
        
        # Reset all buckets with new config
        self._buckets.clear()
        logger.info(
            f"RateLimiter config updated — "
            f"max={self.max_requests}/{self.window_seconds}s, "
            f"cooldown={self.cooldown_seconds}s"
        )
