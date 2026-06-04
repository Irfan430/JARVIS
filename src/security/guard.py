"""
Security guard for rate limiting, input validation, and access control.
"""

import time
import re
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        """Check if a request is allowed for the given user."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self._requests[user_id] = [
            t for t in self._requests[user_id] if t > window_start
        ]

        if len(self._requests[user_id]) >= self.max_requests:
            return False

        self._requests[user_id].append(now)
        return True

    def get_remaining(self, user_id: str) -> int:
        """Get remaining requests for a user."""
        now = time.time()
        window_start = now - self.window_seconds
        recent = [t for t in self._requests[user_id] if t > window_start]
        return max(0, self.max_requests - len(recent))

    def reset(self, user_id: Optional[str] = None) -> None:
        """Reset rate limits."""
        if user_id:
            self._requests.pop(user_id, None)
        else:
            self._requests.clear()


class InputValidator:
    """Validate and sanitize user input."""

    MAX_LENGTH = 10000
    BLOCKED_PATTERNS = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
    ]

    def __init__(self, max_length: int = 4096):
        self.max_length = max_length

    def validate(self, text: str) -> tuple[bool, str]:
        """Validate input text. Returns (is_valid, error_message)."""
        if not text or not text.strip():
            return False, "Input cannot be empty"

        if len(text) > self.max_length:
            return False, f"Input exceeds maximum length of {self.max_length}"

        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, "Input contains blocked content"

        return True, ""

    def sanitize(self, text: str) -> str:
        """Sanitize input text."""
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove null bytes
        text = text.replace("\x00", "")
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def hash_input(self, text: str) -> str:
        """Create a SHA256 hash of input for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class SecurityGuard:
    """Unified security interface."""

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        validator: Optional[InputValidator] = None,
        allowed_users: Optional[Set[int]] = None,
    ):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.validator = validator or InputValidator()
        self.allowed_users: Set[int] = allowed_users or set()
        self._blocked_users: Set[int] = set()

    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        return self.rate_limiter.is_allowed(user_id)

    def validate_input(self, text: str) -> tuple[bool, str]:
        """Validate user input."""
        return self.validator.validate(text)

    def is_user_allowed(self, user_id: int) -> bool:
        """Check if a user is allowed to use the bot."""
        if user_id in self._blocked_users:
            return False
        if not self.allowed_users:
            return True  # No allowlist = everyone allowed
        return user_id in self.allowed_users

    def block_user(self, user_id: int) -> None:
        """Block a user."""
        self._blocked_users.add(user_id)

    def unblock_user(self, user_id: int) -> None:
        """Unblock a user."""
        self._blocked_users.discard(user_id)

    def is_user_blocked(self, user_id: int) -> bool:
        """Check if a user is blocked."""
        return user_id in self._blocked_users

    def get_security_report(self) -> Dict[str, Any]:
        """Get security status report."""
        return {
            "allowed_users": len(self.allowed_users),
            "blocked_users": len(self._blocked_users),
            "rate_limit": {
                "max_requests": self.rate_limiter.max_requests,
                "window_seconds": self.rate_limiter.window_seconds,
            },
            "input_max_length": self.validator.max_length,
        }
