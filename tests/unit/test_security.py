"""
Tests for the Security module.
"""

import pytest
import time
from src.security.guard import SecurityGuard, RateLimiter, InputValidator


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_creation(self):
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60

    def test_allows_within_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("user1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False

    def test_separate_users(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False
        # Different user should still be allowed
        assert limiter.is_allowed("user2") is True

    def test_get_remaining(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.get_remaining("user1") == 5
        limiter.is_allowed("user1")
        assert limiter.get_remaining("user1") == 4

    def test_reset_single_user(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")
        assert limiter.is_allowed("user1") is False
        limiter.reset("user1")
        assert limiter.is_allowed("user1") is True

    def test_reset_all(self):
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.is_allowed("user1")
        limiter.is_allowed("user2")
        limiter.reset()
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user2") is True


class TestInputValidator:
    """Tests for InputValidator."""

    def test_creation(self):
        validator = InputValidator(max_length=1000)
        assert validator.max_length == 1000

    def test_valid_input(self):
        validator = InputValidator()
        is_valid, error = validator.validate("Hello, JARVIS!")
        assert is_valid is True
        assert error == ""

    def test_empty_input(self):
        validator = InputValidator()
        is_valid, error = validator.validate("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_whitespace_only(self):
        validator = InputValidator()
        is_valid, error = validator.validate("   ")
        assert is_valid is False

    def test_too_long(self):
        validator = InputValidator(max_length=10)
        is_valid, error = validator.validate("a" * 11)
        assert is_valid is False
        assert "length" in error.lower()

    def test_script_injection(self):
        validator = InputValidator()
        is_valid, error = validator.validate("<script>alert('xss')</script>")
        assert is_valid is False
        assert "blocked" in error.lower()

    def test_javascript_url(self):
        validator = InputValidator()
        is_valid, error = validator.validate("javascript:alert(1)")
        assert is_valid is False

    def test_sanitize_html(self):
        validator = InputValidator()
        sanitized = validator.sanitize("<b>Hello</b> <script>alert('x')</script> World")
        assert "<b>" not in sanitized
        assert "<script>" not in sanitized
        assert "Hello" in sanitized
        assert "World" in sanitized

    def test_sanitize_whitespace(self):
        validator = InputValidator()
        sanitized = validator.sanitize("Hello    world   test")
        assert sanitized == "Hello world test"

    def test_hash_input(self):
        validator = InputValidator()
        h1 = validator.hash_input("test input")
        h2 = validator.hash_input("test input")
        h3 = validator.hash_input("different input")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16


class TestSecurityGuard:
    """Tests for SecurityGuard."""

    def test_creation(self):
        guard = SecurityGuard()
        assert guard.rate_limiter is not None
        assert guard.validator is not None

    def test_rate_limit_check(self):
        guard = SecurityGuard(rate_limiter=RateLimiter(max_requests=2, window_seconds=60))
        assert guard.check_rate_limit("user1") is True
        assert guard.check_rate_limit("user1") is True
        assert guard.check_rate_limit("user1") is False

    def test_validate_input(self):
        guard = SecurityGuard()
        is_valid, error = guard.validate_input("Hello")
        assert is_valid is True

    def test_user_allowed_no_allowlist(self):
        guard = SecurityGuard()
        assert guard.is_user_allowed(12345) is True

    def test_user_allowed_with_allowlist(self):
        guard = SecurityGuard(allowed_users={100, 200})
        assert guard.is_user_allowed(100) is True
        assert guard.is_user_allowed(300) is False

    def test_block_user(self):
        guard = SecurityGuard()
        guard.block_user(12345)
        assert guard.is_user_blocked(12345) is True
        assert guard.is_user_allowed(12345) is False

    def test_unblock_user(self):
        guard = SecurityGuard()
        guard.block_user(12345)
        guard.unblock_user(12345)
        assert guard.is_user_blocked(12345) is False
        assert guard.is_user_allowed(12345) is True

    def test_security_report(self):
        guard = SecurityGuard(allowed_users={1, 2, 3})
        guard.block_user(4)
        report = guard.get_security_report()
        assert report["allowed_users"] == 3
        assert report["blocked_users"] == 1
        assert "rate_limit" in report
        assert "input_max_length" in report
