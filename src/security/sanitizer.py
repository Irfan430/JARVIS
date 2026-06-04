"""
JARVIS Input Sanitizer
XSS prevention, command injection prevention, and input length limits.
"""

import re
import html
from typing import Optional
from urllib.parse import urlparse

from loguru import logger


class InputSanitizer:
    """
    Input sanitization for the JARVIS bot.
    Prevents XSS, command injection, and enforces length limits.
    """

    # Patterns for dangerous content
    XSS_PATTERNS = [
        re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),  # onclick=, onerror=, etc.
        re.compile(r"<iframe\b[^>]*>", re.IGNORECASE),
        re.compile(r"<object\b[^>]*>", re.IGNORECASE),
        re.compile(r"<embed\b[^>]*>", re.IGNORECASE),
        re.compile(r"<link\b[^>]*>", re.IGNORECASE),
        re.compile(r"expression\s*\(", re.IGNORECASE),
        re.compile(r"url\s*\(", re.IGNORECASE),
        re.compile(r"data:text/html", re.IGNORECASE),
    ]

    INJECTION_PATTERNS = [
        # Shell injection
        re.compile(r"[;&|`$]"),
        re.compile(r"\$\("),
        re.compile(r"\$\{"),
        # Python code injection
        re.compile(r"__\w+__"),
        re.compile(r"import\s+"),
        re.compile(r"exec\s*\("),
        re.compile(r"eval\s*\("),
        re.compile(r"compile\s*\("),
        # SQL injection basics
        re.compile(r"(?:DROP|DELETE|INSERT|UPDATE|ALTER)\s+", re.IGNORECASE),
        re.compile(r"'\s*OR\s+'", re.IGNORECASE),
        re.compile(r"--\s*$", re.MULTILINE),
    ]

    # Characters that are always stripped
    CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

    def __init__(
        self,
        max_length: int = 4096,
        allow_html: bool = False,
        strict_mode: bool = False,
    ):
        """
        Args:
            max_length: Maximum allowed input length.
            allow_html: Whether to allow HTML tags (escaped).
            strict_mode: If True, reject inputs with any suspicious patterns.
        """
        self.max_length = max_length
        self.allow_html = allow_html
        self.strict_mode = strict_mode
        
        logger.info(
            f"InputSanitizer initialized — "
            f"max_length={max_length}, strict={strict_mode}"
        )

    def sanitize(self, text: str, check_injection: bool = True) -> str:
        """
        Main sanitization pipeline.
        
        Args:
            text: Raw input text.
            check_injection: Whether to check for injection patterns.
        
        Returns:
            Sanitized text.
        
        Raises:
            ValueError: In strict mode if dangerous patterns detected.
        """
        if not isinstance(text, str):
            text = str(text)

        # Step 1: Enforce length limit
        text = self._enforce_length(text)

        # Step 2: Remove control characters
        text = self._remove_control_chars(text)

        # Step 3: Check for XSS patterns
        text = self._sanitize_xss(text)

        # Step 4: Check for injection patterns (may raise in strict mode)
        if check_injection:
            text = self._check_injection(text)

        # Step 5: HTML escape if needed
        if not self.allow_html:
            text = html.escape(text)

        return text

    def sanitize_message(self, text: str) -> str:
        """Sanitize a user message — full pipeline."""
        return self.sanitize(text, check_injection=True)

    def sanitize_command_arg(self, text: str) -> str:
        """Sanitize a command argument — stricter pipeline."""
        text = self._enforce_length(text, max_len=self.max_length // 2)
        text = self._remove_control_chars(text)
        text = self._sanitize_xss(text)
        
        # Command args should not contain special shell characters
        # But we allow them in non-strict mode for natural language
        if self.strict_mode:
            text = self._check_injection(text)
        
        text = html.escape(text)
        return text

    def sanitize_url(self, url: str) -> str:
        """Validate and sanitize a URL."""
        url = url.strip()
        
        # Check length
        if len(url) > 2048:
            raise ValueError("URL too long (max 2048 chars)")
        
        # Parse and validate
        try:
            parsed = urlparse(url)
        except Exception:
            raise ValueError("Invalid URL format")
        
        # Only allow safe schemes
        safe_schemes = {"http", "https", "ftp"}
        if parsed.scheme and parsed.scheme.lower() not in safe_schemes:
            raise ValueError(f"URL scheme '{parsed.scheme}' not allowed")
        
        # Check for javascript: in the full URL
        if "javascript:" in url.lower():
            raise ValueError("URL contains javascript: protocol")
        
        return url

    def _enforce_length(self, text: str, max_len: Optional[int] = None) -> str:
        """Truncate text to max length."""
        limit = max_len or self.max_length
        if len(text) > limit:
            truncated = text[:limit]
            logger.debug(f"Input truncated: {len(text)} -> {limit} chars")
            return truncated
        return text

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters."""
        return self.CONTROL_CHARS.sub("", text)

    def _sanitize_xss(self, text: str) -> str:
        """Check for and sanitize XSS patterns."""
        for pattern in self.XSS_PATTERNS:
            if pattern.search(text):
                if self.strict_mode:
                    raise ValueError(f"Input blocked: contains XSS pattern")
                logger.warning(f"XSS pattern detected in input: {pattern.pattern}")
                text = pattern.sub("", text)
        return text

    def _check_injection(self, text: str) -> str:
        """Check for command injection patterns."""
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                if self.strict_mode:
                    raise ValueError(
                        f"Input blocked: contains injection pattern "
                        f"'{match.group()}'"
                    )
                logger.warning(
                    f"Injection pattern detected in input: {match.group()}"
                )
                # In non-strict mode, we log but don't modify
                # (natural language may contain these characters)
        return text

    def is_safe(self, text: str) -> bool:
        """Quick check if text is safe without modification."""
        try:
            self.sanitize(text, check_injection=True)
            return True
        except ValueError:
            return False

    def get_stats(self) -> dict:
        """Get sanitizer configuration stats."""
        return {
            "max_length": self.max_length,
            "allow_html": self.allow_html,
            "strict_mode": self.strict_mode,
            "xss_patterns_count": len(self.XSS_PATTERNS),
            "injection_patterns_count": len(self.INJECTION_PATTERNS),
        }
