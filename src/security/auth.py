"""
JARVIS Authentication & Session Management
Owner-only command protection, API key validation, session tracking.
"""

import secrets
import time
from typing import Optional, Set, Dict
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class UserSession:
    """Represents an authenticated user session."""
    user_id: int
    username: Optional[str]
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    request_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired based on last activity."""
        # Default 1-hour session timeout; configurable per-instance
        return False  # Handled by SessionManager.check_timeout()

    def touch(self):
        """Update last activity timestamp."""
        self.last_active = time.time()
        self.request_count += 1


class Authenticator:
    """
    Handles user authentication and authorization.
    - Owner-only command protection
    - API key validation
    - Allowlist management
    """

    def __init__(
        self,
        owner_id: int,
        allowed_users: Optional[Set[int]] = None,
        api_key: Optional[str] = None,
    ):
        self.owner_id = owner_id
        self._allowed_users: Set[int] = set(allowed_users or set())
        self._api_key = api_key
        self._api_keys: Set[str] = set()
        
        # Always include the owner
        if owner_id:
            self._allowed_users.add(owner_id)
        
        logger.info(
            f"Authenticator initialized — owner_id={owner_id}, "
            f"allowed_users={len(self._allowed_users)}"
        )

    def is_owner(self, user_id: int) -> bool:
        """Check if user is the bot owner."""
        return user_id == self.owner_id

    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized (owner or in allowlist)."""
        return user_id in self._allowed_users

    def is_allowed_user(self, user_id: int) -> bool:
        """Alias for is_authorized for clarity."""
        return self.is_authorized(user_id)

    def add_allowed_user(self, user_id: int) -> bool:
        """Add a user to the allowlist. Returns True if added."""
        if user_id not in self._allowed_users:
            self._allowed_users.add(user_id)
            logger.info(f"Added user {user_id} to allowlist")
            return True
        return False

    def remove_allowed_user(self, user_id: int) -> bool:
        """Remove a user from the allowlist. Cannot remove owner."""
        if user_id == self.owner_id:
            logger.warning(f"Cannot remove owner {user_id} from allowlist")
            return False
        removed = self._allowed_users.discard(user_id)
        if removed is not None:
            logger.info(f"Removed user {user_id} from allowlist")
            return True
        return False

    def register_api_key(self, key: Optional[str] = None) -> str:
        """Register a new API key. Returns the key."""
        if key is None:
            key = secrets.token_urlsafe(32)
        self._api_keys.add(key)
        logger.info(f"Registered new API key: {key[:8]}...")
        return key

    def validate_api_key(self, key: str) -> bool:
        """Validate an API key."""
        if not self._api_key and not self._api_keys:
            # No API keys configured — validation is disabled
            return True
        return key in self._api_keys or key == self._api_key

    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self._api_keys:
            self._api_keys.discard(key)
            logger.info(f"Revoked API key: {key[:8]}...")
            return True
        return False

    def check_user_access(self, user_id: int, owner_only: bool = False) -> bool:
        """
        Comprehensive access check.
        Returns True if access is granted.
        """
        if owner_only:
            return self.is_owner(user_id)
        return self.is_authorized(user_id)

    def get_allowed_users(self) -> Set[int]:
        """Get the current set of allowed user IDs."""
        return self._allowed_users.copy()


class SessionManager:
    """
    Manages user sessions with timeout support.
    """

    def __init__(self, timeout: int = 3600):
        """
        Args:
            timeout: Session timeout in seconds (default 1 hour).
        """
        self.timeout = timeout
        self._sessions: Dict[int, UserSession] = {}
        logger.info(f"SessionManager initialized — timeout={timeout}s")

    def get_or_create_session(self, user_id: int, username: Optional[str] = None) -> UserSession:
        """Get existing session or create a new one."""
        now = time.time()
        
        if user_id in self._sessions:
            session = self._sessions[user_id]
            # Check timeout
            if now - session.last_active > self.timeout:
                logger.info(f"Session expired for user {user_id}")
                session = UserSession(user_id=user_id, username=username)
                self._sessions[user_id] = session
            else:
                session.touch()
        else:
            session = UserSession(user_id=user_id, username=username)
            self._sessions[user_id] = session

        return session

    def check_timeout(self, user_id: int) -> bool:
        """Check if a session has timed out. Returns True if timed out."""
        if user_id not in self._sessions:
            return True  # No session = timed out
        session = self._sessions[user_id]
        return (time.time() - session.last_active) > self.timeout

    def remove_session(self, user_id: int) -> bool:
        """Remove a user session."""
        if user_id in self._sessions:
            del self._sessions[user_id]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        now = time.time()
        expired = [
            uid for uid, session in self._sessions.items()
            if (now - session.last_active) > self.timeout
        ]
        for uid in expired:
            del self._sessions[uid]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)

    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)

    def get_session(self, user_id: int) -> Optional[UserSession]:
        """Get a user session without creating one."""
        return self._sessions.get(user_id)
