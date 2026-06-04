"""
JARVIS Configuration Management
Loads settings from .env and config.yaml with Pydantic validation.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, model_validator

# Load .env file from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


class TelegramConfig(BaseSettings):
    """Telegram bot configuration."""
    bot_token: str = Field(default="", description="Telegram Bot API token")
    owner_id: int = Field(default=0, description="Telegram user ID of the owner")
    allowed_users: List[int] = Field(default_factory=list, description="List of allowed Telegram user IDs")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL (optional, for production)")
    parse_mode: str = Field(default="HTML", description="Message parse mode: HTML or Markdown")

    @field_validator("allowed_users", mode="before")
    @classmethod
    def parse_allowed_users(cls, v):
        """Handle comma-separated, JSON list, or single int formats."""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json
                return json.loads(v)
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        if isinstance(v, int):
            return [v]
        return v

    model_config = {"env_prefix": "TELEGRAM_"}


class AIConfig(BaseSettings):
    """AI/LLM provider configuration."""
    provider: str = Field(default="mimo", description="AI provider: openai, anthropic, mimo")
    api_key: str = Field(default="", description="AI provider API key")
    model: str = Field(default="mimo-v2.5-pro", description="Model name to use")
    base_url: str = Field(default="https://api.xiaomimimo.com/v1", description="API base URL")
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: str = Field(
        default="You are JARVIS, a highly capable AI assistant. "
                "You are helpful, witty, and slightly formal in tone. "
                "Always strive to be accurate and useful. "
                "Respond in the same language the user writes in.",
        description="System prompt for the AI"
    )

    model_config = {"env_prefix": "AI_"}


class VoiceConfig(BaseSettings):
    """Voice/STT/TTS configuration."""
    enabled: bool = Field(default=True, description="Enable voice features")
    stt_provider: str = Field(default="whisper", description="STT provider: whisper, google, vosk")
    stt_model: str = Field(default="base", description="Whisper model: tiny, base, small, medium, large")
    tts_provider: str = Field(default="edge-tts", description="TTS provider: edge-tts, pyttsx3")
    tts_voice: str = Field(default="en-US-GuyNeural", description="TTS voice name")
    max_audio_duration: int = Field(default=120, description="Max audio duration in seconds")

    model_config = {"env_prefix": "VOICE_"}


class SecurityConfig(BaseSettings):
    """Security and rate limiting configuration."""
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=30, ge=1, description="Max requests per window")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    rate_limit_cooldown: int = Field(default=30, ge=1, description="Cooldown period in seconds")
    max_input_length: int = Field(default=4096, ge=1, le=100000, description="Max input text length")
    api_key_header: str = Field(default="X-API-Key", description="Header for API key auth")
    session_timeout: int = Field(default=3600, ge=60, description="Session timeout in seconds")

    model_config = {"env_prefix": "SECURITY_"}


class SearchConfig(BaseSettings):
    """Web search configuration."""
    enabled: bool = Field(default=True)
    provider: str = Field(default="duckduckgo", description="Search provider: duckduckgo, brave, serpapi")
    api_key: str = Field(default="", description="Search API key (if needed)")
    max_results: int = Field(default=5, ge=1, le=20)

    model_config = {"env_prefix": "SEARCH_"}


class AppConfig(BaseSettings):
    """Root application configuration."""
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    data_dir: Path = Field(default=Path("./data"), description="Data storage directory")
    cache_ttl: int = Field(default=300, ge=0, description="Cache TTL in seconds")

    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"Invalid log level: {v}. Must be one of: {valid}")
        return upper

    @model_validator(mode="after")
    def ensure_data_dir(self) -> "AppConfig":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if config_path is None:
        # Look for config.yaml in the project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
    
    path = Path(config_path)
    if not path.exists():
        return {}
    
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from environment variables and YAML file.
    Environment variables take precedence over YAML.
    """
    yaml_data = load_yaml_config(config_path)
    
    # Flatten nested YAML into env-compatible format
    flat_data = {}
    prefix_map = {
        "telegram": "TELEGRAM_",
        "ai": "AI_",
        "voice": "VOICE_",
        "security": "SECURITY_",
        "search": "SEARCH_",
    }
    
    for section, prefix in prefix_map.items():
        if section in yaml_data and isinstance(yaml_data[section], dict):
            for key, value in yaml_data[section].items():
                env_key = f"{prefix}{key.upper()}"
                # Only set if not already in environment
                if env_key not in os.environ and key.upper() not in os.environ:
                    os.environ[env_key] = str(value)
    
    # Top-level config keys
    for key, value in yaml_data.items():
        if key not in prefix_map and isinstance(value, (str, int, float, bool)):
            env_key = key.upper()
            if env_key not in os.environ:
                os.environ[env_key] = str(value)

    return AppConfig()
