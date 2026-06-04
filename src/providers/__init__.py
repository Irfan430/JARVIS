"""
LLM Providers — Abstract base and registry.

All provider implementations must inherit from `BaseLLMProvider`.
"""

from .base import BaseLLMProvider, ChatMessage, ChatResponse, StreamingChunk
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .mimo_provider import MiMoProvider
from .manager import ProviderManager

__all__ = [
    "BaseLLMProvider",
    "ChatMessage",
    "ChatResponse",
    "StreamingChunk",
    "OpenAIProvider",
    "AnthropicProvider",
    "MiMoProvider",
    "ProviderManager",
    "create_provider",
]


def create_provider(
    provider: str,
    api_key: str,
    model: str = "",
    system_prompt: str = "",
    **kwargs,
) -> BaseLLMProvider:
    """Factory function to create an LLM provider by name."""
    providers_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "mimo": MiMoProvider,
    }
    cls = providers_map.get(provider.lower())
    if cls is None:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers_map.keys())}")
    return cls(api_key=api_key, model=model, system_prompt=system_prompt, **kwargs)
