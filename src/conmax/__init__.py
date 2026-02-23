"""conmax — context maximizer: dynamic just-in-time chat history compression for LLMs."""

from .chat import ConmaxChat, TurnStats
from .compressor import CompressionConfig, CompressionResult, COMPRESSION_MARKER
from .provider import LLMProvider, AnthropicProvider, OpenAIProvider, create_provider
from .tokenizer import (
    Message,
    TokenCounter,
    APPROXIMATE_COUNTER,
    approximate_token_count,
)

__all__ = [
    "ConmaxChat",
    "TurnStats",
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "create_provider",
    "CompressionConfig",
    "CompressionResult",
    "COMPRESSION_MARKER",
    "Message",
    "TokenCounter",
    "APPROXIMATE_COUNTER",
    "approximate_token_count",
]

__version__ = "0.1.0"
