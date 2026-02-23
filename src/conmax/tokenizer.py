from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from collections.abc import Sequence

Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}

# Per-message overhead in tokens to approximate role/formatting tokens
_MESSAGE_OVERHEAD = 4


class TokenCountFn(Protocol):
    def __call__(self, text: str) -> int: ...


def approximate_token_count(text: str) -> int:
    """Estimate tokens as len(text) // 4. No dependencies required."""
    return max(len(text) // 4, 1)


def messages_to_text(messages: Sequence[Message]) -> str:
    """Flatten a message list to a single string for token counting."""
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg.get('role', 'unknown')}: {msg.get('content', '')}")
    return "\n\n".join(parts)


@dataclass(frozen=True)
class TokenCounter:
    count_fn: TokenCountFn
    name: str

    def count(self, text: str) -> int:
        return self.count_fn(text)

    def count_messages(self, messages: Sequence[Message]) -> int:
        text = messages_to_text(messages)
        return self.count_fn(text) + len(messages) * _MESSAGE_OVERHEAD


APPROXIMATE_COUNTER = TokenCounter(
    count_fn=approximate_token_count, name="approximate"
)
