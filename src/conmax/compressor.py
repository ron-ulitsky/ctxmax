from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .tokenizer import Message
from .provider import LLMProvider

COMPRESSION_MARKER = "[SUMMARY]"


@dataclass
class CompressionResult:
    """Result of a compress_history() call."""

    messages: list[Message]
    tokens_before: int
    tokens_after: int
    compressions_performed: int
    segments_compressed: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass
class CompressionConfig:
    """Controls compression chunking and behavior.

    chunk_size: messages per compression chunk (default 6 = ~3 turns)
    min_messages_to_keep: recent messages always kept verbatim (default 4 = ~2 turns)
    summary_ratio: target output = input_tokens * ratio (default 0.3)
    compression_marker: prefix for summary messages
    """

    chunk_size: int = 6
    min_messages_to_keep: int = 4
    summary_ratio: float = 0.3
    compression_marker: str = COMPRESSION_MARKER


def _chunk_messages(
    messages: list[Message], chunk_size: int
) -> list[list[Message]]:
    """Split messages into chunks of chunk_size. Last chunk may be smaller."""
    chunks: list[list[Message]] = []
    for i in range(0, len(messages), chunk_size):
        chunks.append(messages[i : i + chunk_size])
    return chunks


def compress_history(
    history: list[Message],
    available_tokens: int,
    provider: LLMProvider,
    config: CompressionConfig | None = None,
) -> CompressionResult:
    """Compress history to fit within available_tokens.

    Uses a greedy oldest-first strategy: compress one chunk at a time,
    stopping as soon as the history fits. This ensures we compress only
    as much as needed — the core "just-in-time" property.
    """
    if config is None:
        config = CompressionConfig()

    tokens_before = provider.count_messages_tokens(history)

    # Already fits — no compression needed
    if tokens_before <= available_tokens:
        return CompressionResult(
            messages=list(history),
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            compressions_performed=0,
            segments_compressed=0,
        )

    # Split into compressible head and protected tail
    keep = config.min_messages_to_keep
    if keep >= len(history):
        # Nothing to compress — all messages are protected
        return CompressionResult(
            messages=list(history),
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            compressions_performed=0,
            segments_compressed=0,
        )

    protected_tail = history[-keep:]
    compressible = history[:-keep]

    # Chunk the compressible portion, oldest first
    chunks = _chunk_messages(compressible, config.chunk_size)

    compressions_performed = 0
    segments_compressed = 0

    for i, chunk in enumerate(chunks):
        # Recalculate current total
        current_messages = []
        for c in chunks:
            current_messages.extend(c)
        current_messages.extend(protected_tail)
        current_tokens = provider.count_messages_tokens(current_messages)

        if current_tokens <= available_tokens:
            break

        tokens_to_free = current_tokens - available_tokens
        chunk_tokens = provider.count_messages_tokens(chunk)

        # Calculate target: compress enough to fit, but not more than ratio
        target_by_ratio = int(chunk_tokens * config.summary_ratio)
        target_by_need = chunk_tokens - tokens_to_free
        target_tokens = max(min(target_by_ratio, target_by_need), 20)

        summary_text = provider.compress(chunk, target_tokens)

        # Replace chunk with summary message
        chunks[i] = [
            {
                "role": "assistant",
                "content": f"{config.compression_marker} {summary_text}",
            }
        ]

        compressions_performed += 1
        segments_compressed += len(chunk)

    # Reconstruct final message list
    final_messages: list[Message] = []
    for c in chunks:
        final_messages.extend(c)
    final_messages.extend(protected_tail)

    tokens_after = provider.count_messages_tokens(final_messages)

    return CompressionResult(
        messages=final_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        compressions_performed=compressions_performed,
        segments_compressed=segments_compressed,
    )
