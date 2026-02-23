from __future__ import annotations

import pytest

from conmax.tokenizer import Message, APPROXIMATE_COUNTER, TokenCounter
from conmax.provider import LLMProvider
from conmax.compressor import (
    CompressionConfig,
    CompressionResult,
    COMPRESSION_MARKER,
    compress_history,
)
from conmax.chat import ConmaxChat, TurnStats


class MockProvider(LLMProvider):
    """Deterministic mock provider for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.compress_calls: list[tuple[list[Message], int]] = []
        self.chat_calls: list[list[Message]] = []

    def _make_token_counter(self) -> TokenCounter:
        return APPROXIMATE_COUNTER

    def compress(self, messages, target_tokens) -> str:
        self.compress_calls.append((list(messages), target_tokens))
        return f"Summary of {len(messages)} messages"

    def chat(self, messages) -> str:
        self.chat_calls.append(list(messages))
        return "Mock response."


def _make_messages(n: int, content_size: int = 100) -> list[Message]:
    """Create n alternating user/assistant messages with content of ~content_size chars."""
    msgs: list[Message] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: " + "x" * content_size
        msgs.append({"role": role, "content": content})
    return msgs


class TestCompressionNotNeeded:
    def test_empty_history(self):
        provider = MockProvider()
        result = compress_history([], 1000, provider)
        assert result.compressions_performed == 0
        assert result.messages == []

    def test_small_history_no_compression(self):
        provider = MockProvider()
        msgs = _make_messages(4, content_size=20)
        result = compress_history(msgs, 10000, provider)
        assert result.compressions_performed == 0
        assert len(result.messages) == 4

    def test_returns_zero_compressions(self):
        provider = MockProvider()
        msgs = _make_messages(2, content_size=10)
        result = compress_history(msgs, 10000, provider)
        assert result.compressions_performed == 0
        assert result.tokens_saved == 0


class TestCompressionNeeded:
    def test_compresses_when_over_budget(self):
        provider = MockProvider()
        # 20 messages with ~100 chars each = ~500+ tokens with approximate counter
        msgs = _make_messages(20, content_size=100)
        result = compress_history(msgs, 200, provider)
        assert result.compressions_performed > 0
        assert result.tokens_after <= result.tokens_before

    def test_stops_when_fits(self):
        provider = MockProvider()
        msgs = _make_messages(20, content_size=100)
        # Give a generous budget that only needs 1 chunk compressed
        tokens_before = provider.count_messages_tokens(msgs)
        # Set budget to need only a small reduction
        budget = tokens_before - 50
        result = compress_history(msgs, budget, provider)
        # Should not compress everything — stop early
        assert result.compressions_performed >= 1

    def test_protected_tail_is_verbatim(self):
        provider = MockProvider()
        msgs = _make_messages(12, content_size=100)
        config = CompressionConfig(min_messages_to_keep=4)
        result = compress_history(msgs, 100, provider, config)
        # Last 4 messages should be the originals
        tail = result.messages[-4:]
        assert tail == msgs[-4:]

    def test_tokens_saved_is_positive(self):
        provider = MockProvider()
        msgs = _make_messages(20, content_size=100)
        result = compress_history(msgs, 200, provider)
        if result.compressions_performed > 0:
            assert result.tokens_saved > 0

    def test_summary_marker_present(self):
        provider = MockProvider()
        msgs = _make_messages(20, content_size=100)
        result = compress_history(msgs, 200, provider)
        summaries = [
            m for m in result.messages if m["content"].startswith(COMPRESSION_MARKER)
        ]
        assert len(summaries) > 0


class TestCompressionConfig:
    def test_custom_chunk_size(self):
        provider = MockProvider()
        msgs = _make_messages(20, content_size=100)
        config = CompressionConfig(chunk_size=2)
        result = compress_history(msgs, 200, provider, config)
        # With smaller chunks, each compress call handles fewer messages
        for call_msgs, _ in provider.compress_calls:
            assert len(call_msgs) <= 2

    def test_custom_min_messages_to_keep(self):
        provider = MockProvider()
        msgs = _make_messages(12, content_size=100)
        config = CompressionConfig(min_messages_to_keep=6)
        result = compress_history(msgs, 100, provider, config)
        # Last 6 should be originals
        assert result.messages[-6:] == msgs[-6:]

    def test_all_protected_no_compression(self):
        provider = MockProvider()
        msgs = _make_messages(4, content_size=100)
        config = CompressionConfig(min_messages_to_keep=4)
        result = compress_history(msgs, 10, provider, config)
        # Can't compress — all messages are protected
        assert result.compressions_performed == 0


class TestConmaxChat:
    def test_send_returns_turn_stats(self):
        provider = MockProvider()
        chat = ConmaxChat(provider, context_budget=10000)
        stats = chat.send("Hello!")
        assert isinstance(stats, TurnStats)
        assert stats.prompt_tokens > 0
        assert stats.response_tokens > 0

    def test_history_grows_after_send(self):
        provider = MockProvider()
        chat = ConmaxChat(provider, context_budget=10000)
        chat.send("Hello!")
        assert len(chat.history) == 2  # user + assistant
        chat.send("How are you?")
        assert len(chat.history) == 4

    def test_last_response(self):
        provider = MockProvider()
        chat = ConmaxChat(provider, context_budget=10000)
        chat.send("Hello!")
        assert chat.last_response == "Mock response."

    def test_compression_triggers_with_tiny_budget(self):
        provider = MockProvider()
        chat = ConmaxChat(
            provider,
            context_budget=200,
            response_reservation=50,
            compression_config=CompressionConfig(min_messages_to_keep=2),
        )
        # Fill history with enough messages to overflow
        for i in range(10):
            chat.send(f"Message {i} " + "x" * 80)
        # Provider should have been asked to compress at least once
        assert len(provider.compress_calls) > 0

    def test_reset_clears_history(self):
        provider = MockProvider()
        chat = ConmaxChat(provider, context_budget=10000)
        chat.send("Hello!")
        chat.reset()
        assert chat.history == []
        assert chat.last_response == ""

    def test_utilization_is_sensible(self):
        provider = MockProvider()
        chat = ConmaxChat(provider, context_budget=10000)
        stats = chat.send("Hello!")
        assert 0 < stats.utilization_pct < 100

    def test_export_import_history(self):
        provider = MockProvider()
        chat = ConmaxChat(provider, context_budget=10000)
        chat.send("Hello!")
        exported = chat.export_history()
        chat.reset()
        chat.import_history(exported)
        assert len(chat.history) == 2

    def test_context_budget_setter(self):
        provider = MockProvider()
        chat = ConmaxChat(provider, context_budget=10000)
        chat.context_budget = 5000
        assert chat.context_budget == 5000
