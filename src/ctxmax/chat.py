from __future__ import annotations

from dataclasses import dataclass

from .tokenizer import Message
from .provider import LLMProvider
from .compressor import CompressionConfig, CompressionResult, compress_history


@dataclass
class TurnStats:
    """Token accounting for a single chat turn."""

    prompt_tokens: int
    history_tokens_before: int
    history_tokens_after: int
    response_tokens: int
    compression_result: CompressionResult | None
    total_context_tokens: int
    context_budget: int

    @property
    def tokens_saved(self) -> int:
        if self.compression_result:
            return self.compression_result.tokens_saved
        return 0

    @property
    def utilization_pct(self) -> float:
        if self.context_budget == 0:
            return 0.0
        return (self.total_context_tokens / self.context_budget) * 100


class CtxmaxChat:
    """Stateful conversation with automatic just-in-time compression.

    Usage:
        provider = AnthropicProvider()
        chat = CtxmaxChat(provider, context_budget=100_000)
        stats = chat.send("Hello!")
        print(chat.last_response)
    """

    def __init__(
        self,
        provider: LLMProvider,
        context_budget: int,
        response_reservation: int = 2048,
        system_prompt: str | None = None,
        compression_config: CompressionConfig | None = None,
    ) -> None:
        self._provider = provider
        self._context_budget = context_budget
        self._response_reservation = response_reservation
        self._system_prompt = system_prompt
        self._compression_config = compression_config or CompressionConfig()
        self._history: list[Message] = []
        self._last_response: str = ""
        self._system_tokens: int = (
            provider.count_tokens(system_prompt) if system_prompt else 0
        )

    def send(self, user_message: str) -> TurnStats:
        """Send a message, compress if needed, return stats."""
        prompt_tokens = self._provider.count_tokens(user_message)
        history_tokens_before = self._provider.count_messages_tokens(self._history)

        # available = C - system - prompt - response_reservation
        available = (
            self._context_budget
            - self._system_tokens
            - prompt_tokens
            - self._response_reservation
        )

        # Compress history if it exceeds available space
        compression_result: CompressionResult | None = None
        if history_tokens_before > available:
            compression_result = compress_history(
                self._history, available, self._provider, self._compression_config
            )
            self._history = compression_result.messages

        history_tokens_after = self._provider.count_messages_tokens(self._history)

        # Build messages for the API call
        messages: list[Message] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_message})

        # Get response
        response_text = self._provider.chat(messages)
        self._last_response = response_text

        # Append this turn to history
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": response_text})

        response_tokens = self._provider.count_tokens(response_text)
        total_context = (
            self._system_tokens + history_tokens_after + prompt_tokens + response_tokens
        )

        return TurnStats(
            prompt_tokens=prompt_tokens,
            history_tokens_before=history_tokens_before,
            history_tokens_after=history_tokens_after,
            response_tokens=response_tokens,
            compression_result=compression_result,
            total_context_tokens=total_context,
            context_budget=self._context_budget,
        )

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def last_response(self) -> str:
        return self._last_response

    @property
    def context_budget(self) -> int:
        return self._context_budget

    @context_budget.setter
    def context_budget(self, value: int) -> None:
        self._context_budget = value

    def reset(self) -> None:
        self._history = []
        self._last_response = ""

    def export_history(self) -> list[Message]:
        return list(self._history)

    def import_history(self, messages: list[Message]) -> None:
        self._history = list(messages)
