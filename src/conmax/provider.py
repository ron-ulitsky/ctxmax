from __future__ import annotations

import abc
import os
from collections.abc import Sequence

from .tokenizer import Message, TokenCounter, APPROXIMATE_COUNTER, messages_to_text


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    def __init__(self) -> None:
        self._counter: TokenCounter = self._make_token_counter()

    @abc.abstractmethod
    def _make_token_counter(self) -> TokenCounter: ...

    @abc.abstractmethod
    def compress(self, messages: Sequence[Message], target_tokens: int) -> str:
        """Summarize messages into ~target_tokens tokens."""

    @abc.abstractmethod
    def chat(self, messages: Sequence[Message]) -> str:
        """Send messages and return the assistant response text."""

    def count_tokens(self, text: str) -> int:
        return self._counter.count(text)

    def count_messages_tokens(self, messages: Sequence[Message]) -> int:
        return self._counter.count_messages(messages)

    @property
    def counter_name(self) -> str:
        return self._counter.name


_DEFAULT_COMPRESSION_PROMPT = (
    "You are a conversation summarizer. Summarize the provided conversation "
    "history concisely, preserving all key facts, decisions, and context. "
    "Output only the summary text, no preamble."
)


class AnthropicProvider(LLMProvider):
    """Provider using Anthropic's Claude models."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 2048,
        compression_model: str | None = None,
        compression_prompt: str = _DEFAULT_COMPRESSION_PROMPT,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.compression_model = compression_model or model
        self.compression_prompt = compression_prompt

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install conmax[anthropic]"
            )

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        super().__init__()

    def _make_token_counter(self) -> TokenCounter:
        client = self._client

        def count_fn(text: str) -> int:
            try:
                result = client.beta.messages.count_tokens(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": text}],
                )
                return result.input_tokens
            except Exception:
                return APPROXIMATE_COUNTER.count(text)

        return TokenCounter(count_fn=count_fn, name="anthropic-api")

    def compress(self, messages: Sequence[Message], target_tokens: int) -> str:
        formatted = messages_to_text(messages)
        response = self._client.messages.create(
            model=self.compression_model,
            max_tokens=target_tokens,
            system=self.compression_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Summarize this conversation in approximately "
                        f"{target_tokens} tokens:\n\n{formatted}"
                    ),
                }
            ],
        )
        return response.content[0].text

    def chat(self, messages: Sequence[Message]) -> str:
        # Separate system prompt if first message has role "system"
        system = None
        chat_messages = list(messages)
        if chat_messages and chat_messages[0]["role"] == "system":
            system = chat_messages.pop(0)["content"]

        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=chat_messages,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text


class OpenAIProvider(LLMProvider):
    """Provider using OpenAI-compatible APIs."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        compression_model: str | None = None,
        compression_prompt: str = _DEFAULT_COMPRESSION_PROMPT,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.compression_model = compression_model or model
        self.compression_prompt = compression_prompt

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install conmax[openai]"
            )

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        super().__init__()

    def _make_token_counter(self) -> TokenCounter:
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.model)

            def count_fn(text: str) -> int:
                return len(encoding.encode(text))

            return TokenCounter(count_fn=count_fn, name=f"tiktoken/{encoding.name}")
        except Exception:
            return APPROXIMATE_COUNTER

    def compress(self, messages: Sequence[Message], target_tokens: int) -> str:
        formatted = messages_to_text(messages)
        response = self._client.chat.completions.create(
            model=self.compression_model,
            max_tokens=target_tokens,
            messages=[
                {"role": "system", "content": self.compression_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Summarize this conversation in approximately "
                        f"{target_tokens} tokens:\n\n{formatted}"
                    ),
                },
            ],
        )
        return response.choices[0].message.content

    def chat(self, messages: Sequence[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content


def create_provider(provider: str, **kwargs) -> LLMProvider:
    """Factory function. provider is one of 'anthropic', 'openai'."""
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
    }
    if provider not in providers:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list(providers.keys())}"
        )
    return providers[provider](**kwargs)
