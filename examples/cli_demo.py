#!/usr/bin/env python3
"""Interactive CLI demo for conmax — context maximizer."""

from __future__ import annotations

import os
import sys

# Add src to path for running without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conmax.chat import ConmaxChat, TurnStats
from conmax.compressor import CompressionConfig, COMPRESSION_MARKER
from conmax.provider import LLMProvider, create_provider


def _has_rich() -> bool:
    try:
        import rich
        return True
    except ImportError:
        return False


def pick_provider() -> LLMProvider:
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_anthropic and not has_openai:
        print("No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        sys.exit(1)

    available: list[str] = []
    if has_anthropic:
        available.append("anthropic")
    if has_openai:
        available.append("openai")

    if len(available) == 1:
        choice = available[0]
        print(f"Using {choice} (API key found)")
    else:
        print("Available providers:")
        for i, name in enumerate(available, 1):
            print(f"  {i}. {name}")
        while True:
            raw = input("Pick a provider [1]: ").strip()
            if raw == "":
                choice = available[0]
                break
            try:
                idx = int(raw) - 1
                choice = available[idx]
                break
            except (ValueError, IndexError):
                print("Invalid choice, try again.")

    return create_provider(choice)


def pick_budget() -> tuple[int, int]:
    print("\nContext budget options:")
    print("  1. Large (100,000 tokens) — normal usage")
    print("  2. Medium (8,000 tokens) — see compression after a few turns")
    print("  3. Demo (2,000 tokens) — forces compression quickly")
    print("  4. Custom")

    presets = {
        "1": (100_000, 2048),
        "2": (8_000, 1024),
        "3": (2_000, 512),
    }

    while True:
        raw = input("Choice [3]: ").strip()
        if raw == "":
            raw = "3"
        if raw in presets:
            budget, reserve = presets[raw]
            print(f"Budget: {budget:,} tokens, response reserve: {reserve:,}")
            return budget, reserve
        if raw == "4":
            try:
                budget = int(input("Context budget (tokens): "))
                reserve = int(input("Response reservation (tokens) [2048]: ") or "2048")
                return budget, reserve
            except ValueError:
                print("Invalid number.")
        else:
            print("Invalid choice.")


def render_stats_plain(stats: TurnStats) -> None:
    compressed = stats.compression_result is not None and stats.compression_result.compressions_performed > 0
    bar_len = 30
    filled = int(bar_len * min(stats.utilization_pct, 100) / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    print()
    print(f"--- Turn Stats ---")
    print(f"  Prompt tokens:     {stats.prompt_tokens:>6,}")
    print(f"  History (before):  {stats.history_tokens_before:>6,}")
    print(f"  History (after):   {stats.history_tokens_after:>6,}")
    print(f"  Response tokens:   {stats.response_tokens:>6,}")
    print(f"  Tokens saved:      {stats.tokens_saved:>6,}")
    print(f"  Context budget:    {stats.context_budget:>6,}")
    print(f"  Utilization: [{bar}] {stats.utilization_pct:.1f}%")

    if compressed:
        cr = stats.compression_result
        print(f"  ** COMPRESSED: {cr.compressions_performed} chunk(s), saved {cr.tokens_saved:,} tokens **")
    print()


def render_stats_rich(stats: TurnStats) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    compressed = stats.compression_result is not None and stats.compression_result.compressions_performed > 0

    bar_len = 30
    filled = int(bar_len * min(stats.utilization_pct, 100) / 100)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    lines = Text()
    lines.append(f"  Prompt tokens:     {stats.prompt_tokens:>6,}\n")
    lines.append(f"  History (before):  {stats.history_tokens_before:>6,}\n")
    lines.append(f"  History (after):   {stats.history_tokens_after:>6,}\n")
    lines.append(f"  Response tokens:   {stats.response_tokens:>6,}\n")
    lines.append(f"  Tokens saved:      {stats.tokens_saved:>6,}\n")
    lines.append(f"  Context budget:    {stats.context_budget:>6,}\n")
    lines.append(f"  Utilization: [{bar}] {stats.utilization_pct:.1f}%\n")

    if compressed:
        cr = stats.compression_result
        lines.append(
            f"  COMPRESSED: {cr.compressions_performed} chunk(s), saved {cr.tokens_saved:,} tokens",
            style="bold yellow",
        )

    style = "yellow" if compressed else "cyan"
    console.print(Panel(lines, title="Turn Stats", border_style=style))


def render_stats(stats: TurnStats) -> None:
    if _has_rich():
        render_stats_rich(stats)
    else:
        render_stats_plain(stats)


def print_history_summary(chat: ConmaxChat) -> None:
    history = chat.history
    total = len(history)
    summaries = sum(1 for m in history if m["content"].startswith(COMPRESSION_MARKER))
    print(f"\nHistory: {total} messages ({summaries} summaries, {total - summaries} verbatim)\n")


def print_full_history(chat: ConmaxChat) -> None:
    for i, msg in enumerate(chat.history):
        role = msg["role"].upper()
        content = msg["content"]
        if len(content) > 120:
            content = content[:120] + "..."
        print(f"  [{i}] {role}: {content}")
    print()


def main() -> None:
    print("=" * 50)
    print("  conmax — context maximizer demo")
    print("=" * 50)
    print()

    provider = pick_provider()
    budget, reserve = pick_budget()

    chat = ConmaxChat(
        provider=provider,
        context_budget=budget,
        response_reservation=reserve,
        system_prompt="You are a helpful assistant. Keep responses concise.",
        compression_config=CompressionConfig(
            chunk_size=6,
            min_messages_to_keep=4,
            summary_ratio=0.3,
        ),
    )

    print(f"\nReady! Type your messages. Commands: /stats /history /reset /budget <N> /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye!")
            break
        elif user_input.lower() == "/stats":
            print_history_summary(chat)
            continue
        elif user_input.lower() == "/history":
            print_full_history(chat)
            continue
        elif user_input.lower() == "/reset":
            chat.reset()
            print("History cleared.\n")
            continue
        elif user_input.lower().startswith("/budget"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    chat.context_budget = int(parts[1])
                    print(f"Budget set to {chat.context_budget:,} tokens.\n")
                except ValueError:
                    print("Usage: /budget <number>\n")
            else:
                print(f"Current budget: {chat.context_budget:,} tokens\n")
            continue

        try:
            stats = chat.send(user_input)
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

        print(f"\nAssistant: {chat.last_response}\n")
        render_stats(stats)


if __name__ == "__main__":
    main()
