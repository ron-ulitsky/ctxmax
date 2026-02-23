#!/usr/bin/env python3
"""Benchmark the ctxmax just-in-time compression strategy.

Runs entirely offline using a deterministic MockProvider — no API keys needed.
The provider generates summaries of exactly the requested token size so that
the approximate token counter produces reproducible, meaningful numbers.

Usage (from repo root):
    python benchmarks/benchmark_compression.py
    python benchmarks/benchmark_compression.py --quick   # fewer turns per scenario
    python benchmarks/benchmark_compression.py --group budget  # one scenario group
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field

# Allow running from repo root without installing the package.
sys.path.insert(0, "src")

from ctxmax.chat import CtxmaxChat, TurnStats
from ctxmax.compressor import CompressionConfig
from ctxmax.provider import LLMProvider
from ctxmax.tokenizer import APPROXIMATE_COUNTER, Message, TokenCounter


# ---------------------------------------------------------------------------
# Deterministic benchmark provider
# ---------------------------------------------------------------------------


class BenchmarkProvider(LLMProvider):
    """No-network provider for benchmarking.

    compress() produces a text whose approximate token count equals
    target_tokens, so token arithmetic in compress_history() is exact.

    chat() returns a fixed-size response that simulates a real model reply.
    """

    def __init__(self, response_chars: int = 400) -> None:
        self._response_chars = response_chars
        self.compress_call_count: int = 0
        super().__init__()

    def _make_token_counter(self) -> TokenCounter:
        return APPROXIMATE_COUNTER

    def compress(self, messages: list[Message], target_tokens: int) -> str:
        self.compress_call_count += 1
        # approximate_token_count = len(text) // 4, so target_tokens*4 chars ≈ target_tokens tokens.
        chars = max(target_tokens * 4, 4)
        return "s" * chars

    def chat(self, messages: list[Message]) -> str:
        return "r" * self._response_chars


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    turns: int  # number of send() calls to simulate
    context_budget: int  # token budget (the model's context window)
    user_msg_chars: int = 300  # chars per user message (~75 tokens)
    response_chars: int = 400  # chars in the mock assistant reply (~100 tokens)
    response_reservation: int = 150  # tokens reserved for the response
    system_prompt: str | None = None
    compression_config: CompressionConfig = field(default_factory=CompressionConfig)


# ---------------------------------------------------------------------------
# Per-scenario results
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    scenario: Scenario
    turn_stats: list[TurnStats]
    elapsed_s: float

    # --- aggregate properties ---

    @property
    def turns_total(self) -> int:
        return len(self.turn_stats)

    @property
    def turns_compressed(self) -> int:
        return sum(1 for s in self.turn_stats if s.compression_result is not None)

    @property
    def compression_rate(self) -> float:
        return self.turns_compressed / max(self.turns_total, 1)

    @property
    def total_tokens_saved(self) -> int:
        return sum(s.tokens_saved for s in self.turn_stats)

    @property
    def avg_utilization_pct(self) -> float:
        if not self.turn_stats:
            return 0.0
        return statistics.mean(s.utilization_pct for s in self.turn_stats)

    @property
    def avg_chunks_per_event(self) -> float:
        """Mean compressions_performed across turns that did compress."""
        events = [
            s.compression_result.compressions_performed
            for s in self.turn_stats
            if s.compression_result is not None
        ]
        return statistics.mean(events) if events else 0.0

    @property
    def avg_compression_ratio(self) -> float:
        """Mean tokens_before / tokens_after on compression turns."""
        ratios = [
            s.compression_result.tokens_before / s.compression_result.tokens_after
            for s in self.turn_stats
            if s.compression_result is not None
            and s.compression_result.tokens_after > 0
        ]
        return statistics.mean(ratios) if ratios else 1.0

    @property
    def multi_chunk_rate(self) -> float:
        """Fraction of compression events that needed >1 chunk compressed.

        A multi-chunk event means one chunk's summary_ratio wasn't enough to
        free the required tokens, so the algorithm had to compress more chunks.
        Higher values indicate a tighter budget relative to message sizes.
        """
        if self.turns_compressed == 0:
            return 0.0
        multi = sum(
            1
            for s in self.turn_stats
            if s.compression_result is not None
            and s.compression_result.compressions_performed > 1
        )
        return multi / self.turns_compressed

    @property
    def total_compress_api_calls(self) -> int:
        """Total summarisation calls made across the whole conversation."""
        return sum(
            s.compression_result.compressions_performed
            for s in self.turn_stats
            if s.compression_result is not None
        )


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------


def run_scenario(scenario: Scenario) -> ScenarioResult:
    provider = BenchmarkProvider(response_chars=scenario.response_chars)
    chat = CtxmaxChat(
        provider=provider,
        context_budget=scenario.context_budget,
        response_reservation=scenario.response_reservation,
        system_prompt=scenario.system_prompt,
        compression_config=scenario.compression_config,
    )
    user_message = "u" * scenario.user_msg_chars
    turn_stats: list[TurnStats] = []

    t0 = time.perf_counter()
    for _ in range(scenario.turns):
        stats = chat.send(user_message)
        turn_stats.append(stats)
    elapsed = time.perf_counter() - t0

    return ScenarioResult(
        scenario=scenario,
        turn_stats=turn_stats,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


_COL_NAME = 38
_COL_NUM = 7
_COL_WIDE = 11


def _header() -> str:
    return (
        f"{'Scenario':<{_COL_NAME}}"
        f"{'Turns':>{_COL_NUM}}"
        f"{'Budget':>{_COL_NUM}}"
        f"{'Comp%':>{_COL_WIDE}}"
        f"{'AvgUtil%':>{_COL_WIDE}}"
        f"{'AvgRatio':>{_COL_WIDE}}"
        f"{'AvgChunks':>{_COL_WIDE}}"
        f"{'MultiCh%':>{_COL_WIDE}}"
        f"{'TotalSaved':>{_COL_WIDE}}"
        f"{'CompCalls':>{_COL_WIDE}}"
    )


def _row(r: ScenarioResult) -> str:
    s = r.scenario
    return (
        f"{s.name:<{_COL_NAME}}"
        f"{r.turns_total:>{_COL_NUM}}"
        f"{s.context_budget:>{_COL_NUM}}"
        f"{r.compression_rate * 100:>{_COL_WIDE}.1f}"
        f"{r.avg_utilization_pct:>{_COL_WIDE}.1f}"
        f"{r.avg_compression_ratio:>{_COL_WIDE}.2f}"
        f"{r.avg_chunks_per_event:>{_COL_WIDE}.2f}"
        f"{r.multi_chunk_rate * 100:>{_COL_WIDE}.1f}"
        f"{r.total_tokens_saved:>{_COL_WIDE}}"
        f"{r.total_compress_api_calls:>{_COL_WIDE}}"
    )


def print_report(results: list[ScenarioResult], group: str | None = None) -> None:
    title = "ctxmax compression benchmark"
    if group:
        title += f" — group: {group}"
    print()
    print(title)
    print()

    # Group rows by scenario name prefix (text before first '/')
    last_group = None
    sep = "-" * len(_header())
    print(sep)
    print(_header())
    print(sep)

    for r in results:
        # Print a blank separator between groups
        group_key = r.scenario.name.split("/")[0]
        if last_group is not None and group_key != last_group:
            print()
        last_group = group_key
        print(_row(r))

    print(sep)
    print()
    print("Column legend:")
    print(f"  {'Comp%':<12} fraction of turns that triggered compression")
    print(f"  {'AvgUtil%':<12} mean context-window utilisation across all turns")
    print(f"  {'AvgRatio':<12} mean tokens_before/tokens_after on compression turns")
    print(f"  {'AvgChunks':<12} mean chunks compressed per compression turn")
    print(f"  {'MultiCh%':<12} % of compression turns needing >1 chunk compressed")
    print(f"  {'TotalSaved':<12} cumulative tokens freed across all turns")
    print(f"  {'CompCalls':<12} total summarisation API calls made")
    print()
    total_elapsed = sum(r.elapsed_s for r in results)
    print(f"Total wall time: {total_elapsed:.3f}s  ({len(results)} scenarios)")


# ---------------------------------------------------------------------------
# Scenario suite
# ---------------------------------------------------------------------------


def build_scenarios(quick: bool = False, group: str | None = None) -> list[Scenario]:
    """Return the scenario list, optionally filtered to a named group."""
    turns = 20 if quick else 60

    # A single turn consumes roughly:
    #   user_msg_chars/4 + response_chars/4 + ~8 overhead ≈ 175 tokens (200+400 chars)
    # Reference turn size ~175 tokens at default message sizes.

    all_scenarios: list[Scenario] = [
        # ── Group: budget ────────────────────────────────────────────────────
        # Vary budget tightness relative to per-turn token cost.
        Scenario(
            name="budget/tight  (budget ≈ 3 turns)",
            turns=turns,
            context_budget=500,
            user_msg_chars=200,
            response_chars=200,
        ),
        Scenario(
            name="budget/medium (budget ≈ 8 turns)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
        ),
        Scenario(
            name="budget/generous (budget ≈ 20 turns)",
            turns=turns,
            context_budget=3_500,
            user_msg_chars=200,
            response_chars=200,
        ),
        # ── Group: msgsize ───────────────────────────────────────────────────
        # Vary message size while keeping the budget/turn ratio constant.
        Scenario(
            name="msgsize/small (50-char msgs, budget=300)",
            turns=turns,
            context_budget=300,
            user_msg_chars=50,
            response_chars=50,
            response_reservation=50,
        ),
        Scenario(
            name="msgsize/medium (200-char msgs, budget=1400)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
        ),
        Scenario(
            name="msgsize/large (800-char msgs, budget=5000)",
            turns=turns,
            context_budget=5_000,
            user_msg_chars=800,
            response_chars=800,
            response_reservation=400,
        ),
        # ── Group: chunk_size ────────────────────────────────────────────────
        # Smaller chunks = more granular, more API calls; larger = coarser, fewer calls.
        Scenario(
            name="chunk_size/2  (fine-grained)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(chunk_size=2),
        ),
        Scenario(
            name="chunk_size/6  (default)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(chunk_size=6),
        ),
        Scenario(
            name="chunk_size/12 (coarse)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(chunk_size=12),
        ),
        # ── Group: keep ──────────────────────────────────────────────────────
        # min_messages_to_keep controls the protected tail.
        Scenario(
            name="keep/2  (small protected tail)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(min_messages_to_keep=2),
        ),
        Scenario(
            name="keep/4  (default)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(min_messages_to_keep=4),
        ),
        Scenario(
            name="keep/8  (large protected tail)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(min_messages_to_keep=8),
        ),
        # ── Group: ratio ─────────────────────────────────────────────────────
        # summary_ratio controls how aggressively each chunk is compressed.
        Scenario(
            name="ratio/0.1 (aggressive — 10% target)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(summary_ratio=0.1),
        ),
        Scenario(
            name="ratio/0.3 (default — 30% target)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(summary_ratio=0.3),
        ),
        Scenario(
            name="ratio/0.5 (conservative — 50% target)",
            turns=turns,
            context_budget=1_400,
            user_msg_chars=200,
            response_chars=200,
            compression_config=CompressionConfig(summary_ratio=0.5),
        ),
    ]

    if group is None:
        return all_scenarios

    filtered = [s for s in all_scenarios if s.name.startswith(group + "/")]
    if not filtered:
        available = sorted({s.name.split("/")[0] for s in all_scenarios})
        print(f"Unknown group '{group}'. Available: {available}", file=sys.stderr)
        sys.exit(1)
    return filtered


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ctxmax just-in-time compression (offline, no API keys)."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 20 turns per scenario instead of 60.",
    )
    parser.add_argument(
        "--group",
        metavar="NAME",
        help="Run only one scenario group: budget | msgsize | chunk_size | keep | ratio",
    )
    args = parser.parse_args()

    scenarios = build_scenarios(quick=args.quick, group=args.group)

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        result = run_scenario(scenario)
        results.append(result)

    print_report(results, group=args.group)


if __name__ == "__main__":
    main()
