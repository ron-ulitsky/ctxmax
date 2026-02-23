#!/usr/bin/env python3
"""
ctxmax benchmark suite.

Three benchmarks run offline (no API keys required):

  1. Compression Profile  — per-turn stats across a simulated conversation
  2. Strategy Comparison  — ctxmax vs truncation vs fixed-threshold compression
  3. Config Sensitivity   — effect of CompressionConfig parameters

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --turns 50 --budget 4000 --msg-tokens 200
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from ctxmax import CtxmaxChat, CompressionConfig
from ctxmax.provider import LLMProvider
from ctxmax.tokenizer import APPROXIMATE_COUNTER, Message, TokenCounter


# ─────────────────────────────────────────────────────────────────────────────
# MockProvider — deterministic, no API keys, faithfully respects target_tokens
# ─────────────────────────────────────────────────────────────────────────────


class MockProvider(LLMProvider):
    """Deterministic mock provider for offline benchmarking.

    compress() returns text whose token count matches target_tokens
    (APPROXIMATE_COUNTER divides char length by 4, so chars = tokens * 4).
    chat() returns a fixed-length response.
    """

    def __init__(self, response_tokens: int = 60) -> None:
        self._response_tokens = response_tokens
        self.compress_calls: int = 0
        self.chat_calls: int = 0
        super().__init__()

    def _make_token_counter(self) -> TokenCounter:
        return APPROXIMATE_COUNTER

    def compress(self, messages: list[Message], target_tokens: int) -> str:
        self.compress_calls += 1
        target_chars = max(target_tokens * 4, 20)
        base = " ".join(m["content"][:20] for m in messages)
        if len(base) < target_chars:
            base = base + " " + "s" * (target_chars - len(base) - 1)
        return base[:target_chars]

    def chat(self, messages: list[Message]) -> str:
        self.chat_calls += 1
        return "r" * (self._response_tokens * 4)

    def reset(self) -> None:
        self.compress_calls = 0
        self.chat_calls = 0


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1: Compression Profile
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TurnRecord:
    turn: int
    history_before: int
    history_after: int
    tokens_saved: int
    chunks_compressed: int
    utilization_pct: float
    compress_calls_this_turn: int


def bench_compression_profile(
    *,
    n_turns: int,
    message_tokens: int,
    context_budget: int,
    response_reservation: int = 300,
    config: CompressionConfig | None = None,
) -> tuple[list[TurnRecord], MockProvider]:
    """Simulate a conversation and record per-turn compression stats."""
    provider = MockProvider(response_tokens=60)
    chat = CtxmaxChat(
        provider,
        context_budget=context_budget,
        response_reservation=response_reservation,
        compression_config=config,
    )

    records: list[TurnRecord] = []
    for i in range(n_turns):
        user_msg = f"T{i + 1}: " + "u" * (message_tokens * 4 - 5)
        calls_before = provider.compress_calls
        stats = chat.send(user_msg)
        records.append(
            TurnRecord(
                turn=i + 1,
                history_before=stats.history_tokens_before,
                history_after=stats.history_tokens_after,
                tokens_saved=stats.tokens_saved,
                chunks_compressed=(
                    stats.compression_result.compressions_performed
                    if stats.compression_result
                    else 0
                ),
                utilization_pct=stats.utilization_pct,
                compress_calls_this_turn=provider.compress_calls - calls_before,
            )
        )

    return records, provider


def report_compression_profile(
    records: list[TurnRecord], provider: MockProvider
) -> None:
    print(_header("Benchmark 1: Compression Profile"))
    print(
        f"{'Turn':>4}  {'Hist-Before':>11}  {'Hist-After':>10}  "
        f"{'Saved':>6}  {'Chunks':>6}  {'Util%':>6}  {'Calls':>5}"
    )
    print(_hline())

    turns_with_compression = 0
    for r in records:
        marker = " *" if r.chunks_compressed > 0 else "  "
        print(
            f"{r.turn:>4}{marker} {r.history_before:>10,}  {r.history_after:>10,}  "
            f"{r.tokens_saved:>6,}  {r.chunks_compressed:>6}  "
            f"{r.utilization_pct:>5.1f}%  {r.compress_calls_this_turn:>5}"
        )
        if r.chunks_compressed > 0:
            turns_with_compression += 1

    print(_hline())
    total_saved = sum(r.tokens_saved for r in records)
    print(f"Turns with compression : {turns_with_compression}/{len(records)}")
    print(f"Total tokens saved     : {total_saved:,}")
    print(f"Total compress() calls : {provider.compress_calls}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2: Strategy Comparison
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StrategyResult:
    name: str
    final_msgs: int
    verbatim_msgs: int
    summary_msgs: int
    final_tokens: int
    compress_calls: int
    avg_utilization: float
    # Fraction of original turns with any representation in the final history.
    # Summarization strategies retain 100%; truncation loses early turns.
    coverage_pct: float = 100.0


def _is_summary(msg: Message) -> bool:
    return msg["content"].startswith("[SUMMARY]")


def _truncate_to_fit(
    history: list[Message], budget: int, p: MockProvider
) -> list[Message]:
    while history and p.count_messages_tokens(history) > budget:
        history = history[1:]
    return history


def _fixed_threshold_compress(
    history: list[Message],
    budget: int,
    threshold: float,
    target_ratio: float,
    p: MockProvider,
) -> list[Message]:
    tokens = p.count_messages_tokens(history)
    if tokens / budget < threshold:
        return history
    keep = 4
    if len(history) <= keep:
        return history
    target = int(budget * target_ratio)
    head, tail = history[:-keep], history[-keep:]
    summary = p.compress(head, target)
    return [{"role": "assistant", "content": f"[SUMMARY] {summary}"}] + tail


def _coverage_pct(verbatim_msgs: int, summary_msgs: int, n_turns: int) -> float:
    """Estimate what fraction of original turns have any representation.

    Strategies that never delete (ctxmax, fixed-threshold) always cover 100%.
    Truncation can only represent the turns whose messages are still verbatim.
    """
    if summary_msgs > 0:
        return 100.0
    # Each turn contributes 2 messages (user + assistant).
    covered_turns = verbatim_msgs // 2
    return min(100.0, 100.0 * covered_turns / n_turns) if n_turns > 0 else 0.0


def bench_strategy_comparison(
    *,
    n_turns: int,
    message_tokens: int,
    context_budget: int,
    response_tokens: int = 60,
) -> list[StrategyResult]:
    """Run all three strategies on the same workload and collect final-state metrics."""
    response_text = "r" * (response_tokens * 4)
    results: list[StrategyResult] = []

    # ── ctxmax ────────────────────────────────────────────────────────────────
    p1 = MockProvider(response_tokens=response_tokens)
    chat1 = CtxmaxChat(p1, context_budget=context_budget, response_reservation=300)
    utils1: list[float] = []
    for i in range(n_turns):
        stats = chat1.send(f"T{i + 1}: " + "u" * (message_tokens * 4 - 5))
        utils1.append(stats.utilization_pct)

    h1 = chat1.history
    verbatim1 = sum(1 for m in h1 if not _is_summary(m))
    summary1 = sum(1 for m in h1 if _is_summary(m))
    results.append(
        StrategyResult(
            name="ctxmax (just-in-time)",
            final_msgs=len(h1),
            verbatim_msgs=verbatim1,
            summary_msgs=summary1,
            final_tokens=p1.count_messages_tokens(h1),
            compress_calls=p1.compress_calls,
            avg_utilization=sum(utils1) / len(utils1),
            coverage_pct=_coverage_pct(verbatim1, summary1, n_turns),
        )
    )

    # ── Truncation ────────────────────────────────────────────────────────────
    p2 = MockProvider(response_tokens=response_tokens)
    h2: list[Message] = []
    utils2: list[float] = []
    for i in range(n_turns):
        h2.append(
            {"role": "user", "content": f"T{i + 1}: " + "u" * (message_tokens * 4 - 5)}
        )
        h2.append({"role": "assistant", "content": response_text})
        h2 = _truncate_to_fit(h2, context_budget, p2)
        utils2.append(100 * p2.count_messages_tokens(h2) / context_budget)

    verbatim2 = sum(1 for m in h2 if not _is_summary(m))
    results.append(
        StrategyResult(
            name="Truncation (drop oldest)",
            final_msgs=len(h2),
            verbatim_msgs=verbatim2,
            summary_msgs=0,
            final_tokens=p2.count_messages_tokens(h2),
            compress_calls=0,
            avg_utilization=sum(utils2) / len(utils2),
            coverage_pct=_coverage_pct(verbatim2, 0, n_turns),
        )
    )

    # ── Fixed-threshold ───────────────────────────────────────────────────────
    p3 = MockProvider(response_tokens=response_tokens)
    h3: list[Message] = []
    utils3: list[float] = []
    for i in range(n_turns):
        h3.append(
            {"role": "user", "content": f"T{i + 1}: " + "u" * (message_tokens * 4 - 5)}
        )
        h3.append({"role": "assistant", "content": response_text})
        h3 = _fixed_threshold_compress(h3, context_budget, 0.80, 0.50, p3)
        utils3.append(100 * p3.count_messages_tokens(h3) / context_budget)

    verbatim3 = sum(1 for m in h3 if not _is_summary(m))
    summary3 = sum(1 for m in h3 if _is_summary(m))
    results.append(
        StrategyResult(
            name="Fixed-threshold (80% → 50%)",
            final_msgs=len(h3),
            verbatim_msgs=verbatim3,
            summary_msgs=summary3,
            final_tokens=p3.count_messages_tokens(h3),
            compress_calls=p3.compress_calls,
            avg_utilization=sum(utils3) / len(utils3),
            coverage_pct=_coverage_pct(verbatim3, summary3, n_turns),
        )
    )

    return results


_STRATEGY_WIDTH = 98


def report_strategy_comparison(results: list[StrategyResult]) -> None:
    print(_header("Benchmark 2: Strategy Comparison", _STRATEGY_WIDTH))
    print(
        f"{'Strategy':<28}  {'Msgs':>5}  {'Verbatim':>8}  {'Summaries':>9}  "
        f"{'Tokens':>7}  {'Cmpr calls':>10}  {'Coverage':>8}  {'Avg Util%':>9}"
    )
    print(_hline(_STRATEGY_WIDTH))
    for r in results:
        print(
            f"{r.name:<28}  {r.final_msgs:>5}  {r.verbatim_msgs:>8}  "
            f"{r.summary_msgs:>9}  {r.final_tokens:>7,}  "
            f"{r.compress_calls:>10}  {r.coverage_pct:>7.0f}%  {r.avg_utilization:>8.1f}%"
        )
    print(_hline(_STRATEGY_WIDTH))
    print("Verbatim  = messages not replaced by summaries (more = more original context)")
    print("Summaries = compressed chunks (ctxmax: fine-grained; fixed-threshold: coarse)")
    print("Coverage  = % of original turns with any representation in the final history")
    print("Cmpr calls = proxy for compression latency overhead")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 3: Config Sensitivity
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConfigResult:
    label: str
    compress_calls: int
    final_msgs: int
    verbatim_msgs: int
    final_tokens: int
    avg_utilization: float


def bench_config_sensitivity(
    *,
    n_turns: int,
    message_tokens: int,
    context_budget: int,
) -> list[ConfigResult]:
    """Vary one CompressionConfig parameter at a time and observe the effect."""
    configs: list[tuple[str, CompressionConfig]] = [
        ("default  (chunk=6, keep=4, ratio=0.30)", CompressionConfig()),
        ("chunk=2  (chunk=2, keep=4, ratio=0.30)", CompressionConfig(chunk_size=2)),
        ("chunk=10 (chunk=10, keep=4, ratio=0.30)", CompressionConfig(chunk_size=10)),
        ("keep=2   (chunk=6, keep=2, ratio=0.30)", CompressionConfig(min_messages_to_keep=2)),
        ("keep=8   (chunk=6, keep=8, ratio=0.30)", CompressionConfig(min_messages_to_keep=8)),
        ("ratio=0.1 (chunk=6, keep=4, ratio=0.10)", CompressionConfig(summary_ratio=0.1)),
        ("ratio=0.5 (chunk=6, keep=4, ratio=0.50)", CompressionConfig(summary_ratio=0.5)),
    ]

    results: list[ConfigResult] = []
    for label, config in configs:
        p = MockProvider(response_tokens=60)
        chat = CtxmaxChat(
            p,
            context_budget=context_budget,
            response_reservation=300,
            compression_config=config,
        )
        utils: list[float] = []
        for i in range(n_turns):
            stats = chat.send(f"T{i + 1}: " + "u" * (message_tokens * 4 - 5))
            utils.append(stats.utilization_pct)

        h = chat.history
        results.append(
            ConfigResult(
                label=label,
                compress_calls=p.compress_calls,
                final_msgs=len(h),
                verbatim_msgs=sum(1 for m in h if not _is_summary(m)),
                final_tokens=p.count_messages_tokens(h),
                avg_utilization=sum(utils) / len(utils),
            )
        )

    return results


_CONFIG_WIDTH = 88


def report_config_sensitivity(results: list[ConfigResult]) -> None:
    print(_header("Benchmark 3: Config Sensitivity", _CONFIG_WIDTH))
    print(
        f"{'Config':<44}  {'Calls':>5}  {'Msgs':>5}  "
        f"{'Verbatim':>8}  {'Tokens':>7}  {'Avg Util%':>9}"
    )
    print(_hline(_CONFIG_WIDTH))
    for r in results:
        print(
            f"{r.label:<44}  {r.compress_calls:>5}  {r.final_msgs:>5}  "
            f"{r.verbatim_msgs:>8}  {r.final_tokens:>7,}  {r.avg_utilization:>8.1f}%"
        )
    print(_hline(_CONFIG_WIDTH))
    print("chunk=  : messages per compression chunk (larger → fewer, coarser compressions)")
    print("keep=   : recent messages always kept verbatim (larger → more protected tail)")
    print("ratio=  : target summary size as fraction of chunk (lower → more aggressive)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────


def _hline(width: int = 82) -> str:
    return "─" * width


def _header(title: str, width: int = 82) -> str:
    pad = (width - len(title) - 2) // 2
    return f"{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ctxmax offline benchmark — no API keys required",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=40,
        metavar="N",
        help="Number of conversation turns to simulate",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=3000,
        metavar="TOKENS",
        help="Context budget in tokens",
    )
    parser.add_argument(
        "--msg-tokens",
        type=int,
        default=150,
        metavar="TOKENS",
        help="Approximate tokens per user message",
    )
    args = parser.parse_args()

    print()
    print("ctxmax benchmark suite (offline — no API keys required)")
    print(f"Parameters: turns={args.turns}, budget={args.budget:,}, msg_tokens={args.msg_tokens}")
    print()

    t0 = time.perf_counter()

    records, provider = bench_compression_profile(
        n_turns=args.turns,
        message_tokens=args.msg_tokens,
        context_budget=args.budget,
    )
    report_compression_profile(records, provider)

    strategy_results = bench_strategy_comparison(
        n_turns=args.turns,
        message_tokens=args.msg_tokens,
        context_budget=args.budget,
    )
    report_strategy_comparison(strategy_results)

    config_results = bench_config_sensitivity(
        n_turns=args.turns,
        message_tokens=args.msg_tokens,
        context_budget=args.budget,
    )
    report_config_sensitivity(config_results)

    print(f"Total elapsed: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
