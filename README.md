# conmax

**Context Maximizer** — dynamic just-in-time chat history compression for LLMs.

Instead of compressing chat history to a fixed size when hitting a threshold, conmax compresses *just enough* on each turn so you retain the maximum possible context.

## The Problem

Most LLM chat applications handle context limits by either:
- **Truncating** old messages (losing context entirely)
- **Compressing to a fixed size** when a threshold is hit (losing more context than necessary)

Both approaches throw away more information than needed.

## The Solution

conmax implements a simple formula: given a context budget `C`, a new prompt of size `p`, and a response reservation `r`, compress history to at most `C - p - r` tokens — and no more.

The compression algorithm:
1. If history already fits, do nothing (zero overhead)
2. Split history into a **protected tail** (recent messages kept verbatim) and **compressible head**
3. Chunk the compressible portion and compress oldest chunks first using an LLM summarizer
4. **Stop as soon as it fits** — never compress more than needed

This means you always retain the maximum amount of context that your budget allows.

## Installation

```bash
pip install conmax
```

With provider extras:

```bash
pip install conmax[anthropic]   # For Claude
pip install conmax[openai]      # For GPT + tiktoken
pip install conmax[all]         # Everything including CLI demo deps
```

## Quick Start

```python
from conmax import ConmaxChat, create_provider

provider = create_provider("anthropic")  # or "openai"
chat = ConmaxChat(provider, context_budget=100_000)

stats = chat.send("Hello! Tell me about quantum computing.")
print(chat.last_response)
print(f"Context utilization: {stats.utilization_pct:.1f}%")

# Keep chatting — compression happens automatically when needed
stats = chat.send("Can you go deeper on quantum entanglement?")
if stats.tokens_saved > 0:
    print(f"Compressed! Saved {stats.tokens_saved} tokens")
```

## Configuration

```python
from conmax import ConmaxChat, CompressionConfig, create_provider

provider = create_provider("anthropic", model="claude-sonnet-4-20250514")

chat = ConmaxChat(
    provider=provider,
    context_budget=100_000,       # Model's context window
    response_reservation=2048,    # Tokens reserved for the response
    system_prompt="You are a helpful assistant.",
    compression_config=CompressionConfig(
        chunk_size=6,              # Messages per compression chunk
        min_messages_to_keep=4,    # Recent messages always kept verbatim
        summary_ratio=0.3,         # Target: 30% of original size
    ),
)
```

## Provider Abstraction

conmax is provider-agnostic. Built-in providers for Anthropic and OpenAI are included, and you can bring your own:

```python
from conmax import LLMProvider, TokenCounter, APPROXIMATE_COUNTER

class MyProvider(LLMProvider):
    def _make_token_counter(self):
        return APPROXIMATE_COUNTER  # or your own

    def compress(self, messages, target_tokens):
        # Summarize messages into ~target_tokens tokens
        ...

    def chat(self, messages):
        # Return assistant response text
        ...
```

### Using with OpenAI-compatible local models (Ollama, etc.)

```python
provider = create_provider(
    "openai",
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
```

## TurnStats

Every call to `chat.send()` returns a `TurnStats` object:

```python
stats = chat.send("Hello!")

stats.prompt_tokens          # Tokens in your message
stats.history_tokens_before  # History size before compression
stats.history_tokens_after   # History size after compression
stats.response_tokens        # Tokens in the response
stats.tokens_saved           # How many tokens compression freed
stats.utilization_pct        # Context window utilization (0-100%)
stats.compression_result     # Detailed CompressionResult, or None
```

## CLI Demo

An interactive demo is included that shows compression in action with live stats:

```bash
pip install conmax[all]
ANTHROPIC_API_KEY=sk-... python examples/cli_demo.py
```

Use the "Demo" budget preset (2,000 tokens) to see compression kick in after just a few messages. Commands available in the demo:

| Command | Description |
|---------|-------------|
| `/stats` | Show history summary |
| `/history` | Print all messages with roles |
| `/reset` | Clear conversation history |
| `/budget <N>` | Change context budget mid-session |
| `/quit` | Exit |

## How Compression Works

```
Turn 1-3: History fits in budget → no compression, zero overhead
Turn 4:   History exceeds budget by 500 tokens
          → Compress oldest chunk (messages 0-5) into a summary
          → Summary frees 600 tokens → fits now, stop
          → Messages 6+ remain verbatim

Turn 7:   History exceeds budget again
          → Oldest chunk is already a summary, compress next chunk
          → Stop as soon as it fits
```

The key property: **compression is lazy and minimal**. Each turn compresses only what's needed, preserving maximum context at all times.

## Tradeoffs: Context Accuracy vs. Response Time

conmax makes an explicit tradeoff compared to traditional approaches: it prioritizes **context accuracy** (retaining maximum information) at the cost of **response latency** (extra LLM calls for compression).

### The core tension

Traditional compression strategies compress aggressively and infrequently — they batch a large chunk of history into a small summary, then coast for many turns before compressing again. This is fast on average but lossy: you throw away more context than you need to, and that information is gone forever.

conmax takes the opposite approach: compress minimally but potentially on every turn. Once the conversation exceeds the context budget, each new message may trigger a small compression pass. This means:

- **More LLM calls per turn** — each compression requires a summarization call *before* the actual chat call, so a turn that triggers compression makes 2+ API calls instead of 1
- **Higher latency** — the user waits for the compression call(s) to complete before their message is processed
- **Higher cost** — more API calls means more tokens processed and more money spent

### When this tradeoff makes sense

conmax is worth it when **context quality matters more than speed**:

- **Long-running agents** where losing early context causes the agent to repeat mistakes or forget decisions
- **Complex multi-step workflows** where the model needs to reference information from much earlier in the conversation
- **Low-volume, high-value conversations** (e.g. consulting, debugging sessions) where the user can tolerate an extra second or two of latency in exchange for better responses
- **Small context windows** (local models, older APIs) where every token of context is precious

### When traditional approaches are better

Stick with fixed-threshold compression or simple truncation when:

- **Latency is critical** — real-time chatbots, customer support, etc.
- **Conversations are short** — if you rarely hit the context limit, the optimization doesn't matter
- **Context isn't that important** — casual chat, simple Q&A where old messages rarely matter
- **Cost sensitivity is high** — the extra compression calls add up at scale

### Mitigating the latency cost

Several strategies can reduce the overhead:

| Strategy | How it helps |
|----------|-------------|
| **Use a cheap/fast model for compression** | Set `compression_model` to a smaller model (e.g. `claude-haiku-4-5-20251001` for compression, `claude-sonnet-4-20250514` for chat). Summarization doesn't need your best model. |
| **Increase `chunk_size`** | Larger chunks mean fewer compression calls per turn, at the cost of coarser-grained compression. |
| **Tune `summary_ratio`** | A lower ratio compresses more aggressively per chunk, meaning you need fewer chunks compressed to fit. |
| **Increase `response_reservation`** | A larger reservation means compression kicks in earlier and has more room, reducing the chance of needing multiple chunk compressions in a single turn. |
| **Use local models for compression** | Route compression calls to a fast local model via Ollama while keeping your main chat on a cloud API. |

### Comparison

| Approach | Context quality | Avg latency | Cost | Compression frequency |
|----------|----------------|-------------|------|----------------------|
| Truncation | Low — old context is deleted | None | None | Never |
| Fixed-threshold compression | Medium — compresses more than needed | Low (amortized) | Low (infrequent) | Infrequent, aggressive |
| **conmax (just-in-time)** | **High — minimal information loss** | Medium (per-turn) | Medium (frequent, small) | Frequent, minimal |

The fundamental insight is that compression is lossy, so the less you compress, the better your context quality. conmax minimizes total information loss by spreading compression across many small, targeted passes rather than a few large destructive ones.

## Development

```bash
git clone https://github.com/yourusername/conmax.git
cd conmax
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
