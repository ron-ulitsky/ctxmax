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

## Development

```bash
git clone https://github.com/yourusername/conmax.git
cd conmax
pip install -e ".[dev]"
pytest tests/
```

## License

MIT
