# Your agent's memory shouldn't answer questions it wasn't asked

*Draft blog post / Show HN write-up for Agent Memory
(`pip install agent-memory-sdk`).*

## The failure mode nobody benchmarks

Every agent-memory system demos the same way: store "How do I reset my
password?", ask "How do I reset my password?", get the answer back. Magic.

Here's the query that breaks most of them:

> Stored memory: **"What payment methods do you support?"** → *"We accept
> Visa, Mastercard, and PayPal."*
>
> User asks: **"Does your platform support two-factor authentication?"**

A retrieve-and-inject pipeline finds the payment memory (it shares the word
*support*), it's the top hit, and the answer about Visa and Mastercard goes
into the prompt — or worse, straight back to the user. The memory system did
exactly what it was built to do: retrieve. Nobody asked it whether the
retrieval should be *used*.

We know this failure mode intimately because an early version of our own
scoring had it: the top hit was normalized to a perfect 1.0, so *any* shared
word replayed a stored answer verbatim at confidence 1.0. We fixed it, wrote
adversarial cases into the eval suite so it can't come back, and then
realized the fix *is* the product.

## Memory needs a decision layer, not just a retriever

Retrieval answers "what did we store about this?". An agent needs a second
question answered: **"should I use it, and how much should I trust it?"**

Agent Memory returns one of four explicit actions from every `resolve()`:

| Action | Meaning | When |
|---|---|---|
| **replay** | Return the stored answer verbatim | Exact/near-exact repeat, high trust |
| **restore** | Inject the memory as context, let the LLM adapt it | Paraphrase, adjacent question |
| **verify** | Surface the memory but validate before reuse | Stale or `requires_verification` facts |
| **none** | Ignore memory entirely | Weak or coincidental matches |

The score behind the action combines similarity, recency, stored confidence,
and usage — and crucially, the non-similarity factors can *lower* it. A
memory stored with `confidence=0.1` never replays verbatim, even on an exact
query match; it restores as context instead. A fact that keeps being
replayed doesn't get artificially fresher; it ages into `verify`.

And it shows its work:

```text
>>> print(memory.resolve("Does the platform support 2FA?").explain())
action: none
confidence: 0.68
reasons:
  - keyword match
  - below restore threshold
scores:
  semantic_score: 0.60
  recency_score: 1.00
  confidence_score: 1.00
  policy_score: 0.68
```

When your agent answers wrongly, you can see *why the memory layer decided
what it decided* — which is more than most of us can say about our RAG
pipelines.

## The numbers (reproducible, no synthetic baselines)

The eval suite includes the trap cases above. Current results, reproducible
with `agent-memory eval` from the repo:

- **34/36 decision-quality cases** (the suite has grown; both misses return
  VERIFY — cautious, never a wrong REPLAY)
- Workload-dependent local retrieval latency with no network round-trip; use
  the reproducible [stress-test benchmark](stress-testing.md) to measure your
  corpus and query mix
- Everything local: SQLite + optional ONNX MiniLM. No API keys, no torch.

We deliberately don't publish a "99% faster than an LLM call" number
computed against a hardcoded baseline. If you pass your own measured
latency, the benchmark compares against that, labeled as user-supplied.

## Try it in 60 seconds

Python:

```python
from agent_memory import Memory, MemoryAction

memory = Memory(persist_dir=".agent_memory")
memory.remember("What payment methods do you support?",
                "We accept Visa, Mastercard, and PayPal.")

decision = memory.resolve("Does the platform support two-factor auth?")
assert decision.action == MemoryAction.NONE   # the whole point
```

Or as an MCP server for Cursor / Claude Code — one config block:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory-mcp",
      "env": { "AGENT_MEMORY_DIR": "~/.agent_memory" }
    }
  }
}
```

Repo: https://github.com/TheProdSDE/agent-memory-sdk — the adversarial eval
dataset is in `benchmarks/datasets/decision_traps.json`, and we take PRs
that add trap cases we fail.
