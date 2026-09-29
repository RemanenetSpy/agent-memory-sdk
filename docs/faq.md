# FAQ

## Decision actions

**Why REPLAY instead of RESTORE?**
A high-confidence exact or near-exact match means the stored answer is directly applicable. REPLAY skips the LLM call entirely — zero tokens, zero latency, and consistent output. RESTORE is used when the memory is relevant but may need adaptation.

**When is VERIFY triggered?**
For memories of type `fact`, `workflow`, or `tool_output` when either:
- Their score falls below `verify_threshold` (default 0.80), or
- They are older than `recency_half_life_days` (default 30 days), or
- `requires_verification=True` was set at store time and the candidate clears the restore threshold. Below that threshold, the correct action is NONE.

**Why does NONE fire even though there's a related memory?**
The composite score (semantic + recency + confidence + usage) is below the restore threshold. This prevents the "shared-word trap" — a memory about payment methods doesn't answer a question about two-factor authentication just because both mention "support".

**What is the trap score threshold?**
By default `restore_threshold=0.70`. Below this, the action is NONE. Raise it to be more conservative, lower it to be more permissive.

---

## Memory management

**Nothing is automatically saved — is that intentional?**
Yes. Auto-saving every turn fills the store with low-quality junk. Your application decides what's worth remembering by calling `remember()` after a validated answer.

**How do I stop a fact from replaying stale data?**
Set `requires_verification=True` at store time. A sufficiently relevant match returns VERIFY instead of REPLAY or RESTORE; a candidate below the restore threshold returns NONE and is not used.

**How do I share memory between multiple processes?**
Point all processes at the same `persist_dir`. SQLite WAL mode makes concurrent reads safe. The MCP server, CLI, and Python SDK can all share one directory.

**How do I isolate memories between users?**
Use a scoped view for per-user isolation. Derive the user ID from authenticated application context; `scope="user"` alone is only a tier label:
```python
alice = memory.scoped(user_id=authenticated_user_id)
decision = alice.resolve(query)
```

---

## Performance

**How fast is `resolve()`?**

It depends on the corpus, query terms, cache state, embedding mode, and
machine. A cache hit measures the in-process cache path; a cache miss measures
retrieval against the configured store. Run the documented stress harness with
your workload before setting a latency expectation.

**How does it scale?**

For SQLite FTS5, query-term document frequency can matter more than total row
count. Measure the corpus and query distribution you plan to ship; see
[stress-testing.md](stress-testing.md) for the reproducible harness.

**When should I use Redis, Postgres, or Qdrant instead of SQLite?**
- **Redis**: multiple services sharing memory, and the lowest latency of the server backends
- **Postgres**: production deployment with existing SQL infrastructure and SQL-native aggregates
- **Qdrant**: the corpus has outgrown a general-purpose database and KNN latency
  is the bottleneck — Qdrant is built for nothing but vector search, and scales
  further than we have measured (our own numbers stop at 20,000 entries; treat
  Qdrant's published limits as theirs, not ours)

All three do real vector KNN, but only with a server-side index: RediSearch on
Redis (8+ / Stack, db 0), the `vector` extension on Postgres, and natively on
Qdrant. Without one — or without the `[semantic]` extra — `search()` quietly
becomes keyword search. `memory.store.semantic_search_enabled` tells you which.
- **SQLite**: a zero-setup local default; validate its behavior against your workload and deployment needs

---

## Debugging

**How do I see why a decision was made?**
```python
print(decision.explain())
# Outputs:
#   action: restore  confidence: 0.76
#   reasons: moderate semantic match, keyword match, recent memory
#   scores:
#     semantic_score:  0.74
#     keyword_score:   0.75
#     recency_score:   1.00
#     confidence_score: 1.00
#     policy_score:    0.76
```

**How do I enable debug logging?**
```python
from agent_memory import configure_debug_logging
configure_debug_logging()
```
This shows every retrieval, cache hit/miss, and decision at DEBUG level from `agent_memory.*` loggers.

**Why are some test queries returning NONE when I expect a match?**
Check `decision.reasons` — common causes:
- Low semantic/keyword overlap → the query and memory don't share enough content words
- Memory is too old → recency score is low, dragging the composite below the threshold
- Confidence is low → update it with `ConfidenceLearner.record_event(entry, ConfidenceEvent.USER_CONFIRMED)`

---

## Integration

**Can I use it with LangChain / LlamaIndex?**
Yes. `AgentMemoryLangChain` (drop-in for `BaseMemory`) and `AgentMemoryLlamaIndex` are included. See [examples/README.md](../examples/README.md).

**Does it work with the MCP protocol (Cursor, Claude Code)?**
Yes. `agent-memory-mcp` exposes 7 MCP tools. See [docs/mcp.md](mcp.md).

**Can multiple agents share one memory store?**
Yes. `MultiAgentMemory` wraps any `Memory` instance with SHARED / NAMESPACED / ISOLATED isolation modes. See [examples/multi_agent.py](../examples/multi_agent.py).

**What happens if I don't call `remember()` after a good answer?**
Nothing is stored. The next identical query will return NONE and the agent will answer from scratch again.
