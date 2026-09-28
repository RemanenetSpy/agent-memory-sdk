# Usage Guide

## The two-call pattern

Nothing is saved automatically. Your agent calls `resolve()` **before** the LLM and `remember()` **after** a good answer:

```python
from agent_memory import Memory, MemoryAction

memory = Memory(persist_dir=".agent_memory")

def handle(user_query: str) -> str:
    decision = memory.resolve(user_query)          # ① BEFORE the LLM

    if decision.action == MemoryAction.REPLAY:
        return decision.response                   # no LLM call — zero cost

    if decision.action == MemoryAction.RESTORE:
        context = memory.format_restore_context(decision)
        answer  = call_llm(user_query, system_extra=context)
    elif decision.action == MemoryAction.VERIFY:
        answer  = revalidate_or_regenerate(decision.memory, user_query)
    else:                                          # NONE — answer from scratch
        answer  = call_llm(user_query)

    memory.remember(user_query, answer)            # ② AFTER a good answer
    return answer
```

## What to remember, and how

| What you're saving | How to save it |
|---|---|
| A validated answer the user accepted | `remember(q, a, confidence=0.95)` |
| An expensive tool / API result | `type="tool_output", ttl="1h"` — eligible for reuse within the hour; verify-type policy applies |
| A fact that can go stale (rate limits, prices) | `type="fact", requires_verification=True` — relevant matches go to VERIFY; low-scoring matches return NONE |
| A user preference | `type="preference", scope="user"` |
| Project conventions | `type="workflow", scope="project"` |
| A low-certainty guess | `confidence=0.4` — may restore as context, never replays verbatim |

## Inspecting a decision

Every decision is transparent:

```python
decision = memory.resolve("How do I reset my password?")

decision.response        # the stored answer being replayed
decision.memory.query    # the original question it matched
decision.memory.confidence, decision.memory.access_count
print(decision.explain())  # full score breakdown
```

## Sharing memory across processes

The store is a SQLite file under `persist_dir`. Every process pointing at the same path shares memories — WAL mode makes concurrent access safe:

- **Your Python app** — `Memory(persist_dir="~/.myapp_memory")`
- **MCP server** — `AGENT_MEMORY_DIR=~/.myapp_memory agent-memory-mcp`
- **CLI** — `agent-memory --data-dir ~/.myapp_memory resolve "…"`

What Cursor learned this morning is replayable from your Python service this afternoon.

## Where it earns its keep

**Support bot** — repeated questions REPLAY (zero LLM cost, identical answers); paraphrases RESTORE the canonical answer; policy facts stored with `requires_verification=True` get re-checked before reuse.

**Coding agent** — project-scoped workflows stop the agent re-deriving your conventions each session, but age into VERIFY when they go stale.

**Tool-output caching with judgment** — API results replay within their TTL; unrelated questions never get polluted by them (trap-query protection).

**Not for document RAG** — this stores query→answer *experiences* and decides whether to trust them. It complements a document store, not replaces one.

---

## API Reference

### Core methods

```python
# Store
memory.remember(query, response, *,
    type="conversation", scope="user",
    tags=[], confidence=1.0,
    requires_verification=False,
    ttl=None, metadata={})

# Retrieve
entry    = memory.get(memory_id)
entries  = memory.list(limit=100, offset=0, *, scope, include_archived, type)

# Decide
decision = memory.resolve(query, *,
    mode="auto",      # "auto" | "replay" | "restore" | "verify"
    top_k=3,
    scope=None,
    enable_verify=True)

# Maintain
memory.forget(memory_id)
memory.archive(memory_id)
memory.cleanup(delete=False)                  # mark / delete expired
memory.consolidate(similarity_threshold=0.95) # merge near-duplicates
memory.stats()                                # aggregate statistics

# Format context for the LLM
memory.format_restore_context(decision)       # → Markdown string
memory.format_verify_context(decision)        # → Markdown string
```

All methods have async counterparts: `aremember`, `aresolve`, `alist`, `aget`, `aforget`, `aarchive`, `acleanup`, `astats`, `aconsolidate`.

### MemoryDecision fields

```python
decision.action        # MemoryAction: REPLAY | RESTORE | VERIFY | NONE
decision.confidence    # float 0.0–1.0
decision.query         # str — the original query
decision.reasons       # list[str] — human-readable reason tags
decision.scores        # dict[str, float] — per-component scores
decision.response      # str | None — for REPLAY
decision.memory        # MemoryEntry | None — for REPLAY / VERIFY
decision.context       # list[RetrievalResult] — for RESTORE / VERIFY
decision.explain()     # → str full score breakdown
```

### MemoryEntry fields

```python
entry.id, entry.query, entry.response, entry.content
entry.type             # MemoryType enum
entry.scope            # MemoryScope enum
entry.confidence       # float 0.0–1.0
entry.tags             # list[str]
entry.metadata         # dict
entry.access_count     # int
entry.created_at, entry.updated_at, entry.last_accessed_at
entry.expires_at       # datetime | None
entry.archived         # bool
entry.state            # MemoryState: ACTIVE | ARCHIVED | EXPIRED
entry.requires_verification  # bool
```
