# Memory Model

Every stored experience is a `MemoryEntry` — a structured record of a question-and-answer pair enriched with metadata that drives retrieval and decision-making.

## MemoryEntry fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | UUID — stable identifier for the entry |
| `query` | `str` | The question or context that prompted this response |
| `response` | `str` | The answer or content to replay/restore |
| `content` | `str` | Searchable text (defaults to `response`) |
| `type` | `MemoryType` | Controls verification semantics — see below |
| `scope` | `MemoryScope` | Isolation boundary — see below |
| `confidence` | `float` | 0.0–1.0; updated by `ConfidenceLearner` events |
| `tags` | `list[str]` | Free-form labels for filtering and graph edges |
| `metadata` | `dict` | Caller-defined key-value pairs |
| `requires_verification` | `bool` | Always returns VERIFY, never silent REPLAY |
| `access_count` | `int` | Incremented on every REPLAY (not RESTORE) |
| `created_at` | `datetime` | Immutable — set once at store time |
| `updated_at` | `datetime` | Set on content edits (not on access) |
| `expires_at` | `datetime \| None` | TTL expiry; `None` = never expires |
| `state` | `MemoryState` | Derived: ACTIVE / ARCHIVED / EXPIRED |

---

## Memory types

The type controls how the decision engine treats the entry — specifically when to demand verification before reuse.

| Type | Default action | Verify when |
|------|---------------|-------------|
| `conversation` | REPLAY / RESTORE | rarely — conversational exchanges |
| `fact` | REPLAY if fresh → VERIFY if stale | confidence < threshold or age > half-life |
| `workflow` | REPLAY if fresh → VERIFY if stale | same as fact |
| `tool_output` | VERIFY always recommended | result may have changed |
| `document` | RESTORE always | too long to replay verbatim |
| `code` | REPLAY / RESTORE | rarely — code doesn't change silently |
| `summary` | REPLAY / RESTORE | consolidated view of several entries |
| `preference` | REPLAY | high-priority recall |

`fact`, `workflow`, and `tool_output` are collectively called *verify types* — they trigger VERIFY when their score falls below `verify_threshold` or when they have aged past `recency_half_life_days`.

```python
memory.remember(
    "API rate limit",
    "1000 req/min per key",
    type="fact",
    requires_verification=True,   # always returns VERIFY, even at 100% confidence
)
```

---

## Memory scopes

Scope is a **tier**, not a tenant identifier. It says how widely a memory
applies — this session, this user, this whole workspace — and it can be used to
narrow retrieval and listing. It does **not** keep two users apart: two users
both writing at `scope="user"` are in the same tier. For per-user and
per-session separation, see [Tenants and sessions](#tenants-and-sessions) below.

Multiple scopes can be queried at once.

| Scope | Typical use |
|-------|-------------|
| `session` | Current conversation only; should be cleaned up at session end |
| `user` | Per-user persistent memory |
| `project` | Shared across a project team |
| `workspace` | Shared across an entire product workspace |
| `team` | Team-level conventions and policies |
| `global` | Application-wide facts visible to every user |

```python
# Store for a specific scope
memory.remember(query, response, scope="project")

# Retrieve only user + global memories
decision = memory.resolve(query, scope=["user", "global"])
entries  = memory.list(scope=["project", "team"])
```

---

## Tenants and sessions

One store can hold many users, each with many sessions, plus a shared tier every
user can read. `Memory.scoped()` returns a view that stamps identity on writes
and applies the matching filter to reads.

```python
memory = Memory(persist_dir=".agent_memory")

alice    = memory.scoped(user_id="alice")                     # all of alice's sessions
alice_s3 = memory.scoped(user_id="alice", session_id="s3")    # one session
shared   = memory.scoped(shared=True)                         # the common tier

alice_s3.remember("Which seat do I prefer?", "Window seat")
shared.remember("What is the refund window?", "30 days")

alice_s3.resolve("seat preference?")   # session -> user -> shared
alice.resolve("seat preference?")      # sees the s3 write
memory.scoped(user_id="bob").resolve("seat preference?")   # NONE
```

Visibility, narrow to wide:

| View | Sees |
|------|------|
| `scoped(user_id=u, session_id=s)` | that session, plus `u`'s session-less memories, plus shared |
| `scoped(user_id=u)` | every session of `u`, plus shared |
| `scoped(shared=True)` | shared only |

Writes never widen: a session view writes into that session, and only a
`shared=True` view writes shared memories. Memories written directly through
`Memory` carry no `user_id` and are invisible to a user view unless it is built
with `include_unscoped=True` — otherwise one process-wide store would leak into
every tenant's view.

The filter is applied **during retrieval, before scoring**, so a decision is
never computed over another tenant's memories. It is a metadata filter rather
than an indexed column, so a narrow view over a large shared store escalates to
a deeper candidate pool when the first pass is crowded out by memories it cannot
see. For agent-to-agent rather than user-to-user separation, see
[`MultiAgentMemory`](usage.md).

### Deleting a tenant or a session

```python
alice.forget_all()                    # every session of one user
alice_s3.forget_all()                 # one session, rest of the user intact
shared.forget_all()                   # only the shared tier

memory.forget_where(metadata={"user_id": "alice"})   # same thing, explicit
memory.forget_where(scope=["session"])               # all session-tier memories
memory.forget_where(tags=["scratch"])
memory.forget_where(all=True)                        # wipe the store (explicit)
```

`forget_where()` includes archived and expired memories: "forget this" must not
leave a copy behind in another state. It requires at least one filter, or
`all=True`, so an empty call cannot wipe a store by accident.

---

## Memory states

State is derived from the entry's fields — it is not set directly.

| State | Derivation | What happens |
|-------|------------|--------------|
| `ACTIVE` | Not archived, not expired | Returned by search and listing |
| `ARCHIVED` | `archived=True` | Hidden from search; retrievable with `include_archived=True` |
| `EXPIRED` | `expires_at` in the past | Hidden by default; removed by `cleanup(delete=True)` |
| `DELETED` | Hard-deleted from store | No longer exists |

```python
# Expire after 24 hours
memory.remember(query, response, ttl="24h")

# Archive (soft-delete, recoverable)
memory.archive(entry_id)

# Hard-delete all expired entries
memory.cleanup(delete=True)
```

---

## Confidence and decay

Confidence starts at `1.0` and is updated by feedback events:

```python
from agent_memory import ConfidenceLearner, ConfidenceEvent

learner = ConfidenceLearner()

# After a user rejects the response
learner.record_event(entry, ConfidenceEvent.USER_REJECTED)   # -0.25
memory.store.update(entry)

# Nightly decay (half-life 90 days)
for e in memory.list():
    learner.decay(e)
    memory.store.update(e)
```

A confidence below the `restore_threshold` (default 0.70) causes the entry to be down-ranked and may push the action from REPLAY to RESTORE or NONE.

---

## TTL format

| Format | Example | Meaning |
|--------|---------|---------|
| String duration | `"30d"` | 30 days |
| String duration | `"2h"` | 2 hours |
| String duration | `"60m"` | 60 minutes |
| Integer seconds | `3600` | 1 hour |
| `None` | `None` | Never expires |

Expiry is enforced on the read path: once `expires_at` has passed, the memory is
not retrieved, resolved, or listed, whether or not `cleanup()` has run.
`cleanup()` reconciles the stored `state` and can delete expired rows
(`cleanup(delete=True)`); it is housekeeping, not the thing that makes expiry
take effect.

---

## How `updated_at` and `access_count` interact

`updated_at` is **not** updated on REPLAY — only on content edits. This is intentional: frequently-accessed stale facts must not appear fresh to the recency scorer. A memory that hasn't been content-edited in 90 days will decay towards VERIFY even if it was replayed 1,000 times. `access_count` is updated on every REPLAY and contributes the `usage_weight` (default 10%) to the policy score.
