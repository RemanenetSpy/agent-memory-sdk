# Getting Started

## Installation

```bash
# Core (SQLite backend, MCP server, CLI)
pip install agent-memory-sdk

# Common extras
pip install "agent-memory-sdk[semantic]"    # vector search (sqlite-vec + fastembed)
pip install "agent-memory-sdk[dashboard]"   # Streamlit dashboard
pip install "agent-memory-sdk[api]"         # FastAPI REST server
pip install "agent-memory-sdk[langchain]"   # LangChain BaseMemory adapter
pip install "agent-memory-sdk[llamaindex]"  # LlamaIndex BaseMemory adapter
```

---

## 5-minute quickstart

```python
from agent_memory import Memory, MemoryAction

memory = Memory(persist_dir=".agent_memory")

# 1. Store an answer you want to remember
memory.remember(
    query    = "How do I reset my password?",
    response = "Go to Settings → Security → Reset Password.",
    type     = "conversation",
    tags     = ["auth", "faq"],
)

# 2. Before every LLM call, check memory first
decision = memory.resolve("I forgot my password")

if decision.action == MemoryAction.REPLAY:
    return decision.response                    # no LLM call needed

if decision.action == MemoryAction.RESTORE:
    context = memory.format_restore_context(decision)
    return call_llm(query, system_extra=context)

# NONE — answer from scratch
return call_llm(query)
```

---

## What each action means

| Action | When | What to do |
|--------|------|-----------|
| **REPLAY** | High-confidence exact match | Return `decision.response` — no LLM call needed |
| **RESTORE** | Moderate match | Pass `format_restore_context(decision)` as system context to the LLM |
| **VERIFY** | Stale or flagged fact | Validate with tools/DB before reusing; the memory may be outdated |
| **NONE** | No relevant match | Answer from scratch and optionally store the result |

---

## Inspect why a decision was made

```python
decision = memory.resolve("What is the API rate limit?")

print(decision.action.value)     # "verify"
print(decision.confidence)       # 0.88
print(decision.reasons)          # ["high semantic match", "requires verification", ...]
print(decision.explain())        # full per-component score breakdown
```

---

## Store facts, workflows, and preferences

```python
# A fact that should be re-verified before replay (prices change)
memory.remember("subscription cost", "$49/month", type="fact", requires_verification=True)

# A user preference (high-priority replay)
memory.remember("UI theme", "dark mode preferred", type="preference", scope="user")

# A workflow that expires in 30 days
memory.remember("deploy process", "Push to main → CI runs → ECS updates",
                type="workflow", ttl="30d")

# A tool output that expires in 1 hour
memory.remember("weather NYC", "72°F sunny", type="tool_output", ttl="1h")
```

---

## CLI

```bash
# Store a memory
agent-memory remember "How do I reset my password?" \
    "Settings → Security → Reset Password" \
    --type conversation --tags auth,faq

# Resolve a query
agent-memory resolve "I forgot my password" --explain

# See what's stored
agent-memory stats

# Run the eval suite
agent-memory eval

# Run the benchmark
agent-memory benchmark --seed --repeat 3

# Clean up expired entries
agent-memory cleanup --delete
```

---

## Dashboard

```bash
pip install "agent-memory-sdk[dashboard]"

# Optional: seed demo data
python scripts/seed_demo.py --data-dir .agent_memory

AGENT_MEMORY_DIR=.agent_memory agent-memory-dashboard
# → http://localhost:8501
```

---

## Backends

```python
# Default: SQLite (zero setup)
mem = Memory(persist_dir=".agent_memory")

# Redis
mem = Memory(backend="redis", url="redis://localhost:6379/0")

# Postgres
mem = Memory(backend="postgres", dsn="postgresql://user:pw@localhost/mydb")

# Qdrant — for collections past what SQLite or Postgres serves
mem = Memory(backend="qdrant", url="http://localhost:6333")
```

Start Redis, Postgres, or Qdrant with one command:
```bash
docker compose -f docker-compose.dev.yml up -d
```

---

## Debug logging

```python
from agent_memory import configure_debug_logging
configure_debug_logging()  # prints DEBUG logs from every agent_memory.* module

# Or configure manually
import logging
logging.getLogger("agent_memory").setLevel(logging.DEBUG)
```

---

## Next steps

- [Usage Guide](usage.md) — integration patterns, async API, consolidation
- [Features](features.md) — all capabilities with code snippets
- [Examples](../examples/README.md) — runnable samples for every backend/adapter
- [Comparison](comparison.md) — how it differs from mem0, Zep, LangMem
