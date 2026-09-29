# Examples

Runnable code samples for every integration.
`agent-memory-sdk` is the central package in all of them — the adapters, servers, and framework connectors all delegate storage, retrieval, and decision logic to `Memory`.

```mermaid
flowchart TD
    M["🧠 agent-memory-sdk\nMemory(persist_dir, backend)\n.remember()  .resolve()"]

    M --> LC["LangChain\nAgentMemoryLangChain\n[langchain]"]
    M --> LI["LlamaIndex\nAgentMemoryLlamaIndex\n[llamaindex]"]
    M --> API["FastAPI REST server\nagent-memory-api\n[api]"]
    M --> DASH["Streamlit dashboard\nagent-memory-dashboard\n[dashboard]"]
    M --> MA["Multi-agent\nMultiAgentMemory\n(NAMESPACED · SHARED · ISOLATED)"]
    M --> BE["Backends\nSQLite · ChromaDB\nRedis · Postgres · Qdrant"]

    style M fill:#4f46e5,color:#fff
    style LC fill:#1e3a5f,color:#fff
    style LI fill:#1e3a5f,color:#fff
    style API fill:#065f46,color:#fff
    style DASH fill:#065f46,color:#fff
    style MA fill:#7c3aed,color:#fff
    style BE fill:#1e293b,color:#fff
```

---

## Index

| Example | Install extra | Run |
|---------|--------------|-----|
| [basic_usage.py](basic_usage.py) | *(none)* | `python examples/basic_usage.py` |
| [langchain_integration.py](langchain_integration.py) | `[langchain]` | `python examples/langchain_integration.py` |
| [llamaindex_integration.py](llamaindex_integration.py) | `[llamaindex]` | `python examples/llamaindex_integration.py` |
| [redis_backend.py](redis_backend.py) | `[redis]` | Needs Redis — see below |
| [postgres_backend.py](postgres_backend.py) | `[postgres]` | Needs Postgres — see below |
| [qdrant_backend.py](qdrant_backend.py) | `[qdrant]` | Needs Qdrant — see below |
| [multi_agent.py](multi_agent.py) | *(none)* | `python examples/multi_agent.py` |
| [rest_api.py](rest_api.py) | `[api]` | Needs server — see below |
| [confidence_and_graph.py](confidence_and_graph.py) | *(none)* | `python examples/confidence_and_graph.py` |
| [benchmark_harness.py](benchmark_harness.py) | *(none)* | `python examples/benchmark_harness.py` |

---

## Setup

### Core (SQLite, no extras needed)

```bash
pip install agent-memory-sdk
python examples/basic_usage.py
python examples/multi_agent.py
python examples/confidence_and_graph.py
python examples/benchmark_harness.py
```

### LangChain adapter

```bash
pip install "agent-memory-sdk[langchain]" langchain-openai
python examples/langchain_integration.py
```

### LlamaIndex adapter

```bash
pip install "agent-memory-sdk[llamaindex]" llama-index-llms-openai
python examples/llamaindex_integration.py
```

### Redis backend

```bash
# Start Redis (or use docker-compose.dev.yml)
docker compose -f docker-compose.dev.yml up -d redis

pip install "agent-memory-sdk[redis]"
python examples/redis_backend.py
```

### PostgreSQL backend

```bash
# Start Postgres
docker compose -f docker-compose.dev.yml up -d postgres

pip install "agent-memory-sdk[postgres]"
python examples/postgres_backend.py
```

### Qdrant backend

```bash
# Start Qdrant
docker compose -f docker-compose.dev.yml up -d qdrant

pip install "agent-memory-sdk[qdrant,semantic]"
python examples/qdrant_backend.py
```

### REST API server

```bash
pip install "agent-memory-sdk[api]"

# Terminal 1 — start the server
export AGENT_MEMORY_API_KEY="replace-with-a-long-random-secret"
AGENT_MEMORY_DIR=.agent_memory agent-memory-api

# Terminal 2 — run the client example
python examples/rest_api.py
```

---

## Example descriptions

### [basic_usage.py](basic_usage.py)
The fundamentals: `remember()`, `resolve()`, all four actions (REPLAY / RESTORE / VERIFY / NONE), and `decision.explain()`.
No extras needed.

---

### [langchain_integration.py](langchain_integration.py)
Drops `AgentMemoryLangChain` into a LangChain `ConversationChain` as a `BaseMemory` replacement.

- `save_context()` → `memory.remember()`
- `load_memory_variables()` → `memory.resolve()` — returns plain text or `list[BaseMessage]`
- Plugs into any chain with `memory=lc_memory`

```python
from agent_memory import Memory
from agent_memory.adapters.langchain_adapter import AgentMemoryLangChain

memory = Memory(persist_dir=".agent_memory")
lc_memory = AgentMemoryLangChain(memory=memory, return_messages=False)

lc_memory.save_context({"input": "What is the rate limit?"}, {"output": "100 req/min."})
ctx = lc_memory.load_memory_variables({"input": "API limits?"})
print(ctx["history"])
```

---

### [llamaindex_integration.py](llamaindex_integration.py)
Wraps `Memory` as a LlamaIndex `BaseMemory` for chat engines and OpenAI agents.

- `put()` buffers messages; flushes USER+ASSISTANT pairs to persistent memory automatically
- `get(input=)` retrieves relevant context via `memory.resolve()`
- `token_limit` trims context to fit your LLM's window
- Plugs into `OpenAIAgent.from_tools(..., memory=li_memory)`

```python
from agent_memory import Memory
from agent_memory.adapters.llamaindex_adapter import AgentMemoryLlamaIndex

memory = Memory(persist_dir=".agent_memory")
li_memory = AgentMemoryLlamaIndex(memory=memory, top_k=5)

msgs = li_memory.get(input="API rate limits?")
```

---

### [redis_backend.py](redis_backend.py)
Uses Redis as the storage engine instead of SQLite. Identical API — only the constructor changes.

- The lowest-latency server backend (see [benchmarks](../docs/benchmarks.md))
- Keys namespaced under a configurable prefix
- RediSearch HNSW vector KNN when the server has the module (Redis 8+ / Redis
  Stack, db 0) and `[semantic]` is installed — and the same index serves the
  keyword half, so neither pass reads the corpus into Python
- Python BM25 as the fallback — plain Redis needs no module at all

```python
from agent_memory import Memory

memory = Memory(backend="redis", url="redis://localhost:6379/0")
memory.remember("password reset", "Settings → Security → Reset Password.")
d = memory.resolve("I forgot my password")
```

---

### [postgres_backend.py](postgres_backend.py)
Uses PostgreSQL with `tsvector` full-text search. Scope-aware queries map cleanly to SQL WHERE clauses.

- `tsvector` GIN index for fast keyword search
- pgvector HNSW KNN for semantic search (install `[pgvector]`), auto-selected
  over IVFFlat on pgvector ≥ 0.7.0
- Scope filtering pushed to SQL — no Python-side filtering

```python
from agent_memory import Memory, MemoryScope

memory = Memory(backend="postgres", dsn="postgresql://user:pw@localhost/mydb")
memory.remember("deadline", "Ships Friday", scope=MemoryScope.PROJECT)
d = memory.resolve("When is the deadline?", scope=[MemoryScope.PROJECT])
```

---

### [qdrant_backend.py](qdrant_backend.py)
Uses Qdrant, a purpose-built vector database, for collections past the point where
SQLite or Postgres keeps up. Qdrant publishes limits far beyond anything this
repo has measured — see [benchmarks](../docs/benchmarks.md) for what we actually
ran (2,000 entries across all backends, 20,000 for the pgvector index).

- Qdrant HNSW graph over the whole corpus; `ef_search` trades latency for recall
- Hybrid retrieval: dense KNN plus BM25 over candidates narrowed by Qdrant's
  full-text payload index, fused with RRF
- Scope, type, archived and TTL filters run server-side on payload indexes
- The entry lives in the point payload, so there is no second store to sync
- `path="./qdrant_data"` runs embedded, with no server at all

```python
from agent_memory import Memory

memory = Memory(backend="qdrant", url="http://localhost:6333", restore_threshold=0.55)
memory.remember("password reset", "Settings → Security → Reset Password.")
d = memory.resolve("I can't get into my account anymore")
```

---

### [multi_agent.py](multi_agent.py)
Multiple agents share one `Memory` store with configurable isolation.

| Mode | `list()` returns | `resolve()` searches |
|------|-----------------|---------------------|
| `NAMESPACED` (default) | own + global | full store |
| `ISOLATED` | own only | full store |
| `SHARED` | everything | full store |

> **Note:** `resolve()` always searches the full store. Use `scope=` to narrow it,
> or use `list()` + `store.keyword_search()` for fully isolated retrieval.

```python
from agent_memory import Memory, MultiAgentMemory

shared = Memory(persist_dir=".agent_memory")
agent_a = MultiAgentMemory(shared, agent_id="agent-a")
agent_b = MultiAgentMemory(shared, agent_id="agent-b")

agent_a.remember("secret", "only a's data")
agent_b.broadcast("company name", "Acme Corp")   # visible to all

print(agent_a.list())   # sees: agent-a's memories + broadcasts
print(agent_b.list())   # sees: agent-b's memories + broadcasts
```

---

### [rest_api.py](rest_api.py)
Two patterns: calling the REST server with `httpx`, and embedding it inside your own FastAPI app.

```python
# Call the running server
import httpx
client = httpx.Client(base_url="http://localhost:8000")
client.post("/memories", json={"query": "...", "response": "..."})
data = client.post("/resolve", json={"query": "..."}).json()

# Or embed in your own app
from fastapi import FastAPI
from agent_memory.api.server import create_app

app = FastAPI()
app.mount("/memory", create_app(persist_dir=".agent_memory"))
```

Full endpoint list: `POST /memories`, `GET /memories`, `GET /memories/{id}`,
`DELETE /memories/{id}`, `POST /memories/{id}/archive`, `POST /resolve`,
`GET /stats`, `POST /cleanup`, `POST /consolidate`.

---

### [confidence_and_graph.py](confidence_and_graph.py)
Two advanced features that build on the core `Memory` store.

**Confidence learning** — update confidence based on feedback events:

```python
from agent_memory import ConfidenceLearner, ConfidenceEvent

learner = ConfidenceLearner()
update = learner.record_event(entry, ConfidenceEvent.USER_CONFIRMED)
# entry.confidence is updated in-place; persist with memory.store.update(entry)

# Nightly decay: half-life 90 days
for e in memory.list():
    learner.decay(e)
    memory.store.update(e)
```

**Memory graph** — discover relationships between memories:

```python
from agent_memory import MemoryGraph

graph = MemoryGraph.build(memory.store, similarity_threshold=0.4)
neighbours = graph.neighbors(entry_id)          # related by similarity or tags
path = graph.path(source_id, target_id)         # BFS shortest path
clusters = graph.clusters()                     # connected components
scores = graph.importance_scores()              # PageRank
```

---

### [benchmark_harness.py](benchmark_harness.py)
Measures retrieval quality using a LongMemEval- or LoCoMo-style dataset.

Metrics: **Recall@k**, **MRR**, **content recall**, **action accuracy**, **P95 latency**.

```python
from agent_memory import Memory, BenchmarkHarness, BenchmarkDataset

memory  = Memory(persist_dir=".agent_memory")
harness = BenchmarkHarness(memory)
result  = harness.run_from_file("my_dataset.json")
print(result.format())
```

Dataset format — `sessions` (events to seed) + `questions` (queries to evaluate):

```json
{
  "name": "my_bench",
  "sessions": [
    { "session_id": "s1", "events": [
      { "query": "Q", "response": "A", "type": "fact", "tags": ["topic"] }
    ]}
  ],
  "questions": [
    { "query": "related?", "expected_content": "A", "difficulty": "easy" }
  ]
}
```
