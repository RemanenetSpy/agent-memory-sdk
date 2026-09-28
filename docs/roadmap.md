# Roadmap & Current Status

Track live progress on the **[GitHub Project board →](https://github.com/users/theprodsde/projects/2)**

## Current version: v0.3.0-dev

## ✅ Shipped

### Core
- Decision engine (REPLAY / RESTORE / VERIFY / NONE) with adversarial eval suite — 34/36 (94%); both misses fail safe to VERIFY
- `decision.explain()` — full per-component score breakdown
- Hybrid retrieval: BM25 FTS5 + coverage scaling + RRF fusion; latency is measured by the reproducible stress harness
- Optional vector search via sqlite-vec + fastembed ONNX (`[semantic]` extra)
- TTL, memory states (active / archived / expired), near-duplicate consolidation
- SQL-aggregate `stats()` / `cleanup()` — no Python-side row loading
- WAL mode + busy timeout for concurrent MCP / CLI / app access
- Full async API (`aremember`, `aresolve`, `alist`, `astats`, …)

### Backends

| Backend | Extra | Retrieval |
|---------|-------|-----------|
| SQLite (default) | *(none)* | FTS5 BM25 + optional sqlite-vec KNN |
| ChromaDB | *(bundled)* | Vector embeddings + Python BM25 |
| Redis | `[redis]` | Python BM25 (sorted-set index) |
| PostgreSQL | `[postgres]` | tsvector FTS + optional pgvector KNN |

Dev infrastructure: `docker-compose.dev.yml` spins up Redis + Postgres with one command.

### Interfaces

| Interface | Entry point | Extra |
|-----------|------------|-------|
| CLI | `agent-memory` | *(none)* |
| MCP server | `agent-memory-mcp` / `uvx agent-memory-sdk` | *(bundled)* |
| FastAPI REST server | `agent-memory-api` | `[api]` |
| Streamlit dashboard | `agent-memory-dashboard` | `[dashboard]` |

### Framework adapters

| Adapter | Class | Extra |
|---------|-------|-------|
| LangChain | `AgentMemoryLangChain(BaseMemory)` | `[langchain]` |
| LlamaIndex | `AgentMemoryLlamaIndex(BaseMemory)` | `[llamaindex]` |

### Advanced features
- **Memory graph** — similarity + tag-overlap edges, BFS path-finding, connected-component clustering, PageRank importance scores
- **Confidence learning** — event-driven deltas + half-life temporal decay, batch updates
- **Multi-agent support** — SHARED / NAMESPACED / ISOLATED isolation, broadcast, transfer ownership
- **Benchmark harness** — Recall@k, MRR, content-recall, action-accuracy, P95 latency; LongMemEval + LoCoMo dataset format

### Quality
- 196 tests across Python 3.10–3.13 (ruff · enforced mypy · pytest)
- CI matrix: core + semantic + Redis + API + Postgres + LangChain + feature jobs
- Release pipeline gates on full test matrix before PyPI publish
- Branch protection: all CI status checks required, 1 PR review, conversation resolution
- Auto-reviewer: `@theprodsde` requested on every PR via `.github/CODEOWNERS`

---

## 🔜 Up next

| Feature | Why |
|---------|-----|
| pgvector KNN on Postgres | Semantic search on the Postgres backend |
| Redis VSS (vector search) | KNN on Redis without a Python-side BM25 fallback |
| Dashboard graph explorer tab | Graph visualisation inside the Streamlit dashboard |
| Streaming resolve | `aresolve()` yields partial results as memories are scored |
| Memory compression | Auto-summarise old sessions to reduce store size |

---

## 💬 Have input on priorities?

Open a [Discussion](https://github.com/TheProdSDE/agent-memory-sdk/discussions) or add a comment on the relevant [GitHub Project card](https://github.com/users/theprodsde/projects/2).
