# How agent-memory-sdk compares

## The landscape (September 2026)

Agent memory is a validated, crowded category. Star counts as of 2026-09-23:

| Project | Stars | Category | Read-time decision layer? |
|---|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | ~65.9k | Memory layer (cloud-first) | ❌ — LLM decides ADD/UPDATE/DELETE at *write* time; retrieval is search-and-inject |
| [Graphiti / Zep](https://github.com/getzep/graphiti) | ~31.1k | Temporal knowledge graph | ❌ — temporal fact invalidation, but no replay/inject/skip decision |
| [Cognee](https://github.com/topoteretes/cognee) | ~30.9k | Knowledge-graph memory | ❌ |
| [Supermemory](https://github.com/supermemoryai/supermemory) | ~30.8k | Memory API | ❌ |
| [Letta (MemGPT)](https://github.com/letta-ai/letta) | ~24.9k | Stateful agent platform | ⚠️ — agent self-manages tiers via LLM tool calls |
| [GPTCache](https://github.com/zilliztech/GPTCache) | ~8.2k | Semantic cache | ⚠️ — replay only; no inject/verify/skip graduation |
| [LangMem](https://github.com/langchain-ai/langmem) | ~1.7k | LangGraph memory utils | ❌ |

Two observations from this table:

1. **The problem is validated** — five projects above 24k stars all exist to stop agents
   from losing or mismanaging context.
2. **The read-side decision layer is an open gap.** Semantic caches do REPLAY only.
   Memory layers do RESTORE only (unconditionally). Nobody graduates between them, and
   nobody has VERIFY. Getting all four actions today means wiring GPTCache + Mem0 +
   custom staleness logic yourself.

## Feature matrix

**How to read this table.** Cells about other projects describe *capabilities*
checked against that project's own source or docs, dated below — not measured
behaviour. Where we have not run the system ourselves, the cell says
`❔ not measured` rather than `❌`; an absent feature and an untested one are
different claims. Capability checks for this revision were made on 2026-09-27
(mem0 against `mem0/memory/main.py` on `main`). Corrections from maintainers are
welcome and will be applied — see the
[benchmark RFC](competitive-benchmark-rfc.md) for the review process.

| Dimension | Redis (raw) | mem0 | Zep | LangMem | LlamaIndex memory | ChromaDB / Pinecone | MemGPT / Letta | **agent-memory-sdk** |
|-----------|-------------|------|-----|---------|-------------------|---------------------|----------------|----------------------|
| **Core model** | Key-value; no memory schema | Entity extraction + vector store; user/session hierarchy | Conversation turns + entity graph + vector search | Message history + LLM-driven summary/extraction | Chat buffer or LLM-summarised window | Embedding vectors; chunk-level similarity | Paged context: main + archival + recall tiers | query→response experience pairs; typed + scoped |
| **Decision intelligence** | ❌ None — caller decides everything | ❌ None — always retrieves; caller decides | ❌ None — inject is caller's job | ⚠️ Partial — LLM decides what to compress | ❌ None — returns window contents | ❌ None — nearest neighbours regardless of relevance | ⚠️ Partial — LLM function calls move data between tiers | ✅ Explicit: REPLAY / RESTORE / VERIFY / NONE with scored rationale |
| **Explainability** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ `decision.explain()` → per-component scores + reason tags |
| **Trap-query protection** | ❔ not measured | ❔ not measured | ❔ not measured | ❔ not measured | ❔ not measured | ❔ not measured | ❔ not measured | ✅ 34/36 adversarial trap cases; misses fail safe to VERIFY, never wrong REPLAY |
| **Local / offline** | ✅ self-hosted | ⚠️ self-hostable (local vector store + local LLM); cloud is the documented default | ⚠️ open-source but needs server + OpenAI | ⚠️ needs LLM provider | ⚠️ needs LLM provider | ✅ self-hostable | ⚠️ heavy; LLM call per op | ✅ SQLite + ONNX MiniLM, zero API keys; workload-specific latency |
| **Confidence / trust model** | ❌ | ❌ | ❌ | ❌ | ❌ | Cosine only | ❌ | ✅ per-entry confidence; event-driven updates; half-life decay |
| **Verification semantics** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ `requires_verification=True` → VERIFY for relevant matches; low-scoring matches return NONE |
| **TTL / expiry** | ✅ native | ✅ `expiration_date` (day granularity; expired memories hidden) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ any entry, second granularity; `ttl="30d"` / `ttl=3600` |
| **Multi-agent isolation** | Manual prefix | User/agent hierarchy | Session-scoped | ❌ | ❌ | Manual collection | ❌ | ✅ NAMESPACED / ISOLATED / SHARED; broadcast; transfer |
| **Memory graph** | ❌ | Internal entity graph | ✅ entity + knowledge graph | ❌ | ❌ | ❌ | ❌ | ✅ similarity + tag edges; BFS; clusters; PageRank |
| **Entity extraction** | ❌ | ✅ auto from conversation | ✅ auto from conversation | ⚠️ LLM-dependent | ❌ | ❌ | ✅ | ✅ `memory.from_conversation(human, assistant)` auto-extracts facts, preferences, entities |
| **LLM calls per memory op** | 0 | 1+ (extraction) | 1+ (embedding + entity) | 1+ (summary) | 1+ (summary) | 0 (embedding only) | 1+ (always) | 0 (optional: 0 with SQLite-only mode) |
| **Framework adapters** | Manual | LangChain, LlamaIndex | LangChain | Native LangChain | Native LlamaIndex | Community | Standalone | LangChain + LlamaIndex `BaseMemory`; MCP for Cursor/Claude Code |

---

## Where agent-memory-sdk is genuinely better

### 1. Explicit, scored decision layer
Every competitor either returns results and leaves the caller to decide (Redis, ChromaDB, LlamaIndex, LangMem) or silently injects past context (mem0, Zep). agent-memory-sdk is the only tool where the memory system itself returns a **typed action** (REPLAY / RESTORE / VERIFY / NONE) with a per-component score breakdown.

This directly prevents the most common memory failure: replaying an answer to a superficially similar but semantically different question.

```
Stored memory: "What payment methods do you support?"
Query:         "Does the platform support two-factor authentication?"

mem0 / Zep / ChromaDB → return the payment answer (shared word: "support")
agent-memory-sdk       → NONE  confidence=0.61  reason: "below restore threshold"
```

### 2. Full explainability
`decision.explain()` returns semantic score, recency score, confidence score, usage score, and the final policy-weighted total — for every query, every time. No other tool in this list exposes this.

### 3. Offline-first with locally measured latency
Zero API keys. SQLite + FTS5 runs in-process; the optional ONNX embedding model loads locally. MemGPT/Letta makes an LLM API call per memory operation. mem0 and Zep default to cloud. Benchmark latency against your corpus using the documented harness.

### 4. Verification semantics
Facts, workflows, and tool outputs use verification rules when they are stale or fall below the verification threshold. An explicit `requires_verification=True` routes a sufficiently relevant match to VERIFY rather than replay; a candidate below the restore threshold returns NONE. No other tool has this concept.

### 5. Automatic entity extraction from conversation
`memory.from_conversation(human, assistant)` runs regex + optional spaCy NER over a raw conversation turn and auto-stores facts, preferences, and named entities — no manual `remember()` needed.

```python
memory.from_conversation(
    human="My name is Karan, I prefer Python, and I work at Acme.",
    assistant="Got it!",
)
# Automatically stores: name fact, Python preference, Acme work-at fact
```

### 6. Hierarchical paged context management (in-context + recall + archival)
`memory.paged()` wraps the store with three tiers matching MemGPT's model:
- **In-context buffer** — last N turns always returned without search
- **Recall** — standard semantic search over older entries
- **Archival** — explicit search over archived long-term entries

```python
paged = memory.paged(context_size=20)
paged.add_turn("What is Python?", "A language.")  # → in-context buffer
ctx = paged.get_context("Python")  # → in_context + recalled
print(ctx.format_for_llm())        # Markdown context block for the LLM
```

Old entries are automatically paged out to recall when the buffer fills.

### 7. Semantic knowledge graph with typed entities + relations
`memory.knowledge_graph()` builds a `KnowledgeGraph` with typed entities (PERSON, ORG, LOCATION, CONCEPT) and typed relations (works_at, located_in, uses, created_by, …) extracted from memory content — not just similarity edges.

```python
kg = memory.knowledge_graph()
entity = kg.find_entity("Karan")          # case-insensitive
rels   = kg.relations_for(entity.id)      # typed relations
path   = kg.path("Karan", "Acme Corp")   # BFS path through graph
kg.merge_entities(keep_id, dupe_id)       # deduplicate aliases
```

### 8. Per-entry confidence with adaptive learning
Confidence is a first-class field that is updated via feedback events (`VERIFIED_CORRECT`, `USER_REJECTED`, etc.) and decays with time (half-life model). Other tools use retrieval similarity as a proxy for trust; agent-memory-sdk separates the two.

---

## Where agent-memory-sdk is worse

### No hosted multi-tenant API
mem0 and Zep offer managed cloud APIs with auth, multi-tenancy, and dashboards. agent-memory-sdk has a FastAPI server and Streamlit dashboard, but no built-in auth layer for a public-facing deployment.

### No published head-to-head accuracy benchmark yet
mem0 and Zep publish LOCOMO / LongMemEval *end-to-end accuracy* (LLM answering + LLM judge). We publish a LongMemEval **retrieval proxy** on independent cleaned-release haystacks ([report](../benchmarks/longmemeval/REPORT.md)); it is neither an end-to-end result nor a direct comparison with the paper's original-release session-index baselines.

### Smaller community
The projects above have 25–65k stars, funded teams, and large contributor bases. This is a focused single-purpose SDK, not a platform.

---

## When to choose which

| Situation | Best fit |
|-----------|---------|
| You want zero-setup local memory with explicit decisions | **agent-memory-sdk** |
| You want memory auto-extracted from raw conversation text | mem0 or Zep |
| You need a managed cloud memory API with multi-tenancy | mem0 Cloud or Zep Cloud |
| You're building a very-long-context agent and need paged memory tiers | MemGPT / Letta |
| You need a knowledge graph of entities from conversation | Zep |
| You need a vector database for document RAG | ChromaDB / Pinecone / Qdrant |
| You want memory + explicit decision + explainability + offline | **agent-memory-sdk** |

---

## One-sentence differentiator

> agent-memory-sdk is designed to treat "should I use this memory, and how much should I trust it" as a first-class, scored, explainable decision — not as something the caller figures out after retrieval — while remaining fully local in SQLite-only mode. Measure latency for your workload.
