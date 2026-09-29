# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (vector index batch — issues #18, #19, #20)
- **Redis vector search (RedisVSS)** — the Redis backend now builds an HNSW index
  over a `FT.CREATE` VECTOR field and `search()` is real cosine KNN, pre-filtered
  server-side on scope and archived TAGs. The index scan itself is
  sub-millisecond; end to end `search()` is ~6 ms at 2,000 memories, of which
  ~2.8 ms is embedding the query — see [docs/benchmarks.md](docs/benchmarks.md). Needs RediSearch
  (Redis 8+ or Redis Stack) plus `[semantic]`; without either it falls back to
  Python BM25 exactly as before. Entries stored before the upgrade are embedded
  automatically at init, and the index is rebuilt if the embedding model changes.
  RediSearch only indexes db 0 — on any other database the store logs a warning
  and stays lexical.
- **PostgreSQL HNSW index** — the Postgres backend now picks `hnsw` over
  `ivfflat` when the server's pgvector is ≥ 0.7.0. HNSW accepts online inserts
  and its scan cost stays roughly flat as the table grows, where IVFFlat wants a
  populated table at build time and slows linearly. Measured: at 20,000 rows HNSW
  scanned in 0.58ms against IVFFlat's 4.76ms — though IVFFlat was the faster of
  the two at 2,000 rows, which is why "auto" exists. Built `CONCURRENTLY` so it does not lock writes; an
  existing IVFFlat index is dropped once HNSW replaces it. Override with
  `PostgresMemoryStore(vector_index="hnsw" | "ivfflat")`, and tune recall with
  `ef_search`. New `vector_index_type` and `pgvector_version` properties.
- **Qdrant backend** (`backend="qdrant"`, `[qdrant]` extra) for collections past
  what SQLite or Postgres serves. Verified at 2,000 entries; Qdrant's own
  published limits go much further and are not claims this repo has tested.
  Hybrid retrieval:
  dense KNN in Qdrant plus BM25 over candidates narrowed by Qdrant's full-text
  payload index, fused with the existing RRF retriever. Scope, type, archived and
  TTL filters all run server-side on payload indexes; the whole entry lives in
  the point payload, so there is no second store to keep in sync. Supports Qdrant
  Cloud and an embedded `path=` mode with no server.
- `touch()` overrides on the Redis and Qdrant stores: REPLAY's hot path patches
  the usage counters in place instead of re-embedding the entry.
- `docker-compose.dev.yml` now ships a vector index in every service — `redis:8`
  (bundled query engine), the `pgvector/pgvector` Postgres image, and Qdrant.

- **Shared, injectable vector-index config** (`VectorIndexConfig`,
  `IvfFlatConfig`): `m`, `ef_construction`, `ef_search` and the KNN over-fetch
  policy are now one validated, frozen dataclass accepted by all three server
  backends (`vector_config=`), instead of module constants no caller could reach.
  Each store still ships its own engine's defaults, so passing nothing changes
  nothing. Exposed as `store.vector_config`.
- **Backend registry** (`agent_memory.backends`): `Memory(backend=...)` resolves
  through a registry instead of an if/elif chain, so adding a backend no longer
  edits `Memory.__init__` or its error message. `register_backend()` lets a
  third-party store plug in — `Memory(backend="mystore")` — and factories import
  lazily, so `import agent_memory` still pulls in no optional driver.
- `MemoryStore.semantic_search_enabled` is part of the abstract interface
  (defaulting to False) rather than a duck-typed attribute the retriever probed
  with `getattr`.
- `scripts/backend_benchmark.py` — compares every backend on one corpus, one
  query set and one embedder, reporting `resolve()` / `search()` /
  `keyword_search()` latency separately, plus **Recall@k** (a rank cutoff over
  what the index found) kept distinct from **answer rate** (what the decision
  layer did at a given `restore_threshold`). Measured as Recall@k, every vector
  backend retrieves equally well (92.2% R@1 fused) and vector search is worth ~8
  points over lexical-only — the spread that shows up in answer rate is a
  threshold effect, not a retrieval difference. Results and the HNSW-vs-IVFFlat
  crossover: [docs/benchmarks.md](docs/benchmarks.md).
- `scripts/stress_test.py --backend / --backend-opt` — the stress harness can now
  measure any registered backend, not just SQLite.

### Performance
- **Redis reads were O(N) round trips.** `list_all()` issued one `GET` per entry,
  so every keyword search cost one round trip per stored memory. Batched into a
  single `MGET`: `resolve()` at 2,000 memories went from ~86 ms to ~8 ms.
- **Redis keyword search now uses RediSearch.** The index carries a `TEXT` field,
  so the BM25 half of every hybrid query is served by the inverted index instead
  of reading the whole corpus back into Python and re-ranking it — O(N) work
  behind an O(log N) vector index. `keyword_search()` p50 dropped from ~44 ms to
  ~1.5 ms. Stores without RediSearch keep the Python BM25 path.
- **Postgres opened a new connection per statement.** Every read paid a TCP
  handshake plus authentication, putting a ~20 ms floor under each query.
  Connections are now cached per thread (with `close()` to release them);
  `resolve()` p50 fell from ~57 ms to ~14 ms and writes went from 29/s to 78/s.
- Postgres reads no longer `SELECT *`, which was streaming the 384-dim embedding
  and the tsvector back on every row the caller never looked at.
- `IvfFlatConfig` now defaults to `probes=3` against `lists=10`. It was
  `probes=10`, which probes *every* list — a brute-force scan dressed as an index.

### Fixed
- **Postgres keyword search could not match a paraphrase.** It used
  `plainto_tsquery()`, which ANDs every term, so "how to change my login
  credentials" only matched documents containing all five words — while the
  SQLite, Redis and Qdrant backends all rank an OR recall set. Now builds an OR
  `to_tsquery()` from tokenised content terms. Labelled-query recall on the
  Postgres backend went from 23.5% to ~47%.
- **Postgres embedded different text than every other backend** (`query` +
  `content`, omitting tags), so the same corpus ranked differently there. All
  backends now index one shared `search_document()`.
- **Postgres scoped search was broken.** `keyword_search()` and the pgvector KNN
  path bound positional parameters in the wrong order whenever `scopes` (or
  `memory_type`) narrowed the query, because the placeholder in the SELECT list
  binds ahead of the WHERE clause's. Scoped keyword search returned results for
  the wrong filter and scoped vector search raised
  `InvalidTextRepresentation: invalid input syntax for type vector`.
- **Postgres vector search failed outright with pgvector-python ≥ 0.4**, which
  removed `register_vector_globally()`. Registration is now per connection, runs
  after `CREATE EXTENSION`, and is best-effort — every query casts explicitly
  with `::vector`, so an unregistered connection still reads and writes correctly.
- `Memory(backend=...)` now forwards `embedder` and `enable_embeddings` to the
  Redis, Postgres and Qdrant stores; previously both were silently dropped and
  those backends could never be given a custom embedder through `Memory`.

### Added (v0.2 performance & semantic search batch)
- **SQLite FTS5 keyword index** with built-in BM25 ranking: keyword search no
  longer loads up to 10k rows and rebuilds a Python BM25 index per query
  (~12ms resolve at 5,000 memories, previously ~1s). Existing databases are
  backfilled automatically on first open.
- **Optional vector search on the default backend** via the `semantic` extra
  (`pip install agent-memory-sdk[semantic]`): sqlite-vec + fastembed (ONNX
  MiniLM, no torch). Auto-detected; `Memory(enable_embeddings=True)` to force.
  Handles paraphrases with zero shared words.
- **SQL-aggregate `stats()` and `cleanup()`** — no more silent 10k row cap;
  cleanup also removes FTS/vector index rows.
- `requires_verification=True` memories now always VERIFY (previously a high
  score replayed them silently, defeating the flag).
- `enable_verify=False` now degrades would-be VERIFY decisions to RESTORE
  instead of NONE.
- `decision_traps.json` eval dataset: 13 adversarial cases including
  shared-word traps; eval accepts VERIFY where RESTORE is expected (both
  surface the memory; verify is more cautious).
- Test suites: MCP reply contract over bulk data (`tests/test_mcp_server.py`),
  scale/latency regression tests (`tests/test_scale.py`), semantic backend
  tests (`tests/test_semantic_sqlite.py`). 119 tests total.
- `mcp_server.reset_memory()` and lazy env reads for testability.
- CONTRIBUTING.md with good-first-issues; docs/why-decision-layer.md.
- README demo GIF (regenerate via `docs/assets/record-demo.sh`).
- MCP replay replies now include `matched_query`, `stored_at`, and
  `times_reused` so clients can see what was remembered, not just the answer.
- CLI `resolve` output is now human-readable: action, confidence, the
  remembered query (with stored date and reuse count on replay), and the
  response/reason — instead of the raw Decision repr.
- README "Wiring It Into Your Agent" section: the resolve/remember loop,
  write rules, multi-process sharing, and when to use (and not use) this.
- MCP Registry manifest (`server.json`, validated against the 2025-09-29
  schema) plus an `agent-memory-sdk` console-script alias so
  `uvx agent-memory-sdk` launches the MCP server; `mcp-name` ownership
  marker in the README. See docs/launch-checklist.md.

### Changed (v0.2 batch)
- `consolidate()` now issues one search per entry instead of one per pair
  (was O(N²) searches).
- Benchmark no longer invents an 800ms "no-memory baseline"; the comparison
  section only appears against a user-supplied `--baseline-ms`, labeled as
  such. docs/benchmarks.md publishes only reproducible numbers.
- SQLite `store()` uses an UPSERT (stable rowids for the FTS/vector indexes).
- CI: `actions/checkout`, mypy enforced (no `|| true`), ruff on all packages,
  a dedicated semantic-extras job, and an eval job. mypy is clean across the
  codebase.

### Added (multi-tenancy & decision-safety benchmark)
- **`Memory.scoped(user_id=..., session_id=..., shared=...)`** — per-user and
  per-session views over one store. `scope` is a tier (`user`/`project`/...), not
  a tenant id, so many users in one store needed this. Reads are hierarchical
  (session → user → shared) and filtered **during retrieval, before scoring**;
  writes stay in the view's own namespace. Memories written directly through
  `Memory` are invisible to a user view unless `include_unscoped=True`.
- **`Memory.forget_where(...)`** — bulk delete by scope, type, tags, metadata, or
  predicate, including archived and expired copies. Requires an explicit filter
  or `all=True`. `MultiAgentMemory.forget_all()` and `MemoryView.forget_all()`
  build on it, giving a per-tenant "delete everything for this user".
- **`Memory.resolve(where=...)`** and `from_conversation(metadata=...)` for
  callers that can only see part of a store.
- **Decision Safety Suite v2** (`benchmarks/decision_safety/`): six op-script
  batteries — deletion durability, TTL expiry, poisoned writes, state
  invalidation, provenance re-assertion, point-in-time queries — with a runner,
  an adapter contract, and paired-metric enforcement in the result validator.
  See `docs/decision-safety-suite.md`.

### Fixed
- **Cross-agent memory leak on the decision path**: `MultiAgentMemory.resolve()`
  ignored the isolation mode, so an `ISOLATED` agent could have another agent's
  memory replayed to it verbatim while `list()` correctly showed nothing.
  Isolation is now applied during retrieval, and a filtered candidate list is
  never cached for another caller.
- **Expired memories served from the retrieval cache**: a memory whose TTL was
  shorter than the retriever's 5-second result cache could be replayed after
  expiry on a repeated query, even though the store's read path filtered it.
- **Critical scoring bug**: the top BM25 hit was always normalized to a perfect 1.0,
  so any query sharing a single word with a stored memory replayed that memory's
  answer verbatim at confidence 1.0. Keyword scores are now scaled by query-term
  coverage, so weak overlaps score low and unrelated queries return `NONE`.
- **Decision score floor removed**: thresholds previously used
  `max(policy_score, raw_semantic)`, so confidence, recency, and usage could never
  lower a decision below the raw retrieval score (a confidence-0.1 memory still
  replayed). Low-confidence memories now RESTORE as context instead of replaying.
- Replaying a memory no longer resets its `updated_at`, so frequently-replayed
  stale facts correctly age into VERIFY instead of looking perpetually fresh.
- ChromaDB backend now stores the query in metadata; multiline queries are no
  longer corrupted on read-back.
- README repo links pointed at the old `agent-memory` repository name.
- **MCP server crashed on fresh installs**: `mcp>=1.0.0` resolves to mcp 2.x,
  which renamed `FastMCP` to `MCPServer`. The server now imports either API.

### Added
- Query-term coverage scoring with a shared stop-word list and plural folding
- SQLite WAL mode + 30s busy timeout for concurrent access (MCP server + CLI + app
  sharing one database file)
- `ttl` parameter on the MCP `remember_memory` tool
- Regression tests for decision quality (`tests/test_decision_quality.py`)
- PyPI metadata: authors, keywords, classifiers, project URLs

### Changed
- Hybrid scoring now lets the stronger retrieval channel (semantic or keyword)
  dominate, improving paraphrase handling on the ChromaDB backend
- README documents backend trade-offs honestly: the default `sqlite` backend is
  lexical-only; use `chromadb` for semantic paraphrase matching

### Added (pre-existing unreleased items)
- SQLite backend (`SqliteMemoryStore`) as default lightweight storage
- Async API support (`aremember`, `aresolve`, `alist`, `aget`, `aforget`, `aarchive`, `acleanup`, `astats`, `aconsolidate`)
- Backend selection via `backend` parameter in `Memory` constructor (`"chromadb"` or `"sqlite"`)
- Comprehensive test coverage for both ChromaDB and SQLite backends
- Package distribution configuration (wheel, sdist)
- Docker support with both backends

### Changed
- **Default backend changed from ChromaDB to SQLite** for lightweight deployments
- Version bumped to `0.1.0-alpha` (was incorrectly `0.3.0` in code/docs)
- Updated roadmap to reflect completed features

### Fixed
- Version inconsistency across `pyproject.toml`, `agent_memory/__init__.py`, and `README.md`
- Clean code principles and SOLID OOP compliance across all modules

## [0.1.0-alpha] - 2026-06-28

### Added
- Initial release of Agent Memory
- Decision-based memory layer (Replay / Restore / Verify / None)
- Hybrid retrieval: BM25 keyword search + Vector semantic search with RRF fusion
- Multi-factor scoring policy (semantic 70%, recency 15%, confidence 20%, frequency 10%)
- Structured memory with types (conversation, fact, workflow, document, tool_output, code, summary, preference)
- Scoped memory (session, user, project, workspace, team, global)
- TTL support with flexible duration strings (e.g., "30d", "2h")
- Full observability via `decision.explain()`
- CLI with remember, resolve, stats, benchmark, eval, cleanup commands
- MCP server integration for Cursor, VS Code, and other MCP clients
- Docker support with multi-stage build
- Comprehensive documentation (architecture, benchmarks, examples, FAQ, getting started, memory model, policies)
- Benchmark and evaluation datasets (coding_agent, customer_support, research_agent)
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks (ruff, mypy, black)

### Security
- No known vulnerabilities

---

## Release Notes Template

### [X.Y.Z] - YYYY-MM-DD

#### Added
- New features

#### Changed
- Changes in existing functionality

#### Deprecated
- Soon-to-be removed features

#### Removed
- Removed features

#### Fixed
- Bug fixes

#### Security
- Security fixes