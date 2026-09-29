# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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