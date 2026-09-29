from __future__ import annotations

import json
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_memory.embeddings import Embedder, embedding_dimension, get_default_embedder
from agent_memory.exceptions import BackendConnectionError
from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType
from agent_memory.search_index import BloomFilter, DynamicStopWords
from agent_memory.store import MemoryStore, _tokenize, bm25_scores, query_coverage

log = get_logger(__name__)

_VECTOR_BACKFILL_BATCH_SIZE = 64


class SqliteMemoryStore(MemoryStore):
    """SQLite-backed persistent memory storage.

    Keyword search uses an FTS5 index with SQLite's built-in BM25 ranking, so
    queries don't load the table into Python. With the ``semantic`` extra
    installed (sqlite-vec + an embedding model), ``search()`` becomes true
    vector search; otherwise it falls back to the lexical index.
    """

    def __init__(
        self,
        persist_dir: str | Path = ".agent_memory",
        collection_name: str = "agent_memories",
        embedder: Embedder | None = None,
        enable_embeddings: bool | str = "auto",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.persist_dir / f"{collection_name}.db"
        self._fts_enabled = False
        self._vec_enabled = False
        self._embedder: Embedder | None = None
        self._vec_dim = 0
        # Thread-local connection cache: one live connection per thread avoids
        # the ~2–3 ms overhead of sqlite3.connect() on every query.
        self._local = threading.local()
        # Protects concurrent mutations of _bloom, _dyn_stop, and the pending buffers.
        self._index_lock = threading.Lock()
        # Ensures only one _warm_indexes run executes at a time.
        # Non-force calls skip if the lock is held; force calls (rebuild_indexes)
        # block until the in-progress warm finishes before starting their own.
        self._warm_lock = threading.Lock()
        # Search-acceleration indexes (both maintained incrementally on write).
        # Bloom filter:   O(k) NONE fast-path — skips FTS5 when no token can match.
        # DynamicStopWords: extends STOP_WORDS with corpus-saturated terms at runtime.
        self._bloom: BloomFilter = BloomFilter(capacity=100_000)
        # min_docs=5000: don't activate dynamic stop words until the store is
        # large enough that high-frequency terms genuinely saturate the corpus.
        # Below 5K entries, even "common" domain terms appear in high fractions
        # of a tiny corpus and filtering them would hurt recall.
        self._dyn_stop: DynamicStopWords = DynamicStopWords(idf_threshold=0.05, min_docs=5_000)
        self._indexes_warm = False   # True after first load from existing DB
        # Pending buffers: when not None, store() mirrors writes here so
        # _warm_indexes can replay them into the new filter/stop-words at swap time.
        self._warm_pending: set[str] | None = None
        self._dyn_stop_pending: list[tuple[str, str, list[str]]] | None = None
        try:
            self._init_db()
        except sqlite3.Error as exc:
            raise BackendConnectionError("sqlite", str(exc)) from exc
        self._warm_indexes()
        log.info("SQLite store ready  path=%s  bloom=%d tokens  dyn_stop=%d terms",
                 self.db_path, self._bloom.estimated_count, self._dyn_stop.dynamic_count)
        if enable_embeddings is True or enable_embeddings == "auto":
            self._init_embeddings(embedder, required=enable_embeddings is True)

    def _connect(self) -> sqlite3.Connection:
        """Return a cached per-thread SQLite connection.

        Re-uses the same live connection within a thread to avoid the overhead
        of sqlite3.connect() on every query.  The cached connection is trusted;
        if a genuine closed-connection error surfaces from a query, callers
        should call close() and retry.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        # Performance pragmas applied once per connection:
        #   cache_size  — 32 MB page cache (default ≈ 2 MB)
        #   temp_store  — temp tables in memory, not on disk
        #   mmap_size   — 128 MB memory-mapped I/O for reads
        conn.execute("PRAGMA cache_size = -32000")
        conn.execute("PRAGMA temp_store = memory")
        conn.execute("PRAGMA mmap_size = 134217728")
        if self._vec_enabled:
            self._load_vec_extension(conn)
        self._local.conn = conn
        return conn

    def rebuild_indexes(self) -> None:
        """Force-rebuild the Bloom filter and DynamicStopWords from the current DB.

        Use after bulk imports that bypass the normal ``store()`` path.
        """
        self._warm_indexes(force=True)

    def _warm_indexes(self, *, force: bool = False) -> None:
        """Populate the Bloom filter and DynamicStopWords from the existing DB.

        Only one warm runs at a time (_warm_lock): non-force calls skip if a
        warm is already running; force calls (rebuild_indexes) wait for it to
        finish before starting their own.  Pending buffers capture every
        store() write that arrives during the build window so no tokens are
        lost when the new filter is swapped in.
        """
        acquired = self._warm_lock.acquire(blocking=force)
        if not acquired:
            return  # another warm is in progress; let it complete

        try:
            with self._index_lock:
                if self._indexes_warm and not force:
                    return
                # Open both pending buffers.  store() will mirror writes here
                # so we can replay them at swap time.
                self._warm_pending = set()
                self._dyn_stop_pending = []
                self._indexes_warm = False

            term_freq: Counter[str] = Counter()
            doc_count = 0

            # Dedicated read connection — never touches the thread-local connection
            # so we cannot interfere with (or be blocked by) any open transaction.
            read_conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                read_conn.execute("BEGIN DEFERRED")
                doc_count = read_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                local_bloom = BloomFilter(capacity=max(doc_count * 2, 100_000))
                local_dyn_stop = DynamicStopWords(idf_threshold=0.05, min_docs=5_000)
                if doc_count:
                    for query, content, tags_json in read_conn.execute(
                        "SELECT query, content, tags FROM memories"
                    ):
                        try:
                            tags = json.loads(tags_json) if tags_json else []
                        except Exception:
                            tags = []
                        text = f"{query} {content} {' '.join(tags)}"
                        tokens = {t for t in _tokenize(text) if len(t) > 1}
                        for t in tokens:
                            local_bloom.add(t)
                            term_freq[t] += 1
                read_conn.execute("ROLLBACK")
            finally:
                read_conn.close()

            local_dyn_stop.rebuild_from_counter(term_freq, doc_count)

            with self._index_lock:
                # Replay exact tokens added by concurrent store() calls.
                if self._warm_pending:
                    for t in self._warm_pending:
                        local_bloom.add(t)
                # Replay dyn_stop add_entry calls missed by the DB snapshot.
                if self._dyn_stop_pending:
                    for q, c, tags in self._dyn_stop_pending:
                        local_dyn_stop.add_entry(q, c, tags)
                self._warm_pending = None
                self._dyn_stop_pending = None
                self._bloom = local_bloom
                self._dyn_stop = local_dyn_stop
                self._indexes_warm = True

            log.debug("Indexes warmed  n=%d  bloom_tokens=%d  dyn_stop=%d",
                      doc_count, local_bloom.estimated_count, local_dyn_stop.dynamic_count)
        finally:
            self._warm_lock.release()

    def close(self) -> None:
        """Explicitly release the cached connection for this thread."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            # WAL is persistent in the DB file, so it only needs to be set
            # once. Switching modes needs a brief exclusive lock, which can
            # fail when several processes initialize the same DB at once —
            # whichever one wins has set it, so the losers can move on.
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    tags TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    requires_verification INTEGER NOT NULL DEFAULT 0,
                    archived INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'active',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
            for index_sql in (
                "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)",
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)",
                "CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived)",
                "CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)",
                "CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at)",
            ):
                conn.execute(index_sql)
            self._init_fts(conn)
            conn.commit()

    def _init_fts(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(search_text)"
            )
        except sqlite3.OperationalError:
            # SQLite built without FTS5; keyword search falls back to Python BM25.
            self._fts_enabled = False
            return
        self._fts_enabled = True
        # Backfill for databases created before the FTS index existed (or
        # written by an older version of this library).
        memories_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        if fts_count != memories_count:
            conn.execute("DELETE FROM memories_fts")
            conn.execute(
                """
                INSERT INTO memories_fts(rowid, search_text)
                SELECT rowid, query || char(10) || content || char(10) || tags
                FROM memories
                """
            )

    # ------------------------------------------------------------------
    # Optional vector search (sqlite-vec + embedding model)
    # ------------------------------------------------------------------

    def _init_embeddings(self, embedder: Embedder | None, *, required: bool) -> None:
        try:
            import sqlite_vec  # noqa: F401
        except ImportError:
            if required:
                raise ImportError(
                    "enable_embeddings=True requires sqlite-vec. "
                    "Install with: pip install agent-memory-sdk[semantic]"
                ) from None
            return

        resolved = embedder or get_default_embedder()
        if resolved is None:
            if required:
                raise ImportError(
                    "enable_embeddings=True requires an embedding model. "
                    "Install with: pip install agent-memory-sdk[semantic]"
                )
            return

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            if not self._load_vec_extension(conn):
                if required:
                    raise RuntimeError(
                        "This Python's sqlite3 cannot load extensions, "
                        "so sqlite-vec is unavailable."
                    )
                return
            self._embedder = resolved
            self._vec_dim = embedding_dimension(resolved)
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                    embedding float[{self._vec_dim}] distance_metric=cosine
                )
                """
            )
            self._vec_enabled = True
            self._backfill_vectors(conn)
            conn.commit()
        finally:
            conn.close()
        # The thread-local cached connection (created before _vec_enabled was set)
        # has no vec extension loaded.  Invalidate it so the next _connect() call
        # creates a fresh connection that goes through the vec extension setup.
        self.close()

    @staticmethod
    def _load_vec_extension(conn: sqlite3.Connection) -> bool:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return True
        except (ImportError, AttributeError, sqlite3.OperationalError):
            return False

    def _backfill_vectors(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT m.rowid, m.query, m.content, m.tags FROM memories m
            WHERE m.rowid NOT IN (SELECT rowid FROM memories_vec)
            """
        ).fetchall()
        if not rows or self._embedder is None:
            return
        import sqlite_vec

        for start in range(0, len(rows), _VECTOR_BACKFILL_BATCH_SIZE):
            batch = rows[start : start + _VECTOR_BACKFILL_BATCH_SIZE]
            texts = [f"{query}\n{content}\n{tags}" for _, query, content, tags in batch]
            vectors = self._embedder(texts)
            for (rowid, *_), vector in zip(batch, vectors):
                conn.execute(
                    "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(vector)),
                )

    @property
    def semantic_search_enabled(self) -> bool:
        """True when search() uses real embeddings instead of lexical ranking."""
        return self._vec_enabled

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            return int(cursor.fetchone()[0])

    def store(self, entry: MemoryEntry) -> MemoryEntry:  # type: ignore[override]
        log.debug("store  id=%s  type=%s  scope=%s", entry.id[:8], entry.type.value, entry.scope.value)
        entry.refresh_state()
        with self._connect() as conn:
            # Upsert (not INSERT OR REPLACE) so the rowid stays stable —
            # the FTS and vector tables are keyed by it.
            conn.execute(
                """
                INSERT INTO memories (
                    id, query, response, content, type, scope, metadata, tags,
                    confidence, requires_verification, archived, state, access_count,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    query = excluded.query,
                    response = excluded.response,
                    content = excluded.content,
                    type = excluded.type,
                    scope = excluded.scope,
                    metadata = excluded.metadata,
                    tags = excluded.tags,
                    confidence = excluded.confidence,
                    requires_verification = excluded.requires_verification,
                    archived = excluded.archived,
                    state = excluded.state,
                    access_count = excluded.access_count,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    entry.id,
                    entry.query,
                    entry.response,
                    entry.content,
                    entry.type.value,
                    entry.scope.value,
                    json.dumps(entry.metadata),
                    json.dumps(entry.tags),
                    entry.confidence,
                    int(entry.requires_verification),
                    int(entry.archived),
                    entry.state.value,
                    entry.access_count,
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                    entry.expires_at.isoformat() if entry.expires_at else None,
                ),
            )
            rowid = conn.execute(
                "SELECT rowid FROM memories WHERE id = ?", (entry.id,)
            ).fetchone()[0]
            if self._fts_enabled:
                conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO memories_fts(rowid, search_text) VALUES (?, ?)",
                    (rowid, self._search_document(entry)),
                )
            if self._vec_enabled and self._embedder is not None:
                import sqlite_vec

                vector = self._embedder([self._search_document(entry)])[0]
                # vec0 tables don't support INSERT OR REPLACE; delete first.
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO memories_vec(rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(vector)),
                )
            conn.commit()

        # Update search-acceleration indexes under lock — bytearray |= and
        # Counter updates are not atomic; concurrent store() calls would race.
        # Also mirror into _warm_pending when a _warm_indexes() call is in
        # progress so those tokens are not lost when the new filter is swapped in.
        text = f"{entry.query} {entry.content} {' '.join(entry.tags)}"
        tokens = {t for t in _tokenize(text) if len(t) > 1}
        with self._index_lock:
            self._bloom.add_many(tokens)
            self._dyn_stop.add_entry(entry.query, entry.content, entry.tags)
            if self._warm_pending is not None:
                self._warm_pending.update(tokens)
            if self._dyn_stop_pending is not None:
                self._dyn_stop_pending.append(
                    (entry.query, entry.content, list(entry.tags))
                )

        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        finally:
            conn.row_factory = None
        if not row:
            return None
        return self._row_to_entry(row)

    def touch(self, memory_id: str) -> bool:
        """Fast access-count increment — no FTS5 reindex.

        Only ``access_count`` is updated; content fields and indexes are
        untouched.  Commits immediately so the write transaction releases the
        write lock for concurrent writers (SQLite allows only one writer at a
        time, even in WAL mode).

        Limitation: this issues ``conn.commit()`` on the shared thread-local
        connection.  Do not call ``touch()`` from inside an outer write
        transaction on the same thread — the commit here would prematurely
        flush that transaction.  All current internal call-sites are
        single-statement; this note is for future callers.
        """
        conn = self._connect()
        cursor = conn.execute(
            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
            (memory_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def update(self, entry: MemoryEntry) -> MemoryEntry:
        return self.store(entry)

    def delete(self, memory_id: str) -> bool:
        log.debug("delete  id=%s", memory_id[:8])
        with self._connect() as conn:
            self._delete_index_rows(conn, "id = ?", [memory_id])
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _delete_index_rows(self, conn: sqlite3.Connection, where: str, params: list) -> None:
        """Remove FTS/vector rows for memories matching a WHERE clause."""
        if self._fts_enabled:
            conn.execute(
                f"DELETE FROM memories_fts WHERE rowid IN "
                f"(SELECT rowid FROM memories WHERE {where})",
                params,
            )
        if self._vec_enabled:
            conn.execute(
                f"DELETE FROM memories_vec WHERE rowid IN "
                f"(SELECT rowid FROM memories WHERE {where})",
                params,
            )

    # ------------------------------------------------------------------
    # Listing and filters
    # ------------------------------------------------------------------

    def _build_filters(
        self,
        *,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
        memory_type: MemoryType | None = None,
        table_alias: str = "",
    ) -> tuple[list[str], list[Any]]:
        prefix = f"{table_alias}." if table_alias else ""
        where_clauses: list[str] = []
        params: list[Any] = []
        if not include_archived:
            where_clauses.append(f"{prefix}archived = 0")
        if scopes:
            placeholders = ",".join("?" * len(scopes))
            where_clauses.append(f"{prefix}scope IN ({placeholders})")
            params.extend(s.value for s in scopes)
        if memory_type:
            where_clauses.append(f"{prefix}type = ?")
            params.append(memory_type.value)
        if not include_expired:
            where_clauses.append(
                f"({prefix}state != 'expired' AND "
                f"({prefix}expires_at IS NULL OR {prefix}expires_at > ?))"
            )
            params.append(datetime.now(timezone.utc).isoformat())
        return where_clauses, params

    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        where_clauses, params = self._build_filters(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
            memory_type=memory_type,
        )
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        params.extend([limit, offset])

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    f"SELECT * FROM memories {where_sql} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    params,
                ).fetchall()
            finally:
                conn.row_factory = None
            return [self._row_to_entry(row) for row in rows]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if self._vec_enabled:
            return self._vector_search(
                query,
                top_k=top_k,
                scopes=scopes,
                include_archived=include_archived,
                include_expired=include_expired,
            )
        # Without embeddings, "semantic" search is lexical (BM25 scaled by
        # query-term coverage). Install agent-memory-sdk[semantic] or use the
        # chromadb backend when paraphrase robustness matters.
        return self.keyword_search(
            query,
            top_k=top_k,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if not self._fts_enabled:
            return self._python_keyword_search(
                query,
                top_k=top_k,
                scopes=scopes,
                include_archived=include_archived,
                include_expired=include_expired,
            )

        # ── Bloom filter NONE fast-path ───────────────────────────────────────
        # If none of the query's content tokens are in the Bloom filter, no FTS5
        # query can return any matches.  Skip the SQL round-trip entirely.
        # O(k·|tokens|) in Python memory — typically < 0.1ms.
        if self._indexes_warm and self._bloom.estimated_count > 0:
            raw_tokens = _tokenize(query)
            content_tokens = self._dyn_stop.filter_tokens(raw_tokens) or raw_tokens
            if content_tokens and not self._bloom.any_of(content_tokens):
                log.debug("keyword_search bloom_miss  query=%r", query)
                return []

        match_expr = self._fts_match_expression(query)
        if not match_expr:
            return []

        where_clauses, params = self._build_filters(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
            table_alias="m",
        )
        filter_sql = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""

        # Use the cached connection directly (no context manager) — reads don't
        # need commit() and skipping it removes ~2ms of overhead per query.
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                f"""
                SELECT m.*, bm25(memories_fts) AS fts_rank
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ?{filter_sql}
                ORDER BY fts_rank
                LIMIT ?
                """,
                [match_expr, *params, top_k + 10],
            ).fetchall()
        finally:
            conn.row_factory = None

        if not rows:
            return []

        # bm25() is a rank (more negative = better); convert to a positive
        # relevance score, normalize, and scale by query-term coverage so a
        # weak best match cannot score a perfect 1.0.
        raw = [max(0.0, -float(row["fts_rank"])) for row in rows]
        max_raw = max(raw)
        results: list[tuple[MemoryEntry, float]] = []
        for row, raw_score in zip(rows, raw):
            entry = self._row_to_entry(row)
            coverage = query_coverage(query, self._search_document(entry))
            if max_raw > 0:
                score = (raw_score / max_raw) * (0.5 + 0.5 * coverage)
            else:
                score = 0.5 + 0.5 * coverage if coverage > 0 else 0.0
            if score > 0:
                results.append((entry, score))
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:top_k]

    def _python_keyword_search(
        self,
        query: str,
        *,
        top_k: int,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
    ) -> list[tuple[MemoryEntry, float]]:
        entries = self.list_all(
            limit=10_000,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        if not entries:
            return []
        documents = [self._search_document(e) for e in entries]
        return bm25_scores(query, entries, documents, top_k)

    def _vector_search(
        self,
        query: str,
        *,
        top_k: int,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
    ) -> list[tuple[MemoryEntry, float]]:
        import sqlite_vec

        assert self._embedder is not None
        query_vector = self._embedder([query])[0]

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            try:
                # Over-fetch so post-KNN filtering still yields top_k results.
                knn = conn.execute(
                    """
                    SELECT rowid, distance FROM memories_vec
                    WHERE embedding MATCH ? AND k = ?
                    """,
                    (sqlite_vec.serialize_float32(query_vector), max(top_k * 4, 20)),
                ).fetchall()
                if not knn:
                    return []

                distances = {row["rowid"]: float(row["distance"]) for row in knn}
                placeholders = ",".join("?" * len(distances))
                where_clauses, params = self._build_filters(
                    scopes=scopes,
                    include_archived=include_archived,
                    include_expired=include_expired,
                )
                filter_sql = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""
                rows = conn.execute(
                    f"SELECT rowid, * FROM memories WHERE rowid IN ({placeholders}){filter_sql}",
                    [*distances.keys(), *params],
                ).fetchall()
            finally:
                conn.row_factory = None

        matches = [
            (self._row_to_entry(row), max(0.0, 1.0 - distances[row["rowid"]]))
            for row in rows
        ]
        matches.sort(key=lambda pair: pair[1], reverse=True)
        return matches[:top_k]

    def _fts_expressions(self, query: str) -> tuple[str, str]:
        """Return (and_expr, or_expr) FTS5 MATCH expressions filtered by dynamic stop words."""
        tokens = _tokenize(query)
        if not tokens:
            return "", ""

        # Apply dynamic corpus-frequency stop words (extends static STOP_WORDS)
        content = self._dyn_stop.filter_tokens(tokens)
        if not content:
            content = [t for t in tokens if len(t) > 1]
        if not content:
            return "", ""

        and_expr = " ".join(f'"{t}"' for t in content)
        or_expr  = " OR ".join(f'"{t}"' for t in content)
        return and_expr, or_expr

    def _fts_match_expression(self, query: str) -> str:
        """Returns the OR FTS5 expression with dynamic stop-word filtering."""
        _, or_expr = self._fts_expressions(query)
        return or_expr

    # ------------------------------------------------------------------
    # Aggregates (pure SQL — no row loading)
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            total, total_access = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(access_count), 0) FROM memories"
            ).fetchone()
            # Effective state: archived wins, then expiry (marked or by
            # timestamp), then active — mirrors MemoryEntry.refresh_state().
            state_rows = conn.execute(
                """
                SELECT CASE
                    WHEN archived = 1 THEN 'archived'
                    WHEN state = 'expired'
                         OR (expires_at IS NOT NULL AND expires_at <= ?) THEN 'expired'
                    ELSE 'active'
                END AS effective_state, COUNT(*)
                FROM memories GROUP BY effective_state
                """,
                (now,),
            ).fetchall()
            type_rows = conn.execute(
                "SELECT type, COUNT(*) FROM memories GROUP BY type"
            ).fetchall()
        return {
            "total": total,
            "by_state": dict(state_rows),
            "by_type": dict(type_rows),
            "total_access_count": total_access,
        }

    def cleanup_expired(self, *, delete: bool = False) -> dict[str, int]:
        now = datetime.now(timezone.utc).isoformat()
        expired_where = "(state = 'expired' OR (expires_at IS NOT NULL AND expires_at <= ?))"
        with self._connect() as conn:
            if delete:
                self._delete_index_rows(conn, expired_where, [now])
                cursor = conn.execute(f"DELETE FROM memories WHERE {expired_where}", (now,))
                conn.commit()
                return {"expired": 0, "deleted": cursor.rowcount}

            conn.execute(
                """
                UPDATE memories SET state = 'expired'
                WHERE archived = 0 AND state != 'expired'
                  AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (now,),
            )
            expired = conn.execute(
                f"SELECT COUNT(*) FROM memories WHERE {expired_where}", (now,)
            ).fetchone()[0]
            conn.commit()
        return {"expired": expired, "deleted": 0}

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _search_document(entry: MemoryEntry) -> str:
        return f"{entry.query}\n{entry.content}\n{' '.join(entry.tags)}"

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            query=row["query"],
            response=row["response"],
            content=row["content"],
            type=MemoryType(row["type"]),
            scope=MemoryScope(row["scope"]),
            metadata=json.loads(row["metadata"]),
            tags=json.loads(row["tags"]),
            confidence=row["confidence"],
            requires_verification=bool(row["requires_verification"]),
            archived=bool(row["archived"]),
            state=MemoryState(row["state"]),
            access_count=row["access_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
        )
