from __future__ import annotations

import json
import re
import threading
from typing import Any

from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType
from agent_memory.store import (
    STOP_WORDS,
    MemoryStore,
    _tokenize,
    query_coverage,
    search_document,
)
from agent_memory.vector_index import IvfFlatConfig, VectorIndexConfig

log = get_logger(__name__)

# pgvector gained `hnsw` in 0.5.0, but 0.7.0 is the first release we treat as
# production-ready for it (halfvec/sparsevec storage, much faster builds, and
# the iterative index scans that keep recall high under selective filters).
# Below that floor "auto" stays on ivfflat; pass vector_index="hnsw" to override.
_HNSW_MIN_PGVECTOR = (0, 7, 0)

# pgvector's own recommended defaults. Pass vector_config= / ivfflat_config= to
# override either index's parameters.
DEFAULT_VECTOR_CONFIG = VectorIndexConfig(m=16, ef_construction=64, ef_search=64)
DEFAULT_IVFFLAT_CONFIG = IvfFlatConfig(lists=10, probes=3)


# Exactly the columns _row_to_entry reads. Listed explicitly rather than using a
# star projection, which also streams back the 384-dim embedding and the tsvector
# on every read — kilobytes per row the caller never looks at.
_ENTRY_COLUMNS = (
    "id, query, response, content, type, scope, metadata, tags, confidence, "
    "requires_verification, archived, state, access_count, created_at, updated_at, "
    "last_accessed_at, expires_at"
)


def _parse_pgvector_version(raw: str) -> tuple[int, ...]:
    """Parse an extversion string like ``0.8.6`` into a comparable tuple."""
    parts: list[int] = []
    for chunk in raw.split("."):
        match = re.match(r"\d+", chunk)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


class PostgresMemoryStore(MemoryStore):
    """PostgreSQL-backed persistent memory store.

    Requires: pip install agent-memory-sdk[postgres]

    For vector semantic search add pgvector:
        pip install agent-memory-sdk[postgres,pgvector]

    The table uses a ``tsvector`` column for full-text keyword search and an
    optional ``embedding vector(N)`` column (via pgvector) for KNN lookup.

    The KNN index is HNSW when the server's pgvector is new enough. HNSW accepts
    online inserts and its scan cost stays roughly flat as the table grows, where
    ivfflat wants a populated table at build time and slows linearly because each
    probed list holds more vectors (measured in ``docs/benchmarks.md``: at 20,000
    rows HNSW scanned in 0.58ms against ivfflat's 4.76ms, having been the slower
    of the two at 2,000). Set ``vector_index`` to force one or the other.
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/agent_memory",
        table_name: str = "agent_memories",
        embedder: Any | None = None,
        enable_embeddings: bool | str = "auto",
        connection: Any | None = None,
        vector_index: str = "auto",
        vector_config: VectorIndexConfig | None = None,
        ivfflat_config: IvfFlatConfig | None = None,
    ) -> None:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "Postgres backend requires psycopg2. "
                "Install with: pip install agent-memory-sdk[postgres]"
            ) from None

        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        self._dsn = dsn
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,44}", table_name):
            raise ValueError(
                "table_name must be a simple SQL identifier of at most 45 characters"
            )
        self._table = table_name
        if vector_index not in ("auto", "hnsw", "ivfflat"):
            raise ValueError(
                f"vector_index must be 'auto', 'hnsw' or 'ivfflat', got {vector_index!r}"
            )
        self._vector_index_pref = vector_index
        self._vector_config = vector_config or DEFAULT_VECTOR_CONFIG
        self._ivfflat_config = ivfflat_config or DEFAULT_IVFFLAT_CONFIG
        self._embedder: Any | None = None
        self._vec_dim = 0
        self._vec_enabled = False
        self._vec_index_type: str | None = None
        self._pgvector_version: tuple[int, ...] | None = None
        self._register_vector: Any | None = None
        self._external_conn = connection
        self._local = threading.local()

        self._init_db()
        if enable_embeddings is True or enable_embeddings == "auto":
            self._init_embeddings(embedder, required=enable_embeddings is True)

    def _connect(self) -> Any:
        """Return this thread's connection, opening one the first time.

        Connections are cached per thread rather than opened per statement: a
        fresh psycopg2.connect() is a TCP handshake plus authentication, which
        costs far more than any query here and would put a 20ms floor under
        every single read. Per *thread* because a psycopg2 connection cannot be
        used concurrently from two of them.
        """
        if self._external_conn is not None:
            return self._external_conn
        conn = getattr(self._local, "conn", None)
        if conn is not None and not self._reusable(conn):
            self._discard(conn)
            conn = None
        if conn is None:
            conn = self._psycopg2.connect(self._dsn)
            self._register_vector_types(conn)
            self._local.conn = conn
        return conn

    def _reusable(self, conn: Any) -> bool:
        """True if *conn* is healthy and outside a transaction.

        Call sites commit on success and let exceptions propagate, so a failed
        statement can leave the transaction INERROR. On a per-statement
        connection that did not matter — it was about to be closed. On a reused
        one every later statement would fail with InFailedSqlTransaction, so the
        transaction is cleared here instead of in eleven `finally` blocks.
        """
        if conn.closed:
            return False
        try:
            if conn.get_transaction_status() != self._psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                conn.rollback()
        except self._psycopg2.Error:
            return False
        return True

    def _discard(self, conn: Any) -> None:
        try:
            conn.close()
        except Exception:  # already broken; nothing to salvage
            pass
        self._local.conn = None

    def _register_vector_types(self, conn: Any) -> None:
        """Teach a fresh connection about pgvector's types, if it can be taught.

        Only affects how values are adapted and decoded — every query here casts
        explicitly with ``::vector``, so a connection that cannot be registered
        still reads and writes correctly.
        """
        if self._register_vector is None:
            return
        try:
            self._register_vector(conn)
        except Exception as exc:  # extension missing, old server, …
            log.debug("pgvector type registration skipped: %s", exc)

    def _release(self, conn: Any) -> None:
        """End the transaction an operation left open on the cached connection.

        Runs in every operation's ``finally``, so it is the one hook that fires
        after both success and failure. Writes have already committed by then, so
        this only ends *read* transactions — but it has to: a cached connection
        left "idle in transaction" holds locks that block DDL from any other
        connection (a DROP TABLE waits forever) and pins a snapshot open, which
        stops vacuum from reclaiming anything.

        A caller-supplied connection is left alone; its owner drives its
        transactions.
        """
        if self._external_conn is not None:
            return
        if conn is not None and not conn.closed:
            try:
                conn.rollback()
            except self._psycopg2.Error:
                self._discard(conn)

    def _settle(self) -> None:
        """End any transaction open on this thread's cached connection."""
        conn = getattr(self._local, "conn", None)
        if conn is not None and not conn.closed:
            try:
                conn.rollback()
            except self._psycopg2.Error:
                self._discard(conn)

    def close(self) -> None:
        """Close this thread's cached connection, if it has one."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            self._discard(conn)

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        content TEXT NOT NULL,
                        type TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}',
                        tags JSONB NOT NULL DEFAULT '[]',
                        confidence REAL NOT NULL DEFAULT 1.0,
                        requires_verification BOOLEAN NOT NULL DEFAULT FALSE,
                        archived BOOLEAN NOT NULL DEFAULT FALSE,
                        state TEXT NOT NULL DEFAULT 'active',
                        access_count INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        last_accessed_at TIMESTAMPTZ,
                        expires_at TIMESTAMPTZ,
                        search_vector TSVECTOR
                    )
                """)
                for sql in (
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_scope ON {self._table}(scope)",
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_type ON {self._table}(type)",
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_archived ON {self._table}(archived)",
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_state ON {self._table}(state)",
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_expires ON {self._table}(expires_at)",
                    f"CREATE INDEX IF NOT EXISTS idx_{self._table}_fts ON {self._table} USING GIN(search_vector)",
                ):
                    cur.execute(sql)
            conn.commit()
        finally:
            self._release(conn)

    def _init_embeddings(self, embedder: Any | None, *, required: bool) -> None:
        try:
            import pgvector.psycopg2
        except ImportError:
            if required:
                raise ImportError(
                    "enable_embeddings=True requires pgvector. "
                    "Install with: pip install agent-memory-sdk[postgres,pgvector]"
                ) from None
            return

        # pgvector-python dropped register_vector_globally() in 0.4; the modern
        # API registers per connection, and needs the extension to already exist
        # (it looks up the type's OID), so it runs after CREATE EXTENSION below.
        self._register_vector = getattr(pgvector.psycopg2, "register_vector", None)
        legacy_global = getattr(pgvector.psycopg2, "register_vector_globally", None)
        if legacy_global is not None:
            legacy_global()
            self._register_vector = None

        from agent_memory.embeddings import embedding_dimension, get_default_embedder

        resolved = embedder or get_default_embedder()
        if resolved is None:
            if required:
                raise ImportError(
                    "enable_embeddings=True requires an embedding model. "
                    "Install with: pip install agent-memory-sdk[semantic]"
                )
            return

        self._embedder = resolved
        self._vec_dim = embedding_dimension(resolved)

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                row = cur.fetchone()
                self._pgvector_version = _parse_pgvector_version(row[0]) if row else None
                cur.execute(f"""
                    ALTER TABLE {self._table}
                    ADD COLUMN IF NOT EXISTS embedding vector({self._vec_dim})
                """)
            conn.commit()
            self._register_vector_types(conn)
            # register_vector() runs a catalogue lookup to find the type OID,
            # which opens a transaction it does not close. On a cached connection
            # that snapshot stays open — and CREATE INDEX CONCURRENTLY below waits
            # for every snapshot older than itself, so leaving it would hang.
            conn.commit()
        finally:
            self._release(conn)

        self._vec_index_type = self._resolve_index_type()
        self._create_vector_index(self._vec_index_type)
        self._vec_enabled = True

    def _resolve_index_type(self) -> str:
        if self._vector_index_pref != "auto":
            return self._vector_index_pref
        version = self._pgvector_version
        if version is not None and version >= _HNSW_MIN_PGVECTOR:
            return "hnsw"
        log.info(
            "pgvector %s is below %s; using ivfflat for KNN",
            ".".join(str(p) for p in version) if version else "unknown",
            ".".join(str(p) for p in _HNSW_MIN_PGVECTOR),
        )
        return "ivfflat"

    def _create_vector_index(self, index_type: str) -> None:
        """Build the KNN index, preferring a non-blocking CONCURRENTLY build.

        CONCURRENTLY cannot run inside a transaction, so it needs a connection
        of its own in autocommit mode. When the caller handed us a connection we
        do not own its transaction state, so fall back to the blocking form.
        """
        if index_type == "hnsw":
            name = f"idx_{self._table}_hnsw"
            using = (
                f"USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {self._vector_config.m}, "
                f"ef_construction = {self._vector_config.ef_construction})"
            )
        else:
            name = f"idx_{self._table}_embedding"
            using = (
                f"USING ivfflat (embedding vector_cosine_ops) "
                f"WITH (lists = {self._ivfflat_config.lists})"
            )

        sql = f"CREATE INDEX {{concurrently}} IF NOT EXISTS {name} ON {self._table} {using}"

        if self._external_conn is not None:
            conn = self._external_conn
            with conn.cursor() as cur:
                cur.execute(sql.format(concurrently=""))
            conn.commit()
        else:
            # CONCURRENTLY waits out every snapshot older than itself, so this
            # store's own cached connection must not be sitting in a transaction
            # while we wait — that would be a deadlock against ourselves.
            self._settle()
            conn = self._psycopg2.connect(self._dsn)
            try:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql.format(concurrently="CONCURRENTLY"))
            finally:
                conn.close()

        if index_type == "hnsw":
            # The pre-0.2.x layout built an ivfflat index under this name. Keeping
            # both would double every insert's index maintenance for no gain.
            self._execute_commit(f"DROP INDEX IF EXISTS idx_{self._table}_embedding")

    def _execute_commit(self, sql: str) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        finally:
            self._release(conn)

    @property
    def semantic_search_enabled(self) -> bool:
        return self._vec_enabled

    @property
    def vector_index_type(self) -> str | None:
        """``"hnsw"``, ``"ivfflat"``, or None when vector search is off."""
        return self._vec_index_type

    @property
    def pgvector_version(self) -> tuple[int, ...] | None:
        """The server's pgvector version as a tuple, e.g. ``(0, 8, 6)``."""
        return self._pgvector_version

    @property
    def vector_config(self) -> VectorIndexConfig:
        """The HNSW parameters this store builds and queries its index with."""
        return self._vector_config

    @property
    def ivfflat_config(self) -> IvfFlatConfig:
        """The IVFFlat parameters used when HNSW is unavailable or overridden."""
        return self._ivfflat_config

    @property
    def count(self) -> int:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table}")
                return int(cur.fetchone()[0])
        finally:
            self._release(conn)

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        entry.refresh_state()
        search_text = f"{entry.query} {entry.content} {' '.join(entry.tags)}"
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._table} (
                        id, query, response, content, type, scope, metadata, tags,
                        confidence, requires_verification, archived, state, access_count,
                        created_at, updated_at, last_accessed_at, expires_at, search_vector
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, to_tsvector('english', %s)
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        query = EXCLUDED.query,
                        response = EXCLUDED.response,
                        content = EXCLUDED.content,
                        type = EXCLUDED.type,
                        scope = EXCLUDED.scope,
                        metadata = EXCLUDED.metadata,
                        tags = EXCLUDED.tags,
                        confidence = EXCLUDED.confidence,
                        requires_verification = EXCLUDED.requires_verification,
                        archived = EXCLUDED.archived,
                        state = EXCLUDED.state,
                        access_count = EXCLUDED.access_count,
                        created_at = EXCLUDED.created_at,
                        updated_at = EXCLUDED.updated_at,
                        last_accessed_at = EXCLUDED.last_accessed_at,
                        expires_at = EXCLUDED.expires_at,
                        search_vector = EXCLUDED.search_vector
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
                        entry.requires_verification,
                        entry.archived,
                        entry.state.value,
                        entry.access_count,
                        entry.created_at,
                        entry.updated_at,
                        entry.last_accessed_at,
                        entry.expires_at,
                        search_text,
                    ),
                )
                if self._vec_enabled and self._embedder is not None:
                    vector = self._embedder([search_document(entry)])[0]
                    cur.execute(
                        f"UPDATE {self._table} SET embedding = %s::vector WHERE id = %s",
                        (vector, entry.id),
                    )
            conn.commit()
        finally:
            self._release(conn)
        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=self._extras.DictCursor) as cur:
                cur.execute(f"SELECT {_ENTRY_COLUMNS} FROM {self._table} WHERE id = %s", (memory_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_entry(row)
        finally:
            self._release(conn)

    def update(self, entry: MemoryEntry) -> MemoryEntry:
        return self.store(entry)

    def delete(self, memory_id: str) -> bool:
        conn = self._connect()
        deleted: bool
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._table} WHERE id = %s", (memory_id,))
                deleted = bool(cur.rowcount > 0)
            conn.commit()
        finally:
            self._release(conn)
        return deleted

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
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=self._extras.DictCursor) as cur:
                cur.execute(
                    f"SELECT {_ENTRY_COLUMNS} FROM {self._table} {where_sql} "
                    f"ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                    params,
                )
                return [self._row_to_entry(row) for row in cur.fetchall()]
        finally:
            self._release(conn)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if self._vec_enabled and self._embedder is not None:
            return self._vector_search(
                query,
                top_k=top_k,
                scopes=scopes,
                include_archived=include_archived,
                include_expired=include_expired,
            )
        return self.keyword_search(
            query,
            top_k=top_k,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )

    @staticmethod
    def _or_tsquery(query: str) -> str | None:
        """Build an OR tsquery from *query*, or None if it has no usable terms.

        ``plainto_tsquery`` ANDs every term, so "how to change my login
        credentials" would only match a document containing all five — which is
        why paraphrases found nothing here while the SQLite, Redis and Qdrant
        backends all rank an OR recall set. This matches them: recall broadly,
        then let ts_rank and the coverage scaling below do the discriminating.

        Terms come from ``_tokenize`` (``\\w+`` only), so none can carry tsquery
        operators and the joined string is safe to pass to ``to_tsquery``.
        """
        tokens = [t for t in _tokenize(query) if t not in STOP_WORDS and len(t) > 1]
        if not tokens:
            tokens = [t for t in _tokenize(query) if len(t) > 1]
        if not tokens:
            return None
        return " | ".join(dict.fromkeys(tokens))

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if not query.strip():
            return []

        tsquery = self._or_tsquery(query)
        if tsquery is None:
            return []

        where_clauses, filter_params = self._build_filters(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        where_clauses.append("search_vector @@ to_tsquery('english', %s)")
        where_sql = "WHERE " + " AND ".join(where_clauses)
        # Positional placeholders bind in SQL order, and the ts_rank() call in
        # the SELECT list comes before the WHERE clause — so its parameter has to
        # lead, ahead of the filter params.
        params = [tsquery, *filter_params, tsquery, top_k * 3]

        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=self._extras.DictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT {_ENTRY_COLUMNS}, ts_rank(search_vector, to_tsquery('english', %s)) AS rank
                    FROM {self._table} {where_sql}
                    ORDER BY rank DESC LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        finally:
            self._release(conn)

        if not rows:
            return []

        entries = [self._row_to_entry(row) for row in rows]
        raw_scores = [max(0.0, float(row["rank"])) for row in rows]
        max_score = max(raw_scores) if raw_scores else 0.0

        results: list[tuple[MemoryEntry, float]] = []
        for entry, raw in zip(entries, raw_scores):
            doc = f"{entry.query}\n{entry.content}"
            cov = query_coverage(query, doc)
            if max_score > 0:
                score = (raw / max_score) * (0.5 + 0.5 * cov)
            else:
                score = (0.5 + 0.5 * cov) if cov > 0 else 0.0
            if score > 0:
                results.append((entry, score))
        results.sort(key=lambda p: p[1], reverse=True)
        return results[:top_k]

    def _vector_search(
        self,
        query: str,
        *,
        top_k: int,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
    ) -> list[tuple[MemoryEntry, float]]:
        assert self._embedder is not None
        query_vector = self._embedder([query])[0]

        where_clauses, filter_params = self._build_filters(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        where_clauses.append("embedding IS NOT NULL")
        where_sql = "WHERE " + " AND ".join(where_clauses)
        # The distance expression in the SELECT list binds before the WHERE
        # clause's own params, so the query vector has to lead.
        params = [query_vector, *filter_params, query_vector, top_k]

        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=self._extras.DictCursor) as cur:
                self._apply_search_params(cur)
                cur.execute(
                    f"""
                    SELECT {_ENTRY_COLUMNS}, 1 - (embedding <=> %s::vector) AS similarity
                    FROM {self._table} {where_sql}
                    ORDER BY embedding <=> %s::vector LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        finally:
            self._release(conn)

        return [
            (self._row_to_entry(row), max(0.0, float(row["similarity"])))
            for row in rows
        ]

    def _apply_search_params(self, cur: Any) -> None:
        """Set the index's recall knob for this query.

        ``ef_search`` is HNSW's search-time candidate list: too small and the
        walk stops early, dropping true neighbours that a filtered query then
        cannot recover. SET LOCAL scopes it to the surrounding transaction, so
        it never leaks into other users of a shared connection.

        Both names are extension-namespaced, which Postgres accepts as a custom
        placeholder even on a server where pgvector never loaded — so this needs
        no version guard and cannot poison the transaction.
        """
        guc, value = (
            ("hnsw.ef_search", self._vector_config.ef_search)
            if self._vec_index_type == "hnsw"
            else ("ivfflat.probes", self._ivfflat_config.probes)
        )
        cur.execute(f"SET LOCAL {guc} = %s", (value,))

    def _build_filters(
        self,
        *,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
        memory_type: MemoryType | None = None,
    ) -> tuple[list[str], list[Any]]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if not include_archived:
            where_clauses.append("archived = FALSE")
        if scopes:
            placeholders = ",".join(["%s"] * len(scopes))
            where_clauses.append(f"scope IN ({placeholders})")
            params.extend(s.value for s in scopes)
        if memory_type:
            where_clauses.append("type = %s")
            params.append(memory_type.value)
        if not include_expired:
            where_clauses.append(
                "(state != 'expired' AND (expires_at IS NULL OR expires_at > NOW()))"
            )
        return where_clauses, params

    def stats(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*), COALESCE(SUM(access_count), 0) FROM {self._table}"
                )
                total, total_access = cur.fetchone()
                cur.execute(f"""
                    SELECT CASE
                        WHEN archived THEN 'archived'
                        WHEN state = 'expired'
                             OR (expires_at IS NOT NULL AND expires_at <= NOW()) THEN 'expired'
                        ELSE 'active'
                    END AS effective_state, COUNT(*)
                    FROM {self._table} GROUP BY effective_state
                """)
                by_state = dict(cur.fetchall())
                cur.execute(f"SELECT type, COUNT(*) FROM {self._table} GROUP BY type")
                by_type = dict(cur.fetchall())
        finally:
            self._release(conn)
        return {
            "total": total,
            "by_state": by_state,
            "by_type": by_type,
            "total_access_count": int(total_access),
        }

    def cleanup_expired(self, *, delete: bool = False) -> dict[str, int]:
        expired_where = (
            "(state = 'expired' OR (expires_at IS NOT NULL AND expires_at <= NOW()))"
        )
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                if delete:
                    cur.execute(f"DELETE FROM {self._table} WHERE {expired_where}")
                    deleted = cur.rowcount
                    conn.commit()
                    return {"expired": 0, "deleted": deleted}
                cur.execute(f"""
                    UPDATE {self._table} SET state = 'expired'
                    WHERE archived = FALSE AND state != 'expired'
                      AND expires_at IS NOT NULL AND expires_at <= NOW()
                """)
                cur.execute(
                    f"SELECT COUNT(*) FROM {self._table} WHERE {expired_where}"
                )
                expired = cur.fetchone()[0]
            conn.commit()
        finally:
            self._release(conn)
        return {"expired": expired, "deleted": 0}

    @staticmethod
    def _row_to_entry(row: Any) -> MemoryEntry:
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            metadata = json.loads(metadata)
        tags = row["tags"]
        if not isinstance(tags, list):
            tags = json.loads(tags)
        return MemoryEntry(
            id=row["id"],
            query=row["query"],
            response=row["response"],
            content=row["content"],
            type=MemoryType(row["type"]),
            scope=MemoryScope(row["scope"]),
            metadata=metadata,
            tags=tags,
            confidence=float(row["confidence"]),
            requires_verification=bool(row["requires_verification"]),
            archived=bool(row["archived"]),
            state=MemoryState(row["state"]),
            access_count=int(row["access_count"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            expires_at=row["expires_at"],
        )
