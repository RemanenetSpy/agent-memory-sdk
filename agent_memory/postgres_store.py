from __future__ import annotations

import json
import re
from typing import Any

from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType
from agent_memory.store import MemoryStore, query_coverage


class PostgresMemoryStore(MemoryStore):
    """PostgreSQL-backed persistent memory store.

    Requires: pip install agent-memory-sdk[postgres]

    For vector semantic search add pgvector:
        pip install agent-memory-sdk[postgres,pgvector]

    The table uses a ``tsvector`` column for full-text keyword search and an
    optional ``embedding vector(N)`` column (via pgvector) for KNN lookup.
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/agent_memory",
        table_name: str = "agent_memories",
        embedder: Any | None = None,
        enable_embeddings: bool | str = "auto",
        connection: Any | None = None,
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
        self._embedder: Any | None = None
        self._vec_dim = 0
        self._vec_enabled = False
        self._external_conn = connection

        self._init_db()
        if enable_embeddings is True or enable_embeddings == "auto":
            self._init_embeddings(embedder, required=enable_embeddings is True)

    def _connect(self) -> Any:
        if self._external_conn is not None:
            return self._external_conn
        return self._psycopg2.connect(self._dsn)

    def _should_close(self) -> bool:
        return self._external_conn is None

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
            if self._should_close():
                conn.close()

    def _init_embeddings(self, embedder: Any | None, *, required: bool) -> None:
        try:
            import pgvector.psycopg2

            pgvector.psycopg2.register_vector_globally()
        except ImportError:
            if required:
                raise ImportError(
                    "enable_embeddings=True requires pgvector. "
                    "Install with: pip install agent-memory-sdk[postgres,pgvector]"
                ) from None
            return

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
                cur.execute(f"""
                    ALTER TABLE {self._table}
                    ADD COLUMN IF NOT EXISTS embedding vector({self._vec_dim})
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._table}_embedding
                    ON {self._table} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 10)
                """)
            conn.commit()
        finally:
            if self._should_close():
                conn.close()

        self._vec_enabled = True

    @property
    def semantic_search_enabled(self) -> bool:
        return self._vec_enabled

    @property
    def count(self) -> int:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table}")
                return int(cur.fetchone()[0])
        finally:
            if self._should_close():
                conn.close()

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
                    vector = self._embedder([f"{entry.query}\n{entry.content}"])[0]
                    cur.execute(
                        f"UPDATE {self._table} SET embedding = %s WHERE id = %s",
                        (vector, entry.id),
                    )
            conn.commit()
        finally:
            if self._should_close():
                conn.close()
        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=self._extras.DictCursor) as cur:
                cur.execute(f"SELECT * FROM {self._table} WHERE id = %s", (memory_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return self._row_to_entry(row)
        finally:
            if self._should_close():
                conn.close()

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
            if self._should_close():
                conn.close()
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
                    f"SELECT * FROM {self._table} {where_sql} "
                    f"ORDER BY updated_at DESC LIMIT %s OFFSET %s",
                    params,
                )
                return [self._row_to_entry(row) for row in cur.fetchall()]
        finally:
            if self._should_close():
                conn.close()

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

        where_clauses, params = self._build_filters(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        where_clauses.append("search_vector @@ plainto_tsquery('english', %s)")
        params.append(query)
        where_sql = "WHERE " + " AND ".join(where_clauses)
        params.append(query)
        params.append(top_k * 3)

        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=self._extras.DictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT *, ts_rank(search_vector, plainto_tsquery('english', %s)) AS rank
                    FROM {self._table} {where_sql}
                    ORDER BY rank DESC LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        finally:
            if self._should_close():
                conn.close()

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

        where_clauses, params = self._build_filters(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        where_clauses.append("embedding IS NOT NULL")
        where_sql = "WHERE " + " AND ".join(where_clauses)
        params.extend([query_vector, query_vector, top_k])

        conn = self._connect()
        try:
            with conn.cursor(cursor_factory=self._extras.DictCursor) as cur:
                cur.execute(
                    f"""
                    SELECT *, 1 - (embedding <=> %s::vector) AS similarity
                    FROM {self._table} {where_sql}
                    ORDER BY embedding <=> %s::vector LIMIT %s
                    """,
                    params,
                )
                rows = cur.fetchall()
        finally:
            if self._should_close():
                conn.close()

        return [
            (self._row_to_entry(row), max(0.0, float(row["similarity"])))
            for row in rows
        ]

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
            if self._should_close():
                conn.close()
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
            if self._should_close():
                conn.close()
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
