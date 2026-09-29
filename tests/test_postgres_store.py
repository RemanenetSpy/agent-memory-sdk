"""Tests for the PostgreSQL memory store backend.

Requires a running Postgres instance.  The connection string is read from
``AGENT_MEMORY_POSTGRES_DSN`` (default: ``postgresql://localhost/agent_memory_test``).
The entire module is skipped when psycopg2 is not installed or the server is
unreachable.
"""
from __future__ import annotations

import os

import pytest

POSTGRES_DSN = os.environ.get(
    "AGENT_MEMORY_POSTGRES_DSN",
    "postgresql://localhost/agent_memory_test",
)

try:
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    conn.close()
    POSTGRES_AVAILABLE = True
except Exception:
    POSTGRES_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not POSTGRES_AVAILABLE, reason="postgres not available"
)

TABLE = "test_memories"


@pytest.fixture()
def store():
    from agent_memory.postgres_store import PostgresMemoryStore

    s = PostgresMemoryStore(dsn=POSTGRES_DSN, table_name=TABLE, enable_embeddings=False)
    yield s
    # Teardown: drop the test table
    conn = psycopg2.connect(POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_store_and_get(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="hello postgres", response="it works")
    stored = store.store(entry)
    fetched = store.get(stored.id)
    assert fetched is not None
    assert fetched.query == "hello postgres"


def test_count(store):
    from agent_memory.models import MemoryEntry

    assert store.count == 0
    store.store(MemoryEntry(query="a", response="b"))
    assert store.count == 1


def test_update(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="original", response="old")
    store.store(entry)
    entry.response = "new response"
    store.update(entry)
    assert store.get(entry.id).response == "new response"


def test_delete(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="delete me", response="gone")
    store.store(entry)
    assert store.delete(entry.id) is True
    assert store.get(entry.id) is None
    assert store.delete(entry.id) is False


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def test_keyword_search(store):
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="machine learning algorithms", response="gradient descent"))
    store.store(MemoryEntry(query="cooking recipes pasta", response="boil water"))

    results = store.keyword_search("machine learning", top_k=5)
    assert len(results) > 0
    assert "machine" in results[0][0].query.lower() or "learning" in results[0][0].query.lower()


def test_keyword_search_empty(store):
    assert store.keyword_search("nothing here") == []


# ---------------------------------------------------------------------------
# Scope filtering
# ---------------------------------------------------------------------------


def test_scope_filter(store):
    from agent_memory.models import MemoryEntry, MemoryScope

    store.store(MemoryEntry(query="user", response="u", scope=MemoryScope.USER))
    store.store(MemoryEntry(query="global", response="g", scope=MemoryScope.GLOBAL))

    user_entries = store.list_all(scopes=[MemoryScope.USER])
    assert len(user_entries) == 1
    assert user_entries[0].scope == MemoryScope.USER


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats(store):
    from agent_memory.models import MemoryEntry, MemoryType

    store.store(MemoryEntry(query="q1", response="r1", type=MemoryType.FACT))
    store.store(MemoryEntry(query="q2", response="r2", type=MemoryType.CODE))
    s = store.stats()
    assert s["total"] == 2
    assert "fact" in s["by_type"]
    assert "code" in s["by_type"]


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_expired(store):
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="old", response="r", expires_at=past)
    e.refresh_state()
    store.store(e)
    result = store.cleanup_expired()
    assert result["expired"] >= 1


def test_cleanup_delete(store):
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="old", response="r", expires_at=past)
    e.refresh_state()
    store.store(e)
    result = store.cleanup_expired(delete=True)
    assert result["deleted"] >= 1
    assert store.count == 0


# ---------------------------------------------------------------------------
# Scoped search — regression coverage for positional parameter binding
#
# The SELECT list carries its own placeholder (ts_rank / the distance operator),
# which binds ahead of the WHERE clause's. Getting that order wrong only shows up
# once a filter adds params, i.e. exactly when scopes are passed.
# ---------------------------------------------------------------------------


def test_keyword_search_with_scope_filter(store):
    from agent_memory.models import MemoryEntry, MemoryScope

    store.store(
        MemoryEntry(query="machine learning algorithms", response="r", scope=MemoryScope.GLOBAL)
    )
    store.store(
        MemoryEntry(query="machine learning notes", response="r", scope=MemoryScope.USER)
    )

    global_hits = store.keyword_search(
        "machine learning", top_k=5, scopes=[MemoryScope.GLOBAL]
    )
    assert [e.scope for e, _ in global_hits] == [MemoryScope.GLOBAL]

    both = store.keyword_search(
        "machine learning", top_k=5, scopes=[MemoryScope.GLOBAL, MemoryScope.USER]
    )
    assert len(both) == 2


# ---------------------------------------------------------------------------
# pgvector HNSW
#
# Needs the `vector` extension on the server (use the pgvector/pgvector image),
# the pgvector Python package, and an embedding model.
# ---------------------------------------------------------------------------

VECTOR_TABLE = "test_memories_vec"


def _pgvector_ready() -> bool:
    if not POSTGRES_AVAILABLE:
        return False
    try:
        import pgvector.psycopg2  # noqa: F401

        from agent_memory.embeddings import get_default_embedder

        if get_default_embedder() is None:
            return False
        conn = psycopg2.connect(POSTGRES_DSN)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


PGVECTOR_AVAILABLE = _pgvector_ready()

needs_pgvector = pytest.mark.skipif(
    not PGVECTOR_AVAILABLE, reason="pgvector extension / package / embedding model unavailable"
)


def _drop(table: str) -> None:
    conn = psycopg2.connect(POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()


def _vector_indexes(table: str) -> list[str]:
    conn = psycopg2.connect(POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = %s AND indexdef ILIKE '%%embedding%%'",
            (table,),
        )
        names = sorted(row[0] for row in cur.fetchall())
    conn.close()
    return names


@pytest.fixture()
def vector_store():
    from agent_memory.postgres_store import PostgresMemoryStore

    _drop(VECTOR_TABLE)
    store = PostgresMemoryStore(
        dsn=POSTGRES_DSN, table_name=VECTOR_TABLE, enable_embeddings=True
    )
    yield store
    _drop(VECTOR_TABLE)


@needs_pgvector
def test_auto_picks_hnsw_on_modern_pgvector(vector_store):
    from agent_memory.postgres_store import _HNSW_MIN_PGVECTOR

    version = vector_store.pgvector_version
    assert version is not None
    expected = "hnsw" if version >= _HNSW_MIN_PGVECTOR else "ivfflat"
    assert vector_store.vector_index_type == expected
    assert vector_store.semantic_search_enabled is True
    assert _vector_indexes(VECTOR_TABLE) == [
        f"idx_{VECTOR_TABLE}_hnsw" if expected == "hnsw" else f"idx_{VECTOR_TABLE}_embedding"
    ]


@needs_pgvector
def test_vector_search_finds_paraphrase(vector_store):
    from agent_memory.models import MemoryEntry

    vector_store.store(
        MemoryEntry(query="How do I reset my password?", response="Settings → Security")
    )
    vector_store.store(MemoryEntry(query="Sourdough starter care", response="Feed daily"))

    # No content words in common with the stored query — only vector search can
    # surface it, so this fails if the store silently fell back to BM25.
    results = vector_store.search("I forgot my login credentials", top_k=1)
    assert results
    assert results[0][0].query == "How do I reset my password?"
    assert results[0][1] > 0.4


@needs_pgvector
def test_vector_search_respects_scope_filter(vector_store):
    from agent_memory.models import MemoryEntry, MemoryScope

    vector_store.store(MemoryEntry(query="user note", response="r", scope=MemoryScope.USER))
    vector_store.store(
        MemoryEntry(query="global note", response="r", scope=MemoryScope.GLOBAL)
    )

    hits = vector_store.search("note", top_k=5, scopes=[MemoryScope.GLOBAL])
    assert [e.scope for e, _ in hits] == [MemoryScope.GLOBAL]


@needs_pgvector
def test_vector_index_choice_can_be_forced():
    from agent_memory.postgres_store import PostgresMemoryStore

    _drop(VECTOR_TABLE)
    try:
        ivf = PostgresMemoryStore(
            dsn=POSTGRES_DSN,
            table_name=VECTOR_TABLE,
            enable_embeddings=True,
            vector_index="ivfflat",
        )
        assert ivf.vector_index_type == "ivfflat"
        assert _vector_indexes(VECTOR_TABLE) == [f"idx_{VECTOR_TABLE}_embedding"]

        # Re-opening with HNSW replaces the ivfflat index rather than stacking
        # a second one on the same column.
        hnsw = PostgresMemoryStore(
            dsn=POSTGRES_DSN,
            table_name=VECTOR_TABLE,
            enable_embeddings=True,
            vector_index="hnsw",
        )
        assert hnsw.vector_index_type == "hnsw"
        assert _vector_indexes(VECTOR_TABLE) == [f"idx_{VECTOR_TABLE}_hnsw"]
    finally:
        _drop(VECTOR_TABLE)


@needs_pgvector
def test_reopening_is_idempotent(vector_store):
    from agent_memory.models import MemoryEntry
    from agent_memory.postgres_store import PostgresMemoryStore

    entry = vector_store.store(MemoryEntry(query="persisted note", response="r"))
    before = _vector_indexes(VECTOR_TABLE)

    reopened = PostgresMemoryStore(
        dsn=POSTGRES_DSN, table_name=VECTOR_TABLE, enable_embeddings=True
    )
    assert _vector_indexes(VECTOR_TABLE) == before
    assert [e.id for e, _ in reopened.search("persisted note", top_k=1)] == [entry.id]


def test_keyword_search_matches_any_token(store):
    """OR recall, not AND: a paraphrase must not need every word present.

    plainto_tsquery() ANDs every term, which made scoped paraphrase queries
    return nothing here while every other backend ranked an OR recall set.
    """
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="machine learning algorithms", response="gradient descent"))
    store.store(MemoryEntry(query="cooking recipes pasta", response="boil water"))

    matched = {e.query for e, _ in store.keyword_search("learning pasta", top_k=5)}
    assert matched == {"machine learning algorithms", "cooking recipes pasta"}

    # A query where only some words appear anywhere still finds the entry.
    partial = store.keyword_search("how do I pick learning algorithms", top_k=5)
    assert [e.query for e, _ in partial][:1] == ["machine learning algorithms"]

    assert store.keyword_search("zzzz qqqq nonexistent", top_k=5) == []


def test_keyword_search_ignores_stopword_only_queries(store):
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="machine learning", response="r"))
    # No content terms at all — nothing to rank on, so nothing comes back.
    assert store.keyword_search("a", top_k=5) == []


# ---------------------------------------------------------------------------
# Connection reuse
#
# Connections are cached per thread rather than opened per statement. That is a
# large behavioural change to a shared resource, and the failure modes are the
# quiet kind: a transaction left open holds locks that block DDL from any other
# connection, and one thread's connection used from another corrupts protocol
# state. Both are covered here.
# ---------------------------------------------------------------------------

CONCURRENT_TABLE = "test_memories_concurrent"


@pytest.fixture()
def concurrent_store():
    from agent_memory.postgres_store import PostgresMemoryStore

    _drop(CONCURRENT_TABLE)
    store = PostgresMemoryStore(
        dsn=POSTGRES_DSN, table_name=CONCURRENT_TABLE, enable_embeddings=False
    )
    yield store
    store.close()
    _drop(CONCURRENT_TABLE)


def _open_transactions(table: str) -> int:
    """Backends idling inside a transaction that touched *table*."""
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                "AND state = 'idle in transaction'"
            )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def test_reads_do_not_leave_a_transaction_open(concurrent_store):
    """A read must not idle in a transaction — that holds locks other writers need."""
    from agent_memory.models import MemoryEntry

    entry = concurrent_store.store(MemoryEntry(query="lock probe", response="r"))

    # Every read path, including the ones that only SELECT.
    concurrent_store.get(entry.id)
    concurrent_store.list_all()
    concurrent_store.keyword_search("lock probe")
    assert concurrent_store.count == 1

    assert _open_transactions(CONCURRENT_TABLE) == 0


def test_another_connection_can_take_an_exclusive_lock(concurrent_store):
    """The regression that hung the suite: DROP TABLE waited on a stale snapshot."""
    from agent_memory.models import MemoryEntry

    concurrent_store.store(MemoryEntry(query="ddl probe", response="r"))
    concurrent_store.get(concurrent_store.list_all()[0].id)

    # ACCESS EXCLUSIVE, which cannot be granted while another backend holds a
    # lock on the table. lock_timeout turns a deadlock into a failure.
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '5s'")
            cur.execute(f"LOCK TABLE {CONCURRENT_TABLE} IN ACCESS EXCLUSIVE MODE")
        conn.commit()
    finally:
        conn.close()


def test_recovers_from_a_failed_statement(concurrent_store):
    """A failed statement must not poison every later query on the reused connection."""
    from agent_memory.models import MemoryEntry

    concurrent_store.store(MemoryEntry(query="before failure", response="r"))

    conn = concurrent_store._connect()
    with pytest.raises(psycopg2.Error):
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM a_table_that_does_not_exist")

    # Without the rollback in _connect(), this would raise InFailedSqlTransaction.
    assert concurrent_store.count == 1
    stored = concurrent_store.store(MemoryEntry(query="after failure", response="r"))
    assert concurrent_store.get(stored.id) is not None


def test_concurrent_writes_from_many_threads(concurrent_store):
    """Each thread gets its own connection; every write must survive."""
    import threading

    from agent_memory.models import MemoryEntry

    n_threads, per_thread = 6, 15
    errors: list[str] = []

    def writer(thread_id: int) -> None:
        for i in range(per_thread):
            try:
                concurrent_store.store(
                    MemoryEntry(
                        query=f"thread {thread_id} item {i}",
                        response="r",
                        tags=[f"t{thread_id}"],
                    )
                )
            except Exception as exc:  # noqa: BLE001 - reported below
                errors.append(f"thread={thread_id} i={i}: {exc!r}")

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert concurrent_store.count == n_threads * per_thread
    assert _open_transactions(CONCURRENT_TABLE) == 0


def test_concurrent_reads_and_writes(concurrent_store):
    """Readers and writers on separate connections must not interfere."""
    import threading

    from agent_memory.models import MemoryEntry

    errors: list[str] = []
    for i in range(20):
        concurrent_store.store(MemoryEntry(query=f"seed item {i}", response="r"))

    def writer(offset: int) -> None:
        for i in range(10):
            try:
                concurrent_store.store(
                    MemoryEntry(query=f"late item {offset + i}", response="r")
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"writer {offset}: {exc!r}")

    def reader() -> None:
        for _ in range(10):
            try:
                concurrent_store.keyword_search("item", top_k=5)
                concurrent_store.list_all(limit=10)
                concurrent_store.stats()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"reader: {exc!r}")

    threads = [threading.Thread(target=writer, args=(100 + t * 10,)) for t in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert concurrent_store.count == 50


def test_close_releases_the_cached_connection(concurrent_store):
    from agent_memory.models import MemoryEntry

    concurrent_store.store(MemoryEntry(query="reopen probe", response="r"))
    first = concurrent_store._connect()
    concurrent_store.close()
    assert first.closed

    # The next operation transparently opens a fresh one.
    assert concurrent_store.count == 1
    assert concurrent_store._connect() is not first
