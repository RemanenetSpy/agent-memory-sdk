"""Tests for the Redis memory store backend.

The CRUD / BM25 tests use fakeredis, so no real Redis server is required in CI.
The vector-search tests need RediSearch, which fakeredis does not implement;
they skip unless a real Redis 8+ / Redis Stack instance is reachable.
Skip the whole module if fakeredis is not installed.
"""
from __future__ import annotations

import os
import zlib
from uuid import uuid4

import pytest

try:
    import fakeredis

    FAKEREDIS_AVAILABLE = True
except ImportError:
    FAKEREDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FAKEREDIS_AVAILABLE, reason="fakeredis not installed"
)


def make_store():
    from agent_memory.redis_store import RedisMemoryStore

    fake_client = fakeredis.FakeRedis(decode_responses=True)
    return RedisMemoryStore(redis_client=fake_client)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_store_and_get():
    store = make_store()
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="hello", response="world")
    stored = store.store(entry)
    fetched = store.get(stored.id)
    assert fetched is not None
    assert fetched.query == "hello"
    assert fetched.response == "world"


def test_count_increments():
    store = make_store()
    from agent_memory.models import MemoryEntry

    assert store.count == 0
    store.store(MemoryEntry(query="a", response="b"))
    assert store.count == 1
    store.store(MemoryEntry(query="c", response="d"))
    assert store.count == 2


def test_update():
    store = make_store()
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="original", response="old response")
    store.store(entry)
    entry.response = "updated response"
    store.update(entry)
    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.response == "updated response"


def test_delete():
    store = make_store()
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="to delete", response="bye")
    store.store(entry)
    assert store.count == 1
    assert store.delete(entry.id) is True
    assert store.get(entry.id) is None
    assert store.count == 0
    assert store.delete(entry.id) is False  # idempotent


def test_get_missing():
    store = make_store()
    assert store.get("nonexistent-id") is None


# ---------------------------------------------------------------------------
# Scope filtering
# ---------------------------------------------------------------------------


def test_list_scope_filter():
    store = make_store()
    from agent_memory.models import MemoryEntry, MemoryScope

    store.store(MemoryEntry(query="user q", response="r", scope=MemoryScope.USER))
    store.store(MemoryEntry(query="global q", response="r", scope=MemoryScope.GLOBAL))

    user_entries = store.list_all(scopes=[MemoryScope.USER])
    assert len(user_entries) == 1
    assert user_entries[0].scope == MemoryScope.USER

    global_entries = store.list_all(scopes=[MemoryScope.GLOBAL])
    assert len(global_entries) == 1


def test_list_archived_excluded_by_default():
    store = make_store()
    from agent_memory.models import MemoryEntry

    e = MemoryEntry(query="archived", response="r", archived=True)
    store.store(e)
    assert store.list_all() == []
    assert len(store.list_all(include_archived=True)) == 1


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def test_keyword_search_finds_match():
    store = make_store()
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="Python async programming", response="use asyncio"))
    store.store(MemoryEntry(query="How to bake bread", response="flour and yeast"))

    results = store.keyword_search("async programming", top_k=5)
    assert len(results) > 0
    best_entry, best_score = results[0]
    assert "async" in best_entry.query.lower() or "async" in best_entry.response.lower()
    assert best_score > 0


def test_keyword_search_empty_store():
    store = make_store()
    assert store.keyword_search("anything") == []


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


def test_expired_excluded_by_default():
    store = make_store()
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="expired", response="old", expires_at=past)
    e.refresh_state()
    store.store(e)
    assert store.list_all() == []
    assert len(store.list_all(include_expired=True)) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats():
    store = make_store()
    from agent_memory.models import MemoryEntry, MemoryType

    store.store(MemoryEntry(query="q1", response="r1", type=MemoryType.FACT))
    store.store(MemoryEntry(query="q2", response="r2", type=MemoryType.CONVERSATION))
    s = store.stats()
    assert s["total"] == 2
    assert "fact" in s["by_type"]
    assert "conversation" in s["by_type"]


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_marks_expired():
    store = make_store()
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="old", response="r", expires_at=past)
    e.refresh_state()
    store.store(e)
    result = store.cleanup_expired()
    assert result["expired"] >= 1


def test_cleanup_delete():
    store = make_store()
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
# Vector search (RediSearch / RedisVSS)
#
# These need a real server with the RediSearch module — Redis 8+ or Redis Stack.
# fakeredis has no vector index, so with it the store stays on the BM25 path
# (exercised by every test above). Point AGENT_MEMORY_TEST_REDIS_URL at a live
# instance to run them:
#
#   docker compose -f docker-compose.dev.yml up -d redis
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("AGENT_MEMORY_TEST_REDIS_URL", "redis://localhost:6379/9")

try:
    import redis as _redis

    _probe = _redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
    _probe.ping()
    _probe.execute_command("FT._LIST")
    REDISEARCH_AVAILABLE = True
except Exception:
    REDISEARCH_AVAILABLE = False

needs_redisearch = pytest.mark.skipif(
    not REDISEARCH_AVAILABLE, reason="no Redis with the RediSearch module at AGENT_MEMORY_TEST_REDIS_URL"
)


def fake_embedder(dim: int = 16):
    """Hashed bag-of-words embedder — no model download needed.

    Shared words pull vectors together, so exact matches rank first, which is all
    the index-mechanics tests need. crc32, not hash(): str hashing is salted per
    process, which would make the scores differ from run to run.
    """

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * dim
            for token in text.lower().split():
                vec[zlib.crc32(token.encode()) % dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    return embed


@pytest.fixture()
def vss_store():
    """A store on a real RediSearch instance, isolated by key prefix."""
    from agent_memory.redis_store import RedisMemoryStore

    client = _redis.from_url(REDIS_URL, decode_responses=True)
    prefix = f"amtest_{uuid4().hex[:8]}"
    store = RedisMemoryStore(
        redis_client=client, key_prefix=prefix, embedder=fake_embedder(), enable_embeddings=True
    )
    yield store
    client.execute_command("FT.DROPINDEX", f"{prefix}:idx", "DD")
    keys = client.keys(f"{prefix}:*")
    if keys:
        client.delete(*keys)


@needs_redisearch
def test_vss_enables_semantic_search(vss_store):
    assert vss_store.semantic_search_enabled is True
    shape = vss_store._index_shape()
    assert shape is not None
    assert shape.dim == 16
    # The same index serves keyword_search, so the text field must be there.
    assert shape.has_text is True


@needs_redisearch
def test_vss_knn_ranks_best_match_first(vss_store):
    from agent_memory.models import MemoryEntry

    vss_store.store(MemoryEntry(query="Python async programming", response="use asyncio"))
    vss_store.store(MemoryEntry(query="How to bake bread", response="flour and yeast"))

    results = vss_store.search("Python async programming", top_k=2)
    assert results
    assert results[0][0].query == "Python async programming"
    # Cosine similarity, not a BM25 score. It falls short of 1.0 even for an
    # exact query hit, because the indexed document is query + content + tags.
    assert results[0][1] > results[1][1]
    assert 0.0 <= results[0][1] <= 1.0


@needs_redisearch
def test_vss_respects_scope_filter(vss_store):
    from agent_memory.models import MemoryEntry, MemoryScope

    vss_store.store(MemoryEntry(query="user note", response="r", scope=MemoryScope.USER))
    vss_store.store(MemoryEntry(query="global note", response="r", scope=MemoryScope.GLOBAL))

    hits = vss_store.search("note", top_k=5, scopes=[MemoryScope.GLOBAL])
    assert [entry.scope for entry, _ in hits] == [MemoryScope.GLOBAL]


@needs_redisearch
def test_vss_excludes_archived_and_expired(vss_store):
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    archived = MemoryEntry(query="archived note", response="r", archived=True)
    expired = MemoryEntry(
        query="expired note", response="r", expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc)
    )
    expired.refresh_state()
    live = MemoryEntry(query="live note", response="r")
    for entry in (archived, expired, live):
        vss_store.store(entry)

    assert [e.id for e, _ in vss_store.search("note", top_k=5)] == [live.id]
    visible = {e.id for e, _ in vss_store.search("note", top_k=5, include_archived=True)}
    assert archived.id in visible


@needs_redisearch
def test_vss_delete_removes_vector(vss_store):
    from agent_memory.models import MemoryEntry

    entry = vss_store.store(MemoryEntry(query="temporary note", response="r"))
    assert vss_store._client.exists(vss_store._vec_key(entry.id))

    vss_store.delete(entry.id)
    assert not vss_store._client.exists(vss_store._vec_key(entry.id))
    assert vss_store.search("temporary note", top_k=5) == []


@needs_redisearch
def test_vss_backfills_entries_written_without_vectors(vss_store):
    """Reopening a pre-vector store embeds whatever is already in Redis."""
    from agent_memory.models import MemoryEntry
    from agent_memory.redis_store import RedisMemoryStore

    client = vss_store._client
    prefix = vss_store._prefix
    lexical = RedisMemoryStore(redis_client=client, key_prefix=prefix, enable_embeddings=False)
    entry = lexical.store(MemoryEntry(query="legacy note", response="written before vectors"))
    assert not client.exists(lexical._vec_key(entry.id))

    upgraded = RedisMemoryStore(
        redis_client=client, key_prefix=prefix, embedder=fake_embedder(), enable_embeddings=True
    )
    assert client.exists(upgraded._vec_key(entry.id))
    assert [e.id for e, _ in upgraded.search("legacy note", top_k=1)] == [entry.id]


@needs_redisearch
def test_vss_rebuilds_index_when_embedding_dim_changes(vss_store):
    from agent_memory.models import MemoryEntry
    from agent_memory.redis_store import RedisMemoryStore

    vss_store.store(MemoryEntry(query="dimension change", response="r"))

    rebuilt = RedisMemoryStore(
        redis_client=vss_store._client,
        key_prefix=vss_store._prefix,
        embedder=fake_embedder(dim=32),
        enable_embeddings=True,
    )
    assert rebuilt._index_shape().dim == 32
    # The old vectors were dropped with the index and re-embedded at the new dim.
    assert [e.query for e, _ in rebuilt.search("dimension change", top_k=1)] == [
        "dimension change"
    ]


@needs_redisearch
def test_vss_touch_does_not_re_embed(vss_store):
    """REPLAY's hot path must not pay for an embedding call."""
    from agent_memory.models import MemoryEntry
    from agent_memory.redis_store import RedisMemoryStore

    calls: list[int] = []

    def counting_embedder(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return fake_embedder()(texts)

    store = RedisMemoryStore(
        redis_client=vss_store._client,
        key_prefix=f"{vss_store._prefix}_touch",
        embedder=counting_embedder,
        enable_embeddings=True,
    )
    entry = store.store(MemoryEntry(query="hot path", response="r"))
    calls.clear()

    assert store.touch(entry.id) is True
    assert calls == []
    assert store.get(entry.id).access_count == 1
    assert store.touch("missing-id") is False


@needs_redisearch
def test_vss_requires_redisearch_when_forced():
    """enable_embeddings=True fails loudly on a server without the module."""
    from agent_memory.redis_store import RedisMemoryStore

    client = fakeredis.FakeRedis(decode_responses=True)
    with pytest.raises((RuntimeError, ImportError)):
        RedisMemoryStore(redis_client=client, enable_embeddings=True)


# ---------------------------------------------------------------------------
# Pure helpers — no server needed
# ---------------------------------------------------------------------------


def test_pack_float32_little_endian():
    import struct

    from agent_memory.redis_store import _pack_float32

    packed = _pack_float32([1.0, -2.0, 0.5])
    assert len(packed) == 12
    assert struct.unpack("<3f", packed) == (1.0, -2.0, 0.5)


def test_parse_search_reply_handles_resp2_and_resp3():
    from agent_memory.redis_store import RedisMemoryStore

    parse = RedisMemoryStore._parse_search_reply

    resp2 = [2, "p:vec:a", ["__dist", "0.25"], "p:vec:b", ["__dist", "0.75"]]
    assert parse(resp2) == [("p:vec:a", 0.25), ("p:vec:b", 0.75)]

    resp3 = {
        "results": [
            {"id": "p:vec:a", "extra_attributes": {"__dist": "0.25"}},
            {"id": "p:vec:b", "extra_attributes": {"__dist": "0.75"}},
        ]
    }
    assert parse(resp3) == [("p:vec:a", 0.25), ("p:vec:b", 0.75)]

    # A hit without the distance field is skipped rather than crashing.
    assert parse([1, "p:vec:a", ["other", "1"]]) == []
    assert parse([0]) == []


@needs_redisearch
def test_vss_falls_back_to_bm25_on_non_zero_db(caplog):
    """RediSearch cannot index db != 0; the store degrades instead of failing."""
    from agent_memory.models import MemoryEntry
    from agent_memory.redis_store import RedisMemoryStore

    # from_url ignores a db kwarg when the URL carries a path, so rewrite the path.
    other_db_url = REDIS_URL.rsplit("/", 1)[0] + "/9"
    client = _redis.from_url(other_db_url, decode_responses=True)
    assert client.connection_pool.connection_kwargs["db"] == 9
    prefix = f"amtest_{uuid4().hex[:8]}"
    store = RedisMemoryStore(
        redis_client=client, key_prefix=prefix, embedder=fake_embedder(), enable_embeddings="auto"
    )
    try:
        assert store.semantic_search_enabled is False
        store.store(MemoryEntry(query="Python async programming", response="use asyncio"))
        assert [e.query for e, _ in store.search("async programming", top_k=1)] == [
            "Python async programming"
        ]
    finally:
        keys = client.keys(f"{prefix}:*")
        if keys:
            client.delete(*keys)

    with pytest.raises(RuntimeError, match="db 0"):
        RedisMemoryStore(
            redis_client=client,
            key_prefix=prefix,
            embedder=fake_embedder(),
            enable_embeddings=True,
        )


@needs_redisearch
def test_vss_keyword_search_uses_the_index(vss_store):
    """The BM25 half is served by RediSearch, not by reading the whole corpus."""
    from agent_memory.models import MemoryEntry

    vss_store.store(MemoryEntry(query="machine learning algorithms", response="gradient descent"))
    vss_store.store(MemoryEntry(query="cooking recipes pasta", response="boil water"))

    results = vss_store.keyword_search("machine learning", top_k=5)
    assert [e.query for e, _ in results] == ["machine learning algorithms"]
    assert 0.0 < results[0][1] <= 1.0

    # OR over tokens: a query spanning both entries must return both.
    matched = {e.query for e, _ in vss_store.keyword_search("learning pasta", top_k=5)}
    assert matched == {"machine learning algorithms", "cooking recipes pasta"}

    assert vss_store.keyword_search("zzzz qqqq nonexistent", top_k=5) == []


@needs_redisearch
def test_vss_keyword_search_respects_filters(vss_store):
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry, MemoryScope

    vss_store.store(
        MemoryEntry(query="machine learning notes", response="r", scope=MemoryScope.GLOBAL)
    )
    vss_store.store(
        MemoryEntry(query="machine learning drafts", response="r", scope=MemoryScope.USER)
    )
    archived = MemoryEntry(query="machine learning archive", response="r", archived=True)
    vss_store.store(archived)
    expired = MemoryEntry(
        query="machine learning expired",
        response="r",
        expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )
    expired.refresh_state()
    vss_store.store(expired)

    visible = {e.query for e, _ in vss_store.keyword_search("machine learning", top_k=10)}
    assert visible == {"machine learning notes", "machine learning drafts"}

    scoped = vss_store.keyword_search(
        "machine learning", top_k=10, scopes=[MemoryScope.GLOBAL]
    )
    assert [e.query for e, _ in scoped] == ["machine learning notes"]

    with_archived = {
        e.query
        for e, _ in vss_store.keyword_search(
            "machine learning", top_k=10, include_archived=True
        )
    }
    assert "machine learning archive" in with_archived


@needs_redisearch
def test_index_without_text_field_is_rebuilt(vss_store):
    """An index built before keyword search was indexed must not be reused."""
    from agent_memory.models import MemoryEntry
    from agent_memory.redis_store import RedisMemoryStore

    client = vss_store._client
    prefix = vss_store._prefix
    vss_store.store(MemoryEntry(query="machine learning algorithms", response="r"))

    # Rebuild the index the way the pre-text-field version did: vector + tags only.
    client.execute_command("FT.DROPINDEX", f"{prefix}:idx")
    client.execute_command(
        "FT.CREATE", f"{prefix}:idx", "ON", "HASH", "PREFIX", "1", f"{prefix}:vec:",
        "SCHEMA", "scope", "TAG", "archived", "TAG",
        "embedding", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", "16",
        "DISTANCE_METRIC", "COSINE",
    )
    assert vss_store._index_shape().has_text is False

    upgraded = RedisMemoryStore(
        redis_client=client, key_prefix=prefix, embedder=fake_embedder(), enable_embeddings=True
    )
    assert upgraded._index_shape().has_text is True
    # Dropped with DD, so the entries were re-embedded and re-indexed from JSON.
    assert [e.query for e, _ in upgraded.keyword_search("machine learning", top_k=1)] == [
        "machine learning algorithms"
    ]


def test_parse_scored_reply_handles_resp2_and_resp3():
    from agent_memory.redis_store import RedisMemoryStore

    parse = RedisMemoryStore._parse_scored_reply

    assert parse([2, "p:vec:a", "1.5", "p:vec:b", "0.5"]) == [
        ("p:vec:a", 1.5),
        ("p:vec:b", 0.5),
    ]
    resp3 = {"results": [{"id": "p:vec:a", "score": 1.5}, {"id": "p:vec:b", "score": 0.5}]}
    assert parse(resp3) == [("p:vec:a", 1.5), ("p:vec:b", 0.5)]
    assert parse([0]) == []


@needs_redisearch
def test_vss_keyword_search_finds_terms_across_field_boundaries(vss_store):
    """RediSearch does not split on newlines, so the indexed text must not use them.

    search_document() joins query / content / tags with "\\n". Indexed verbatim,
    the words either side of each newline fuse into one term — so the last word of
    the query and the first word of the response both become unsearchable.
    """
    from agent_memory.models import MemoryEntry

    vss_store.store(
        MemoryEntry(
            query="machine learning algorithms",
            response="gradient descent",
            tags=["optimisation"],
        )
    )

    # "algorithms" ends the query, "gradient" starts the response, "optimisation"
    # is the tag — every one of them sits against a newline in the raw document.
    for term in ("machine", "algorithms", "gradient", "descent", "optimisation"):
        hits = vss_store.keyword_search(term, top_k=5)
        assert hits, f"{term!r} is not searchable"
        assert hits[0][0].query == "machine learning algorithms"
