"""Tests for the Qdrant memory store backend.

Requires a running Qdrant instance. The URL is read from
``AGENT_MEMORY_TEST_QDRANT_URL`` (default: ``http://localhost:6333``):

    docker compose -f docker-compose.dev.yml up -d qdrant

The module is skipped when qdrant-client is not installed or the server is
unreachable. Each test gets its own collection, so nothing leaks between them.
"""
from __future__ import annotations

import os
import zlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

QDRANT_URL = os.environ.get("AGENT_MEMORY_TEST_QDRANT_URL", "http://localhost:6333")

try:
    from qdrant_client import QdrantClient

    _probe = QdrantClient(url=QDRANT_URL, timeout=3)
    _probe.get_collections()
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant not available")


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
def client():
    return QdrantClient(url=QDRANT_URL, timeout=30)


@pytest.fixture()
def collection_name(client):
    name = f"amtest_{uuid4().hex[:8]}"
    yield name
    if client.collection_exists(name):
        client.delete_collection(name)


@pytest.fixture()
def store(client, collection_name):
    from agent_memory.qdrant_store import QdrantMemoryStore

    return QdrantMemoryStore(
        client=client, collection_name=collection_name, embedder=fake_embedder()
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_store_and_get(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="hello qdrant", response="it works", tags=["a", "b"])
    stored = store.store(entry)
    fetched = store.get(stored.id)
    assert fetched is not None
    assert fetched.query == "hello qdrant"
    assert fetched.response == "it works"
    assert fetched.tags == ["a", "b"]


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
    entry.content = "new response"
    store.update(entry)
    assert store.get(entry.id).response == "new response"
    assert store.count == 1


def test_delete(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="delete me", response="gone")
    store.store(entry)
    assert store.delete(entry.id) is True
    assert store.get(entry.id) is None
    assert store.delete(entry.id) is False


def test_get_missing(store):
    assert store.get(str(uuid4())) is None


def test_non_uuid_ids_round_trip(store):
    """Qdrant only accepts UUID / int point IDs; a slug ID must still work."""
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(id="user-42:pref", query="slug id", response="r")
    store.store(entry)
    fetched = store.get("user-42:pref")
    assert fetched is not None
    assert fetched.id == "user-42:pref"
    assert store.delete("user-42:pref") is True


def test_metadata_round_trips(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(
        query="q", response="r", metadata={"session_id": "s1", "nested": {"n": 1}}
    )
    store.store(entry)
    assert store.get(entry.id).metadata == {"session_id": "s1", "nested": {"n": 1}}


# ---------------------------------------------------------------------------
# Listing and filtering
# ---------------------------------------------------------------------------


def test_list_scope_filter(store):
    from agent_memory.models import MemoryEntry, MemoryScope

    store.store(MemoryEntry(query="user q", response="r", scope=MemoryScope.USER))
    store.store(MemoryEntry(query="global q", response="r", scope=MemoryScope.GLOBAL))

    user_entries = store.list_all(scopes=[MemoryScope.USER])
    assert [e.scope for e in user_entries] == [MemoryScope.USER]
    assert len(store.list_all()) == 2


def test_list_type_filter(store):
    from agent_memory.models import MemoryEntry, MemoryType

    store.store(MemoryEntry(query="a fact", response="r", type=MemoryType.FACT))
    store.store(MemoryEntry(query="a chat", response="r"))

    assert [e.type for e in store.list_all(memory_type=MemoryType.FACT)] == [MemoryType.FACT]


def test_list_orders_by_updated_at_desc(store):
    from agent_memory.models import MemoryEntry

    old = MemoryEntry(query="older", response="r")
    old.updated_at = datetime.now(timezone.utc) - timedelta(days=1)
    store.store(old)
    store.store(MemoryEntry(query="newer", response="r"))

    assert [e.query for e in store.list_all()] == ["newer", "older"]
    assert [e.query for e in store.list_all(limit=1, offset=1)] == ["older"]


def test_list_archived_excluded_by_default(store):
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="archived", response="r", archived=True))
    assert store.list_all() == []
    assert len(store.list_all(include_archived=True)) == 1


# ---------------------------------------------------------------------------
# Search — dense KNN and BM25
# ---------------------------------------------------------------------------


def test_semantic_search_enabled(store):
    assert store.semantic_search_enabled is True


def test_vector_search_ranks_best_match_first(store):
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="Python async programming", response="use asyncio"))
    store.store(MemoryEntry(query="How to bake bread", response="flour and yeast"))

    results = store.search("Python async programming", top_k=2)
    assert results
    assert results[0][0].query == "Python async programming"
    assert results[0][1] > results[1][1]
    assert 0.0 <= results[0][1] <= 1.0


def test_vector_search_respects_scope_filter(store):
    from agent_memory.models import MemoryEntry, MemoryScope

    store.store(MemoryEntry(query="user note", response="r", scope=MemoryScope.USER))
    store.store(MemoryEntry(query="global note", response="r", scope=MemoryScope.GLOBAL))

    hits = store.search("note", top_k=5, scopes=[MemoryScope.GLOBAL])
    assert [e.scope for e, _ in hits] == [MemoryScope.GLOBAL]


def test_vector_search_excludes_archived_and_expired(store):
    from agent_memory.models import MemoryEntry

    archived = MemoryEntry(query="archived note", response="r", archived=True)
    expired = MemoryEntry(
        query="expired note", response="r", expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc)
    )
    expired.refresh_state()
    live = MemoryEntry(query="live note", response="r")
    for entry in (archived, expired, live):
        store.store(entry)

    assert [e.id for e, _ in store.search("note", top_k=5)] == [live.id]
    assert archived.id in {e.id for e, _ in store.search("note", top_k=5, include_archived=True)}
    assert expired.id in {e.id for e, _ in store.search("note", top_k=5, include_expired=True)}


def test_keyword_search_finds_match(store):
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="machine learning algorithms", response="gradient descent"))
    store.store(MemoryEntry(query="cooking recipes pasta", response="boil water"))

    results = store.keyword_search("machine learning", top_k=5)
    assert results
    assert results[0][0].query == "machine learning algorithms"
    assert results[0][1] > 0


def test_keyword_search_matches_any_token(store):
    """Full-text filtering must be OR, not AND, or partial queries find nothing."""
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="machine learning algorithms", response="gradient descent"))
    store.store(MemoryEntry(query="cooking recipes pasta", response="boil water"))

    matched = {e.query for e, _ in store.keyword_search("learning pasta", top_k=5)}
    assert matched == {"machine learning algorithms", "cooking recipes pasta"}


def test_keyword_search_respects_scope_filter(store):
    from agent_memory.models import MemoryEntry, MemoryScope

    store.store(
        MemoryEntry(query="machine learning notes", response="r", scope=MemoryScope.GLOBAL)
    )
    store.store(
        MemoryEntry(query="machine learning drafts", response="r", scope=MemoryScope.USER)
    )

    hits = store.keyword_search("machine learning", top_k=5, scopes=[MemoryScope.GLOBAL])
    assert [e.scope for e, _ in hits] == [MemoryScope.GLOBAL]


def test_keyword_search_empty_store(store):
    assert store.keyword_search("anything") == []


def test_keyword_search_no_match(store):
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="machine learning", response="r"))
    assert store.keyword_search("zzzz qqqq nonexistent") == []


def test_keyword_only_mode_falls_back(client, collection_name):
    """Without an embedding model, search() degrades to the BM25 path."""
    from agent_memory.models import MemoryEntry
    from agent_memory.qdrant_store import QdrantMemoryStore

    store = QdrantMemoryStore(
        client=client, collection_name=collection_name, enable_embeddings=False
    )
    assert store.semantic_search_enabled is False
    store.store(MemoryEntry(query="machine learning algorithms", response="r"))
    assert [e.query for e, _ in store.search("machine learning", top_k=2)] == [
        "machine learning algorithms"
    ]


# ---------------------------------------------------------------------------
# Usage counters
# ---------------------------------------------------------------------------


def test_touch_does_not_re_embed(client, collection_name):
    """REPLAY's hot path must not pay for an embedding call."""
    from agent_memory.models import MemoryEntry
    from agent_memory.qdrant_store import QdrantMemoryStore

    calls: list[int] = []

    def counting_embedder(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return fake_embedder()(texts)

    store = QdrantMemoryStore(
        client=client, collection_name=collection_name, embedder=counting_embedder
    )
    entry = store.store(MemoryEntry(query="hot path", response="r"))
    calls.clear()

    assert store.touch(entry.id) is True
    assert calls == []
    refreshed = store.get(entry.id)
    assert refreshed.access_count == 1
    assert refreshed.last_accessed_at is not None
    assert store.touch(str(uuid4())) is False


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


def test_expired_excluded_by_default(store):
    from agent_memory.models import MemoryEntry

    expired = MemoryEntry(
        query="expired", response="old", expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc)
    )
    expired.refresh_state()
    store.store(expired)
    store.store(MemoryEntry(query="never expires", response="r"))
    store.store(
        MemoryEntry(
            query="expires later",
            response="r",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )

    assert {e.query for e in store.list_all()} == {"never expires", "expires later"}
    assert len(store.list_all(include_expired=True)) == 3


def test_cleanup_marks_expired(store):
    from agent_memory.models import MemoryEntry, MemoryState

    entry = MemoryEntry(
        query="old", response="r", expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc)
    )
    store.store(entry)

    assert store.cleanup_expired()["expired"] >= 1
    assert store.get(entry.id).state == MemoryState.EXPIRED
    assert store.count == 1


def test_cleanup_delete(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(
        query="old", response="r", expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc)
    )
    entry.refresh_state()
    store.store(entry)
    store.store(MemoryEntry(query="keep me", response="r"))

    assert store.cleanup_expired(delete=True)["deleted"] == 1
    assert [e.query for e in store.list_all()] == ["keep me"]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats(store):
    from agent_memory.models import MemoryEntry, MemoryType

    store.store(MemoryEntry(query="q1", response="r1", type=MemoryType.FACT))
    store.store(MemoryEntry(query="q2", response="r2", type=MemoryType.CODE))
    archived = store.store(MemoryEntry(query="q3", response="r3", archived=True))
    store.touch(archived.id)

    stats = store.stats()
    assert stats["total"] == 3
    assert stats["by_type"] == {"fact": 1, "code": 1, "conversation": 1}
    assert stats["by_state"] == {"archived": 1, "active": 2}
    assert stats["total_access_count"] == 1


# ---------------------------------------------------------------------------
# Collection configuration
# ---------------------------------------------------------------------------


def test_rejects_mismatched_embedding_dimension(client, collection_name):
    from agent_memory.qdrant_store import QdrantMemoryStore

    QdrantMemoryStore(client=client, collection_name=collection_name, embedder=fake_embedder(16))

    with pytest.raises(ValueError, match="16-dim"):
        QdrantMemoryStore(
            client=client, collection_name=collection_name, embedder=fake_embedder(32)
        )


def test_payload_only_collection_stays_keyword_only(client, collection_name):
    """A collection created without vectors cannot suddenly gain KNN."""
    from agent_memory.qdrant_store import QdrantMemoryStore

    QdrantMemoryStore(client=client, collection_name=collection_name, enable_embeddings=False)

    reopened = QdrantMemoryStore(
        client=client, collection_name=collection_name, embedder=fake_embedder()
    )
    assert reopened.semantic_search_enabled is False


def test_requires_embedding_model_when_forced(client, collection_name, monkeypatch):
    import agent_memory.qdrant_store as qdrant_store
    from agent_memory.qdrant_store import QdrantMemoryStore

    monkeypatch.setattr(qdrant_store, "get_default_embedder", lambda: None)
    with pytest.raises(ImportError, match="embedding model"):
        QdrantMemoryStore(
            client=client, collection_name=collection_name, enable_embeddings=True
        )


# ---------------------------------------------------------------------------
# Pure helpers — no server needed
# ---------------------------------------------------------------------------


def test_point_id_passes_uuids_through_and_hashes_the_rest():
    from agent_memory.qdrant_store import QdrantMemoryStore

    point_id = QdrantMemoryStore._point_id
    generated = str(uuid4())
    assert point_id(generated) == generated

    slug = point_id("user-42:pref")
    assert slug != "user-42:pref"
    assert point_id("user-42:pref") == slug  # stable across calls
    assert point_id("user-43:pref") != slug
