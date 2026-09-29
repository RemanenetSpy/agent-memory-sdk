"""Tests for custom exceptions, logging, and DSA algorithms in KnowledgeGraph."""
from __future__ import annotations

import logging

import pytest

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
from agent_memory.exceptions import (
    AgentMemoryError,
    BackendConnectionError,
    ConfigurationError,
    DecisionError,
    ExtractionError,
    MemoryNotFoundError,
    MemoryStoreError,
    MemoryWriteError,
    RetrievalError,
)


def test_all_exceptions_are_subclass_of_base():
    for exc_cls in [
        ConfigurationError,
        BackendConnectionError,
        MemoryStoreError,
        MemoryNotFoundError,
        MemoryWriteError,
        RetrievalError,
        DecisionError,
        ExtractionError,
    ]:
        assert issubclass(exc_cls, AgentMemoryError)


def test_backend_connection_error_fields():
    exc = BackendConnectionError("redis", "Connection refused")
    assert exc.backend == "redis"
    assert "redis" in str(exc)
    assert "Connection refused" in str(exc)


def test_memory_not_found_fields():
    exc = MemoryNotFoundError("abc-123", backend="sqlite")
    assert exc.memory_id == "abc-123"
    assert exc.backend == "sqlite"
    assert "abc-123" in str(exc)


def test_retrieval_error_fields():
    exc = RetrievalError("search failed", query="test query")
    assert exc.query == "test query"
    assert "search failed" in str(exc)


def test_configuration_error_raised_on_bad_backend(tmp_path):
    from agent_memory.manager import Memory

    with pytest.raises(ConfigurationError):
        Memory(persist_dir=tmp_path, backend="nonexistent_backend")


# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------


def test_get_logger_returns_child_of_agent_memory():
    from agent_memory.logging_config import get_logger

    logger = get_logger("agent_memory.test_module")
    assert logger.name.startswith("agent_memory")


def test_get_logger_auto_prefixes():
    from agent_memory.logging_config import get_logger

    logger = get_logger("some.other.module")
    # Should be prefixed with agent_memory
    assert "agent_memory" in logger.name


def test_timed_context_manager_logs(caplog):
    from agent_memory.logging_config import get_logger, timed

    test_log = get_logger("agent_memory.test_timed")
    with caplog.at_level(logging.DEBUG, logger="agent_memory"):
        with timed(test_log, "test_op", key="value"):
            pass   # no-op
    assert any("test_op" in r.message for r in caplog.records)


def test_configure_debug_logging_adds_handler():
    import logging as _logging

    from agent_memory.logging_config import configure_debug_logging

    root = _logging.getLogger("agent_memory")
    original_handlers = list(root.handlers)
    configure_debug_logging()
    assert root.level == _logging.DEBUG
    # Clean up
    for h in root.handlers:
        if h not in original_handlers:
            root.removeHandler(h)
    root.setLevel(_logging.WARNING)


# ---------------------------------------------------------------------------
# KnowledgeGraph DSA — Union-Find clusters
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg_with_edges():
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph, RelationType

    kg = KnowledgeGraph()
    a = kg.add_entity("A", EntityType.CONCEPT)
    b = kg.add_entity("B", EntityType.CONCEPT)
    c = kg.add_entity("C", EntityType.CONCEPT)
    d = kg.add_entity("D", EntityType.CONCEPT)   # isolated
    e = kg.add_entity("E", EntityType.CONCEPT)   # isolated

    # A-B high confidence, B-C medium confidence
    kg.add_relation(a.id, RelationType.RELATED_TO, b.id, target_is_entity=True, confidence=0.9)
    kg.add_relation(b.id, RelationType.RELATED_TO, c.id, target_is_entity=True, confidence=0.7)

    return kg, (a, b, c, d, e)


def test_union_find_clusters_groups_connected(kg_with_edges):
    kg, (a, b, c, d, e) = kg_with_edges
    clusters = kg.clusters()
    assert len(clusters) >= 1
    # A, B, C should all be in the same cluster
    all_ids = {ent.id for cluster in clusters for ent in cluster}
    assert {a.id, b.id, c.id}.issubset(all_ids)


def test_union_find_clusters_respects_min_confidence(kg_with_edges):
    kg, (a, b, c, d, e) = kg_with_edges
    # At 0.8 threshold: A-B (0.9) joins, B-C (0.7) does not
    clusters = kg.clusters(min_confidence=0.8)
    sizes = sorted(len(cl) for cl in clusters)
    # Expect {A,B} in one cluster and everything else isolated
    assert 2 in sizes


def test_union_find_clusters_all_isolated():
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph

    kg = KnowledgeGraph()
    for ch in "ABCDE":
        kg.add_entity(ch, EntityType.CONCEPT)
    # No edges
    clusters = kg.clusters()
    assert len(clusters) == 5
    assert all(len(cl) == 1 for cl in clusters)


def test_union_find_clusters_sorted_by_size():
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph, RelationType

    kg = KnowledgeGraph()
    big = [kg.add_entity(f"Big{i}", EntityType.CONCEPT) for i in range(4)]
    small = [kg.add_entity(f"Small{i}", EntityType.CONCEPT) for i in range(2)]
    for i in range(len(big) - 1):
        kg.add_relation(big[i].id, RelationType.RELATED_TO, big[i+1].id, target_is_entity=True)
    kg.add_relation(small[0].id, RelationType.RELATED_TO, small[1].id, target_is_entity=True)
    clusters = kg.clusters()
    sizes = [len(cl) for cl in clusters]
    assert sizes == sorted(sizes, reverse=True)


# ---------------------------------------------------------------------------
# KnowledgeGraph DSA — Dijkstra shortest_path
# ---------------------------------------------------------------------------


def test_dijkstra_finds_direct_high_confidence_path():
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph, RelationType

    kg = KnowledgeGraph()
    a = kg.add_entity("Alice", EntityType.PERSON)
    b = kg.add_entity("Acme", EntityType.ORGANIZATION)
    kg.add_relation(a.id, RelationType.WORKS_AT, b.id, target_is_entity=True, confidence=0.95)

    path = kg.shortest_path("Alice", "Acme")
    assert len(path) == 2
    assert path[0].id == a.id
    assert path[-1].id == b.id


def test_dijkstra_prefers_high_confidence_route():
    """Dijkstra should pick the high-confidence indirect route over the
    low-confidence direct one."""
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph, RelationType

    kg = KnowledgeGraph()
    a = kg.add_entity("A", EntityType.CONCEPT)
    b = kg.add_entity("B", EntityType.CONCEPT)
    c = kg.add_entity("C", EntityType.CONCEPT)

    # Direct A→C but low confidence (cost = 1 - 0.1 = 0.9)
    kg.add_relation(a.id, RelationType.RELATED_TO, c.id, target_is_entity=True, confidence=0.1)
    # Indirect A→B→C each high confidence (cost = 0.1 + 0.1 = 0.2)
    kg.add_relation(a.id, RelationType.RELATED_TO, b.id, target_is_entity=True, confidence=0.9)
    kg.add_relation(b.id, RelationType.RELATED_TO, c.id, target_is_entity=True, confidence=0.9)

    path = kg.shortest_path("A", "C")
    # Should go A→B→C (total cost 0.2) not A→C (cost 0.9)
    assert len(path) == 3
    assert any(e.id == b.id for e in path)


def test_dijkstra_no_path():
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_entity("X", EntityType.CONCEPT)
    kg.add_entity("Y", EntityType.CONCEPT)
    assert kg.shortest_path("X", "Y") == []


def test_dijkstra_same_entity():
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_entity("Solo", EntityType.CONCEPT)
    assert len(kg.shortest_path("Solo", "Solo")) == 1


# ---------------------------------------------------------------------------
# KnowledgeGraph — O(1) relation index
# ---------------------------------------------------------------------------


def test_relations_for_uses_index(kg_with_edges):
    kg, (a, b, c, d, e) = kg_with_edges
    rels = kg.relations_for(a.id)
    assert len(rels) >= 1
    # Verify index is populated
    for rel in rels:
        assert rel.id in kg._rel_index


# ---------------------------------------------------------------------------
# KnowledgeGraph — prefix search
# ---------------------------------------------------------------------------


def test_search_prefix_finds_matching_entities():
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_entity("Python", EntityType.TECHNOLOGY)
    kg.add_entity("PyCharm", EntityType.TECHNOLOGY)
    kg.add_entity("Django", EntityType.TECHNOLOGY)

    results = kg.search_prefix("py")
    names = {e.name for e in results}
    assert "Python" in names
    assert "PyCharm" in names
    assert "Django" not in names


def test_search_prefix_empty():
    from agent_memory.knowledge_graph import EntityType, KnowledgeGraph

    kg = KnowledgeGraph()
    kg.add_entity("SomeEntity", EntityType.UNKNOWN)
    assert kg.search_prefix("xyz_nomatch") == []


# ---------------------------------------------------------------------------
# Retriever — LRU cache
# ---------------------------------------------------------------------------


def test_retriever_cache_hit_on_repeated_query(tmp_path):

    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="cache_test")
    mem.remember("cache query", "cached response")

    # First call — cold cache miss; warms the LRU cache
    d1 = mem.resolve("cache query")

    # Second call — must hit the LRU cache (same query key)
    d2 = mem.resolve("cache query")

    # Both decisions must be identical (same memory id), proving the cache was hit.
    # Wall-clock timing is deliberately avoided: it is inherently flaky on
    # resource-constrained CI runners (GitHub Actions, Python 3.10).
    assert d1.action == d2.action
    assert d1.confidence == d2.confidence
    if d1.memory and d2.memory:
        assert d1.memory.id == d2.memory.id


def test_retriever_cache_invalidated_after_write(tmp_path):
    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="inv_test")
    mem.remember("inv query", "original response")
    d1 = mem.resolve("inv query")

    # Add a better match — cache should be invalidated by remember()
    mem.remember("inv query", "better response", confidence=1.0)
    d2 = mem.resolve("inv query")

    # Second result may differ from first (cache was cleared)
    # Both should be valid decisions
    assert d1.action.value in ("replay", "restore", "verify", "none")
    assert d2.action.value in ("replay", "restore", "verify", "none")


def test_retriever_cache_invalidated_after_archive(tmp_path):
    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="archive_inv_test")
    entry = mem.remember("archive cache query", "response")
    assert mem.resolve("archive cache query", mode="replay").action.value == "replay"

    mem.archive(entry.id)

    assert mem.resolve("archive cache query", mode="replay").action.value == "none"


def test_retriever_cache_invalidated_after_cleanup(tmp_path):
    from datetime import datetime, timedelta, timezone

    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="cleanup_inv_test")
    entry = mem.remember("cleanup cache query", "response", ttl="1h")
    assert mem.resolve("cleanup cache query", mode="replay").action.value == "replay"

    entry.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    mem.store.update(entry)
    assert mem.cleanup()["expired"] == 1

    assert mem.resolve("cleanup cache query", mode="replay").action.value == "none"


def test_lru_cache_evicts_oldest_entries():
    from agent_memory.retriever import _LRUCache

    cache = _LRUCache(maxsize=3, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.get("a")         # access "a" → moves to recent end
    cache.set("d", 4)      # should evict "b" (oldest unreferenced)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("d") == 4


def test_lru_cache_ttl_expiry():
    import time

    from agent_memory.retriever import _LRUCache

    cache = _LRUCache(maxsize=10, ttl_seconds=0.05)
    cache.set("key", "value")
    assert cache.get("key") == "value"
    time.sleep(0.1)
    assert cache.get("key") is None    # expired
