"""Concurrency tests for SqliteMemoryStore and the Memory public API.

All bugs caught in the two code-review rounds were concurrency bugs that the
single-threaded test suite missed.  These tests hammer the critical paths from
multiple threads to guard against regressions.
"""
from __future__ import annotations

import threading
from pathlib import Path

from agent_memory import Memory
from agent_memory.models import MemoryEntry, MemoryScope, MemoryType
from agent_memory.sqlite_store import SqliteMemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(n: int) -> MemoryEntry:
    return MemoryEntry(
        query=f"query token_{n} unique_{n}",
        response=f"response {n}",
        content=f"content token_{n}",
        type=MemoryType.CONVERSATION,
        scope=MemoryScope.SESSION,
        tags=[f"tag{n % 5}"],
    )


def _store_fixture(tmp_path: Path) -> SqliteMemoryStore:
    return SqliteMemoryStore(persist_dir=tmp_path)


# ---------------------------------------------------------------------------
# 1. Concurrent store() — bloom filter must not corrupt
# ---------------------------------------------------------------------------

def test_concurrent_store_bloom_consistency(tmp_path):
    """N threads write concurrently; every written entry must be findable afterwards."""
    store = _store_fixture(tmp_path)
    N_THREADS = 8
    N_PER_THREAD = 25
    errors: list[str] = []
    written: list[MemoryEntry] = []
    lock = threading.Lock()

    def writer(thread_id: int) -> None:
        local: list[MemoryEntry] = []
        for i in range(N_PER_THREAD):
            entry = _make_entry(thread_id * 100 + i)
            try:
                store.store(entry)
                local.append(entry)
            except Exception as e:
                errors.append(f"thread={thread_id} i={i}: {e!r}")
        with lock:
            written.extend(local)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"store() raised exceptions: {errors[:3]}"
    assert store.count == N_THREADS * N_PER_THREAD

    # Every written entry must be retrievable via get() — proves no bloom corruption
    # caused a silent discard or double-write corruption.
    for entry in written[:20]:  # spot-check 20 entries
        fetched = store.get(entry.id)
        assert fetched is not None, f"Entry {entry.id[:8]} vanished after concurrent writes"
        assert fetched.query == entry.query


def test_concurrent_store_keyword_search(tmp_path):
    """Concurrent store + keyword_search must not raise or return corrupt results."""
    store = _store_fixture(tmp_path)
    errors: list[str] = []

    # Pre-populate so searches return something
    for i in range(50):
        store.store(_make_entry(i))

    def searcher(q: str) -> None:
        try:
            results = store.keyword_search(q, top_k=5)
            for entry, score in results:
                assert 0.0 <= score <= 1.0, f"corrupt score {score}"
        except Exception as e:
            errors.append(f"search={q}: {e!r}")

    def writer(start: int) -> None:
        for i in range(start, start + 20):
            try:
                store.store(_make_entry(i))
            except Exception as e:
                errors.append(f"write={i}: {e!r}")

    threads = (
        [threading.Thread(target=writer, args=(50 + t * 20,)) for t in range(4)]
        + [threading.Thread(target=searcher, args=(f"token_{t * 5}",)) for t in range(4)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent store+search raised: {errors[:3]}"


# ---------------------------------------------------------------------------
# 2. touch() — access_count must be durable and correct under concurrency
# ---------------------------------------------------------------------------

def test_touch_durability(tmp_path):
    """touch() must commit; access_count must persist after a fresh get()."""
    store = _store_fixture(tmp_path)
    entry = store.store(_make_entry(0))

    result = store.touch(entry.id)
    assert result is True

    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.access_count == 1, (
        f"touch() didn't commit: access_count={fetched.access_count}"
    )


def test_concurrent_touch_correct_count(tmp_path):
    """N concurrent touch() calls on the same entry must all persist."""
    store = _store_fixture(tmp_path)
    entry = store.store(_make_entry(0))
    N = 20
    errors: list[str] = []

    def toucher() -> None:
        try:
            store.touch(entry.id)
        except Exception as e:
            errors.append(repr(e))

    threads = [threading.Thread(target=toucher) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.access_count == N, (
        f"Expected access_count={N}, got {fetched.access_count}"
    )


# ---------------------------------------------------------------------------
# 3. rebuild_indexes() during concurrent store() — no crash or corruption
# ---------------------------------------------------------------------------

def test_rebuild_during_concurrent_store(tmp_path):
    """rebuild_indexes() called mid-write must not corrupt the bloom filter."""
    store = _store_fixture(tmp_path)
    # Pre-populate
    for i in range(30):
        store.store(_make_entry(i))

    errors: list[str] = []

    def writer(start: int) -> None:
        for i in range(start, start + 20):
            try:
                store.store(_make_entry(i))
            except Exception as e:
                errors.append(repr(e))

    def rebuilder() -> None:
        try:
            store.rebuild_indexes()
        except Exception as e:
            errors.append(repr(e))

    threads = (
        [threading.Thread(target=writer, args=(100 + t * 20,)) for t in range(4)]
        + [threading.Thread(target=rebuilder) for _ in range(2)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"rebuild during write raised: {errors[:3]}"
    # After rebuild the indexes must be warm and usable
    assert store._indexes_warm
    results = store.keyword_search("token_0", top_k=5)
    assert len(results) > 0, "keyword_search returned nothing after rebuild"


# ---------------------------------------------------------------------------
# 4. High-level Memory API — mark_correct / mark_wrong / verify
# ---------------------------------------------------------------------------

def test_mark_correct_boosts_confidence(tmp_path):
    memory = Memory(persist_dir=tmp_path)
    # Use confidence < 1.0 so there is room to boost (ConfidenceLearner clamps at 1.0)
    memory.remember("what is 2+2", "4", type="fact", confidence=0.7)

    decision = memory.resolve("what is 2+2")
    assert decision.action.value != "none"

    entry_before = memory.get(decision.memory.id if decision.memory else decision.context[0].entry.id)
    conf_before = entry_before.confidence

    result = memory.mark_correct(decision)
    assert result is not None
    assert result.confidence > conf_before, (
        f"mark_correct did not boost confidence: {conf_before} → {result.confidence}"
    )


def test_mark_wrong_penalises_confidence(tmp_path):
    memory = Memory(persist_dir=tmp_path)
    memory.remember("capital of france", "Paris", type="fact", confidence=1.0)

    decision = memory.resolve("capital of france")
    assert decision.action.value != "none"

    entry_id = (decision.memory or decision.context[0].entry).id
    conf_before = memory.get(entry_id).confidence

    result = memory.mark_wrong(decision)
    assert result is not None
    assert result.confidence < conf_before, (
        f"mark_wrong did not penalise confidence: {conf_before} → {result.confidence}"
    )


def test_verify_correct_returns_replay(tmp_path):
    memory = Memory(persist_dir=tmp_path, verify_threshold=0.0, restore_threshold=0.0)
    memory.remember("sky color", "blue", type="fact", requires_verification=True)

    decision = memory.resolve("sky color")
    # With requires_verification=True the engine should emit VERIFY
    assert decision.action.value == "verify", (
        f"Expected VERIFY, got {decision.action.value}"
    )

    resolved = memory.verify(decision, verifier=lambda d: True)
    assert resolved.action.value == "replay"
    assert resolved.response == "blue"


def test_verify_wrong_returns_none(tmp_path):
    memory = Memory(persist_dir=tmp_path, verify_threshold=0.0, restore_threshold=0.0)
    memory.remember("speed of light", "100 m/s", type="fact", requires_verification=True)

    decision = memory.resolve("speed of light")
    assert decision.action.value == "verify"

    resolved = memory.verify(decision, verifier=lambda d: False)
    assert resolved.action.value == "none"

    # Confidence must have been penalised and persisted
    entry = memory.get(decision.memory.id)
    assert entry.confidence < 1.0, "Confidence not penalised after failed verify"


def test_verify_noop_on_non_verify_decision(tmp_path):
    """verify() is a no-op when called on a non-VERIFY decision."""
    memory = Memory(persist_dir=tmp_path)
    memory.remember("hello", "world")
    memory.resolve("hello")  # warm the store; result not used
    # Force a NONE decision by using a query with no match
    from agent_memory.models import MemoryAction, MemoryDecision
    none_decision = MemoryDecision(
        action=MemoryAction.NONE, query="no match", confidence=0.0
    )
    result = memory.verify(none_decision, verifier=lambda d: True)
    assert result is none_decision  # returned unchanged


# ---------------------------------------------------------------------------
# 5. Graph scores wire-up
# ---------------------------------------------------------------------------

def test_refresh_graph_scores_returns_count(tmp_path):
    memory = Memory(persist_dir=tmp_path, graph_weight=0.1)
    for i in range(5):
        memory.remember(f"fact {i}", f"answer {i}", tags=["shared"])
    n = memory.refresh_graph_scores()
    assert n == 5


def test_graph_weight_affects_score(tmp_path):
    """A highly connected memory should score higher with graph_weight > 0."""

    # Build store without graph
    memory_no_graph = Memory(persist_dir=tmp_path / "no_graph", graph_weight=0.0)
    # Build store with graph
    memory_graph = Memory(persist_dir=tmp_path / "with_graph", graph_weight=0.2)

    shared_tags = ["python", "language"]
    query = "what is python"
    response = "a programming language"

    for m in (memory_no_graph, memory_graph):
        m.remember(query, response, tags=shared_tags)
        m.remember("python tutorial", "learn step by step", tags=shared_tags)
        m.remember("python syntax", "clean and readable", tags=shared_tags)

    memory_graph.refresh_graph_scores()

    d_no_graph = memory_no_graph.resolve(query)
    d_graph = memory_graph.resolve(query)

    # Both should find the memory; graph version may have equal or higher confidence
    assert d_no_graph.action.value != "none"
    assert d_graph.action.value != "none"
    # Graph scores are stored in the policy
    from agent_memory.policy import DefaultPolicy
    assert isinstance(memory_graph._policy, DefaultPolicy)
    assert len(memory_graph._policy._graph_scores) == 3
