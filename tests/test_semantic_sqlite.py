"""Tests for the optional sqlite-vec semantic backend.

Skipped automatically when the ``semantic`` extra (sqlite-vec + fastembed /
sentence-transformers) is not installed, so the base test suite stays light.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from agent_memory import Memory, MemoryAction  # noqa: E402
from agent_memory.embeddings import get_default_embedder  # noqa: E402

if get_default_embedder() is None:
    pytest.skip("no embedding model installed", allow_module_level=True)


@pytest.fixture
def semantic_memory(tmp_path: Path) -> Memory:
    # restore_threshold=0.55 — semantic embeddings score paraphrases lower than
    # lexical exact matches (bge-small-en-v1.5 gives ~0.52–0.65 for semantically
    # related but lexically different queries).  The default 0.70 is tuned for
    # the lexical backend where high scores are easy to achieve; the semantic
    # fixture uses a lower threshold to exercise the paraphrase-retrieval path.
    mem = Memory(
        persist_dir=tmp_path / "sem",
        enable_embeddings=True,
        restore_threshold=0.55,
    )
    assert mem.store.semantic_search_enabled  # type: ignore[union-attr]
    return mem


def test_paraphrase_with_no_shared_words(semantic_memory: Memory) -> None:
    """Vector search finds semantically related entries that share no keywords."""
    semantic_memory.remember(
        "How do I reset my password?",
        "Go to Settings → Security → Reset Password.",
    )
    decision = semantic_memory.resolve(
        "I can't remember my login credentials, help me regain access"
    )
    assert decision.action in (MemoryAction.RESTORE, MemoryAction.VERIFY), (
        f"Expected RESTORE or VERIFY but got {decision.action.value} "
        f"(confidence={decision.confidence:.2f}). "
        "The semantic backend should surface the password-reset memory for this paraphrase."
    )
    assert decision.context
    assert decision.context[0].entry.query == "How do I reset my password?"


def test_exact_match_still_replays(semantic_memory: Memory) -> None:
    semantic_memory.remember("What is the deploy freeze window?", "Fridays after 3pm UTC.")
    decision = semantic_memory.resolve("What is the deploy freeze window?")
    assert decision.action == MemoryAction.REPLAY
    assert decision.response == "Fridays after 3pm UTC."


def test_unrelated_query_returns_none(semantic_memory: Memory) -> None:
    semantic_memory.remember("What is the deploy freeze window?", "Fridays after 3pm UTC.")
    decision = semantic_memory.resolve("What is the capital city of Mongolia?")
    assert decision.action == MemoryAction.NONE


def test_vector_rows_follow_entry_lifecycle(semantic_memory: Memory) -> None:
    entry = semantic_memory.remember("lifecycle probe", "vector should be cleaned up")
    # Update re-embeds without violating the vec table's primary key.
    entry.response = "updated response"
    entry.content = "updated response"
    semantic_memory.store.update(entry)

    assert semantic_memory.forget(entry.id) is True
    decision = semantic_memory.resolve("lifecycle probe")
    assert decision.action == MemoryAction.NONE


def test_persists_and_backfills_across_instances(tmp_path: Path) -> None:
    persist = tmp_path / "reopen"
    m1 = Memory(persist_dir=persist, enable_embeddings=True)
    m1.remember("Where is the runbook for incident response?", "wiki/runbooks/incidents")

    m2 = Memory(persist_dir=persist, enable_embeddings=True)
    decision = m2.resolve("Where is the runbook for incident response?")
    assert decision.action == MemoryAction.REPLAY


def test_vector_backfill_uses_bounded_batches(tmp_path: Path) -> None:
    from agent_memory.sqlite_store import _VECTOR_BACKFILL_BATCH_SIZE

    persist = tmp_path / "batched-backfill"
    lexical = Memory(persist_dir=persist, enable_embeddings=False)
    for index in range(_VECTOR_BACKFILL_BATCH_SIZE + 1):
        lexical.remember(f"query {index}", f"response {index}")
    lexical.store.close()

    batches: list[int] = []

    def embed(texts: list[str]) -> list[list[float]]:
        batches.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    semantic = Memory(persist_dir=persist, enable_embeddings=True, embedder=embed)

    assert semantic.store.semantic_search_enabled  # type: ignore[union-attr]
    assert batches[0] == 1  # embedding_dimension() probe
    assert batches[1:] == [_VECTOR_BACKFILL_BATCH_SIZE, 1]


def test_scope_filtering_applies_after_knn(semantic_memory: Memory) -> None:
    semantic_memory.remember("team secret", "team data", scope="team")
    semantic_memory.remember("user note", "user data", scope="user")
    results = semantic_memory.store.search("team secret", top_k=5, scopes=None)
    assert results
    from agent_memory.models import MemoryScope

    team_only = semantic_memory.store.search(
        "team secret", top_k=5, scopes=[MemoryScope.USER]
    )
    assert all(entry.scope == MemoryScope.USER for entry, _ in team_only)
