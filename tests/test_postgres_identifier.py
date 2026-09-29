"""PostgreSQL configuration validation that does not require a database."""

from __future__ import annotations

import pytest

psycopg2 = pytest.importorskip("psycopg2")


@pytest.mark.parametrize(
    "table_name",
    [
        "memories; DROP TABLE users;--",
        "public.memories",
        'memories"; SELECT 1;--',
        "a" * 46,
    ],
)
def test_postgres_rejects_unsafe_table_names_before_connect(table_name, monkeypatch) -> None:
    from agent_memory.postgres_store import PostgresMemoryStore

    def unexpected_connect(*args, **kwargs):
        pytest.fail("invalid table name must be rejected before connecting")

    monkeypatch.setattr(psycopg2, "connect", unexpected_connect)

    with pytest.raises(ValueError, match="table_name must be a simple SQL identifier"):
        PostgresMemoryStore(table_name=table_name)


def test_postgres_rejects_unknown_vector_index_before_connect(monkeypatch) -> None:
    from agent_memory.postgres_store import PostgresMemoryStore

    def unexpected_connect(*args, **kwargs):
        pytest.fail("invalid vector_index must be rejected before connecting")

    monkeypatch.setattr(psycopg2, "connect", unexpected_connect)

    with pytest.raises(ValueError, match="vector_index must be"):
        PostgresMemoryStore(vector_index="annoy")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.8.6", (0, 8, 6)),
        ("0.7.0", (0, 7, 0)),
        ("0.6", (0, 6)),
        ("1.0.0-rc1", (1, 0, 0)),
        ("0.5.1dev", (0, 5, 1)),
        ("", ()),
        ("unknown", ()),
    ],
)
def test_parses_pgvector_extversion(raw, expected) -> None:
    from agent_memory.postgres_store import _parse_pgvector_version

    assert _parse_pgvector_version(raw) == expected


def test_hnsw_floor_matches_the_documented_threshold() -> None:
    """Auto-detection must not silently drift off the 0.7.0 floor."""
    from agent_memory.postgres_store import _HNSW_MIN_PGVECTOR, _parse_pgvector_version

    assert _HNSW_MIN_PGVECTOR == (0, 7, 0)
    assert _parse_pgvector_version("0.6.2") < _HNSW_MIN_PGVECTOR
    assert _parse_pgvector_version("0.7.0") >= _HNSW_MIN_PGVECTOR
    assert _parse_pgvector_version("0.8.6") >= _HNSW_MIN_PGVECTOR


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("machine learning", "machine | learning"),
        ("How do I reset my password?", "reset | password"),
        ("learning learning algorithms", "learning | algorithms"),
        # All stop words: falls back to the raw tokens rather than giving up, the
        # same fallback the SQLite, Redis and Qdrant keyword paths use. Postgres's
        # own english config then strips them, so the query matches nothing.
        ("a the of", "the | of"),
        # Nothing longer than one character survives tokenisation.
        ("a b", None),
        ("", None),
    ],
)
def test_builds_or_tsquery_from_content_terms(query, expected) -> None:
    from agent_memory.postgres_store import PostgresMemoryStore

    assert PostgresMemoryStore._or_tsquery(query) == expected


def test_or_tsquery_cannot_inject_tsquery_operators() -> None:
    """Terms come from \\w+ tokenisation, so operators never survive."""
    from agent_memory.postgres_store import PostgresMemoryStore

    built = PostgresMemoryStore._or_tsquery("password & (secret | admin):*")
    assert built is not None
    assert set(built.split(" | ")) == {"password", "secret", "admin"}
