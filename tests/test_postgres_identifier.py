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
