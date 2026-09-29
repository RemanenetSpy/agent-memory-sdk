"""Tests for the backend registry that Memory resolves `backend=` through."""
from __future__ import annotations

import pytest

from agent_memory.backends import (
    BackendContext,
    available_backends,
    build_store,
    register_backend,
)
from agent_memory.exceptions import ConfigurationError


@pytest.fixture()
def clean_registry():
    """Snapshot and restore the registry so a test cannot leak a backend."""
    import agent_memory.backends as backends

    original = dict(backends._REGISTRY)
    yield
    backends._REGISTRY.clear()
    backends._REGISTRY.update(original)


def test_every_shipped_backend_is_registered():
    assert available_backends() == ["chromadb", "postgres", "qdrant", "redis", "sqlite"]


def test_unknown_backend_raises_and_names_the_alternatives():
    with pytest.raises(ConfigurationError) as exc_info:
        build_store("annoy", BackendContext())

    message = str(exc_info.value)
    assert "annoy" in message
    for name in available_backends():
        assert name in message


def test_builds_the_default_backend(tmp_path):
    from agent_memory.sqlite_store import SqliteMemoryStore

    store = build_store(
        "sqlite", BackendContext(persist_dir=tmp_path, enable_embeddings=False)
    )
    assert isinstance(store, SqliteMemoryStore)


def test_context_defaults_do_not_require_arguments():
    ctx = BackendContext()
    assert ctx.collection_name == "agent_memories"
    assert ctx.enable_embeddings == "auto"
    assert ctx.embedder is None
    assert ctx.kwargs == {}


def test_register_makes_a_custom_backend_available_to_memory(tmp_path, clean_registry):
    """A third-party store plugs in without editing Memory."""
    from agent_memory import Memory
    from agent_memory.sqlite_store import SqliteMemoryStore

    seen: dict[str, object] = {}

    def factory(ctx: BackendContext) -> SqliteMemoryStore:
        seen["kwargs"] = dict(ctx.kwargs)
        seen["enable_embeddings"] = ctx.enable_embeddings
        return SqliteMemoryStore(
            persist_dir=tmp_path / "custom", enable_embeddings=False
        )

    register_backend("mystore", factory)
    assert "mystore" in available_backends()

    memory = Memory(backend="mystore", enable_embeddings=False, host="example.invalid")
    assert isinstance(memory.store, SqliteMemoryStore)
    # Unrecognised kwargs are handed to the factory rather than swallowed.
    assert seen["kwargs"] == {"host": "example.invalid"}
    assert seen["enable_embeddings"] is False

    memory.remember("registered backend", "works")
    assert memory.store.count == 1


def test_register_refuses_to_shadow_a_shipped_backend(clean_registry):
    with pytest.raises(ValueError, match="already registered"):
        register_backend("sqlite", lambda ctx: None)  # type: ignore[arg-type,return-value]


def test_register_can_replace_deliberately(tmp_path, clean_registry):
    from agent_memory.sqlite_store import SqliteMemoryStore

    marker = SqliteMemoryStore(persist_dir=tmp_path / "replaced", enable_embeddings=False)
    register_backend("sqlite", lambda ctx: marker, replace=True)
    assert build_store("sqlite", BackendContext()) is marker


def test_register_rejects_an_empty_name(clean_registry):
    with pytest.raises(ValueError, match="non-empty"):
        register_backend("", lambda ctx: None)  # type: ignore[arg-type,return-value]


def test_importing_agent_memory_does_not_import_optional_drivers():
    """Registry factories import lazily, so `import agent_memory` stays light."""
    import subprocess
    import sys

    code = (
        "import sys; import agent_memory; "
        "leaked = [m for m in ('psycopg2', 'qdrant_client', 'redis') if m in sys.modules]; "
        "print(leaked)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", f"optional drivers imported eagerly: {out.stdout}"
