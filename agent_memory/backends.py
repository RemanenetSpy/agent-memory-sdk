"""Backend registry — maps a backend name to the factory that builds its store.

:class:`~agent_memory.manager.Memory` used to pick a store with an if/elif chain,
which meant every new backend edited the constructor and its error message. Here
a backend is a registry entry instead, so :class:`Memory` is closed for
modification and third parties can plug in a store the SDK has never heard of::

    from agent_memory.backends import register_backend

    register_backend("mystore", lambda ctx: MyStore(**ctx.kwargs))
    Memory(backend="mystore", host="…")

Factories are imported lazily inside each function so that an optional
dependency is only required by the backend that actually needs it — importing
``agent_memory`` must never pull in psycopg2 or qdrant-client.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_memory.exceptions import ConfigurationError
from agent_memory.store import MemoryStore

__all__ = [
    "BackendContext",
    "StoreFactory",
    "available_backends",
    "build_store",
    "register_backend",
]


@dataclass(frozen=True)
class BackendContext:
    """Everything a store factory may need, normalised across backends.

    Local stores use *persist_dir* / *collection_name*; server stores ignore them
    and read their connection settings out of *kwargs*. Both get the embedding
    settings, so any backend can be handed a custom embedder.
    """

    persist_dir: str | Path = ".agent_memory"
    collection_name: str = "agent_memories"
    embedder: Any | None = None
    enable_embeddings: bool | str = "auto"
    kwargs: dict[str, Any] = field(default_factory=dict)


StoreFactory = Callable[[BackendContext], MemoryStore]


def _sqlite(ctx: BackendContext) -> MemoryStore:
    from agent_memory.sqlite_store import SqliteMemoryStore

    return SqliteMemoryStore(
        persist_dir=ctx.persist_dir,
        collection_name=ctx.collection_name,
        embedder=ctx.embedder,
        enable_embeddings=ctx.enable_embeddings,
    )


def _chromadb(ctx: BackendContext) -> MemoryStore:
    from agent_memory.store import ChromaDBStore

    # Chroma embeds documents itself, so it takes neither embedder nor toggle.
    return ChromaDBStore(
        persist_dir=ctx.persist_dir, collection_name=ctx.collection_name
    )


def _redis(ctx: BackendContext) -> MemoryStore:
    from agent_memory.redis_store import RedisMemoryStore

    return RedisMemoryStore(
        embedder=ctx.embedder, enable_embeddings=ctx.enable_embeddings, **ctx.kwargs
    )


def _postgres(ctx: BackendContext) -> MemoryStore:
    from agent_memory.postgres_store import PostgresMemoryStore

    return PostgresMemoryStore(
        embedder=ctx.embedder, enable_embeddings=ctx.enable_embeddings, **ctx.kwargs
    )


def _qdrant(ctx: BackendContext) -> MemoryStore:
    from agent_memory.qdrant_store import QdrantMemoryStore

    return QdrantMemoryStore(
        embedder=ctx.embedder, enable_embeddings=ctx.enable_embeddings, **ctx.kwargs
    )


_REGISTRY: dict[str, StoreFactory] = {
    "sqlite": _sqlite,
    "chromadb": _chromadb,
    "redis": _redis,
    "postgres": _postgres,
    "qdrant": _qdrant,
}


def register_backend(name: str, factory: StoreFactory, *, replace: bool = False) -> None:
    """Register *factory* under *name* for use as ``Memory(backend=name)``.

    Refuses to shadow an existing name unless *replace* is True, so a typo in a
    plugin cannot silently redirect everyone's ``backend="sqlite"``.
    """
    if not name:
        raise ValueError("backend name must be a non-empty string")
    if name in _REGISTRY and not replace:
        raise ValueError(
            f"backend {name!r} is already registered; pass replace=True to override it"
        )
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    """Registered backend names, sorted for stable error messages and docs."""
    return sorted(_REGISTRY)


def build_store(backend: str, ctx: BackendContext) -> MemoryStore:
    """Build the store for *backend*, or raise ConfigurationError if unknown."""
    try:
        factory = _REGISTRY[backend]
    except KeyError:
        choices = ", ".join(repr(name) for name in available_backends())
        raise ConfigurationError(
            f"Unknown backend: {backend!r}. Choices: {choices}"
        ) from None
    return factory(ctx)
