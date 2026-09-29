"""Tuning knobs for the vector indexes behind the server backends.

Redis, Postgres and Qdrant all build the same kind of index — an HNSW graph —
so the same handful of numbers describes every one of them. Keeping them in one
injectable dataclass means a caller tunes recall/latency the same way whichever
backend they run, and adding a fourth HNSW backend needs no new config type.

Each store ships the defaults its own engine recommends, so passing nothing keeps
the engine's own tuning::

    from agent_memory import VectorIndexConfig

    # Higher recall, slower queries.
    Memory(backend="qdrant", vector_config=VectorIndexConfig(ef_search=512))

    # Denser graph: slower to build, faster and more accurate to search.
    Memory(backend="redis", vector_config=VectorIndexConfig(m=32, ef_construction=400))
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

__all__ = ["IvfFlatConfig", "VectorIndexConfig"]


@dataclass(frozen=True)
class VectorIndexConfig:
    """HNSW build/query parameters and KNN over-fetch sizing.

    Attributes:
        m: Graph out-degree — how many neighbours each node links to. Higher
            means better recall and faster search, at the cost of index size and
            build time.
        ef_construction: Build-time candidate list. Higher builds a better graph
            (and so raises recall permanently) but slows every insert.
        ef_search: Query-time candidate list. The cheapest recall knob, because
            it costs nothing until a query runs — raise it when a filtered query
            returns fewer hits than it should.
        overfetch: Multiplier on ``top_k`` when the store has to filter results
            in Python after the index has already ranked them (expiry, mainly).
            Without it a full candidate pool of soon-to-be-filtered rows can
            leave fewer than ``top_k`` survivors.
        min_candidates: Floor on that pool, so a ``top_k=1`` lookup still gives
            the graph enough room to walk.

    Frozen so a config can be shared between stores without one mutating it
    under the other; use :meth:`tuned` to derive a variant.
    """

    m: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    overfetch: int = 4
    min_candidates: int = 20

    def __post_init__(self) -> None:
        for field_name in ("m", "ef_construction", "ef_search", "overfetch", "min_candidates"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"VectorIndexConfig.{field_name} must be an int, got {value!r}")
            if value < 1:
                raise ValueError(f"VectorIndexConfig.{field_name} must be >= 1, got {value}")

    def candidate_pool(self, top_k: int) -> int:
        """How many neighbours to ask the index for, to end up with *top_k*."""
        return max(top_k * self.overfetch, self.min_candidates)

    def tuned(self, **overrides: Any) -> VectorIndexConfig:
        """Return a copy with *overrides* applied, validated like a fresh one."""
        return replace(self, **overrides)


@dataclass(frozen=True)
class IvfFlatConfig:
    """IVFFlat parameters, for Postgres servers whose pgvector predates HNSW.

    IVFFlat partitions vectors into ``lists`` clusters and scans ``probes`` of
    them per query, so recall depends on both: too few lists and each is a linear
    scan, too few probes and the true neighbour's cluster goes unvisited.

    Attributes:
        lists: Number of clusters. pgvector's guidance is ``rows / 1000`` up to
            1M rows, then ``sqrt(rows)``. The default suits a small table; raise
            it as the corpus grows or each list becomes a linear scan.
        probes: Clusters scanned per query, following pgvector's ``sqrt(lists)``
            starting point. Recall rises with ``probes``, but at
            ``probes == lists`` every list is visited and the index degenerates
            into a brute-force scan — slower than HNSW with nothing to show for
            it. Keep it well below ``lists``.
    """

    lists: int = 10
    probes: int = 3

    def __post_init__(self) -> None:
        for field_name in ("lists", "probes"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"IvfFlatConfig.{field_name} must be an int, got {value!r}")
            if value < 1:
                raise ValueError(f"IvfFlatConfig.{field_name} must be >= 1, got {value}")

    def tuned(self, **overrides: Any) -> IvfFlatConfig:
        """Return a copy with *overrides* applied, validated like a fresh one."""
        return replace(self, **overrides)
