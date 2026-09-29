"""Tests for the shared vector-index configuration.

No server needed — this is the injectable config every HNSW backend takes, plus
the per-engine defaults each store ships.
"""
from __future__ import annotations

import pytest

from agent_memory.vector_index import IvfFlatConfig, VectorIndexConfig

# ---------------------------------------------------------------------------
# Defaults and derivation
# ---------------------------------------------------------------------------


def test_defaults_are_usable_without_arguments():
    config = VectorIndexConfig()
    assert config.m == 16
    assert config.ef_construction == 200
    assert config.ef_search == 64
    assert config.overfetch == 4
    assert config.min_candidates == 20


def test_tuned_overrides_only_what_it_is_given():
    base = VectorIndexConfig()
    tuned = base.tuned(ef_search=512)
    assert tuned.ef_search == 512
    assert tuned.m == base.m
    assert tuned.ef_construction == base.ef_construction
    # The original is untouched — configs are shareable between stores.
    assert base.ef_search == 64


def test_config_is_frozen():
    config = VectorIndexConfig()
    with pytest.raises(Exception):  # dataclasses raises FrozenInstanceError
        config.m = 32  # type: ignore[misc]


@pytest.mark.parametrize(
    ("top_k", "expected"),
    [
        (1, 20),    # floor applies
        (5, 20),    # 5 * 4 == 20, ties the floor
        (10, 40),   # multiplier wins
        (100, 400),
    ],
)
def test_candidate_pool_applies_multiplier_then_floor(top_k, expected):
    assert VectorIndexConfig().candidate_pool(top_k) == expected


def test_candidate_pool_honours_custom_overfetch():
    config = VectorIndexConfig(overfetch=10, min_candidates=5)
    assert config.candidate_pool(10) == 100
    assert config.candidate_pool(1) == 10


# ---------------------------------------------------------------------------
# Validation — a bad knob must fail at construction, not at query time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": 0},
        {"ef_construction": 0},
        {"ef_search": -1},
        {"overfetch": 0},
        {"min_candidates": 0},
    ],
)
def test_rejects_non_positive_values(kwargs):
    with pytest.raises(ValueError, match="must be >= 1"):
        VectorIndexConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"m": 16.0}, {"ef_search": "64"}, {"overfetch": True}])
def test_rejects_non_int_values(kwargs):
    with pytest.raises(TypeError, match="must be an int"):
        VectorIndexConfig(**kwargs)


def test_tuned_revalidates():
    with pytest.raises(ValueError, match="must be >= 1"):
        VectorIndexConfig().tuned(ef_search=0)


# ---------------------------------------------------------------------------
# IVFFlat config
# ---------------------------------------------------------------------------


def test_ivfflat_probes_stay_below_lists():
    """probes == lists visits every cluster, which is a brute-force scan."""
    config = IvfFlatConfig()
    assert config.probes < config.lists


def test_ivfflat_defaults_and_validation():
    assert IvfFlatConfig() == IvfFlatConfig(lists=10, probes=3)
    assert IvfFlatConfig().tuned(lists=500).lists == 500

    with pytest.raises(ValueError, match="must be >= 1"):
        IvfFlatConfig(lists=0)
    with pytest.raises(TypeError, match="must be an int"):
        IvfFlatConfig(probes="10")


# ---------------------------------------------------------------------------
# Every vector backend exposes the config it is actually using
# ---------------------------------------------------------------------------


def test_each_backend_ships_its_engines_defaults():
    """Defaults are per-engine, so importing one store must not retune another."""
    from agent_memory.postgres_store import (
        DEFAULT_IVFFLAT_CONFIG,
    )
    from agent_memory.postgres_store import (
        DEFAULT_VECTOR_CONFIG as PG_DEFAULT,
    )
    from agent_memory.redis_store import DEFAULT_VECTOR_CONFIG as REDIS_DEFAULT

    assert REDIS_DEFAULT.ef_construction == 200      # RediSearch build setting
    assert PG_DEFAULT.ef_construction == 64          # pgvector's own default
    assert DEFAULT_IVFFLAT_CONFIG.lists == 10
    # All three agree on the graph out-degree — the one value every engine shares.
    assert REDIS_DEFAULT.m == PG_DEFAULT.m == 16


def test_qdrant_default_config_matches_its_engine():
    pytest.importorskip("qdrant_client")
    from agent_memory.qdrant_store import DEFAULT_VECTOR_CONFIG as QDRANT_DEFAULT

    assert QDRANT_DEFAULT.m == 16
    assert QDRANT_DEFAULT.ef_construction == 100
    # Qdrant filters server-side, so it can afford a deeper query-time list.
    assert QDRANT_DEFAULT.ef_search == 128


def test_redis_store_accepts_a_custom_config():
    """The config reaches the store even when no server is available."""
    fakeredis = pytest.importorskip("fakeredis")
    from agent_memory.redis_store import RedisMemoryStore

    config = VectorIndexConfig(m=32, ef_construction=400, ef_search=256)
    store = RedisMemoryStore(
        redis_client=fakeredis.FakeRedis(decode_responses=True),
        vector_config=config,
        enable_embeddings=False,
    )
    assert store.vector_config is config
