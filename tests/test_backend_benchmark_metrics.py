"""Tests for the backend benchmark's metric definitions.

These pin the distinction the metric names claim to draw: Recall@k is a *rank*
cutoff over what retrieval found, while answer rate is what the decision layer did
at a *score* threshold. Conflating them once made a pure threshold effect look
like a retrieval difference between backends.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "backend_benchmark.py"


@pytest.fixture(scope="module")
def bench():
    """Import the benchmark script by path — it lives in scripts/, not the package."""
    spec = importlib.util.spec_from_file_location("backend_benchmark", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__], so the
    # module has to be registered before its body runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _entry(tags: list[str], query: str = "q"):
    from agent_memory.models import MemoryEntry

    return MemoryEntry(query=query, response="r", tags=tags)


# ---------------------------------------------------------------------------
# Relevance judgement
# ---------------------------------------------------------------------------


def test_relevance_is_a_tag_lookup(bench):
    assert bench._is_relevant(_entry(["auth", "password"]), "auth") is True
    assert bench._is_relevant(_entry(["billing"]), "auth") is False
    assert bench._is_relevant(_entry([]), "auth") is False


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


def test_recall_at_k_is_a_rank_cutoff_not_a_threshold(bench):
    """k counts ranks: a relevant hit at rank 3 counts for R@5 but not R@1."""
    labelled = [{"query": "q", "category": "auth", "expected": "hit"}]
    ranked = [_entry(["billing"]), _entry(["sdk"]), _entry(["auth"])]

    recall = bench._recall_at_k(lambda q, k: ranked[:k], labelled)
    assert recall[1] == 0.0    # rank 1 is irrelevant
    assert recall[5] == 1.0    # rank 3 is inside the top 5
    assert recall[10] == 1.0


def test_recall_at_k_counts_at_least_one_relevant_hit(bench):
    """Hit-rate semantics: one relevant memory in the top k is enough."""
    labelled = [{"query": "q", "category": "auth", "expected": "hit"}]
    # Only one of ten is relevant, yet R@5 is a full hit, not 1/10.
    ranked = [_entry(["auth"])] + [_entry(["other"]) for _ in range(9)]
    assert bench._recall_at_k(lambda q, k: ranked[:k], labelled)[5] == 1.0


def test_recall_at_k_averages_over_queries(bench):
    labelled = [
        {"query": "a", "category": "auth", "expected": "hit"},
        {"query": "b", "category": "billing", "expected": "hit"},
    ]

    def retrieve(query: str, k: int):
        return [_entry(["auth"])] if query == "a" else [_entry(["sdk"])]

    assert bench._recall_at_k(retrieve, labelled)[1] == 0.5


def test_recall_at_k_excludes_out_of_domain_queries(bench):
    """Out-of-domain queries have no relevant memory, so they cannot score recall."""
    labelled = [
        {"query": "a", "category": "auth", "expected": "hit"},
        {"query": "b", "category": bench._OUT_OF_DOMAIN, "expected": "none"},
    ]
    recall = bench._recall_at_k(lambda q, k: [_entry(["auth"])][:k], labelled)
    # 1/1 in-domain query, not 1/2 — the ood row is not in the denominator.
    assert recall[1] == 1.0


def test_recall_at_k_is_empty_without_in_domain_queries(bench):
    labelled = [{"query": "b", "category": bench._OUT_OF_DOMAIN, "expected": "none"}]
    assert bench._recall_at_k(lambda q, k: [], labelled) == {}


# ---------------------------------------------------------------------------
# Answer rate / refusal precision — decision-layer metrics
# ---------------------------------------------------------------------------


def test_quality_separates_answer_rate_from_refusal_precision(bench):
    quality = bench.Quality(
        answerable_total=10, answered=7, none_total=4, none_refused=4
    )
    assert quality.answer_rate == 0.7
    assert quality.refusal_precision == 1.0
    assert quality.accuracy == 11 / 14


def test_quality_metrics_are_zero_on_an_empty_query_set(bench):
    quality = bench.Quality()
    assert quality.answer_rate == 0.0
    assert quality.refusal_precision == 0.0
    assert quality.accuracy == 0.0


def test_quality_has_no_attribute_named_recall(bench):
    """Guards the naming: these count decisions, not retrieval ranks."""
    quality = bench.Quality()
    assert not hasattr(quality, "recall")
    assert not hasattr(quality, "precision")


# ---------------------------------------------------------------------------
# Ground truth in the shipped data
# ---------------------------------------------------------------------------


def test_labelled_query_set_supports_the_relevance_judgement(bench):
    """Every in-domain query category must be a tag some seeded memory carries."""
    data_dir = _SCRIPT.parent.parent / "benchmarks" / "stress"
    memories = bench._load_jsonl(data_dir / "memories.jsonl")
    queries = bench._load_jsonl(data_dir / "queries.jsonl")

    tags = {tag for m in memories for tag in m.get("tags", [])}
    categories = {
        row["category"] for row in queries if row["category"] != bench._OUT_OF_DOMAIN
    }
    assert categories <= tags, f"no memory carries tag(s): {sorted(categories - tags)}"


def test_out_of_domain_queries_are_exactly_the_unanswerable_ones(bench):
    """The ood category and the `none` label must agree, or both metrics skew."""
    data_dir = _SCRIPT.parent.parent / "benchmarks" / "stress"
    queries = bench._load_jsonl(data_dir / "queries.jsonl")

    ood = {row["query"] for row in queries if row["category"] == bench._OUT_OF_DOMAIN}
    expect_none = {row["query"] for row in queries if row["expected"] == "none"}
    assert ood == expect_none
