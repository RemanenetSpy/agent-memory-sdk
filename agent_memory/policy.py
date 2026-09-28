from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from agent_memory.logging_config import get_logger
from agent_memory.models import (
    VERIFY_TYPES,
    MemoryAction,
    MemoryEntry,
    RetrievalResult,
)

log = get_logger(__name__)


class DecisionPolicy(ABC):
    """Scoring and action-selection policy for memory resolution."""

    @abstractmethod
    def score(self, entry: MemoryEntry, semantic: float, keyword: float) -> float:
        ...

    @abstractmethod
    def select_action(
        self,
        results: list[RetrievalResult],
        *,
        replay_threshold: float,
        restore_threshold: float,
        verify_threshold: float,
    ) -> tuple[MemoryAction, float, str]:
        ...


class DefaultPolicy(DecisionPolicy):
    """
    Default policy combining:
      semantic score + recency + confidence + usage + optional graph importance
    """

    def __init__(
        self,
        semantic_weight: float = 0.55,
        recency_weight: float = 0.15,
        confidence_weight: float = 0.20,
        usage_weight: float = 0.10,
        recency_half_life_days: float = 30.0,
        usage_cap: int = 20,
        graph_weight: float = 0.0,
        min_relevance: float = 0.0,
        flat_crowd_relevance: float = 0.78,
        flat_crowd_margin: float = 0.0,
        replay_margin: float = 0.05,
        exact_relevance: float = 0.95,
    ) -> None:
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.confidence_weight = confidence_weight
        self.usage_weight = usage_weight
        self.recency_half_life_days = recency_half_life_days
        self.usage_cap = usage_cap
        self.graph_weight = graph_weight
        # Relevance gating.  The composite score includes components (recency,
        # confidence, usage) that are identical across candidates in a fresh
        # store, so they carry no evidence that a memory answers *this* query.
        # These gates apply to the discriminating signal only:
        #   min_relevance        — hard floor on hybrid relevance (0 = off;
        #                          absolute floors don't transfer across
        #                          backends, so this is opt-in tuning).
        #   flat_crowd_*         — mediocre relevance AND no separation from
        #                          the best *disagreeing* runner-up → topical
        #                          noise, not an answer → NONE.  Off by
        #                          default (margin=0): calibration on
        #                          LongMemEval_S showed lexical signals can't
        #                          separate "answer absent" from "answer
        #                          present" in dense conversational stores
        #                          without unacceptable false abstention
        #                          (see benchmarks/longmemeval/REPORT.md).
        #   replay_margin        — REPLAY demands the top hit stand out from
        #                          the runner-up; otherwise degrade to RESTORE.
        #   exact_relevance      — at/above this the match is (near-)exact and
        #                          replay_margin is bypassed: a dense store of
        #                          neighbours must not block a direct hit.
        self.min_relevance = min_relevance
        self.flat_crowd_relevance = flat_crowd_relevance
        self.flat_crowd_margin = flat_crowd_margin
        self.replay_margin = replay_margin
        self.exact_relevance = exact_relevance
        # Populated by Memory.refresh_graph_scores(); maps memory_id → normalized [0,1] score.
        self._graph_scores: dict[str, float] = {}

    def update_graph_scores(self, scores: dict[str, float]) -> None:
        """Replace the cached graph importance scores (normalized [0,1]).

        Call Memory.refresh_graph_scores() to build and inject these
        automatically.  Setting graph_weight=0 (the default) means this dict
        has no effect on scoring even when populated.
        """
        self._graph_scores = dict(scores)

    def score(
        self,
        entry: MemoryEntry,
        semantic: float,
        keyword: float,
        *,
        now: datetime | None = None,
    ) -> float:
        return self.score_breakdown(entry, semantic, keyword, now=now)["policy_score"]

    def score_breakdown(
        self,
        entry: MemoryEntry,
        semantic: float,
        keyword: float,
        *,
        now: datetime | None = None,
    ) -> dict[str, float]:
        """Compute per-component scores.

        *now* is shared across all entries in a single retrieve() call so
        recency is consistent and we avoid N ``datetime.now()`` syscalls — a
        simple but correct form of memoisation.
        """
        # Let the stronger retrieval channel dominate: embeddings score
        # paraphrases conservatively, keyword coverage scores them lexically —
        # solid evidence from either channel should carry the match.
        hybrid_semantic = max(
            0.7 * semantic + 0.3 * keyword,
            0.7 * keyword + 0.3 * semantic,
        )
        recency = self._recency_score(entry.updated_at, now=now)
        usage = min(entry.access_count, self.usage_cap) / self.usage_cap
        confidence = entry.confidence
        # Graph importance (normalized [0,1] PageRank).  When graph_weight > 0
        # the base weights are scaled by (1 - graph_weight) so the total score
        # remains in [0, 1] and the graph component has genuine discriminating
        # power rather than being swallowed by the min(1.0) clamp.
        graph_score = self._graph_scores.get(entry.id, 0.0) if self._graph_scores else 0.0
        if self.graph_weight > 0:
            base_scale = 1.0 - self.graph_weight
            policy_score = (
                base_scale * self.semantic_weight * hybrid_semantic
                + base_scale * self.recency_weight * recency
                + base_scale * self.confidence_weight * confidence
                + base_scale * self.usage_weight * usage
                + self.graph_weight * graph_score
            )
        else:
            policy_score = (
                self.semantic_weight * hybrid_semantic
                + self.recency_weight * recency
                + self.confidence_weight * confidence
                + self.usage_weight * usage
            )
        return {
            "semantic_score": hybrid_semantic,
            "keyword_score": keyword,
            "recency_score": recency,
            "confidence_score": confidence,
            "usage_score": usage,
            "graph_score": graph_score,
            "policy_score": policy_score,
            "final_score": policy_score,
        }

    def select_action(
        self,
        results: list[RetrievalResult],
        *,
        replay_threshold: float,
        restore_threshold: float,
        verify_threshold: float,
    ) -> tuple[MemoryAction, float, str]:
        best = results[0]
        score = best.decision_score
        entry = best.entry

        relevance = max(
            0.7 * best.semantic_score + 0.3 * best.keyword_score,
            0.7 * best.keyword_score + 0.3 * best.semantic_score,
        )
        # Margin against the best *disagreeing* runner-up.  A runner-up with
        # the same response corroborates the top hit (duplicate memory) —
        # only a close competitor with a different answer signals ambiguity.
        best_response = (best.entry.response or "").strip().lower()
        margin = 1.0
        for other in results[1:]:
            if (other.entry.response or "").strip().lower() != best_response:
                margin = score - other.decision_score
                break

        # Relevance gates: composite score can clear a threshold on
        # non-discriminating components alone (fresh store → recency and
        # confidence are ~identical for every candidate).  Require the
        # query-relevance signal itself to justify using memory.
        if self.min_relevance > 0 and relevance < self.min_relevance:
            return (
                MemoryAction.NONE,
                score,
                "Relevance below gate — matches are topical, not answering.",
            )
        if (
            self.flat_crowd_margin > 0
            and relevance < self.flat_crowd_relevance
            and margin < self.flat_crowd_margin
        ):
            return (
                MemoryAction.NONE,
                score,
                "Flat crowd — no candidate stands out; treating as no match.",
            )

        # An explicit requires_verification flag outranks replay: the caller
        # marked this memory as needing validation before any reuse.
        if entry.requires_verification and score >= restore_threshold:
            return (
                MemoryAction.VERIFY,
                score,
                "Memory is flagged requires_verification — validate before reuse.",
            )

        if score >= replay_threshold:
            if margin < self.replay_margin and relevance < self.exact_relevance:
                return (
                    MemoryAction.RESTORE,
                    score,
                    "High score but runner-up is close — restoring as context "
                    "instead of replaying verbatim.",
                )
            return (
                MemoryAction.REPLAY,
                score,
                "High composite score — replaying stored response.",
            )

        if score >= restore_threshold:
            if self._should_verify(entry, score, verify_threshold):
                return (
                    MemoryAction.VERIFY,
                    score,
                    "Moderate score on freshness-sensitive memory — verify before reuse.",
                )
            return (
                MemoryAction.RESTORE,
                score,
                "Moderate score — restoring memory as context for synthesis.",
            )

        return (
            MemoryAction.NONE,
            score,
            "Low composite score — no memory applied.",
        )

    def _should_verify(self, entry: MemoryEntry, score: float, verify_threshold: float) -> bool:
        if entry.requires_verification:
            return True
        if entry.type.value in VERIFY_TYPES and score < verify_threshold:
            return True
        age_days = (datetime.now(timezone.utc) - entry.updated_at).total_seconds() / 86400
        if entry.type.value in VERIFY_TYPES and age_days > self.recency_half_life_days:
            return True
        return False

    def _recency_score(
        self, updated_at: datetime, *, now: datetime | None = None
    ) -> float:
        _now = now or datetime.now(timezone.utc)
        ts = updated_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (_now - ts).total_seconds() / 86400)
        return math.exp(-0.693 * age_days / self.recency_half_life_days)
