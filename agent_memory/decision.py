from __future__ import annotations

import time
from collections.abc import Callable

from agent_memory.exceptions import DecisionError
from agent_memory.explain import enrich_decision
from agent_memory.logging_config import get_logger
from agent_memory.models import (
    MemoryAction,
    MemoryDecision,
    MemoryEntry,
    MemoryScope,
    RetrievalResult,
)
from agent_memory.policy import DecisionPolicy, DefaultPolicy
from agent_memory.retriever import MemoryRetriever

log = get_logger(__name__)


class DecisionEngine:
    """
    Decides whether the agent should replay, restore, verify, or ignore memory.
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        policy: DecisionPolicy | None = None,
        replay_threshold: float = 0.85,
        restore_threshold: float = 0.70,
        verify_threshold: float = 0.80,
    ) -> None:
        self._retriever = retriever
        self._policy = policy or DefaultPolicy()
        self.replay_threshold = replay_threshold
        self.restore_threshold = restore_threshold
        self.verify_threshold = verify_threshold

    def decide(
        self,
        query: str,
        *,
        mode: str = "auto",
        top_k: int = 3,
        scopes: list[MemoryScope] | None = None,
        enable_verify: bool = True,
        where: Callable[[MemoryEntry], bool] | None = None,
    ) -> MemoryDecision:
        t0 = time.perf_counter()
        try:
            results = self._retriever.retrieve(query, top_k=top_k, scopes=scopes, where=where)
        except Exception as exc:
            raise DecisionError(f"Retrieval failed during decide: {exc}") from exc

        if not results:
            log.debug("decide  NONE  query=%r  reason='no memories'", query)
            return MemoryDecision(
                action=MemoryAction.NONE,
                query=query,
                confidence=0.0,
                reason="No memories stored yet.",
                reasons=["no memories stored"],
            )

        if mode == "replay":
            decision = self._decide_replay(query, results[0])
        elif mode == "restore":
            decision = self._decide_restore(query, results)
        elif mode == "verify":
            decision = self._decide_verify(query, results)
        else:
            decision = self._decide_auto(query, results, enable_verify=enable_verify)

        enriched = enrich_decision(
            decision,
            self._policy,
            replay_threshold=self.replay_threshold,
            restore_threshold=self.restore_threshold,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug(
            "decide  %.2fms  action=%s  conf=%.2f  query=%r  reasons=%r",
            elapsed_ms,
            enriched.action.value,
            enriched.confidence,
            query,
            enriched.reasons,
        )
        return enriched

    def _decide_auto(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        enable_verify: bool,
    ) -> MemoryDecision:
        action, confidence, reason = self._policy.select_action(
            results,
            replay_threshold=self.replay_threshold,
            restore_threshold=self.restore_threshold,
            verify_threshold=self.verify_threshold if enable_verify else 0.0,
        )

        if action == MemoryAction.REPLAY:
            return self._finalize_replay(query, results[0], reason)
        if action == MemoryAction.VERIFY:
            if enable_verify:
                return self._finalize_verify(query, results[0], reason)
            # Verification disabled: degrade to context injection, not NONE.
            action = MemoryAction.RESTORE
            reason = "Verification disabled — restoring memory as context instead."
        if action == MemoryAction.RESTORE:
            return MemoryDecision(
                action=MemoryAction.RESTORE,
                query=query,
                confidence=confidence,
                context=results,
                reason=reason,
            )

        return MemoryDecision(
            action=MemoryAction.NONE,
            query=query,
            confidence=confidence,
            context=results,
            reason=reason,
        )

    def _decide_replay(self, query: str, best: RetrievalResult) -> MemoryDecision:
        if best.decision_score >= self.replay_threshold:
            return self._finalize_replay(query, best, "Replay mode — match found.")
        return MemoryDecision(
            action=MemoryAction.NONE,
            query=query,
            confidence=best.decision_score,
            context=[best],
            reason="Replay mode — no match above replay threshold.",
        )

    def _decide_restore(self, query: str, results: list[RetrievalResult]) -> MemoryDecision:
        best = results[0]
        if best.decision_score >= self.restore_threshold:
            return MemoryDecision(
                action=MemoryAction.RESTORE,
                query=query,
                confidence=best.decision_score,
                context=results,
                reason="Restore mode — returning relevant memories as context.",
            )
        return MemoryDecision(
            action=MemoryAction.NONE,
            query=query,
            confidence=best.decision_score,
            context=results,
            reason="Restore mode — no match above restore threshold.",
        )

    def _decide_verify(self, query: str, results: list[RetrievalResult]) -> MemoryDecision:
        best = results[0]
        if best.decision_score >= self.restore_threshold:
            return self._finalize_verify(query, best, "Verify mode — memory requires validation.")
        return MemoryDecision(
            action=MemoryAction.NONE,
            query=query,
            confidence=best.decision_score,
            context=results,
            reason="Verify mode — no match above restore threshold.",
        )

    def _finalize_replay(self, query: str, best: RetrievalResult, reason: str) -> MemoryDecision:
        updated = self._retriever.record_access(best.entry)
        best.entry = updated
        log.debug(
            "REPLAY  conf=%.2f  matched=%r  accesses=%d",
            best.decision_score, updated.query[:60], updated.access_count,
        )
        return MemoryDecision(
            action=MemoryAction.REPLAY,
            query=query,
            confidence=best.decision_score,
            response=updated.response,
            memory=updated,
            context=[best],
            reason=reason,
        )

    def _finalize_verify(self, query: str, best: RetrievalResult, reason: str) -> MemoryDecision:
        log.debug(
            "VERIFY  conf=%.2f  matched=%r  type=%s",
            best.decision_score, best.entry.query[:60], best.entry.type.value,
        )
        return MemoryDecision(
            action=MemoryAction.VERIFY,
            query=query,
            confidence=best.decision_score,
            memory=best.entry,
            context=[best],
            reason=reason,
        )
