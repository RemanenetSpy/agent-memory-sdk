"""Decision Safety Suite v2 adapter for this repository's own SDK.

Published first, and with its own gaps declared, so the suite is not graded on a
curve it wrote for itself. Three things are worth reading before the numbers:

* **Scope isolation uses the SDK's scoped views.** One store is shared per
    battery, and each battery scope maps to a ``MemoryView`` identity. Cross-scope
    assertions therefore exercise the SDK's read filter, not separate stores.
* **``advance_clock`` is emulated** by back-dating stored timestamps, because
  there is no injectable clock. TTL itself is enforced on the read path, so the
  following ``cleanup()`` only reconciles stored state.
* **``as_of`` is unsupported**, so ``point_in_time`` is reported unsupported
  rather than scored.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from agent_memory.manager import Memory
from agent_memory.models import MemoryAction, MemoryScope
from agent_memory.scoped import MemoryView
from benchmarks.decision_safety.adapters.base import (
    CapabilityUnsupported,
    DecisionSafetyAdapter,
    QueryOutcome,
    WriteOutcome,
)

_TIERS = {tier.value for tier in MemoryScope}


class AgentMemoryAdapter(DecisionSafetyAdapter):
    name = "agent-memory-sdk"
    deployment = "local"

    capabilities = {
        "delete_by_id": "supported",          # Memory.forget(memory_id)
        "delete_scope": "supported",          # MemoryView.forget_all() deletes only owned entries
        "tombstones": "partial",              # archive()/MemoryState.DELETED exist; forget() is a hard delete
        "raw_message_store": "none",          # no separate transcript table; turns become ordinary memories
        "ttl": "supported",                   # expires_at, enforced on the read path
        "explicit_supersession": "unsupported",  # recency/confidence/consolidate() only
        "as_of_query": "unsupported",         # created_at is stored, but there is no point-in-time read path
        "explicit_abstention": "supported",   # MemoryAction.NONE
    }

    emulated_ops = {"advance_clock"}

    def __init__(
        self,
        *,
        enable_embeddings: bool = False,
        ingest: str = "extract",
        replay_threshold: float = 0.85,
        restore_threshold: float = 0.70,
        verify_threshold: float = 0.80,
    ) -> None:
        if ingest not in {"extract", "verbatim"}:
            raise ValueError("ingest must be 'extract' or 'verbatim'")
        # SDK defaults. They gate abstention, so they are the single biggest
        # lever on this suite's paired metrics and belong in the artifact.
        self.thresholds = {
            "replay": replay_threshold,
            "restore": restore_threshold,
            "verify": verify_threshold,
        }
        self.enable_embeddings = enable_embeddings
        self.ingest = ingest
        # SDK default embedder: fastembed BAAI/bge-small-en-v1.5, else
        # sentence-transformers all-MiniLM-L6-v2 (agent_memory/embeddings.py).
        self.embedding_model = "sdk-default" if enable_embeddings else "none"
        self._root: Path | None = None
        self._memory: Memory | None = None
        self._offset = timedelta(0)

        from agent_memory._version import __version__

        self.version = __version__
        self.notes = [
            "One Memory store is shared by all scopes in a battery; each scope maps to "
            "a MemoryView user_id and is filtered by the SDK's scoped-view architecture.",
            "advance_clock emulated by back-dating stored timestamps, then calling cleanup() "
            "to reconcile stored state; TTL itself is enforced on the read path.",
            f"Ingest path: {self.ingest} "
            + (
                "(Memory.from_conversation, the path that decides what to store)"
                if self.ingest == "extract"
                else "(Memory.remember, which stores unconditionally)"
            ),
            "Retrieval: "
            + (
                "embeddings enabled (hybrid semantic + lexical)"
                if self.enable_embeddings
                else "lexical/BM25 only (embeddings disabled for determinism); "
                "near-tie behaviour under vector similarity is untested in this run"
            ),
            "Decision thresholds: "
            + ", ".join(f"{k}={v}" for k, v in self.thresholds.items())
            + ". These gate abstention, so they move every paired metric here.",
        ]

    # -- lifecycle ---------------------------------------------------------

    def setup(self, *, battery: str) -> None:
        self.teardown()
        self._root = Path(tempfile.mkdtemp(prefix=f"ds-{battery}-"))
        self._memory = Memory(
            persist_dir=self._root,
            enable_embeddings=self.enable_embeddings,
            replay_threshold=self.thresholds["replay"],
            restore_threshold=self.thresholds["restore"],
            verify_threshold=self.thresholds["verify"],
        )
        self._offset = timedelta(0)

    def teardown(self) -> None:
        self._memory = None
        if self._root and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = None

    # -- scope mapping -----------------------------------------------------

    def _tier(self, scope: str) -> MemoryScope:
        tier = scope.split(":", 1)[0]
        return MemoryScope(tier) if tier in _TIERS else MemoryScope.USER

    def _view(self, scope: str) -> MemoryView:
        assert self._memory is not None, "setup() was not called"
        return self._memory.scoped(user_id=scope)

    # -- ops ---------------------------------------------------------------

    def write(
        self,
        *,
        op_id: str,
        scope: str,
        human: str,
        assistant: str,
        at: datetime | None = None,
        ttl: int | None = None,
    ) -> WriteOutcome:
        view = self._view(scope)
        memory = view.memory
        tier = self._tier(scope)

        if ttl is not None:
            # from_conversation() has no TTL parameter, so a TTL op writes
            # verbatim through remember(). Recorded in the op's raw output.
            entries = [view.remember(human, assistant, scope=tier, ttl=ttl)]
        elif self.ingest == "extract":
            entries = view.from_conversation(human, assistant, scope=tier)
        else:
            entries = [view.remember(human, assistant, scope=tier)]

        for entry in entries:
            entry.metadata["ds_op"] = op_id
            entry.metadata["raw_turn"] = entry.query.strip() == human.strip()
            if at is not None:
                entry.created_at = at
                entry.updated_at = at
            memory.store.update(entry)
        if entries:
            memory.retriever.invalidate_cache()

        return WriteOutcome(
            stored=bool(entries),
            refs=[entry.id for entry in entries],
            stored_texts=[f"{entry.query} -> {entry.response}" for entry in entries],
            raw={
                "extracted": len(entries),
                "ingest": "verbatim" if ttl is not None else self.ingest,
                "ttl_seconds": ttl,
            },
        )

    def delete(self, *, op_id: str, refs: list[str]) -> int:
        deleted = 0
        if self._memory is not None:
            for ref in refs:
                deleted += int(self._memory.forget(ref))
        return deleted

    def delete_scope(self, *, scope: str) -> int:
        """Delete only the entries owned by the scoped view."""
        return self._view(scope).forget_all()

    def advance_clock(self, *, seconds: int, now: datetime) -> None:
        """Emulated: shift stored timestamps back, then apply lazy TTL expiry."""
        delta = timedelta(seconds=seconds)
        self._offset += delta
        if self._memory is None:
            return
        for entry in self._memory.list(limit=10_000, include_archived=True):
            entry.created_at = entry.created_at - delta
            entry.updated_at = entry.updated_at - delta
            if entry.last_accessed_at is not None:
                entry.last_accessed_at = entry.last_accessed_at - delta
            if entry.expires_at is not None:
                entry.expires_at = entry.expires_at - delta
            entry.refresh_state()
            self._memory.store.update(entry)
        self._memory.cleanup()
        self._memory.retriever.invalidate_cache()

    def query(self, *, scope: str, text: str, top_k: int) -> QueryOutcome:
        decision = self._view(scope).resolve(text, top_k=top_k)

        # Score only what the caller is actually handed. On MemoryAction.NONE the
        # SDK's answer is "use nothing", even though `decision.context` still
        # carries the near-miss candidates — counting those would score an
        # abstention as both a leak and a hit. The candidates stay in `hits` for
        # diagnostics, which is where the near-tie story is readable.
        abstained = decision.action == MemoryAction.NONE
        texts: list[str] = []
        if not abstained:
            if decision.response:
                texts.append(decision.response)
            texts.extend(
                f"{result.entry.query} {result.entry.response}" for result in decision.context
            )

        return QueryOutcome(
            abstained=abstained,
            texts=texts,
            hits=[
                {
                    "query": result.entry.query,
                    "response": result.entry.response,
                    "score": round(result.final_score, 4),
                    "semantic": round(result.semantic_score, 4),
                    "keyword": round(result.keyword_score, 4),
                    "created_at": result.entry.created_at.isoformat(),
                }
                for result in decision.context
            ],
            raw={
                "action": decision.action.value,
                "confidence": round(decision.confidence, 4),
                "reasons": decision.reasons,
            },
        )

    def query_as_of(
        self, *, scope: str, text: str, as_of: datetime, top_k: int
    ) -> QueryOutcome:
        raise CapabilityUnsupported(
            "as_of_query", "the SDK has no point-in-time read path; not emulated here"
        )

    def inspect(self, *, scope: str) -> dict[str, Any]:
        view = self._view(scope)
        memory = view.memory
        # `raw_turns` counts every *retained* verbatim turn, including archived
        # and expired ones: the battery's question is what could still be
        # re-extracted, and an expired row is still a row. `live_raw_turns` is
        # the subset the read path would serve.
        retained = [
            entry
            for entry in memory.store.list_all(
                limit=10_000, include_archived=True, include_expired=True
            )
            if view.owns(entry)
        ]
        live = [
            entry
            for entry in view.list(limit=10_000, include_archived=True)
            if view.owns(entry)
        ]
        return {
            "raw_turns": sum(1 for e in retained if e.metadata.get("raw_turn")),
            "live_raw_turns": sum(1 for e in live if e.metadata.get("raw_turn")),
            "stored_memories": len(retained),
            "note": "no separate transcript store; verbatim turns are ordinary "
            "memories and are removed when those memories are deleted",
        }
