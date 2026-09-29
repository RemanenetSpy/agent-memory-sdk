from __future__ import annotations

import asyncio
import builtins
from collections.abc import Callable
from pathlib import Path

from agent_memory.backends import BackendContext, build_store
from agent_memory.confidence import ConfidenceEvent, ConfidenceLearner
from agent_memory.decision import DecisionEngine
from agent_memory.entity_extractor import EntityExtractor, ExtractedMemory
from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryAction, MemoryDecision, MemoryEntry, MemoryScope, MemoryType
from agent_memory.policy import DecisionPolicy, DefaultPolicy
from agent_memory.retriever import MemoryRetriever
from agent_memory.scoped import MemoryView
from agent_memory.store import MemoryStore
from agent_memory.ttl import parse_ttl

log = get_logger(__name__)


class Memory:
    """High-level SDK for persistent agent memory with a decision layer."""

    def __init__(
        self,
        persist_dir: str | Path = ".agent_memory",
        collection_name: str = "agent_memories",
        policy: DecisionPolicy | None = None,
        replay_threshold: float = 0.85,
        restore_threshold: float = 0.70,
        verify_threshold: float = 0.80,
        # Any name in agent_memory.backends.available_backends()
        backend: str = "sqlite",
        embedder: object | None = None,
        enable_embeddings: bool | str = "auto",
        store: MemoryStore | None = None,
        graph_weight: float = 0.0,
        **backend_kwargs: object,
    ) -> None:
        if store is not None:
            # Accept a pre-built store directly (useful for testing / custom backends)
            self.store: MemoryStore = store
        else:
            # Resolved through the registry rather than an if/elif chain, so a new
            # backend is a registration rather than an edit here.
            self.store = build_store(
                backend,
                BackendContext(
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    embedder=embedder,
                    enable_embeddings=enable_embeddings,
                    kwargs=backend_kwargs,
                ),
            )
        log.info("Memory initialised  backend=%s", backend)
        import warnings as _warnings
        if not 0.0 <= graph_weight <= 1.0:
            _warnings.warn(
                f"graph_weight={graph_weight!r} is outside [0, 1]. "
                "Values outside this range may produce unexpected scoring.",
                UserWarning,
                stacklevel=2,
            )
        if policy is not None and graph_weight != 0.0:
            _warnings.warn(
                "graph_weight is ignored when a custom policy is provided. "
                "Set graph_weight directly on your policy instance.",
                UserWarning,
                stacklevel=2,
            )
        self._policy = policy or DefaultPolicy(graph_weight=graph_weight)
        self._learner = ConfidenceLearner()
        self.retriever = MemoryRetriever(self.store, policy=self._policy)
        self.decision_engine = DecisionEngine(
            self.retriever,
            policy=self._policy,
            replay_threshold=replay_threshold,
            restore_threshold=restore_threshold,
            verify_threshold=verify_threshold,
        )

    def remember(
        self,
        query: str,
        response: str,
        *,
        content: str | None = None,
        type: MemoryType | str = MemoryType.CONVERSATION,
        scope: MemoryScope | str = MemoryScope.USER,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        requires_verification: bool = False,
        ttl: str | int | float | None = None,
    ) -> MemoryEntry:
        memory_type = MemoryType(type) if isinstance(type, str) else type
        memory_scope = MemoryScope(scope) if isinstance(scope, str) else scope
        entry = MemoryEntry(
            query=query,
            response=response,
            content=content or response,
            type=memory_type,
            scope=memory_scope,
            metadata=metadata or {},
            tags=tags or [],
            confidence=confidence,
            requires_verification=requires_verification,
            expires_at=parse_ttl(ttl),
        )
        entry.refresh_state()
        stored = self.store.store(entry)
        self.retriever.invalidate_cache()
        log.debug("remember  id=%s  type=%s", stored.id[:8], stored.type.value)
        return stored

    async def aremember(
        self,
        query: str,
        response: str,
        *,
        content: str | None = None,
        type: MemoryType | str = MemoryType.CONVERSATION,
        scope: MemoryScope | str = MemoryScope.USER,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        requires_verification: bool = False,
        ttl: str | int | float | None = None,
    ) -> MemoryEntry:
        """Async version of remember()."""
        return await asyncio.to_thread(
            self.remember,
            query,
            response,
            content=content,
            type=type,
            scope=scope,
            metadata=metadata,
            tags=tags,
            confidence=confidence,
            requires_verification=requires_verification,
            ttl=ttl,
        )

    def resolve(
        self,
        query: str,
        *,
        mode: str = "auto",
        top_k: int = 3,
        scope: list[MemoryScope | str] | None = None,
        enable_verify: bool = True,
        where: Callable[[MemoryEntry], bool] | None = None,
    ) -> MemoryDecision:
        """Decide what to do with memory for *query*.

        *where* restricts which stored memories may be considered, applied
        before scoring. Callers that can only see part of the store — one tenant,
        one agent, one session — must pass it, because ``scope`` is a tier
        (``user``/``project``/...), not a tenant identifier.
        """
        scopes = None
        if scope:
            scopes = [MemoryScope(s) if isinstance(s, str) else s for s in scope]
        return self.decision_engine.decide(
            query,
            mode=mode,
            top_k=top_k,
            scopes=scopes,
            enable_verify=enable_verify,
            where=where,
        )

    async def aresolve(
        self,
        query: str,
        *,
        mode: str = "auto",
        top_k: int = 3,
        scope: list[MemoryScope | str] | None = None,
        enable_verify: bool = True,
    ) -> MemoryDecision:
        """Async version of resolve()."""
        return await asyncio.to_thread(
            self.resolve,
            query,
            mode=mode,
            top_k=top_k,
            scope=scope,
            enable_verify=enable_verify,
        )

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scope: list[MemoryScope | str] | None = None,
        include_archived: bool = False,
        type: MemoryType | str | None = None,
    ) -> list[MemoryEntry]:
        scopes = [MemoryScope(s) if isinstance(s, str) else s for s in scope] if scope else None
        memory_type = MemoryType(type) if isinstance(type, str) else type
        return self.store.list_all(
            limit=limit,
            offset=offset,
            scopes=scopes,
            include_archived=include_archived,
            memory_type=memory_type,
        )

    async def alist(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scope: builtins.list[MemoryScope | str] | None = None,
        include_archived: bool = False,
        type: MemoryType | str | None = None,
    ) -> builtins.list[MemoryEntry]:
        """Async version of list()."""
        return await asyncio.to_thread(
            self.list,
            limit=limit,
            offset=offset,
            scope=scope,
            include_archived=include_archived,
            type=type,
        )

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self.store.get(memory_id)

    async def aget(self, memory_id: str) -> MemoryEntry | None:
        """Async version of get()."""
        return await asyncio.to_thread(self.get, memory_id)

    def forget(self, memory_id: str) -> bool:
        deleted = self.store.delete(memory_id)
        if deleted:
            self.retriever.invalidate_cache()
            log.debug("forget  id=%s", memory_id[:8])
        return deleted

    async def aforget(self, memory_id: str) -> bool:
        """Async version of forget()."""
        return await asyncio.to_thread(self.forget, memory_id)

    def forget_where(
        self,
        *,
        scope: builtins.list[MemoryScope | str] | None = None,
        type: MemoryType | str | None = None,
        tags: builtins.list[str] | None = None,
        metadata: dict | None = None,
        where: Callable[[MemoryEntry], bool] | None = None,
        all: bool = False,
        limit: int = 100_000,
    ) -> int:
        """Bulk-delete every memory matching *all* of the given criteria.

        Returns the number of memories deleted. Archived and expired memories
        are included: "forget this" must not leave a copy behind in another
        state.

        *tags* matches entries carrying every listed tag; *metadata* matches
        entries whose metadata contains every given key/value pair. Pass
        ``all=True`` to delete everything in the store — required explicitly,
        so an empty filter cannot wipe a store by accident.

        Usage::

            memory.forget_where(metadata={"user_id": "alice"})
            memory.forget_where(scope=["session"])
            memory.forget_where(all=True)
        """
        criteria = (scope, type, tags, metadata, where)
        if not any(value is not None for value in criteria) and not all:
            raise ValueError(
                "forget_where() needs at least one of scope/type/tags/metadata/where, "
                "or all=True to delete every memory in the store."
            )

        scopes = [MemoryScope(s) if isinstance(s, str) else s for s in scope] if scope else None
        memory_type = MemoryType(type) if isinstance(type, str) else type

        entries = self.store.list_all(
            limit=limit,
            scopes=scopes,
            include_archived=True,
            include_expired=True,
            memory_type=memory_type,
        )

        deleted = 0
        for entry in entries:
            if tags and not set(tags).issubset(entry.tags):
                continue
            if metadata and any(entry.metadata.get(k) != v for k, v in metadata.items()):
                continue
            if where is not None and not where(entry):
                continue
            deleted += int(self.store.delete(entry.id))

        if deleted:
            self.retriever.invalidate_cache()
        log.debug("forget_where  deleted=%d  of=%d", deleted, len(entries))
        return deleted

    async def aforget_where(self, **kwargs: object) -> int:
        """Async version of forget_where()."""
        return await asyncio.to_thread(lambda: self.forget_where(**kwargs))  # type: ignore[arg-type]

    def scoped(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        shared: bool = False,
        include_unscoped: bool = False,
    ) -> MemoryView:
        """Return a per-user or per-session view over this store.

        ``scope`` is a tier (``user``/``project``/...), not a tenant id, so many
        users in one store need this instead. Reads are hierarchical (session →
        user → shared) and filtered during retrieval; writes stay in the view's
        own namespace. See :class:`~agent_memory.scoped.MemoryView`.

        Usage::

            alice = memory.scoped(user_id="alice")
            alice.session("s3").remember("Which seat?", "Window")
            alice.resolve("seat preference?")
            memory.scoped(shared=True).remember("Refund window?", "30 days")
        """
        return MemoryView(
            memory=self,
            user_id=user_id,
            session_id=session_id,
            shared=shared,
            include_unscoped=include_unscoped,
        )

    def archive(self, memory_id: str) -> MemoryEntry | None:
        entry = self.store.get(memory_id)
        if not entry:
            return None
        entry.archived = True
        entry.refresh_state()
        archived = self.store.update(entry)
        self.retriever.invalidate_cache()
        return archived

    async def aarchive(self, memory_id: str) -> MemoryEntry | None:
        """Async version of archive()."""
        return await asyncio.to_thread(self.archive, memory_id)

    def cleanup(self, *, delete: bool = False) -> dict[str, int]:
        """
        Mark expired memories as expired, optionally deleting them.

        Returns counts: {"expired": N, "deleted": M}
        """
        result = self.store.cleanup_expired(delete=delete)
        if result["expired"] or result["deleted"]:
            self.retriever.invalidate_cache()
        return result

    async def acleanup(self, *, delete: bool = False) -> dict[str, int]:
        """Async version of cleanup()."""
        return await asyncio.to_thread(self.cleanup, delete=delete)

    def stats(self) -> dict:
        """Return aggregate memory and usage statistics."""
        return self.store.stats()

    async def astats(self) -> dict:
        """Async version of stats()."""
        return await asyncio.to_thread(self.stats)

    def consolidate(self, similarity_threshold: float = 0.95) -> builtins.list[MemoryEntry]:
        """
        Merge near-duplicate active memories into summary entries.
        Returns newly created summary memories.
        """
        entries = self.store.list_all(limit=10_000)
        created: list[MemoryEntry] = []
        seen: set[str] = set()

        by_id = {entry.id: entry for entry in entries}
        for entry in entries:
            if entry.id in seen or entry.archived or entry.type == MemoryType.SUMMARY:
                continue

            # One search per entry (not per pair): near-duplicates of this
            # entry's query are its top search hits by construction.
            duplicates = [entry]
            hits = self.store.search(entry.query, top_k=10)
            for hit, score in hits:
                other = by_id.get(hit.id)
                if other is None or other.id == entry.id or other.id in seen:
                    continue
                if other.archived or other.type != entry.type or other.scope != entry.scope:
                    continue
                if score >= similarity_threshold:
                    duplicates.append(other)

            if len(duplicates) < 2:
                continue

            for dup in duplicates:
                seen.add(dup.id)
                dup.archived = True
                self.store.update(dup)

            merged_query = duplicates[0].query
            merged_content = "\n".join(f"- {d.response}" for d in duplicates)
            summary = self.remember(
                query=merged_query,
                response=merged_content,
                content=merged_content,
                type=MemoryType.SUMMARY,
                scope=duplicates[0].scope,
                tags=list({tag for d in duplicates for tag in d.tags}),
                metadata={"consolidated_from": [d.id for d in duplicates]},
            )
            created.append(summary)

        return created

    async def aconsolidate(self, similarity_threshold: float = 0.95) -> builtins.list[MemoryEntry]:
        """Async version of consolidate()."""
        return await asyncio.to_thread(self.consolidate, similarity_threshold=similarity_threshold)

    def format_restore_context(self, decision: MemoryDecision) -> str:
        if not decision.context:
            return ""

        lines = ["## Retrieved Memory Context", ""]
        for result in decision.context:
            entry = result.entry
            lines.append(
                f"### Memory {result.rank} (score: {result.final_score:.2f}, type: {entry.type.value})"
            )
            lines.append(f"**Prior query:** {entry.query}")
            lines.append(f"**Prior response:** {entry.response}")
            if entry.tags:
                lines.append(f"**Tags:** {', '.join(entry.tags)}")
            lines.append("")
        return "\n".join(lines)

    def format_verify_context(self, decision: MemoryDecision) -> str:
        if not decision.memory:
            return ""
        entry = decision.memory
        return (
            "## Memory Pending Verification\n\n"
            f"**Query:** {entry.query}\n"
            f"**Stored response:** {entry.response}\n"
            f"**Type:** {entry.type.value}\n"
            f"**Last updated:** {entry.updated_at.isoformat()}\n\n"
            "Validate this memory with available tools before reusing or regenerating."
        )

    # ------------------------------------------------------------------
    # Entity extraction — auto-remember from raw conversation text
    # ------------------------------------------------------------------

    def from_conversation(
        self,
        human: str,
        assistant: str,
        *,
        scope: MemoryScope | str = MemoryScope.USER,
        min_confidence: float = 0.75,
        extractor: EntityExtractor | None = None,
        metadata: dict | None = None,
        source_trusted: bool = True,
    ) -> builtins.list[MemoryEntry]:
        """Extract and store memories from a single conversation turn.

        Automatically identifies facts, preferences, and entities in
        *human* + *assistant* text and calls :meth:`remember` for each.
        Only candidates above *min_confidence* are stored.

        *metadata* is merged into every stored candidate, which is how a
        :class:`~agent_memory.scoped.MemoryView` stamps tenant identity onto
        extracted memories.

        Set *source_trusted* to ``False`` for tool output, retrieved documents,
        or other externally controlled turns. Those turns are never persisted;
        pattern-based injection detection is only a supplemental defense.

        Returns the list of newly stored :class:`~agent_memory.models.MemoryEntry` objects.

        Usage::

            entries = memory.from_conversation(
                human="My name is Karan and I prefer Python.",
                assistant="Got it!",
            )
        """
        if not source_trusted:
            return []
        mem_scope = MemoryScope(scope) if isinstance(scope, str) else scope
        ext = extractor or EntityExtractor()
        candidates: builtins.list[ExtractedMemory] = ext.extract_from_turn(
            human, assistant, scope=mem_scope
        )
        stored: builtins.list[MemoryEntry] = []
        log.debug("from_conversation  candidates=%d  min_conf=%.2f", len(candidates), min_confidence)
        for c in candidates:
            if "injection-suspected" in c.tags:
                continue
            if c.confidence < min_confidence:
                continue
            entry = self.remember(
                c.query,
                c.response,
                type=c.memory_type,
                scope=mem_scope,
                tags=c.tags,
                confidence=c.confidence,
                requires_verification=c.requires_verification,
                metadata={**(c.metadata or {}), **(metadata or {})},
            )
            stored.append(entry)
        return stored

    async def afrom_conversation(
        self,
        human: str,
        assistant: str,
        *,
        scope: MemoryScope | str = MemoryScope.USER,
        min_confidence: float = 0.75,
        extractor: EntityExtractor | None = None,
        metadata: dict | None = None,
        source_trusted: bool = True,
    ) -> builtins.list[MemoryEntry]:
        """Async version of :meth:`from_conversation`."""
        return await asyncio.to_thread(
            self.from_conversation,
            human,
            assistant,
            scope=scope,
            min_confidence=min_confidence,
            extractor=extractor,
            metadata=metadata,
            source_trusted=source_trusted,
        )

    # ------------------------------------------------------------------
    # Paged memory — hierarchical context tiers
    # ------------------------------------------------------------------

    def paged(
        self,
        *,
        context_size: int = 20,
        recall_top_k: int = 5,
        scope: MemoryScope | str = MemoryScope.USER,
    ) -> PagedMemory:
        """Return a :class:`~agent_memory.paged_memory.PagedMemory` wrapper.

        The wrapper adds an in-context buffer on top of this store so that
        recent turns are always available without a search, while older
        entries are automatically paged out to recall / archival tiers.

        Usage::

            paged = memory.paged(context_size=20)
            paged.add_turn("What is Python?", "A programming language.")
            ctx = paged.get_context("Tell me about Python")
            print(ctx.format_for_llm())
        """
        from agent_memory.paged_memory import PagedMemory

        mem_scope = MemoryScope(scope) if isinstance(scope, str) else scope
        return PagedMemory(
            self,
            context_size=context_size,
            recall_top_k=recall_top_k,
            scope=mem_scope,
        )

    # ------------------------------------------------------------------
    # Knowledge graph — typed entities + relations
    # ------------------------------------------------------------------

    def knowledge_graph(self) -> KnowledgeGraph:
        """Build and return a :class:`~agent_memory.knowledge_graph.KnowledgeGraph`.

        The graph is built from all active memories in this store, extracting
        named entities (people, orgs, locations, concepts) and typed
        relationships between them.

        Usage::

            kg = memory.knowledge_graph()
            entity = kg.find_entity("Karan")
            for rel in kg.relations_for(entity.id):
                print(rel.relation, "→", rel.target)
        """

        from agent_memory.knowledge_graph import GraphBuilder  # noqa: PLC0415

        return GraphBuilder().build(self.store)

    # ------------------------------------------------------------------
    # Feedback — close the confidence learning loop
    # ------------------------------------------------------------------

    def mark_correct(self, decision: MemoryDecision) -> MemoryEntry | None:
        """Signal that a REPLAY or RESTORE decision was correct.

        Boosts the matched memory's confidence score (+0.15) and persists the
        update.  The retriever cache is invalidated so the next resolve() sees
        the fresh score.

        Returns the updated entry, or None if the decision has no memory.
        """
        entry = decision.memory or (decision.context[0].entry if decision.context else None)
        if entry is None:
            return None
        # Work on a copy so the original decision.memory is not mutated if
        # store.update() fails (e.g. disk full, DB locked).
        candidate = entry.model_copy()
        self._learner.record_event(candidate, ConfidenceEvent.USER_CONFIRMED, persist_timestamp=True)
        updated = self.store.update(candidate)
        self.retriever.invalidate_cache()
        log.debug("mark_correct  id=%s  conf=%.2f", updated.id[:8], updated.confidence)
        return updated

    async def amark_correct(self, decision: MemoryDecision) -> MemoryEntry | None:
        """Async version of mark_correct()."""
        return await asyncio.to_thread(self.mark_correct, decision)

    def mark_wrong(self, decision: MemoryDecision) -> MemoryEntry | None:
        """Signal that a REPLAY or RESTORE decision was wrong.

        Penalises the matched memory's confidence score (−0.25) and persists
        the update.  The retriever cache is invalidated so the penalised entry
        is less likely to surface on the next resolve().

        Returns the updated entry, or None if the decision has no memory.
        """
        entry = decision.memory or (decision.context[0].entry if decision.context else None)
        if entry is None:
            return None
        candidate = entry.model_copy()
        self._learner.record_event(candidate, ConfidenceEvent.USER_REJECTED, persist_timestamp=True)
        updated = self.store.update(candidate)
        self.retriever.invalidate_cache()
        log.debug("mark_wrong  id=%s  conf=%.2f", updated.id[:8], updated.confidence)
        return updated

    async def amark_wrong(self, decision: MemoryDecision) -> MemoryEntry | None:
        """Async version of mark_wrong()."""
        return await asyncio.to_thread(self.mark_wrong, decision)

    # ------------------------------------------------------------------
    # Verify — complete the VERIFY decision cycle
    # ------------------------------------------------------------------

    def _apply_verification(
        self,
        entry: MemoryEntry,
        is_valid: bool,
        decision: MemoryDecision,
    ) -> MemoryDecision:
        """Shared result-builder for both sync and async verify paths.

        Applies the confidence event, persists the entry, invalidates the
        cache, and returns the resolved MemoryDecision.  Keeping this in one
        place ensures both paths stay in sync for future changes.
        """
        # Work on a copy so the caller's entry is not mutated if store.update() fails.
        candidate = entry.model_copy()
        event = ConfidenceEvent.VERIFIED_CORRECT if is_valid else ConfidenceEvent.VERIFIED_INCORRECT
        self._learner.record_event(candidate, event, persist_timestamp=True)
        updated = self.store.update(candidate)
        self.retriever.invalidate_cache()
        if is_valid:
            log.debug("verify correct  id=%s  conf=%.2f", updated.id[:8], updated.confidence)
            return MemoryDecision(
                action=MemoryAction.REPLAY,
                query=decision.query,
                confidence=decision.confidence,
                response=updated.response,
                memory=updated,
                context=decision.context,
                reason="Verified correct — replaying stored response.",
            )
        log.debug("verify failed  id=%s  conf=%.2f", updated.id[:8], updated.confidence)
        return MemoryDecision(
            action=MemoryAction.NONE,
            query=decision.query,
            confidence=0.0,
            memory=updated,   # expose the penalised entry so callers can inspect it
            reason="Verification failed — memory is outdated or incorrect.",
        )

    def verify(
        self,
        decision: MemoryDecision,
        verifier: builtins.object,
    ) -> MemoryDecision:
        """Run *verifier* on a VERIFY decision and return a resolved decision.

        *verifier* is any callable with signature ``(MemoryDecision) -> bool``.
        Return ``True`` if the memory is still valid, ``False`` if it is stale
        or wrong.

        On success  → updates confidence (+0.10), returns REPLAY decision.
        On failure  → updates confidence (−0.20), returns NONE decision with
                      ``memory`` set so callers can inspect the updated entry.

        The entry is persisted and the retriever cache is invalidated in both
        cases so confidence changes affect future resolve() calls immediately.

        Example::

            def my_verifier(decision):
                # Call your LLM / tool / API here
                return check_still_true(decision.memory.response)

            resolved = memory.verify(decision, my_verifier)
            if resolved.action == MemoryAction.REPLAY:
                return resolved.response
        """
        if decision.action != MemoryAction.VERIFY or decision.memory is None:
            return decision
        is_valid: bool = verifier(decision)  # type: ignore[operator]
        return self._apply_verification(decision.memory, is_valid, decision)

    async def averify(
        self,
        decision: MemoryDecision,
        verifier: builtins.object,
    ) -> MemoryDecision:
        """Async version of verify().

        If *verifier* is a coroutine function it is awaited directly; otherwise
        it is run in a thread so blocking verifiers don't stall the event loop.
        """
        if decision.action != MemoryAction.VERIFY or decision.memory is None:
            return decision

        import inspect
        if inspect.iscoroutinefunction(verifier):
            # Snapshot the entry before awaiting — a concurrent mark_correct/wrong
            # call could mutate decision.memory.confidence between the await and
            # _apply_verification running in a thread.
            entry_snapshot = decision.memory.model_copy()
            is_valid: bool = await verifier(decision)  # type: ignore[operator]
            return await asyncio.to_thread(
                self._apply_verification, entry_snapshot, is_valid, decision
            )
        return await asyncio.to_thread(self.verify, decision, verifier)

    # ------------------------------------------------------------------
    # Graph importance — wire PageRank into scoring
    # ------------------------------------------------------------------

    def refresh_graph_scores(self) -> int:
        """Build the memory relationship graph and update importance scores.

        Computes PageRank-style importance for every active memory based on
        similarity edges and tag overlap.  Scores are normalised to [0, 1] and
        injected into the policy so subsequent resolve() calls incorporate graph
        connectivity.  The effect is gated by ``graph_weight`` (set at
        ``Memory.__init__``); the default is 0.0 so calling this without a
        non-zero ``graph_weight`` has no effect on decisions.

        Returns the number of memories scored.  Call this after bulk imports
        or whenever the graph topology changes significantly.

        Example::

            memory = Memory(graph_weight=0.10)
            memory.remember(...)
            memory.refresh_graph_scores()   # now resolve() uses graph boost
        """
        from agent_memory.graph import MemoryGraph

        graph = MemoryGraph.build(self.store)
        scores = graph.importance_scores()
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        if isinstance(self._policy, DefaultPolicy):
            self._policy.update_graph_scores(scores)
            log.debug("refresh_graph_scores  n=%d  graph_weight=%.2f",
                      len(scores), self._policy.graph_weight)
            return len(scores)

        import warnings as _w
        _w.warn(
            "refresh_graph_scores() has no effect because the active policy is not "
            "a DefaultPolicy. Implement update_graph_scores() on your custom policy "
            "or use DefaultPolicy(graph_weight=...).",
            UserWarning,
            stacklevel=2,
        )
        return 0

    async def arefresh_graph_scores(self) -> int:
        """Async version of refresh_graph_scores()."""
        return await asyncio.to_thread(self.refresh_graph_scores)

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def list_memories(self, limit: int = 100, offset: int = 0) -> builtins.list[MemoryEntry]:
        return self.list(limit=limit, offset=offset)


MemoryManager = Memory

# Lazy forward references resolved at import time
from agent_memory.knowledge_graph import KnowledgeGraph  # noqa: E402
from agent_memory.paged_memory import PagedMemory  # noqa: E402
