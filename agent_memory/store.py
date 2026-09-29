from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# High-frequency English function words excluded from coverage scoring.
STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "after", "again", "against", "an", "and", "any", "are", "as", "at",
    "be", "been", "being", "before", "between", "but", "by",
    "can", "could", "did", "do", "does", "down", "during",
    "for", "from", "get", "had", "has", "have", "he", "her", "hers", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "may", "me", "might", "mine", "must", "my",
    "no", "not", "now", "of", "on", "or", "our", "ours", "out", "over",
    "s", "she", "should", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "then", "these", "they",
    "this", "those", "through", "to", "under", "up", "us",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "whose", "why", "will", "with", "without", "would",
    "you", "your", "yours",
})


def _normalize_token(token: str) -> str:
    # Cheap plural folding so "tests" matches "test".
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    tokens = {_normalize_token(t) for t in _tokenize(text) if t not in STOP_WORDS}
    return tokens


def search_document(entry: MemoryEntry) -> str:
    """The text every backend indexes for an entry — keyword and vector alike.

    One definition so a paraphrase scores the same whichever backend answers it;
    if the query, content and tags were combined differently per backend, the
    same corpus would rank differently on each.
    """
    return f"{entry.query}\n{entry.content}\n{' '.join(entry.tags)}"


def query_coverage(query: str, document: str) -> float:
    """Fraction of the query's content words that appear in the document."""
    query_tokens = _content_tokens(query)
    if not query_tokens:
        query_tokens = {_normalize_token(t) for t in _tokenize(query)}
    if not query_tokens:
        return 0.0
    doc_tokens = _content_tokens(document) | {_normalize_token(t) for t in _tokenize(document)}
    return len(query_tokens & doc_tokens) / len(query_tokens)


def bm25_scores(
    query: str,
    entries: list[MemoryEntry],
    documents: list[str],
    top_k: int,
) -> list[tuple[MemoryEntry, float]]:
    """Rank entries by BM25, scaled by query-term coverage.

    Raw BM25 scores are relative to the corpus, so normalizing by the max
    would always give the best hit a perfect 1.0 — even when it shares a
    single word with the query. Scaling by coverage keeps exact matches near
    1.0 while weak overlaps score low enough that the decision layer skips them.
    """
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    max_score = max(scores) if len(scores) else 0.0
    if max_score <= 0:
        # BM25 degenerates on tiny corpora (IDF <= 0); rank by coverage alone,
        # mapped through the same 0.5 + 0.5*coverage transform as below.
        ranked_cov = sorted(
            ((entry, query_coverage(query, doc)) for entry, doc in zip(entries, documents)),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
        return [(entry, 0.5 + 0.5 * cov) for entry, cov in ranked_cov if cov > 0]

    ranked = sorted(
        zip(entries, documents, scores),
        key=lambda item: item[2],
        reverse=True,
    )[: top_k * 2]

    results: list[tuple[MemoryEntry, float]] = []
    for entry, doc, score in ranked:
        if score <= 0:
            continue
        coverage = query_coverage(query, doc)
        normalized = float(score / max_score)
        results.append((entry, normalized * (0.5 + 0.5 * coverage)))
    results.sort(key=lambda pair: pair[1], reverse=True)
    return results[:top_k]


class MemoryStore(ABC):
    """Abstract base class for memory storage backends.
    This defines the common interface that all memory store implementations
    must follow, enabling interchangeability between backends (ChromaDB, SQLite, etc.)
    while maintaining SOLID principles - specifically the Liskov Substitution Principle
    and Dependency Inversion Principle.
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> MemoryEntry:
        """Store a memory entry."""
        ...

    @abstractmethod
    def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a memory entry by ID."""
        ...

    @abstractmethod
    def update(self, entry: MemoryEntry) -> MemoryEntry:
        """Update an existing memory entry."""
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by ID. Returns True if deleted."""
        ...

    @abstractmethod
    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        """List memory entries with optional filtering."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search memories using semantic/vector similarity."""
        ...

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        """Search memories using keyword/BM25 matching."""
        ...

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored memories."""
        ...

    @property
    def semantic_search_enabled(self) -> bool:
        """True when :meth:`search` ranks by embeddings rather than lexically.

        Part of the interface rather than a duck-typed extra, because
        :class:`~agent_memory.retriever.MemoryRetriever` branches on it: when a
        store reports False its ``search()`` is known to return the same list as
        ``keyword_search()``, so the retriever skips the duplicate query instead
        of fusing a list with itself. Default False — a backend that gains a
        vector index overrides it (see the Redis, Postgres and Qdrant stores).
        """
        return False

    def touch(self, memory_id: str) -> bool:
        """Increment access_count and set last_accessed_at for *memory_id*.

        This is a lightweight write used on REPLAY — it must NOT update the
        FTS5 / vector indexes (they are keyed on content, not usage metadata).
        The default implementation fetches, mutates, and updates; backends
        should override with a direct SQL UPDATE for speed.
        Returns True if the entry was found, False otherwise.
        """
        entry = self.get(memory_id)
        if entry is None:
            return False
        entry.touch()
        self.update(entry)
        return True

    def stats(self) -> dict[str, Any]:
        """Aggregate memory statistics.

        Default implementation loads entries into Python; backends with a
        query engine (e.g. SQLite) override this with real aggregates.
        """
        entries = self.list_all(limit=1_000_000, include_archived=True, include_expired=True)
        by_state: dict[str, int] = {}
        by_type: dict[str, int] = {}
        total_access = 0
        for entry in entries:
            entry.refresh_state()
            by_state[entry.state.value] = by_state.get(entry.state.value, 0) + 1
            by_type[entry.type.value] = by_type.get(entry.type.value, 0) + 1
            total_access += entry.access_count
        return {
            "total": len(entries),
            "by_state": by_state,
            "by_type": by_type,
            "total_access_count": total_access,
        }

    def cleanup_expired(self, *, delete: bool = False) -> dict[str, int]:
        """Mark expired memories as expired, optionally deleting them.

        Returns counts: {"expired": N, "deleted": M}. Default implementation
        loads entries into Python; backends with a query engine override it.
        """
        entries = self.list_all(limit=1_000_000, include_archived=True, include_expired=True)
        expired_count = 0
        deleted_count = 0
        for entry in entries:
            entry.refresh_state()
            if not entry.is_expired:
                continue
            if delete:
                if self.delete(entry.id):
                    deleted_count += 1
            else:
                if entry.state != MemoryState.EXPIRED:
                    entry.state = MemoryState.EXPIRED
                    self.update(entry)
                expired_count += 1
        return {"expired": expired_count, "deleted": deleted_count}


class ChromaDBStore(MemoryStore):
    """ChromaDB-backed persistent memory storage with scope filtering."""

    def __init__(
        self,
        persist_dir: str | Path = ".agent_memory",
        collection_name: str = "agent_memories",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return int(self._collection.count())

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        entry.refresh_state()
        self._collection.upsert(
            ids=[entry.id],
            documents=[search_document(entry)],
            metadatas=[self._entry_to_metadata(entry)],
        )
        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        result = self._collection.get(ids=[memory_id], include=["metadatas", "documents"])
        if not result["ids"]:
            return None
        return self._metadata_to_entry(
            memory_id=result["ids"][0],
            document=result["documents"][0] or "",
            metadata=result["metadatas"][0] or {},
        )

    def update(self, entry: MemoryEntry) -> MemoryEntry:
        return self.store(entry)

    def delete(self, memory_id: str) -> bool:
        if not self.get(memory_id):
            return False
        self._collection.delete(ids=[memory_id])
        return True

    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        where = self._build_where(scopes=scopes, include_archived=include_archived, memory_type=memory_type)
        kwargs: dict[str, Any] = {
            "include": ["metadatas", "documents"],
            "limit": limit,
            "offset": offset,
        }
        if where:
            kwargs["where"] = where

        result = self._collection.get(**kwargs)
        entries: list[MemoryEntry] = []
        for memory_id, document, metadata in zip(
            result["ids"],
            result["documents"] or [],
            result["metadatas"] or [],
        ):
            entries.append(
                self._metadata_to_entry(
                    memory_id=memory_id,
                    document=document or "",
                    metadata=metadata or {},
                )
            )
        return self._filter_entries(entries, include_archived=include_archived, include_expired=include_expired)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if self.count == 0:
            return []

        where = self._build_where(scopes=scopes, include_archived=include_archived)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, self.count),
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)

        matches: list[tuple[MemoryEntry, float]] = []
        ids = result["ids"][0] if result["ids"] else []
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        for memory_id, doc, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - float(distance))
            entry = self._metadata_to_entry(
                memory_id=memory_id,
                document=doc or "",
                metadata=metadata or {},
            )
            matches.append((entry, similarity))

        return self._filter_matches(matches, include_archived=include_archived, include_expired=include_expired)

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        entries = self.list_all(
            limit=10_000,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        if not entries:
            return []

        documents = [search_document(e) for e in entries]
        return bm25_scores(query, entries, documents, top_k)

    def _build_where(
        self,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        memory_type: MemoryType | None = None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []

        if not include_archived:
            clauses.append({"archived": False})

        if scopes:
            scope_values = [s.value for s in scopes]
            if len(scope_values) == 1:
                clauses.append({"scope": scope_values[0]})
            else:
                clauses.append({"scope": {"$in": scope_values}})

        if memory_type:
            clauses.append({"type": memory_type.value})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _entry_to_metadata(self, entry: MemoryEntry) -> dict[str, Any]:
        return {
            "query": entry.query,
            "response": entry.response,
            "content": entry.content,
            "type": entry.type.value,
            "scope": entry.scope.value,
            "metadata_json": json.dumps(entry.metadata),
            "tags": json.dumps(entry.tags),
            "confidence": entry.confidence,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "access_count": entry.access_count,
            "last_accessed_at": entry.last_accessed_at.isoformat() if entry.last_accessed_at else "",
            "archived": entry.archived,
            "requires_verification": entry.requires_verification,
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else "",
            "state": entry.state.value,
        }

    def _metadata_to_entry(self, memory_id: str, document: str, metadata: dict[str, Any]) -> MemoryEntry:
        last_accessed = metadata.get("last_accessed_at") or None
        expires_raw = metadata.get("expires_at") or None
        # Entries written before v0.1.5 lack the query in metadata; recover it
        # from the search document's first line.
        query = metadata.get("query") or (document.split("\n", 1)[0] if document else "")
        entry = MemoryEntry(
            id=memory_id,
            query=query,
            response=metadata.get("response", ""),
            content=metadata.get("content", metadata.get("response", "")),
            type=MemoryType(metadata.get("type", MemoryType.CONVERSATION.value)),
            scope=MemoryScope(metadata.get("scope", MemoryScope.USER.value)),
            metadata=json.loads(metadata.get("metadata_json", "{}")),
            tags=json.loads(metadata.get("tags", "[]")),
            confidence=float(metadata.get("confidence", 1.0)),
            created_at=datetime.fromisoformat(metadata["created_at"])
            if metadata.get("created_at")
            else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(metadata["updated_at"])
            if metadata.get("updated_at")
            else datetime.now(timezone.utc),
            access_count=int(metadata.get("access_count", 0)),
            last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
            archived=bool(metadata.get("archived", False)),
            requires_verification=bool(metadata.get("requires_verification", False)),
            expires_at=datetime.fromisoformat(expires_raw) if expires_raw else None,
            state=MemoryState(metadata.get("state", MemoryState.ACTIVE.value)),
        )
        entry.refresh_state()
        return entry

    @staticmethod
    def _filter_entries(
        entries: list[MemoryEntry],
        *,
        include_archived: bool,
        include_expired: bool,
    ) -> list[MemoryEntry]:
        filtered: list[MemoryEntry] = []
        for entry in entries:
            if entry.archived and not include_archived:
                continue
            if entry.is_expired and not include_expired:
                continue
            filtered.append(entry)
        return filtered

    @staticmethod
    def _filter_matches(
        matches: list[tuple[MemoryEntry, float]],
        *,
        include_archived: bool,
        include_expired: bool,
    ) -> list[tuple[MemoryEntry, float]]:
        return [
            (entry, score)
            for entry, score in matches
            if (include_archived or not entry.archived) and (include_expired or not entry.is_expired)
        ]
