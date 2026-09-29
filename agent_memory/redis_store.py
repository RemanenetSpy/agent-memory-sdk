from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agent_memory.embeddings import Embedder, embedding_dimension, get_default_embedder
from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType
from agent_memory.store import (
    STOP_WORDS,
    MemoryStore,
    _tokenize,
    bm25_scores,
    query_coverage,
    search_document,
)
from agent_memory.vector_index import VectorIndexConfig

log = get_logger(__name__)

# RediSearch's own default out-degree, with a higher-recall build setting for
# ef_construction. Pass vector_config= to override any of it.
DEFAULT_VECTOR_CONFIG = VectorIndexConfig(m=16, ef_construction=200, ef_search=64)

# Number of entries embedded per batch when backfilling vectors at init.
_VECTOR_BACKFILL_BATCH_SIZE = 64


@dataclass(frozen=True)
class _IndexShape:
    """The parts of an existing RediSearch index we must agree with to reuse it."""

    dim: int
    has_text: bool


def _pack_float32(vector: list[float]) -> bytes:
    """Serialise a vector as the little-endian FLOAT32 blob RedisVSS expects."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _as_str(value: Any) -> str:
    """Decode a Redis reply element, tolerating clients without decode_responses."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


class RedisMemoryStore(MemoryStore):
    """Redis-backed persistent memory store.

    Requires: pip install agent-memory-sdk[redis]

    With RediSearch available (Redis 8+, or Redis Stack) and an embedding model
    installed, ``search()`` becomes true vector KNN over an HNSW index, and
    ``keyword_search()`` is served by the same index's TEXT field rather than by
    re-ranking the corpus in Python. Without either, both fall back to Python
    BM25 over ``list_all()``, exactly as before.

    The index scan itself is sub-millisecond; end to end, a ``search()`` call is
    dominated by embedding the query (~3 ms for bge-small). See
    ``docs/benchmarks.md``.

    Note that RediSearch can only index database 0, so vector search needs the
    default ``db=0``; on any other database the store logs a warning and stays
    lexical.

    Storage layout (all keys prefixed with ``key_prefix``):
      - ``{prefix}:entry:{id}``  — JSON-encoded entry (source of truth)
      - ``{prefix}:ids``          — sorted set of all IDs scored by created_at timestamp
      - ``{prefix}:scope:{s}``   — set of IDs for each scope value
      - ``{prefix}:vec:{id}``    — HASH holding the embedding blob + filter tags

    Vectors live in their own HASH rather than in the entry key so that the
    entry JSON stays readable and stores written before vector support was
    added keep working untouched (they are backfilled at init).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        url: str | None = None,
        key_prefix: str = "agent_memory",
        redis_client: Any | None = None,
        embedder: Embedder | None = None,
        enable_embeddings: bool | str = "auto",
        index_name: str | None = None,
        vector_config: VectorIndexConfig | None = None,
    ) -> None:
        if redis_client is not None:
            self._client = redis_client
        else:
            try:
                import redis
            except ImportError:
                raise ImportError(
                    "Redis backend requires redis-py. "
                    "Install with: pip install agent-memory-sdk[redis]"
                ) from None
            if url:
                self._client = redis.from_url(url, decode_responses=True)
            else:
                self._client = redis.Redis(
                    host=host, port=port, db=db, password=password, decode_responses=True
                )
        self._prefix = key_prefix
        self._index = index_name or f"{key_prefix}:idx"
        self._vector_config = vector_config or DEFAULT_VECTOR_CONFIG
        self._embedder: Embedder | None = None
        self._vec_dim = 0
        self._vec_enabled = False
        self._client.ping()

        if enable_embeddings is True or enable_embeddings == "auto":
            self._init_embeddings(embedder, required=enable_embeddings is True)

    def _key(self, memory_id: str) -> str:
        return f"{self._prefix}:entry:{memory_id}"

    def _ids_key(self) -> str:
        return f"{self._prefix}:ids"

    def _scope_key(self, scope: str) -> str:
        return f"{self._prefix}:scope:{scope}"

    def _vec_key(self, memory_id: str) -> str:
        return f"{self._prefix}:vec:{memory_id}"

    # ------------------------------------------------------------------
    # Optional vector search (RediSearch / RedisVSS + embedding model)
    # ------------------------------------------------------------------

    def _init_embeddings(self, embedder: Embedder | None, *, required: bool) -> None:
        if not self._redisearch_available():
            if required:
                raise RuntimeError(
                    "enable_embeddings=True requires the RediSearch module. "
                    "Use Redis 8+ or Redis Stack (docker run -p 6379:6379 redis:8-alpine)."
                )
            return

        resolved = embedder or get_default_embedder()
        if resolved is None:
            if required:
                raise ImportError(
                    "enable_embeddings=True requires an embedding model. "
                    "Install with: pip install agent-memory-sdk[semantic]"
                )
            return

        self._embedder = resolved
        self._vec_dim = embedding_dimension(resolved)
        try:
            self._ensure_index()
        except Exception as exc:
            # RediSearch refuses to index any database but 0, and a shared server
            # may withhold FT.CREATE outright. Neither is a reason to break an
            # otherwise working store, so fall back to BM25 unless the caller
            # explicitly demanded embeddings.
            self._embedder = None
            self._vec_dim = 0
            if required:
                raise RuntimeError(
                    f"Could not create the Redis vector index {self._index!r}: {exc}. "
                    "RediSearch only indexes db 0 — use db=0 for vector search."
                ) from exc
            log.warning(
                "Redis vector index %s unavailable (%s); falling back to BM25",
                self._index, exc,
            )
            return
        self._vec_enabled = True
        self._backfill_vectors()

    def _redisearch_available(self) -> bool:
        try:
            self._client.execute_command("FT._LIST")
        except Exception:
            return False
        return True

    def _ensure_index(self) -> None:
        """Create the index, recreating it if its shape no longer matches ours."""
        shape = self._index_shape()
        if shape is not None:
            reason = None
            if shape.dim != self._vec_dim:
                reason = f"dim={shape.dim} but the embedder produces dim={self._vec_dim}"
            elif not shape.has_text:
                reason = "no text field (built before keyword search was indexed)"
            if reason is not None:
                # Every stored vector/document is keyed to the old shape, so drop
                # the index *and* its hashes (DD) and let the backfill below
                # rewrite them.
                log.warning(
                    "Redis index %s has %s; dropping and rebuilding", self._index, reason
                )
                self._client.execute_command("FT.DROPINDEX", self._index, "DD")
                shape = None

        if shape is not None:
            return

        self._client.execute_command(
            "FT.CREATE", self._index,
            "ON", "HASH",
            "PREFIX", "1", f"{self._prefix}:vec:",
            "SCHEMA",
            # TEXT so the same index also serves keyword_search: without it the
            # BM25 half of every hybrid query has to read the entire corpus back
            # into Python, which is O(N) per query no matter how fast the KNN is.
            "text", "TEXT",
            "scope", "TAG",
            "archived", "TAG",
            "embedding", "VECTOR", "HNSW", "10",
            "TYPE", "FLOAT32",
            "DIM", str(self._vec_dim),
            "DISTANCE_METRIC", "COSINE",
            "M", str(self._vector_config.m),
            "EF_CONSTRUCTION", str(self._vector_config.ef_construction),
        )

    def _index_shape(self) -> _IndexShape | None:
        """Describe the existing index, or None when there isn't one.

        ``dim`` is 0 when the index carries no vector field we recognise, which
        the caller treats as a mismatch and rebuilds.
        """
        try:
            info = self._client.execute_command("FT.INFO", self._index)
        except Exception:
            return None

        # Depending on redis-py version and negotiated protocol, FT.INFO comes
        # back either already mapped to dicts or as the raw flat key/value list.
        if isinstance(info, dict):
            attributes = info.get("attributes") or []
        else:
            flat = [item if isinstance(item, (list, tuple)) else _as_str(item) for item in info]
            attributes = next(
                (v for k, v in zip(flat[::2], flat[1::2]) if k == "attributes"), []
            )

        dim = 0
        names: set[str] = set()
        for attr in attributes:
            if isinstance(attr, dict):
                fields = {str(k): attr[k] for k in attr}
            elif isinstance(attr, (list, tuple)):
                tokens = [_as_str(t) for t in attr]
                fields = dict(zip(tokens[::2], tokens[1::2]))
            else:
                continue
            if "attribute" in fields:
                names.add(_as_str(fields["attribute"]))
            if "dim" in fields:
                dim = int(fields["dim"])
        return _IndexShape(dim=dim, has_text="text" in names)

    def _backfill_vectors(self) -> None:
        """Embed any entry that has no vector hash yet (pre-upgrade data)."""
        if self._embedder is None:
            return
        ids = [_as_str(i) for i in self._client.zrange(self._ids_key(), 0, -1)]
        if not ids:
            return
        # One pipelined round trip instead of one EXISTS per entry — this runs on
        # every open, over the whole store.
        pipeline = self._client.pipeline()
        for mid in ids:
            pipeline.exists(self._vec_key(mid))
        missing = [mid for mid, present in zip(ids, pipeline.execute()) if not present]
        if not missing:
            return
        log.info("Backfilling %d Redis vectors into %s", len(missing), self._index)
        for start in range(0, len(missing), _VECTOR_BACKFILL_BATCH_SIZE):
            batch = self._get_many(missing[start : start + _VECTOR_BACKFILL_BATCH_SIZE])
            if not batch:
                continue
            vectors = self._embedder([search_document(e) for e in batch])
            pipeline = self._client.pipeline()
            for entry, vector in zip(batch, vectors):
                pipeline.hset(self._vec_key(entry.id), mapping=self._vec_fields(entry, vector))
            pipeline.execute()

    @staticmethod
    def _vec_fields(entry: MemoryEntry, vector: list[float]) -> dict[str, Any]:
        return {
            "embedding": _pack_float32(vector),
            # RediSearch's tokeniser does not split on newlines, so the newlines
            # search_document() uses to separate query/content/tags would fuse the
            # words either side of them into one unsearchable term. The embedder
            # is fine with them; the inverted index is not.
            "text": search_document(entry).replace("\n", " "),
            "scope": entry.scope.value,
            "archived": "1" if entry.archived else "0",
        }

    @property
    def semantic_search_enabled(self) -> bool:
        """True when search() uses real embeddings instead of lexical ranking."""
        return self._vec_enabled

    @property
    def vector_config(self) -> VectorIndexConfig:
        """The HNSW parameters this store's index was built and is queried with."""
        return self._vector_config

    @property
    def count(self) -> int:
        return int(self._client.zcard(self._ids_key()))

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        entry.refresh_state()
        data = self._entry_to_dict(entry)
        pipeline = self._client.pipeline()
        pipeline.set(self._key(entry.id), json.dumps(data))
        pipeline.zadd(self._ids_key(), {entry.id: entry.created_at.timestamp()})
        pipeline.sadd(self._scope_key(entry.scope.value), entry.id)
        if self._vec_enabled and self._embedder is not None:
            vector = self._embedder([search_document(entry)])[0]
            pipeline.hset(self._vec_key(entry.id), mapping=self._vec_fields(entry, vector))
        pipeline.execute()
        return entry

    def touch(self, memory_id: str) -> bool:
        """Bump access_count / last_accessed_at without re-embedding the entry.

        REPLAY calls this on every hit, so it must not go through store() —
        that would re-run the embedder for a change the vector index cannot see.
        """
        raw = self._client.get(self._key(memory_id))
        if not raw:
            return False
        data = json.loads(raw)
        entry = self._dict_to_entry(data)
        entry.touch()
        data["access_count"] = entry.access_count
        data["last_accessed_at"] = (
            entry.last_accessed_at.isoformat() if entry.last_accessed_at else None
        )
        self._client.set(self._key(memory_id), json.dumps(data))
        return True

    def get(self, memory_id: str) -> MemoryEntry | None:
        raw = self._client.get(self._key(memory_id))
        if not raw:
            return None
        return self._dict_to_entry(json.loads(raw))

    def _get_many(self, memory_ids: list[str]) -> list[MemoryEntry]:
        """Fetch many entries in one MGET, skipping any that have gone.

        One round trip rather than one per ID. This is the difference between a
        millisecond and a hundred: every keyword search reads the whole candidate
        set, so at N entries the per-ID version cost N round trips per query.
        """
        if not memory_ids:
            return []
        raws = self._client.mget([self._key(mid) for mid in memory_ids])
        return [self._dict_to_entry(json.loads(raw)) for raw in raws if raw]

    def update(self, entry: MemoryEntry) -> MemoryEntry:
        existing = self.get(entry.id)
        if existing and existing.scope != entry.scope:
            # Remove from old scope set if scope changed
            self._client.srem(self._scope_key(existing.scope.value), entry.id)
        return self.store(entry)

    def delete(self, memory_id: str) -> bool:
        existing = self.get(memory_id)
        if not existing:
            return False
        pipeline = self._client.pipeline()
        pipeline.delete(self._key(memory_id))
        pipeline.delete(self._vec_key(memory_id))
        pipeline.zrem(self._ids_key(), memory_id)
        pipeline.srem(self._scope_key(existing.scope.value), memory_id)
        pipeline.execute()
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
        if scopes:
            raw_ids: set[str] = set()
            pipeline = self._client.pipeline()
            for scope in scopes:
                pipeline.smembers(self._scope_key(scope.value))
            for members in pipeline.execute():
                raw_ids.update(members)
        else:
            raw_ids = set(self._client.zrange(self._ids_key(), 0, -1))

        entries: list[MemoryEntry] = []
        for entry in self._get_many(list(raw_ids)):
            if not include_archived and entry.archived:
                continue
            if not include_expired and entry.is_expired:
                continue
            if memory_type and entry.type != memory_type:
                continue
            entries.append(entry)

        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return entries[offset : offset + limit]

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if self._vec_enabled and self._embedder is not None:
            return self._vector_search(
                query,
                top_k=top_k,
                scopes=scopes,
                include_archived=include_archived,
                include_expired=include_expired,
            )
        # Without RediSearch or an embedding model, "semantic" search is lexical.
        # Install agent-memory-sdk[semantic] and run Redis 8+ / Redis Stack for
        # true KNN.
        return self.keyword_search(
            query,
            top_k=top_k,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )

    @staticmethod
    def _tag_filters(
        *, scopes: list[MemoryScope] | None, include_archived: bool
    ) -> list[str]:
        """RediSearch TAG clauses shared by the vector and keyword queries.

        Expiry is deliberately absent and checked in Python instead: it is
        time-dependent, so baking it into the index would mean rewriting every
        hash as the clock moves.
        """
        filters: list[str] = []
        if not include_archived:
            filters.append("@archived:{0}")
        if scopes:
            filters.append("@scope:{" + "|".join(s.value for s in scopes) + "}")
        return filters

    def _vector_search(
        self,
        query: str,
        *,
        top_k: int,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
    ) -> list[tuple[MemoryEntry, float]]:
        assert self._embedder is not None
        query_vector = self._embedder([query])[0]

        # Pre-filter on the indexed TAG fields so the KNN walk only visits
        # candidates the caller can actually see.
        filters = self._tag_filters(scopes=scopes, include_archived=include_archived)
        prefilter = " ".join(filters) if filters else "*"

        # Over-fetch so the Python-side expiry filter still leaves top_k hits.
        knn_k = self._vector_config.candidate_pool(top_k)
        # EF_RUNTIME is a per-query HNSW knob and belongs inside the KNN
        # brackets, not among FT.SEARCH's own arguments. It can never be below
        # the requested k, or RediSearch cannot return that many neighbours.
        ef_runtime = max(self._vector_config.ef_search, knn_k)
        expr = (
            f"({prefilter})=>"
            f"[KNN {knn_k} @embedding $vec EF_RUNTIME {ef_runtime} AS __dist]"
        )
        # RETURN 1 __dist is load-bearing, not an optimisation: the default reply
        # includes every indexed field, and the raw FLOAT32 embedding blob makes
        # a decode_responses=True client raise UnicodeDecodeError while parsing.
        reply = self._client.execute_command(
            "FT.SEARCH", self._index, expr,
            "PARAMS", "2", "vec", _pack_float32(query_vector),
            "SORTBY", "__dist", "ASC",
            "RETURN", "1", "__dist",
            "LIMIT", "0", str(knn_k),
            "DIALECT", "2",
        )

        # Strip exactly the key prefix rather than splitting on ":vec:", which a
        # caller's own key_prefix could also contain.
        key_prefix_len = len(self._vec_key(""))
        distances = {
            key[key_prefix_len:]: distance
            for key, distance in self._parse_search_reply(reply)
        }
        if not distances:
            return []

        matches: list[tuple[MemoryEntry, float]] = []
        # One MGET for the whole KNN result rather than a GET per hit.
        for entry in self._get_many(list(distances)):
            if not include_archived and entry.archived:
                continue
            if not include_expired and entry.is_expired:
                continue
            # RediSearch reports cosine *distance* in [0, 2]; 1 - d is the
            # cosine similarity, matching every other backend's score scale.
            matches.append((entry, max(0.0, 1.0 - distances[entry.id])))

        matches.sort(key=lambda pair: pair[1], reverse=True)
        return matches[:top_k]

    @staticmethod
    def _parse_search_reply(reply: Any) -> list[tuple[str, float]]:
        """Extract (key, distance) pairs from a raw FT.SEARCH reply.

        RESP2 replies are a flat ``[total, key, [field, value, ...], ...]`` list;
        RESP3 replies are a map keyed by ``results``. Handle both so the store
        works whichever protocol the caller's client negotiated.
        """
        if isinstance(reply, dict):
            hits: list[tuple[str, float]] = []
            for result in reply.get("results", []):
                fields = result.get("extra_attributes") or {}
                distance = fields.get("__dist")
                if distance is None:
                    continue
                hits.append((_as_str(result.get("id", "")), float(distance)))
            return hits

        hits = []
        items = list(reply)[1:]  # drop the leading total
        for key, fields in zip(items[::2], items[1::2]):
            tokens = [_as_str(t) for t in fields] if isinstance(fields, (list, tuple)) else []
            if "__dist" not in tokens:
                continue
            hits.append((_as_str(key), float(tokens[tokens.index("__dist") + 1])))
        return hits

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if self._vec_enabled:
            return self._index_keyword_search(
                query,
                top_k=top_k,
                scopes=scopes,
                include_archived=include_archived,
                include_expired=include_expired,
            )
        return self._python_keyword_search(
            query,
            top_k=top_k,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )

    def _index_keyword_search(
        self,
        query: str,
        *,
        top_k: int,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
    ) -> list[tuple[MemoryEntry, float]]:
        """Rank with RediSearch's own text index instead of Python BM25.

        The retriever runs a keyword pass on *every* resolve, so doing it in
        Python means reading and re-scoring the whole corpus per query — O(N) work
        behind an O(log N) vector index. RediSearch already has an inverted index
        over the same hashes; this asks it the question instead.
        """
        tokens = [t for t in _tokenize(query) if t not in STOP_WORDS and len(t) > 1]
        if not tokens:
            tokens = [t for t in _tokenize(query) if len(t) > 1]
        if not tokens:
            return []

        filters = self._tag_filters(scopes=scopes, include_archived=include_archived)
        prefilter = f"({' '.join(filters)}) " if filters else ""
        # OR over tokens: requiring all of them would drop any entry missing one
        # query word, which is recall the scorer is there to rank, not exclude.
        # _tokenize yields \w+ only, so no RediSearch metacharacter can appear.
        expr = f"{prefilter}@text:({'|'.join(tokens)})"

        pool = self._vector_config.candidate_pool(top_k)
        reply = self._client.execute_command(
            "FT.SEARCH", self._index, expr,
            "WITHSCORES",
            "RETURN", "0",          # ids and scores only; entries come via MGET
            "LIMIT", "0", str(pool),
            "DIALECT", "2",
        )

        key_prefix_len = len(self._vec_key(""))
        scores = {
            key[key_prefix_len:]: score
            for key, score in self._parse_scored_reply(reply)
        }
        if not scores:
            return []

        max_score = max(scores.values())
        results: list[tuple[MemoryEntry, float]] = []
        for entry in self._get_many(list(scores)):
            if not include_expired and entry.is_expired:
                continue
            # Same transform the other backends apply to a raw engine score:
            # normalise to the best hit, then scale by query-term coverage so a
            # one-word overlap cannot score a perfect 1.0.
            coverage = query_coverage(query, search_document(entry))
            raw = scores[entry.id]
            score = (raw / max_score) * (0.5 + 0.5 * coverage) if max_score > 0 else 0.0
            if score > 0:
                results.append((entry, score))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:top_k]

    def _python_keyword_search(
        self,
        query: str,
        *,
        top_k: int,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
    ) -> list[tuple[MemoryEntry, float]]:
        """BM25 in Python — the fallback when there is no RediSearch index."""
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

    @staticmethod
    def _parse_scored_reply(reply: Any) -> list[tuple[str, float]]:
        """Extract (key, score) pairs from an FT.SEARCH ... WITHSCORES reply.

        With ``RETURN 0`` the RESP2 shape is a flat ``[total, key, score, …]``
        list — no per-document field list, unlike the KNN reply.
        """
        if isinstance(reply, dict):
            return [
                (_as_str(result.get("id", "")), float(result.get("score", 0.0)))
                for result in reply.get("results", [])
            ]
        items = list(reply)[1:]  # drop the leading total
        return [
            (_as_str(key), float(_as_str(score)))
            for key, score in zip(items[::2], items[1::2])
        ]

    def stats(self) -> dict[str, Any]:
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

    @staticmethod
    def _entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "query": entry.query,
            "response": entry.response,
            "content": entry.content,
            "type": entry.type.value,
            "scope": entry.scope.value,
            "metadata": entry.metadata,
            "tags": entry.tags,
            "confidence": entry.confidence,
            "requires_verification": entry.requires_verification,
            "archived": entry.archived,
            "state": entry.state.value,
            "access_count": entry.access_count,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "last_accessed_at": (
                entry.last_accessed_at.isoformat() if entry.last_accessed_at else None
            ),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
        }

    @staticmethod
    def _dict_to_entry(data: dict[str, Any]) -> MemoryEntry:
        last_accessed = data.get("last_accessed_at")
        expires_at = data.get("expires_at")
        return MemoryEntry(
            id=data["id"],
            query=data["query"],
            response=data["response"],
            content=data.get("content", data["response"]),
            type=MemoryType(data.get("type", "conversation")),
            scope=MemoryScope(data.get("scope", "user")),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            confidence=float(data.get("confidence", 1.0)),
            requires_verification=bool(data.get("requires_verification", False)),
            archived=bool(data.get("archived", False)),
            state=MemoryState(data.get("state", "active")),
            access_count=int(data.get("access_count", 0)),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
            expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
        )
