from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from agent_memory.embeddings import Embedder, embedding_dimension, get_default_embedder
from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType
from agent_memory.store import STOP_WORDS, MemoryStore, _tokenize, bm25_scores, search_document
from agent_memory.vector_index import VectorIndexConfig

log = get_logger(__name__)

# Qdrant's own recommended defaults, with a deeper query-time list because Qdrant
# filters server-side and can afford it. Pass vector_config= to override any of it.
DEFAULT_VECTOR_CONFIG = VectorIndexConfig(m=16, ef_construction=100, ef_search=128)

# Entries that never expire still get an ``expires_at_ts`` so the TTL filter is a
# single indexed range check. A missing key would need an is-empty condition,
# which Qdrant cannot serve from the numeric index.
_NEVER_EXPIRES_TS = 1e18

# Bounds on the candidate pool pulled back for Python-side BM25 scoring, so one
# very common token cannot turn a keyword query into a full scan.
DEFAULT_KEYWORD_CANDIDATE_FLOOR = 64
DEFAULT_KEYWORD_CANDIDATE_CAP = 512


class QdrantMemoryStore(MemoryStore):
    """Qdrant-backed memory store for collections too large for SQLite or Postgres.

    Requires: pip install agent-memory-sdk[qdrant]

    Use it when the corpus has outgrown a general-purpose database. Qdrant's
    published scaling limits are far beyond what this repo measures (see
    ``docs/benchmarks.md``), so treat them as Qdrant's claims rather than ours.

    Retrieval is hybrid, and each half runs where it belongs:

    - ``search()`` is dense HNSW KNN inside Qdrant, filtered server-side on the
      indexed scope / archived / expiry payload fields.
    - ``keyword_search()`` narrows candidates with Qdrant's full-text payload
      index, then scores them with the same BM25 pass every other backend uses,
      so scores stay comparable across backends.

    :class:`~agent_memory.retriever.MemoryRetriever` fuses the two with RRF.

    Unlike the Redis and Postgres backends, Qdrant holds the whole entry in the
    point payload — there is no second system of record to keep in step.
    """

    def __init__(
        self,
        url: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        collection_name: str = "agent_memories",
        embedder: Embedder | None = None,
        enable_embeddings: bool | str = "auto",
        client: Any | None = None,
        path: str | None = None,
        vector_config: VectorIndexConfig | None = None,
        keyword_candidate_floor: int = DEFAULT_KEYWORD_CANDIDATE_FLOOR,
        keyword_candidate_cap: int = DEFAULT_KEYWORD_CANDIDATE_CAP,
        timeout: int = 30,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError:
            raise ImportError(
                "Qdrant backend requires qdrant-client. "
                "Install with: pip install agent-memory-sdk[qdrant]"
            ) from None

        self._models = models
        self._collection = collection_name
        self._vector_config = vector_config or DEFAULT_VECTOR_CONFIG
        if not 1 <= keyword_candidate_floor <= keyword_candidate_cap:
            raise ValueError(
                "keyword_candidate_floor must be >= 1 and <= keyword_candidate_cap, "
                f"got floor={keyword_candidate_floor} cap={keyword_candidate_cap}"
            )
        self._keyword_floor = keyword_candidate_floor
        self._keyword_cap = keyword_candidate_cap
        self._embedder: Embedder | None = None
        self._vec_dim = 0
        self._vec_enabled = False

        if client is not None:
            self._client = client
        elif path is not None:
            # Embedded on-disk mode — handy for tests and single-process apps.
            self._client = QdrantClient(path=path)
        elif url is not None:
            self._client = QdrantClient(url=url, api_key=api_key, timeout=timeout)
        else:
            self._client = QdrantClient(
                host=host, port=port, api_key=api_key, timeout=timeout
            )

        if enable_embeddings is True or enable_embeddings == "auto":
            resolved = embedder or get_default_embedder()
            if resolved is None:
                if enable_embeddings is True:
                    raise ImportError(
                        "enable_embeddings=True requires an embedding model. "
                        "Install with: pip install agent-memory-sdk[semantic]"
                    )
            else:
                self._embedder = resolved
                self._vec_dim = embedding_dimension(resolved)
                self._vec_enabled = True

        self._init_collection()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_collection(self) -> None:
        models = self._models
        if not self._client.collection_exists(self._collection):
            if self._vec_enabled:
                vectors_config: Any = models.VectorParams(
                    size=self._vec_dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=self._vector_config.m,
                        ef_construct=self._vector_config.ef_construction,
                    ),
                )
            else:
                # A payload-only collection: keyword search still works, and a
                # later run with embeddings installed cannot silently attach a
                # vector of the wrong size to it (see the dim check below).
                vectors_config = {}
            self._client.create_collection(
                collection_name=self._collection, vectors_config=vectors_config
            )
        elif self._vec_enabled:
            self._check_vector_dim()

        for field, schema in (
            ("scope", models.PayloadSchemaType.KEYWORD),
            ("type", models.PayloadSchemaType.KEYWORD),
            ("state", models.PayloadSchemaType.KEYWORD),
            ("archived", models.PayloadSchemaType.BOOL),
            ("expires_at_ts", models.PayloadSchemaType.FLOAT),
            ("updated_at_ts", models.PayloadSchemaType.FLOAT),
            ("search_text", models.PayloadSchemaType.TEXT),
        ):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:  # already indexed
                log.debug("payload index %s not created: %s", field, exc)

    def _check_vector_dim(self) -> None:
        """Turn vector search off rather than write vectors the collection rejects."""
        info = self._client.get_collection(self._collection)
        params = info.config.params.vectors
        size = getattr(params, "size", None)
        if size is None:
            log.warning(
                "Qdrant collection %r has no dense vector configured; "
                "falling back to keyword-only search. Recreate it to enable KNN.",
                self._collection,
            )
            self._vec_enabled = False
        elif size != self._vec_dim:
            raise ValueError(
                f"Qdrant collection {self._collection!r} stores {size}-dim vectors "
                f"but the embedder produces {self._vec_dim}. Use a matching "
                f"embedding model, or a different collection_name."
            )

    @property
    def semantic_search_enabled(self) -> bool:
        """True when search() uses real embeddings instead of lexical ranking."""
        return self._vec_enabled

    @property
    def vector_config(self) -> VectorIndexConfig:
        """The HNSW parameters this collection was created and is queried with."""
        return self._vector_config

    # ------------------------------------------------------------------
    # Point identity and payload mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _point_id(memory_id: str) -> str:
        """Map a memory ID onto a Qdrant point ID.

        Qdrant only accepts UUIDs or unsigned ints. MemoryEntry IDs are UUIDs by
        default and pass through unchanged; anything else (a caller-supplied slug)
        is hashed to a stable UUID5, with the original kept in the payload.
        """
        try:
            return str(UUID(memory_id))
        except ValueError:
            return str(uuid5(NAMESPACE_URL, memory_id))

    def _entry_to_payload(self, entry: MemoryEntry) -> dict[str, Any]:
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
            "expires_at_ts": (
                _as_timestamp(entry.expires_at)
                if entry.expires_at is not None
                else _NEVER_EXPIRES_TS
            ),
            "updated_at_ts": _as_timestamp(entry.updated_at),
            "search_text": search_document(entry),
        }

    @staticmethod
    def _payload_to_entry(payload: dict[str, Any]) -> MemoryEntry:
        last_accessed = payload.get("last_accessed_at")
        expires_at = payload.get("expires_at")
        return MemoryEntry(
            id=payload["id"],
            query=payload["query"],
            response=payload["response"],
            content=payload.get("content", payload["response"]),
            type=MemoryType(payload.get("type", "conversation")),
            scope=MemoryScope(payload.get("scope", "user")),
            metadata=payload.get("metadata") or {},
            tags=payload.get("tags") or [],
            confidence=float(payload.get("confidence", 1.0)),
            requires_verification=bool(payload.get("requires_verification", False)),
            archived=bool(payload.get("archived", False)),
            state=MemoryState(payload.get("state", "active")),
            access_count=int(payload.get("access_count", 0)),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
            expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return int(self._client.count(self._collection, exact=True).count)

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        entry.refresh_state()
        vector: Any = {}
        if self._vec_enabled and self._embedder is not None:
            vector = self._embedder([search_document(entry)])[0]
        self._client.upsert(
            collection_name=self._collection,
            points=[
                self._models.PointStruct(
                    id=self._point_id(entry.id),
                    vector=vector,
                    payload=self._entry_to_payload(entry),
                )
            ],
            # Reads must see the write: the decision layer stores a memory and
            # resolves against it in the same breath.
            wait=True,
        )
        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[self._point_id(memory_id)],
            with_payload=True,
        )
        if not points:
            return None
        return self._payload_to_entry(points[0].payload or {})

    def update(self, entry: MemoryEntry) -> MemoryEntry:
        return self.store(entry)

    def delete(self, memory_id: str) -> bool:
        if self.get(memory_id) is None:
            return False
        self._client.delete(
            collection_name=self._collection,
            points_selector=self._models.PointIdsList(points=[self._point_id(memory_id)]),
            wait=True,
        )
        return True

    def touch(self, memory_id: str) -> bool:
        """Bump usage counters with a payload patch — no re-embedding.

        REPLAY calls this on every hit; going through store() would re-run the
        embedder for a change no vector can reflect.
        """
        entry = self.get(memory_id)
        if entry is None:
            return False
        entry.touch()
        self._client.set_payload(
            collection_name=self._collection,
            payload={
                "access_count": entry.access_count,
                "last_accessed_at": (
                    entry.last_accessed_at.isoformat() if entry.last_accessed_at else None
                ),
            },
            points=[self._point_id(memory_id)],
            wait=True,
        )
        return True

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        *,
        scopes: list[MemoryScope] | None,
        include_archived: bool,
        include_expired: bool,
        memory_type: MemoryType | None = None,
    ) -> Any:
        models = self._models
        must: list[Any] = []
        must_not: list[Any] = []

        if not include_archived:
            must.append(
                models.FieldCondition(key="archived", match=models.MatchValue(value=False))
            )
        if scopes:
            must.append(
                models.FieldCondition(
                    key="scope", match=models.MatchAny(any=[s.value for s in scopes])
                )
            )
        if memory_type is not None:
            must.append(
                models.FieldCondition(
                    key="type", match=models.MatchValue(value=memory_type.value)
                )
            )
        if not include_expired:
            must.append(
                models.FieldCondition(
                    key="expires_at_ts",
                    range=models.Range(gt=_as_timestamp(datetime.now(timezone.utc))),
                )
            )
            must_not.append(
                models.FieldCondition(
                    key="state", match=models.MatchValue(value=MemoryState.EXPIRED.value)
                )
            )

        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)

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
        query_filter = self._build_filter(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
            memory_type=memory_type,
        )
        # Qdrant orders server-side on the indexed updated_at_ts, but order_by and
        # its point-ID offset are mutually exclusive — so fetch through the offset
        # and drop the prefix here.
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=query_filter,
            limit=limit + offset,
            with_payload=True,
            with_vectors=False,
            order_by=self._models.OrderBy(key="updated_at_ts", direction="desc"),
        )
        entries = [self._payload_to_entry(p.payload or {}) for p in points]
        return entries[offset : offset + limit]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        if not self._vec_enabled or self._embedder is None:
            # Without an embedding model there is nothing semantic to do.
            # Install agent-memory-sdk[semantic] for true KNN.
            return self.keyword_search(
                query,
                top_k=top_k,
                scopes=scopes,
                include_archived=include_archived,
                include_expired=include_expired,
            )

        query_vector = self._embedder([query])[0]
        query_filter = self._build_filter(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            search_params=self._models.SearchParams(
                hnsw_ef=self._vector_config.ef_search
            ),
        )
        # Distance.COSINE makes Qdrant report cosine similarity directly, already
        # on the 0..1 scale the other backends use (negatives clamp to 0).
        return [
            (self._payload_to_entry(point.payload or {}), max(0.0, float(point.score)))
            for point in response.points
        ]

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        models = self._models
        tokens = [t for t in _tokenize(query) if t not in STOP_WORDS and len(t) > 1]
        if not tokens:
            tokens = [t for t in _tokenize(query) if len(t) > 1]
        if not tokens:
            return []

        base = self._build_filter(
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        # MatchText is AND over the phrase, which would miss any entry lacking one
        # query word. One condition per token in `should` gives OR recall instead
        # (Qdrant requires at least one `should` to match), and BM25 below does
        # the actual ranking.
        text_filter = models.Filter(
            must=list(getattr(base, "must", None) or []) or None,
            must_not=list(getattr(base, "must_not", None) or []) or None,
            should=[
                models.FieldCondition(key="search_text", match=models.MatchText(text=token))
                for token in tokens
            ],
        )

        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=text_filter,
            limit=min(
                max(top_k * self._vector_config.overfetch * 5, self._keyword_floor),
                self._keyword_cap,
            ),
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return []

        entries = [self._payload_to_entry(p.payload or {}) for p in points]
        documents = [search_document(e) for e in entries]
        return bm25_scores(query, entries, documents, top_k)

    # ------------------------------------------------------------------
    # Aggregates and maintenance
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        models = self._models
        now = _as_timestamp(datetime.now(timezone.utc))

        archived = models.Filter(
            must=[models.FieldCondition(key="archived", match=models.MatchValue(value=True))]
        )
        expired = models.Filter(
            must=[
                models.FieldCondition(key="archived", match=models.MatchValue(value=False))
            ],
            should=[
                models.FieldCondition(
                    key="state", match=models.MatchValue(value=MemoryState.EXPIRED.value)
                ),
                models.FieldCondition(
                    key="expires_at_ts", range=models.Range(lte=now)
                ),
            ],
        )

        total = self.count
        archived_n = self._count(archived)
        expired_n = self._count(expired)
        by_state = {
            key: value
            for key, value in (
                ("archived", archived_n),
                ("expired", expired_n),
                ("active", total - archived_n - expired_n),
            )
            if value
        }

        by_type: dict[str, int] = {}
        for memory_type in MemoryType:
            n = self._count(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type", match=models.MatchValue(value=memory_type.value)
                        )
                    ]
                )
            )
            if n:
                by_type[memory_type.value] = n

        # access_count has no payload index to aggregate over, so this is the one
        # figure that needs a scan. Scroll payloads only, never vectors.
        total_access = 0
        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=1_000,
                offset=offset,
                with_payload=["access_count"],
                with_vectors=False,
            )
            total_access += sum(int((p.payload or {}).get("access_count", 0)) for p in points)
            if offset is None:
                break

        return {
            "total": total,
            "by_state": by_state,
            "by_type": by_type,
            "total_access_count": total_access,
        }

    def _count(self, query_filter: Any) -> int:
        return int(
            self._client.count(
                self._collection, count_filter=query_filter, exact=True
            ).count
        )

    def cleanup_expired(self, *, delete: bool = False) -> dict[str, int]:
        models = self._models
        now = _as_timestamp(datetime.now(timezone.utc))
        expired = models.Filter(
            should=[
                models.FieldCondition(
                    key="state", match=models.MatchValue(value=MemoryState.EXPIRED.value)
                ),
                models.FieldCondition(key="expires_at_ts", range=models.Range(lte=now)),
            ]
        )

        if delete:
            deleted = self._count(expired)
            if deleted:
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=models.FilterSelector(filter=expired),
                    wait=True,
                )
            return {"expired": 0, "deleted": deleted}

        due = models.Filter(
            must=[
                models.FieldCondition(key="archived", match=models.MatchValue(value=False)),
                models.FieldCondition(key="expires_at_ts", range=models.Range(lte=now)),
            ],
            must_not=[
                models.FieldCondition(
                    key="state", match=models.MatchValue(value=MemoryState.EXPIRED.value)
                )
            ],
        )
        if self._count(due):
            self._client.set_payload(
                collection_name=self._collection,
                payload={"state": MemoryState.EXPIRED.value},
                points=models.FilterSelector(filter=due).filter,
                wait=True,
            )
        return {"expired": self._count(expired), "deleted": 0}


def _as_timestamp(value: datetime) -> float:
    """Epoch seconds, treating naive datetimes as UTC like MemoryEntry does."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()
