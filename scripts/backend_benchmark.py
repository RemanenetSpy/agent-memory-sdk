#!/usr/bin/env python
"""Compare storage backends on the same corpus, queries, and embedder.

Answers two separate questions that are easy to conflate:

  * **How fast?**  ``resolve()`` end-to-end, plus the two halves it fuses —
    ``store.search()`` (the vector index) and ``store.keyword_search()`` — so a
    slow BM25 half cannot hide behind a fast KNN half.
  * **How good?**  Two different things, kept apart:
      - *Recall@k* — retrieval quality. Of the queries that have a relevant
        memory, how often does one appear in the top k? A property of the index,
        independent of any threshold. Reported for the raw vector index and for
        the RRF-fused retriever.
      - *Answer rate* and *refusal precision* — decision-layer behaviour at a
        given ``restore_threshold``. Not recall: a query can retrieve the right
        memory at rank 1 and still return NONE because the score sat below the
        threshold.

Every backend shares one loaded embedding model and one seeded corpus, and the
retrieval cache is disabled, so the only variable is the storage layer.

Usage:
    # Everything reachable (skips backends whose server is down)
    python scripts/backend_benchmark.py

    python scripts/backend_benchmark.py --memories 5000 --json
    python scripts/backend_benchmark.py --only sqlite,postgres-hnsw,postgres-ivfflat

Start the servers first:
    docker compose -f docker-compose.dev.yml up -d
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_memory.manager import Memory
from agent_memory.models import MemoryAction, MemoryEntry
from agent_memory.store import MemoryStore
from agent_memory.vector_index import VectorIndexConfig

_DATA_DIR = Path(__file__).parent.parent / "benchmarks" / "stress"

# Actions that count as "the memory was surfaced to the agent".
_SURFACED = (MemoryAction.REPLAY, MemoryAction.RESTORE, MemoryAction.VERIFY)

# Rank cutoffs for Recall@k.
_RECALL_K = (1, 5, 10)

# Queries in this category have no relevant memory by construction, so they score
# refusal precision instead of recall.
_OUT_OF_DOMAIN = "ood"

# Populated by main(); only the report reads it.
_LABELLED: list[dict] = []

# Embeddings score a paraphrase lower than a lexical exact match, so the default
# 0.70 (tuned for the lexical backend) hides recall the vector index did find.
# Both settings are reported rather than picking whichever flatters a backend.
_THRESHOLDS = {"default": 0.70, "tuned": 0.55}


# ---------------------------------------------------------------------------
# Backend definitions — each builds a store, or raises if its server is down
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    note: str
    build: Callable[[str], MemoryStore]
    semantic: bool = True


def _variants(args: argparse.Namespace) -> list[Variant]:
    run_id = uuid.uuid4().hex[:8]
    config = VectorIndexConfig(ef_search=args.ef_search)

    def sqlite_lexical(tmp: str) -> MemoryStore:
        from agent_memory.sqlite_store import SqliteMemoryStore

        return SqliteMemoryStore(persist_dir=tmp, enable_embeddings=False)

    def sqlite_vec(tmp: str) -> MemoryStore:
        from agent_memory.sqlite_store import SqliteMemoryStore

        return SqliteMemoryStore(persist_dir=tmp + "/vec", enable_embeddings=True)

    def redis_vss(_tmp: str) -> MemoryStore:
        from agent_memory.redis_store import RedisMemoryStore

        return RedisMemoryStore(
            url=args.redis_url,
            key_prefix=f"bench_{run_id}",
            enable_embeddings=True,
            vector_config=config,
        )

    def postgres(index: str) -> Callable[[str], MemoryStore]:
        def build(_tmp: str) -> MemoryStore:
            from agent_memory.postgres_store import PostgresMemoryStore

            return PostgresMemoryStore(
                dsn=args.postgres_dsn,
                table_name=f"bench_{index}_{run_id}",
                enable_embeddings=True,
                vector_index=index,
                vector_config=config,
            )

        return build

    def qdrant(_tmp: str) -> MemoryStore:
        from agent_memory.qdrant_store import QdrantMemoryStore

        return QdrantMemoryStore(
            url=args.qdrant_url,
            collection_name=f"bench_{run_id}",
            enable_embeddings=True,
            vector_config=config,
        )

    return [
        Variant("sqlite-lexical", "no server, no embeddings", sqlite_lexical, semantic=False),
        Variant("sqlite-vec", "no server, sqlite-vec KNN", sqlite_vec),
        Variant("redis-vss", "RediSearch HNSW", redis_vss),
        Variant("postgres-hnsw", "pgvector HNSW", postgres("hnsw")),
        Variant("postgres-ivfflat", "pgvector IVFFlat", postgres("ivfflat")),
        Variant("qdrant", "Qdrant HNSW", qdrant),
    ]


# ---------------------------------------------------------------------------
# Corpus and query set
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _corpus(n: int) -> list[MemoryEntry]:
    """n entries built from the stress templates, each one distinct."""
    templates = _load_jsonl(_DATA_DIR / "memories.jsonl")
    now = datetime.now(timezone.utc)
    entries = []
    for i in range(n):
        tpl = templates[i % len(templates)]
        entries.append(
            MemoryEntry(
                query=tpl["query"] + (f" [{i}]" if i >= len(templates) else ""),
                response=tpl["response"],
                content=tpl["response"],
                type=tpl.get("type", "conversation"),
                scope=tpl.get("scope", "user"),
                tags=list(tpl.get("tags", [])),
                requires_verification=bool(tpl.get("requires_verification", False)),
                created_at=now,
                updated_at=now,
            )
        )
    return entries


@dataclass
class Quality:
    """Decision-layer behaviour at one ``restore_threshold``.

    Deliberately not called recall: these count what the decision layer *did*,
    which depends on the threshold as much as on what retrieval found.
    """

    answerable_total: int = 0
    answered: int = 0
    none_total: int = 0
    none_refused: int = 0
    wrong_replay: int = 0

    @property
    def answer_rate(self) -> float:
        """Of queries with a relevant memory, how many surfaced one."""
        return self.answered / self.answerable_total if self.answerable_total else 0.0

    @property
    def refusal_precision(self) -> float:
        """Of queries nothing answers, how many correctly returned NONE."""
        return self.none_refused / self.none_total if self.none_total else 0.0

    @property
    def accuracy(self) -> float:
        total = self.answerable_total + self.none_total
        return (self.answered + self.none_refused) / total if total else 0.0


@dataclass
class Measurement:
    name: str
    note: str
    semantic_enabled: bool = False
    write_rate: float = 0.0
    resolve: dict[str, float] = field(default_factory=dict)
    search: dict[str, float] = field(default_factory=dict)
    keyword: dict[str, float] = field(default_factory=dict)
    quality: dict[str, Quality] = field(default_factory=dict)
    # Recall@k for the raw vector index and for the fused retriever.
    recall_vector: dict[int, float] = field(default_factory=dict)
    recall_fused: dict[int, float] = field(default_factory=dict)
    error: str | None = None


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "avg": round(statistics.mean(ordered), 3),
        "p50": round(ordered[n // 2], 3),
        "p95": round(ordered[min(int(n * 0.95), n - 1)], 3),
        "p99": round(ordered[min(int(n * 0.99), n - 1)], 3),
    }


def _time_calls(fn: Callable[[str], Any], queries: list[str], warmup: int) -> list[float]:
    for q in queries[:warmup]:
        fn(q)
    samples = []
    for q in queries[warmup:]:
        t0 = time.perf_counter()
        fn(q)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def _is_relevant(entry: MemoryEntry, category: str) -> bool:
    """Ground truth: a memory answers a query when it carries the query's topic tag.

    The corpus is seeded from templates that already carry topic tags, and every
    non-``ood`` query category is one of those tags — so relevance is a lookup,
    not a judgement call.
    """
    return category in entry.tags


def _recall_at_k(
    retrieve: Callable[[str, int], list[MemoryEntry]], labelled: list[dict]
) -> dict[int, float]:
    """Recall@k over the queries that have a relevant memory.

    "Recall" in the hit-rate sense the repo's own harness uses: did *at least one*
    relevant memory make the top k? The strict IR ratio would be meaningless here
    — the corpus repeats each template many times, so there are dozens of equally
    relevant entries per query and the ratio would only measure how many
    duplicates fit in k.
    """
    in_domain = [r for r in labelled if r["category"] != _OUT_OF_DOMAIN]
    if not in_domain:
        return {}
    hits = dict.fromkeys(_RECALL_K, 0)
    deepest = max(_RECALL_K)
    for row in in_domain:
        ranked = retrieve(row["query"], deepest)
        for k in _RECALL_K:
            if any(_is_relevant(e, row["category"]) for e in ranked[:k]):
                hits[k] += 1
    return {k: hits[k] / len(in_domain) for k in _RECALL_K}


def _score_quality(memory: Memory, labelled: list[dict]) -> Quality:
    quality = Quality()
    for row in labelled:
        decision = memory.resolve(row["query"])
        if row["category"] == _OUT_OF_DOMAIN:
            quality.none_total += 1
            if decision.action == MemoryAction.NONE:
                quality.none_refused += 1
            elif decision.action == MemoryAction.REPLAY:
                # The worst outcome: a confident answer to a question nothing
                # in the store actually answers.
                quality.wrong_replay += 1
        else:
            quality.answerable_total += 1
            if decision.action in _SURFACED:
                quality.answered += 1
    return quality


def measure(variant: Variant, args: argparse.Namespace, labelled: list[dict]) -> Measurement:
    result = Measurement(variant.name, variant.note)
    latency_queries = [
        labelled[i % len(labelled)]["query"] for i in range(args.queries + args.warmup)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        try:
            store = variant.build(tmp)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}".split("\n")[0][:80]
            return result

        try:
            result.semantic_enabled = bool(store.semantic_search_enabled)

            entries = _corpus(args.memories)
            t0 = time.perf_counter()
            for entry in entries:
                store.store(entry)
            elapsed = time.perf_counter() - t0
            result.write_rate = round(args.memories / elapsed, 1)

            # The two halves the retriever fuses, timed independently.
            result.search = _percentiles(
                _time_calls(lambda q: store.search(q, top_k=5), latency_queries, args.warmup)
            )
            result.keyword = _percentiles(
                _time_calls(
                    lambda q: store.keyword_search(q, top_k=5), latency_queries, args.warmup
                )
            )

            # Recall@k on the index itself — no thresholds, no fusion.
            result.recall_vector = _recall_at_k(
                lambda q, k: [e for e, _ in store.search(q, top_k=k)], labelled
            )

            for label, threshold in _THRESHOLDS.items():
                memory = Memory(store=store, restore_threshold=threshold)
                memory.retriever._cache._maxsize = 0  # measure storage, not the cache
                if label == "default":
                    result.resolve = _percentiles(
                        _time_calls(memory.resolve, latency_queries, args.warmup)
                    )
                    # Recall@k after RRF fusion — thresholds do not apply here
                    # either, so one threshold's retriever is enough.
                    result.recall_fused = _recall_at_k(
                        lambda q, k: [
                            r.entry for r in memory.retriever.retrieve(q, top_k=k)
                        ],
                        labelled,
                    )
                result.quality[label] = _score_quality(memory, labelled)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}".split("\n")[0][:80]
        finally:
            _teardown(variant.name, store, args)

    return result


def _teardown(name: str, store: MemoryStore, args: argparse.Namespace) -> None:
    """Drop whatever the run created, so a benchmark leaves no server state."""
    try:
        if name == "redis-vss":
            client = store._client  # type: ignore[attr-defined]
            try:
                client.execute_command("FT.DROPINDEX", store._index, "DD")  # type: ignore[attr-defined]
            except Exception:
                pass
            keys = client.keys(f"{store._prefix}:*")  # type: ignore[attr-defined]
            if keys:
                client.delete(*keys)
        elif name.startswith("postgres"):
            import psycopg2

            conn = psycopg2.connect(args.postgres_dsn)
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {store._table}")  # type: ignore[attr-defined]
            conn.commit()
            conn.close()
        elif name == "qdrant":
            store._client.delete_collection(store._collection)  # type: ignore[attr-defined]
    except Exception as exc:  # teardown must never mask a result
        print(f"  ! teardown for {name} failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _embed_cost() -> float | None:
    """Per-query embedding time — a constant every semantic backend pays."""
    from agent_memory.embeddings import get_default_embedder

    embedder = get_default_embedder()
    if embedder is None:
        return None
    embedder(["warm up"])
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        embedder(["how do I reset my password"])
        samples.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(samples), 2)


def _markdown(results: list[Measurement], args: argparse.Namespace, embed_ms: float | None) -> str:
    ok = [r for r in results if r.error is None]
    lines: list[str] = []

    lines.append(
        f"Corpus: {args.memories:,} memories · {args.queries} timed queries · "
        f"retrieval cache disabled · one shared bge-small-en-v1.5 embedder"
    )
    if embed_ms is not None:
        lines.append(
            f"Embedding one query costs {embed_ms:.2f} ms on this machine; it is "
            f"included in every `search()` and `resolve()` figure below."
        )
    lines.append("")

    lines.append("### Latency (ms, lower is better)")
    lines.append("")
    lines.append(
        "| Backend | Index | Semantic | `resolve()` p50 | `resolve()` p95 | "
        "`search()` p50 | `keyword_search()` p50 | writes/s |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in ok:
        lines.append(
            f"| `{r.name}` | {r.note} | {'yes' if r.semantic_enabled else 'no'} | "
            f"{r.resolve.get('p50', 0):.2f} | {r.resolve.get('p95', 0):.2f} | "
            f"{r.search.get('p50', 0):.2f} | {r.keyword.get('p50', 0):.2f} | "
            f"{r.write_rate:,.0f} |"
        )
    lines.append("")

    in_domain = len([r for r in results if r]) and sum(
        1 for row in _LABELLED if row["category"] != _OUT_OF_DOMAIN
    )
    lines.append(
        f"### Retrieval quality — Recall@k ({in_domain} queries with a relevant memory)"
    )
    lines.append("")
    lines.append(
        "At least one relevant memory in the top k. No score threshold involved — "
        "this is what the index found."
    )
    lines.append("")
    vec_header = " | ".join(f"vector R@{k}" for k in _RECALL_K)
    fused_header = " | ".join(f"fused R@{k}" for k in _RECALL_K)
    lines.append(f"| Backend | {vec_header} | {fused_header} |")
    lines.append("|---|" + "---|" * (2 * len(_RECALL_K)))
    for r in ok:
        vec = " | ".join(f"{r.recall_vector.get(k, 0):.1%}" for k in _RECALL_K)
        fused = " | ".join(f"{r.recall_fused.get(k, 0):.1%}" for k in _RECALL_K)
        lines.append(f"| `{r.name}` | {vec} | {fused} |")
    lines.append("")

    lines.append("### Decision quality (depends on `restore_threshold`, not the index)")
    lines.append("")
    lines.append(
        "| Backend | Answer rate @0.70 | Answer rate @0.55 | Refusal precision | Wrong REPLAY |"
    )
    lines.append("|---|---|---|---|---|")
    for r in ok:
        default, tuned = r.quality.get("default"), r.quality.get("tuned")
        if default is None or tuned is None:
            continue
        lines.append(
            f"| `{r.name}` | {default.answer_rate:.1%} | {tuned.answer_rate:.1%} | "
            f"{default.refusal_precision:.1%} | {default.wrong_replay} |"
        )

    skipped = [r for r in results if r.error is not None]
    if skipped:
        lines.append("")
        lines.append("Skipped (server unreachable or dependency missing):")
        lines.append("")
        for r in skipped:
            lines.append(f"- `{r.name}` — {r.error}")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memories", type=int, default=2_000)
    p.add_argument("--queries", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--ef-search", type=int, default=64, help="HNSW query-time candidate list")
    p.add_argument("--redis-url", default="redis://localhost:6379/0")
    p.add_argument(
        "--postgres-dsn", default="postgresql://agent_memory:agent_memory@localhost:5432/agent_memory"
    )
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--only", default="", help="comma-separated subset of backend names")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    variants = [v for v in _variants(args) if not wanted or v.name in wanted]
    if wanted - {v.name for v in variants}:
        raise SystemExit(f"unknown backend name(s): {sorted(wanted - {v.name for v in variants})}")

    global _LABELLED
    labelled = _load_jsonl(_DATA_DIR / "queries.jsonl")
    _LABELLED = labelled
    embed_ms = _embed_cost()

    results = []
    for variant in variants:
        print(f"→ {variant.name} …", file=sys.stderr, flush=True)
        result = measure(variant, args, labelled)
        if result.error:
            print(f"  skipped: {result.error}", file=sys.stderr)
        results.append(result)

    if args.json:
        print(
            json.dumps(
                {
                    "config": vars(args),
                    "embed_ms": embed_ms,
                    "results": [
                        {
                            **{
                                k: v
                                for k, v in vars(r).items()
                                if k not in ("quality",)
                            },
                            "quality": {
                                label: {
                                    "answer_rate": round(q.answer_rate, 4),
                                    "refusal_precision": round(q.refusal_precision, 4),
                                    "accuracy": round(q.accuracy, 4),
                                    "wrong_replay": q.wrong_replay,
                                }
                                for label, q in r.quality.items()
                            },
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )
    else:
        print()
        print(_markdown(results, args, embed_ms))


if __name__ == "__main__":
    main()
