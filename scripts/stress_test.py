#!/usr/bin/env python
"""Stress test — measure resolve() latency, CPU, and memory at 10K / 100K / 1M.

Usage:
    python scripts/stress_test.py --memories 10000 --no-cache
    python scripts/stress_test.py --memories 100000 --queries 1000 --no-cache
    python scripts/stress_test.py --memories 1000000 --fast-seed
    python scripts/stress_test.py --memories 10000 --json

Data files (JSONL — extend as needed):
    benchmarks/stress/memories.jsonl
    benchmarks/stress/queries.jsonl

Always use --no-cache for accurate storage-layer latency.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

from agent_memory.manager import Memory

_DATA_DIR = Path(__file__).parent.parent / "benchmarks" / "stress"
_JSON_OUTPUT = False


def _progress(message: str) -> None:
    print(message, file=sys.stderr if _JSON_OUTPUT else sys.stdout, flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get_templates() -> list[dict]:
    p = _DATA_DIR / "memories.jsonl"
    return _load_jsonl(p) if p.exists() else [
        {"query": f"Q{i}", "response": f"A{i}", "type": "conversation", "scope": "global", "tags": []}
        for i in range(16)
    ]


def _get_queries() -> list[str]:
    p = _DATA_DIR / "queries.jsonl"
    return [r["query"] for r in _load_jsonl(p)] if p.exists() else ["What is the API rate limit?"]


class _Snap:
    def __init__(self) -> None:
        if _PSUTIL:
            proc = psutil.Process(os.getpid())
            self.cpu = proc.cpu_times().user
            self.rss = proc.memory_info().rss / 1_048_576
        else:
            self.cpu = self.rss = 0.0

    def delta(self, other: _Snap) -> dict:
        return {
            "cpu_user_s":  round(other.cpu - self.cpu, 3),
            "rss_start_mb": round(self.rss, 1),
            "rss_end_mb":   round(other.rss, 1),
            "rss_delta_mb": round(other.rss - self.rss, 1),
        }


def _seed_normal(memory: Memory, n: int, templates: list[dict]) -> float:
    import uuid
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    store = memory.store
    t0 = time.perf_counter()
    batch: list[MemoryEntry] = []
    now = datetime.now(timezone.utc)

    for i in range(n):
        tpl = templates[i % len(templates)]
        e = MemoryEntry(
            id=str(uuid.uuid4()),
            query=tpl["query"] + (f" [{i}]" if i >= len(templates) else ""),
            response=tpl["response"], content=tpl["response"],
            type=tpl.get("type", "conversation"), scope=tpl.get("scope", "user"),
            tags=list(tpl.get("tags", [])),
            requires_verification=bool(tpl.get("requires_verification", False)),
            confidence=random.uniform(0.75, 1.0),
            created_at=now, updated_at=now,
        )
        batch.append(e)
        if len(batch) >= 500:
            for b in batch:
                store.store(b)
            batch.clear()
            if (i + 1) % 10_000 == 0:
                _progress(f"  {i+1:,}/{n:,}  ({(i+1)/(time.perf_counter()-t0):.0f}/s)")

    for b in batch:
        store.store(b)
    memory.retriever.invalidate_cache()
    return time.perf_counter() - t0


def _seed_fast(memory: Memory, n: int, templates: list[dict]) -> float:
    """Single-transaction bulk insert + FTS5 rebuild — 20-50x faster for SQLite."""
    import uuid as _uuid
    from datetime import datetime, timezone

    store = memory.store
    if not hasattr(store, "_connect"):
        return _seed_normal(memory, n, templates)

    t0 = time.perf_counter()
    conn = store._connect()
    now_iso = datetime.now(timezone.utc).isoformat()

    conn.execute("BEGIN")
    for i in range(n):
        tpl = templates[i % len(templates)]
        conn.execute(
            "INSERT OR IGNORE INTO memories "
            "(id,query,response,content,type,scope,metadata,tags,confidence,"
            "requires_verification,archived,state,access_count,created_at,updated_at,expires_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_uuid.uuid4().hex,
             tpl["query"] + (f" [{i}]" if i >= len(templates) else ""),
             tpl["response"], tpl["response"],
             tpl.get("type","conversation"), tpl.get("scope","user"),
             "{}", json.dumps(tpl.get("tags",[])),
             random.uniform(0.75, 1.0), int(tpl.get("requires_verification",False)),
             0, "active", 0, now_iso, now_iso, None),
        )
        if (i + 1) % 100_000 == 0:
            _progress(f"  {i+1:,}/{n:,}  ({(i+1)/(time.perf_counter()-t0):,.0f}/s)")

    conn.commit()
    _progress("  Rebuilding FTS5 …")
    conn.execute("DELETE FROM memories_fts")
    conn.execute(
        "INSERT INTO memories_fts(rowid,search_text) "
        "SELECT rowid, query||char(10)||content||char(10)||tags FROM memories"
    )
    conn.commit()
    store.close()
    memory.retriever.invalidate_cache()
    # Rebuild Bloom filter and DynamicStopWords after bulk insert bypass
    if hasattr(store, 'rebuild_indexes'):
        store.rebuild_indexes()
    return time.perf_counter() - t0


def _measure(memory: Memory, n_queries: int, variants: list[str], warm_up: int = 50) -> dict:
    queries = [variants[i % len(variants)] for i in range(n_queries + warm_up)]
    for q in queries[:warm_up]:
        memory.resolve(q)

    s0 = _Snap()
    lats: list[float] = []
    actions: dict[str, int] = {}
    cache_hits = 0

    for q in queries[warm_up:]:
        t0 = time.perf_counter()
        d = memory.resolve(q)
        ms = (time.perf_counter() - t0) * 1000
        lats.append(ms)
        actions[d.action.value] = actions.get(d.action.value, 0) + 1
        if ms < 0.5:
            cache_hits += 1

    s1 = _Snap()
    res = s0.delta(s1)
    lats.sort()
    n = len(lats)

    return {
        "n_queries": n,
        "cache_hit_pct":  round(100 * cache_hits / max(n,1), 1),
        "avg_ms":    round(statistics.mean(lats), 3),
        "median_ms": round(lats[n//2], 3),
        "p75_ms":    round(lats[int(n*0.75)], 3),
        "p90_ms":    round(lats[int(n*0.90)], 3),
        "p95_ms":    round(lats[int(n*0.95)], 3),
        "p99_ms":    round(lats[int(n*0.99)], 3),
        "max_ms":    round(max(lats), 3),
        "actions":   actions,
        "hit_rate_pct": round(100*sum(actions.get(a,0) for a in ("replay","restore","verify"))/max(n,1),1),
        "query_cpu_s":      res["cpu_user_s"],
        "query_rss_start_mb": res["rss_start_mb"],
        "query_rss_end_mb":   res["rss_end_mb"],
        "query_rss_delta_mb": res["rss_delta_mb"],
    }


def run(n_memories, *, data_dir=None, n_queries=1000, disable_cache=True,
    fast_seed=False, json_out=False, enable_embeddings: bool | str = "auto") -> dict:
    global _JSON_OUTPUT
    _JSON_OUTPUT = json_out
    templates = _get_templates()
    variants  = _get_queries()

    tmp = tempfile.TemporaryDirectory() if data_dir is None else None
    dir_path = tmp.name if tmp else data_dir

    try:
        memory = Memory(
            persist_dir=dir_path,
            collection_name=f"stress_{n_memories}",
            enable_embeddings=enable_embeddings,
        )
        if disable_cache:
            memory.retriever._cache._maxsize = 0  # type: ignore[attr-defined]

        if not json_out:
            print(f"\n{'='*62}")
            print(f"  STRESS TEST  {n_memories:,} memories  "
                  f"({'no cache' if disable_cache else 'with cache'}"
                  f"{', fast-seed' if fast_seed else ''})")
            print(f"{'='*62}")
            print("\n[1/2] Seeding …")

        s0 = _Snap()
        seed_t = (_seed_fast if fast_seed else _seed_normal)(memory, n_memories, templates)
        s1 = _Snap()
        seed_res = s0.delta(s1)
        count = memory.store.count
        rate = n_memories / seed_t

        if not json_out:
            print(f"  {count:,} entries in {seed_t:.1f}s  ({rate:.0f}/s)  "
                  f"CPU={seed_res['cpu_user_s']:.1f}s  RSS Δ={seed_res['rss_delta_mb']:+.0f} MB")
            print(f"\n[2/2] Measuring {n_queries:,} queries …")

        stats = _measure(memory, n_queries, variants)
        result = {
            "n_memories": n_memories, "actual_count": count,
            "semantic_search_enabled": bool(
                getattr(memory.store, "semantic_search_enabled", False)
            ),
            "cache_enabled": not disable_cache, "fast_seed": fast_seed,
            "seed_s": round(seed_t,2), "seed_rate": round(rate,0),
            "seed_cpu_s":      seed_res["cpu_user_s"],
            "seed_rss_start_mb": seed_res["rss_start_mb"],
            "seed_rss_end_mb":   seed_res["rss_end_mb"],
            "seed_rss_delta_mb": seed_res["rss_delta_mb"],
            **stats,
        }

        if json_out:
            print(json.dumps(result, indent=2))
        else:
            _report(result)
        return result
    finally:
        if tmp:
            tmp.cleanup()


def _report(r: dict) -> None:
    psutil_note = "" if _PSUTIL else " (install psutil for metrics)"
    print(f"\n{'─'*62}")
    print(f"  {r['n_memories']:,} memories{psutil_note}")
    print(f"{'─'*62}")
    print(f"  Seed:      {r['actual_count']:,} entries  {r['seed_s']:.1f}s  {r['seed_rate']:,.0f}/s")
    print(f"  Seed CPU:  {r['seed_cpu_s']:.1f}s user  |  "
          f"RSS: {r['seed_rss_start_mb']:.0f}→{r['seed_rss_end_mb']:.0f} MB  "
          f"(Δ{r['seed_rss_delta_mb']:+.0f} MB)")
    print(f"\n  Queries:   {r['n_queries']:,}  cache={r['cache_hit_pct']:.0f}%  "
          f"hit={r['hit_rate_pct']:.0f}%  actions={r['actions']}")
    print("\n  Latency (ms):")
    for k, latval in [("avg",r['avg_ms']),("p50",r['median_ms']),("p75",r['p75_ms']),
                ("p90",r['p90_ms']),("p95",r['p95_ms']),("p99",r['p99_ms']),("max",r['max_ms'])]:
        print(f"    {k:4}  {latval:8.2f}")
    print(f"\n  Query CPU: {r['query_cpu_s']:.2f}s user  |  "
          f"RSS: {r['query_rss_start_mb']:.0f}→{r['query_rss_end_mb']:.0f} MB  "
          f"(Δ{r['query_rss_delta_mb']:+.0f} MB)")
    print(f"{'─'*62}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--memories",  type=int, default=10_000)
    p.add_argument("--queries",   type=int, default=1_000)
    p.add_argument("--data-dir",  default=None)
    p.add_argument("--no-cache",  action="store_true", default=True)
    p.add_argument("--cache",     action="store_true", help="Enable LRU cache")
    p.add_argument("--fast-seed", action="store_true", help="Bulk insert (SQLite only)")
    p.add_argument(
        "--embeddings",
        choices=("auto", "on", "off"),
        default="auto",
        help="embedding mode: auto (default), on, or off",
    )
    p.add_argument("--json",      action="store_true")
    args = p.parse_args()
    random.seed(42)
    embedding_mode: bool | str = {"auto": "auto", "on": True, "off": False}[args.embeddings]
    run(args.memories, data_dir=args.data_dir, n_queries=args.queries,
        disable_cache=not args.cache, fast_seed=args.fast_seed, json_out=args.json,
        enable_embeddings=embedding_mode)


if __name__ == "__main__":
    main()
