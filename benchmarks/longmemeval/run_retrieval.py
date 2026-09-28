"""LongMemEval retrieval-only benchmark — no LLM calls, no API keys.

Measures what the SDK actually is (a retrieval + decision layer) against the
LongMemEval_S dataset (Wu et al., ICLR 2025): 500 questions, each with its own
~48-session timestamped chat haystack and labelled evidence sessions.

Metrics (all computed offline):
  * Session Recall@k  — fraction of questions where >=1 evidence session
    appears in the top-k retrieved entries (deduped by session).
  * Evidence coverage@k — fraction of a question's evidence sessions found
    in the top-k (matters for multi-session questions).
  * Abstention accuracy — for `_abs` questions the correct behaviour is to
    use NO memory; we count decision.action == NONE as correct.
  * False-abstention rate — NONE fired on a question that *does* have
    evidence in the haystack.
    * Retrieval latency p50/p90/p95/p99 per query, ingestion throughput, and
        whole-run CPU/RSS telemetry.

Methodology notes:
  * One fresh SQLite store per question (haystacks are per-question).
  * Consecutive user->assistant turns are paired into one experience entry;
    dangling turns are stored solo. Session id is kept in entry metadata.
  * Lexical-only by default (enable_embeddings=False) — zero model downloads.
    Pass --semantic to enable the embedding pipeline.

Usage:
    python benchmarks/longmemeval/run_retrieval.py               # full 500
    python benchmarks/longmemeval/run_retrieval.py --limit 25    # smoke run
    python benchmarks/longmemeval/run_retrieval.py --semantic
    python benchmarks/longmemeval/run_retrieval.py --semantic \
        --checkpoint /tmp/longmemeval-semantic-checkpoint.json
    python benchmarks/longmemeval/run_retrieval.py --semantic \
        --checkpoint /tmp/longmemeval-semantic-checkpoint.json --resume
    python benchmarks/longmemeval/run_retrieval.py --semantic \
        --embedding-cache-size 10000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_memory import Memory  # noqa: E402
from agent_memory.embeddings import get_default_embedder  # noqa: E402
from agent_memory.models import MemoryAction  # noqa: E402

DATA = Path(__file__).parent / "data" / "longmemeval_s_cleaned.json"
TOP_KS = (5, 10)

# Load the embedding model once — per-question stores share it.
# BGE-small throughput collapses on long inputs (~5 texts/s at ~430 tokens vs
# ~400/s at ~15 tokens), so embedding input is truncated to EMBED_TRUNC chars
# (documented in REPORT.md methodology).  A content cache exploits the ~1.2x
# session reuse across question haystacks. The cache is disabled by default so
# peak RSS represents the benchmark process rather than retained prior stores.
_EMBEDDER = None
EMBED_TRUNC = 1000
_EMB_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_EMB_CACHE_MAXSIZE = 0


def _percentile(samples: list[float], percentile: int) -> float | None:
    """Return the nearest-rank percentile for a non-empty sample."""
    if not samples:
        return None
    rank = max(1, math.ceil(len(samples) * percentile / 100))
    return sorted(samples)[rank - 1]


def _peak_rss_mb() -> float:
    """Return peak process working-set size in MiB on supported platforms."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("Kernel32.dll")
        psapi = ctypes.WinDLL("Psapi.dll")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        success = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not success:
            raise ctypes.WinError()
        return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    bytes_per_rss_unit = 1 if sys.platform == "darwin" else 1024
    return round(usage.ru_maxrss * bytes_per_rss_unit / (1024 * 1024), 2)


def _write_checkpoint(path: Path, rows: list[dict]) -> None:
    """Atomically persist completed rows so an interrupted run can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps({"rows": rows}, indent=2))
    os.replace(temporary_path, path)


def _load_checkpoint(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Checkpoint {path} has invalid rows data")
    return rows


def configure_embedding_cache(maxsize: int) -> None:
    """Configure the optional bounded cross-haystack embedding cache."""
    global _EMBEDDER, _EMB_CACHE_MAXSIZE
    if maxsize < 0:
        raise ValueError("embedding cache size cannot be negative")
    _EMBEDDER = None
    _EMB_CACHE.clear()
    _EMB_CACHE_MAXSIZE = maxsize


def embedding_cache_stats() -> dict[str, float | int]:
    """Return cache retention separately from whole-process RSS."""
    vector_values = sum(len(vector) for vector in _EMB_CACHE.values())
    return {
        "max_entries": _EMB_CACHE_MAXSIZE,
        "entries": len(_EMB_CACHE),
        # Vector values are logically float32; Python-object overhead is
        # intentionally excluded and captured by process_peak_rss_mb instead.
        "vector_payload_mb": round(vector_values * 4 / (1024 * 1024), 2),
    }


def shared_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        base = get_default_embedder()

        def cached(texts: list[str]) -> list[list[float]]:
            keys = [t[:EMBED_TRUNC] for t in texts]
            if _EMB_CACHE_MAXSIZE == 0:
                return base(keys)

            vectors: dict[str, list[float]] = {}
            missing: list[str] = []
            for key in dict.fromkeys(keys):
                vector = _EMB_CACHE.get(key)
                if vector is None:
                    missing.append(key)
                else:
                    _EMB_CACHE.move_to_end(key)
                    vectors[key] = vector
            if missing:
                for k, vec in zip(missing, base(missing)):
                    vectors[k] = vec
                    _EMB_CACHE[k] = vec
                    _EMB_CACHE.move_to_end(k)
                    while len(_EMB_CACHE) > _EMB_CACHE_MAXSIZE:
                        _EMB_CACHE.popitem(last=False)
            return [vectors[k] for k in keys]

        _EMBEDDER = cached
    return _EMBEDDER


def pair_turns(session: list[dict]) -> list[tuple[str, str]]:
    """Pair consecutive user->assistant turns; dangling turns go in solo."""
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for turn in session:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if turn["role"] == "user":
            if pending_user is not None:
                pairs.append((pending_user, pending_user))
            pending_user = content
        else:  # assistant
            if pending_user is not None:
                pairs.append((pending_user, content))
                pending_user = None
            else:
                pairs.append((content, content))
    if pending_user is not None:
        pairs.append((pending_user, pending_user))
    return pairs


def run_question(q: dict, *, semantic: bool) -> dict:
    evidence = set(q["answer_session_ids"])
    is_abstention = q["question_id"].endswith("_abs")

    with tempfile.TemporaryDirectory() as tmp:
        # Ingest lexical-fast; embeddings (if requested) are batch-backfilled
        # afterwards in a single embedder call — ~50x faster than embedding
        # per-entry during remember().
        memory = Memory(persist_dir=tmp, enable_embeddings=False)

        t0 = time.perf_counter()
        n_entries = 0
        for sid, session in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            for user_text, assistant_text in pair_turns(session):
                memory.remember(
                    user_text,
                    assistant_text,
                    metadata={"session_id": sid},
                )
                n_entries += 1
        if semantic:
            # Re-opening with embeddings enabled triggers the store's batched
            # vector backfill over every entry missing an embedding.
            memory.store.close() if hasattr(memory.store, "close") else None
            memory = Memory(
                persist_dir=tmp,
                enable_embeddings=True,
                embedder=shared_embedder(),
            )
        ingest_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        results = memory.retriever.retrieve(
            q["question"], top_k=max(TOP_KS), diversify_key="session_id"
        )
        retrieve_ms = (time.perf_counter() - t1) * 1000

        decision = memory.resolve(q["question"])

    retrieved_sids: list[str] = []
    for r in results:
        sid = (r.entry.metadata or {}).get("session_id")
        if sid and sid not in retrieved_sids:
            retrieved_sids.append(sid)

    row: dict = {
        "question_id": q["question_id"],
        "question_type": "abstention" if is_abstention else q["question_type"],
        "n_entries": n_entries,
        "ingest_s": round(ingest_s, 3),
        "retrieve_ms": round(retrieve_ms, 3),
        "action": decision.action.value
        if isinstance(decision.action, MemoryAction)
        else str(decision.action),
        "n_evidence": len(evidence),
    }
    for k in TOP_KS:
        # top-k *entries*, deduped to sessions
        sids_at_k: list[str] = []
        for r in results[:k]:
            sid = (r.entry.metadata or {}).get("session_id")
            if sid and sid not in sids_at_k:
                sids_at_k.append(sid)
        hits = evidence & set(sids_at_k)
        row[f"any_hit@{k}"] = bool(hits) if evidence else None
        row[f"coverage@{k}"] = (len(hits) / len(evidence)) if evidence else None
    return row


def summarise(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)

    summary: dict = {"n_questions": len(rows), "per_type": {}, "overall": {}}

    def agg(subset: list[dict]) -> dict:
        out: dict = {"n": len(subset)}
        scored = [r for r in subset if r["question_type"] != "abstention"]
        for k in TOP_KS:
            hits = [r[f"any_hit@{k}"] for r in scored if r[f"any_hit@{k}"] is not None]
            covs = [r[f"coverage@{k}"] for r in scored if r[f"coverage@{k}"] is not None]
            if hits:
                out[f"recall@{k}"] = round(sum(hits) / len(hits), 4)
                out[f"coverage@{k}"] = round(sum(covs) / len(covs), 4)
        abst = [r for r in subset if r["question_type"] == "abstention"]
        if abst:
            out["abstention_none_rate"] = round(
                sum(1 for r in abst if r["action"] == "none") / len(abst), 4
            )
        if scored:
            out["false_abstention_rate"] = round(
                sum(1 for r in scored if r["action"] == "none") / len(scored), 4
            )
        return out

    for qtype, subset in sorted(by_type.items()):
        summary["per_type"][qtype] = agg(subset)
    summary["overall"] = agg(rows)

    lat = [r["retrieve_ms"] for r in rows]
    entries = sum(r["n_entries"] for r in rows)
    ingest = sum(r["ingest_s"] for r in rows)
    summary["latency_ms"] = {
        "p50": round(statistics.median(lat), 2),
        "p90": round(_percentile(lat, 90), 2) if lat else None,
        "p95": round(_percentile(lat, 95), 2) if lat else None,
        "p99": round(_percentile(lat, 99), 2) if lat else None,
        "mean": round(statistics.mean(lat), 2) if lat else None,
    }
    summary["ingestion"] = {
        "total_entries": entries,
        "total_seconds": round(ingest, 1),
        "entries_per_second": round(entries / ingest, 1) if ingest else None,
        "llm_calls": 0,
        "api_cost_usd": 0.0,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    ap.add_argument("--semantic", action="store_true", help="enable embedding pipeline")
    ap.add_argument(
        "--embedding-cache-size",
        type=int,
        default=0,
        help="bounded cross-haystack embedding cache entries; 0 disables it (default)",
    )
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="write completed rows here every --checkpoint-every questions",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume from --checkpoint; completed rows are not rerun",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="checkpoint interval in questions (default: 10)",
    )
    ap.add_argument(
        "--data", default=str(DATA),
        help="dataset JSON (default: longmemeval_s_cleaned.json; pass the _M file for the 500-session variant)",
    )
    args = ap.parse_args()
    if args.resume and not args.checkpoint:
        ap.error("--resume requires --checkpoint")
    if args.checkpoint_every < 1:
        ap.error("--checkpoint-every must be at least 1")
    if args.embedding_cache_size < 0:
        ap.error("--embedding-cache-size cannot be negative")
    if args.semantic:
        configure_embedding_cache(args.embedding_cache_size)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    rows = _load_checkpoint(checkpoint_path) if args.resume and checkpoint_path else []
    completed = len(rows)

    data_path = Path(args.data)
    if data_path.suffix == ".jsonl":
        # Stream one question at a time — the _M variant is 2.5GB and must
        # not be held in memory whole.
        def iter_questions():
            with data_path.open() as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

        n_total = sum(1 for line in data_path.open() if line.strip())
        if completed > n_total:
            raise ValueError("Checkpoint has more rows than the input dataset")

        def remaining_questions():
            for index, question in enumerate(iter_questions()):
                if index >= completed:
                    yield question

        questions = remaining_questions()
    else:
        loaded = json.loads(data_path.read_text())
        if args.limit:
            loaded = loaded[: args.limit]
        n_total = len(loaded)
        if completed > n_total:
            raise ValueError("Checkpoint has more rows than the input dataset")
        questions = iter(loaded[completed:])

    t_start = time.time()
    cpu_start = os.times()
    for i, q in enumerate(questions, completed + 1):
        rows.append(run_question(q, semantic=args.semantic))
        del q
        if checkpoint_path and (i % args.checkpoint_every == 0 or i == n_total):
            _write_checkpoint(checkpoint_path, rows)
        if i % 10 == 0 or i == n_total:
            elapsed = time.time() - t_start
            processed = i - completed
            eta = elapsed / processed * (n_total - i) if processed else 0
            print(
                f"[{i}/{n_total}] elapsed {elapsed:.0f}s  eta {eta:.0f}s",
                flush=True,
            )

    summary = summarise(rows)
    cpu_end = os.times()
    summary["execution"] = {
        "run_wall_seconds": round(time.time() - t_start, 2),
        "run_cpu_user_seconds": round(cpu_end.user - cpu_start.user, 2),
        "run_cpu_system_seconds": round(cpu_end.system - cpu_start.system, 2),
        "process_peak_rss_mb": _peak_rss_mb(),
        "resumed_from_questions": completed,
    }
    if args.semantic:
        summary["embedding_cache"] = embedding_cache_stats()
    mode = "semantic" if args.semantic else "lexical"
    out_path = Path(
        args.out
        or Path(__file__).parent / f"results_{mode}_{len(rows)}q.json"
    )
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    if checkpoint_path:
        _write_checkpoint(checkpoint_path, rows)

    print("\n=== LongMemEval_S retrieval-only summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
