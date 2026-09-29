# Benchmarks & Evaluation

All numbers below are reproducible from this repository — no synthetic
baselines. Run them yourself:

```bash
agent-memory --data-dir /tmp/eval eval                      # decision quality
agent-memory --data-dir /tmp/bench benchmark --seed --repeat 3   # latency & action mix
python -m pytest tests/test_scale.py -v                     # correctness at scale
```

## Decision quality

The eval suite covers 36 cases across 4 datasets, including **adversarial
trap cases** — queries that share a word with a stored memory but ask a
different question, which naive retrieve-and-inject systems answer wrongly.

| Dataset | Cases | What it tests |
|---|---|---|
| `customer_support` | 4 | FAQ replay, paraphrase restore, unrelated → none |
| `coding_agent` | 4 | Workflow replay/restore, fact verification |
| `research_agent` | 4 | Summary replay, cross-document restore |
| `decision_traps` | 24 | Shared-word traps, `requires_verification` facts, paraphrases |

Results (measured 2026-09; the suite has grown to 36 cases):

| Backend | Precision |
|---|---|
| SQLite + `semantic` extra (sqlite-vec + fastembed) | **34/36 (94.4%)** |

Both misses are shared-word traps that return VERIFY instead of NONE — a
cautious failure (VERIFY never uses the memory without validation), never a
wrong REPLAY. External benchmark: see the
[LongMemEval retrieval report](../benchmarks/longmemeval/REPORT.md).

Scoring: an expected `restore` also accepts `replay`/`verify` (all three
surface the memory; verify is simply more cautious). The hard boundaries —
replaying the *wrong* answer, or using memory when `none` was expected —
are never accepted.

Example trap case that must return `none`:

> Stored: "What payment methods do you **support**?" → "Visa, Mastercard, PayPal."
> Query: "Does the platform **support** two-factor authentication?"
> Expected: `none` (a naive top-1 retriever replays the payment answer)

## Latency

Latency and write cost depend on corpus shape, query terms, cache state,
embedding mode, batch size, machine, and operating system. The reproducible
stress harness records p50/p75/p90/p95/p99 latency, CPU, RSS, and seed rate for
the exact workload; see [stress-testing](stress-testing.md). CI uses a
regression bound rather than treating one machine's measurements as a portable
performance guarantee.

Before v0.2, keyword search loaded up to 10,000 rows and rebuilt a Python
BM25 index on **every query** — roughly 1s per resolve at 5,000 memories.
The FTS5 index removed that. `stats()` and `cleanup()` are pure SQL
aggregates with no row cap.

## What we do NOT claim

- No comparison against a hardcoded "LLM baseline". If you pass
  `--baseline-ms` with a latency you measured in your own app, the benchmark
  will compare against that, clearly labeled as user-supplied.
- Token-savings depend entirely on your hit rate and prompt sizes; measure
  them in your own pipeline.

## Competitive benchmark RFC

Before publishing a cross-project comparison, follow the matched-workload,
reproducibility, and maintainer-review rules in the
[competitive benchmark RFC](competitive-benchmark-rfc.md). The companion
[`benchmarks/competitive/`](../benchmarks/competitive/) directory defines the
result contract and adapter requirements.

## Roadmap for external benchmarks

**Done:** LongMemEval retrieval-proxy harness — `_S` (98.1% Recall@5 semantic)
and `_M` (87.0% lexical on independent cleaned-release haystacks). It is not an
end-to-end evaluation or a direct comparison with the paper's original-release
session-index baselines. Full results:
[benchmarks/longmemeval/REPORT.md](../benchmarks/longmemeval/REPORT.md).

**Planned:** the end-to-end stage (LLM answering + the benchmark's official
GPT-4o judge, directly comparable to Zep's published accuracy), LoCoMo (the
benchmark mem0 publishes on), and a repeated-query decision-layer benchmark
(cost/latency/wrong-replay curves — the REPLAY/VERIFY value proposition no
public benchmark covers). Contributions welcome — see
[CONTRIBUTING.md](../CONTRIBUTING.md).
