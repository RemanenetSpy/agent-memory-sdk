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

## Backend comparison

> **Scope — read this before quoting any number below.** This section compares
> **this SDK's storage backends against each other**, on a small synthetic corpus
> built for exactly that purpose. Its Recall@k is *not* a quality result for the
> SDK and is not comparable to anything another project publishes: different
> dataset, different relevance definition, different k.
>
> The externally comparable retrieval number is **LongMemEval session Recall@5 —
> 98.1% semantic / 96.0% lexical** on `_S`, 87.0% on `_M`
> ([report](../benchmarks/longmemeval/REPORT.md)). The decision-quality number is
> **34/36 (94.4%)** on the adversarial trap suite, above. Neither is affected by
> which storage backend you choose, which is why they live in their own sections.
>
> And note what mem0 and Zep publish is **end-to-end QA accuracy** (an LLM answers,
> an LLM judges) — a strictly harder quantity than retrieval recall, which is only
> its ceiling. Zep's own end-to-end figure on LongMemEval_S is 71.2% with gpt-4o.
> Putting our retrieval recall next to their end-to-end accuracy would claim a win
> we have not measured; see [what we do NOT claim](#what-we-do-not-claim).

Reproduce with:

```bash
docker compose -f docker-compose.dev.yml up -d
python scripts/backend_benchmark.py --memories 2000 --queries 300
```

Every backend gets the same corpus, the same labelled query set, and one shared
`bge-small-en-v1.5` embedder, with the retrieval cache disabled — so the only
variable is the storage layer. `search()` and `keyword_search()` are timed
separately, because they are the two halves `resolve()` fuses and a slow keyword
half otherwise hides behind a fast vector index.

Measured 2026-09 on an M-series laptop, 2,000 memories, Docker-hosted servers on
loopback. Embedding one query costs ~2.8 ms and is included in every `search()`
and `resolve()` figure.

### Latency (ms, lower is better)

| Backend | Index | Semantic | `resolve()` p50 | `resolve()` p95 | `search()` p50 | `keyword_search()` p50 | writes/s |
|---|---|---|---|---|---|---|---|
| `sqlite` | none (FTS5 only) | no | **0.73** | 1.12 | 0.61 | 0.61 | **2,238** |
| `sqlite` + `[semantic]` | sqlite-vec | yes | 5.27 | 6.86 | 4.38 | **0.62** | 117 |
| `redis` | RediSearch HNSW | yes | **7.86** | **10.01** | **5.97** | 1.45 | 107 |
| `postgres` | pgvector HNSW | yes | 14.51 | 16.30 | 12.07 | 2.59 | 78 |
| `postgres` | pgvector IVFFlat | yes | 10.26 | 12.51 | 7.62 | 2.49 | 92 |
| `qdrant` | Qdrant HNSW | yes | 13.18 | 19.25 | 7.27 | 5.32 | 82 |

Read this as orders of magnitude, not exact milliseconds — repeat runs on the
same machine moved `postgres` `search()` p50 between 7 and 12 ms.

**SQLite wins on latency, and that is expected.** It is in-process: no socket, no
serialisation, no server. A network backend cannot beat a local file, so the
reason to choose one is never raw single-process latency — it is shared state
across processes, a corpus larger than one machine's disk, or an existing
operational home for the data. Adding `[semantic]` to SQLite costs ~4.5 ms per
resolve and buys ~8 points of Recall@1; that trade, not the backend choice, is
what moves the quality numbers below.

### Retrieval quality — Recall@k (backend-vs-backend only)

Scored over the 51 queries in `benchmarks/stress/queries.jsonl` that have a
relevant memory (the other 12 are out-of-domain and score refusal instead).
Ground truth is a tag lookup: the seeded memories carry topic tags, and every
in-domain query category is one of those tags.

These absolute percentages mean little — the corpus is 32 templates repeated to
2,000 entries and the ground truth is a tag match, so the ceiling is a property of
the fixture. **Only the differences between rows are meaningful.** For an absolute
retrieval number, use the LongMemEval report.

**Recall@k here means "did at least one relevant memory reach the top k".** That
is the hit-rate sense the SDK's own [benchmark
harness](../agent_memory/benchmarks/harness.py) uses. The strict IR ratio —
*fraction of all* relevant documents retrieved — would be meaningless on this
corpus: it repeats 32 templates to reach 2,000 entries, so each query has dozens
of equally relevant memories and the ratio would only measure how many duplicates
fit inside k.

No score threshold is involved. This is purely what the index found.

| Backend | vector R@1 | vector R@5 | fused R@1 | fused R@5 |
|---|---|---|---|---|
| `sqlite` (lexical) | 84.3% | 84.3% | 84.3% | 84.3% |
| `sqlite` + `[semantic]` | **92.2%** | 92.2% | **92.2%** | 92.2% |
| `redis` (HNSW) | **92.2%** | 92.2% | **92.2%** | 92.2% |
| `postgres` (HNSW) | 90.2% | 90.2% | **92.2%** | 92.2% |
| `postgres` (IVFFlat) | 86.3% | 86.3% | **92.2%** | 92.2% |
| `qdrant` (HNSW) | **92.2%** | 92.2% | **92.2%** | 92.2% |

- **Vector search is worth ~8 points of recall** over lexical-only (84.3% →
  92.2%). That is the actual return on the `[semantic]` extra and a vector index.
- **Every vector backend retrieves equally well once fused** — 92.2% across the
  board. There is no retrieval difference between RedisVSS, pgvector and Qdrant
  on this workload; they are the same algorithm over the same embeddings.
- **HNSW recalls more than IVFFlat before fusion** (90.2% vs 86.3% at R@1), which
  is the recall claim in the pgvector upgrade made visible: IVFFlat only visits
  `probes` of its `lists` clusters, so it misses neighbours HNSW's graph walk
  finds. RRF fusion with the keyword half hides the gap here — it would not at a
  scale where the keyword half is weaker.
- **R@1 and R@5 are identical for every backend**, so k does not discriminate on
  this corpus — with ~62 copies of each template, the right template is either at
  rank 1 or absent entirely. R@1 is the informative column. A
  unique-document corpus would be needed to say anything about deeper k.
- The 92.2% ceiling (47/51) is a property of the labelled set, not the backends —
  4 queries have no retrievable relevant memory under this tag-based ground truth.

### Decision quality

Separate from retrieval, and the distinction matters: a query can retrieve the
right memory at rank 1 and still return NONE because the score sat below
`restore_threshold`. *Answer rate* = of the 51 answerable queries, the fraction
where `resolve()` surfaced a memory (REPLAY, RESTORE or VERIFY). *Refusal
precision* = of the 12 out-of-domain queries, the fraction correctly refused.

| Backend | Answer rate @0.70 | Answer rate @0.55 | Refusal precision | Wrong REPLAY |
|---|---|---|---|---|
| `sqlite` (lexical) | 94.1% | 98.0% | 100% | 0 |
| `sqlite` + `[semantic]` | 68.6% | **100%** | 100% | 0 |
| `redis` (HNSW) | 68.6% | **100%** | 100% | 0 |
| `postgres` (HNSW) | 49.0% | **100%** | 100% | 0 |
| `postgres` (IVFFlat) | 49.0% | **100%** | 100% | 0 |
| `qdrant` (HNSW) | 52.9% | **100%** | 100% | 0 |

1. **Every backend refuses correctly, and none ever replays a wrong answer.**
   The decision layer's safety property does not depend on the storage engine.
2. **`restore_threshold` matters far more than the backend.** Recall@k above is
   ~92% everywhere, yet the answer rate at 0.70 ranges from 49% to 69% — so
   roughly a third of correctly-retrieved memories are being discarded by the
   threshold, not missed by the index. The 0.70 default is tuned for lexical
   scoring, where exact word overlap scores near 1.0; embeddings put a genuine
   paraphrase at 0.55–0.65. At 0.55 every semantic backend answers 100% of
   answerable queries with refusal precision intact. **Lower `restore_threshold`
   to ~0.55 whenever KNN is the primary retrieval path.**
3. The residual spread at 0.70 is a scoring-scale artefact, not a retrieval
   difference: FTS5 `bm25()`, RediSearch `BM25STD`, `ts_rank` and Python BM25 all
   normalise differently, so the RRF-fused score lands in a slightly different
   place against one fixed threshold.

### pgvector: HNSW vs IVFFlat

`resolve()` timings above are dominated by embedding and round-trips, which hides
the index. Timing the KNN scan itself with `EXPLAIN (ANALYZE)` — same table, same
`ef_search`/`probes` as the store configures — isolates it:

| Rows | Index | Plan chosen | Index scan p50 | p95 |
|---|---|---|---|---|
| 2,000 | HNSW | Seq Scan | 2.63 ms | 5.05 ms |
| 2,000 | IVFFlat | Index Scan | **0.36 ms** | 0.53 ms |
| 20,000 | HNSW | Index Scan | **0.58 ms** | 0.86 ms |
| 20,000 | IVFFlat | Index Scan | 4.76 ms | 6.19 ms |

There is a crossover, and it is the reason `vector_index="auto"` exists:

- **At 2,000 rows IVFFlat wins**, and the planner does not even use the HNSW
  index — a sequential scan over 2,000 vectors is genuinely cheaper than a graph
  walk. Small corpora pay nothing for having HNSW available.
- **At 20,000 rows HNSW is 8× faster**, and the shape is what matters: growing
  the corpus 10× left HNSW roughly flat (0.36 → 0.58 ms, a graph walk is
  ~O(log N)) while IVFFlat grew ~13× (0.36 → 4.76 ms, because each probed list
  holds 10× more vectors to scan linearly). Extrapolate that to 1M rows and
  IVFFlat is not a contender.

IVFFlat also can't be tuned out of this: raising `probes` for recall scans more
lists, and at `probes == lists` it degenerates into brute force. HNSW additionally
accepts online inserts, where IVFFlat wants a populated table at build time and
drifts from its centroids as data arrives.

## Latency methodology

Latency and write cost depend on corpus shape, query terms, cache state,
embedding mode, batch size, machine, and operating system. The reproducible
stress harness records p50/p75/p90/p95/p99 latency, CPU, RSS, and seed rate for
the exact workload; see [stress-testing](stress-testing.md). CI uses a
regression bound rather than treating one machine's measurements as a portable
performance guarantee.

Any backend can be measured with the same harness:

```bash
python scripts/stress_test.py --backend redis --backend-opt url=redis://localhost:6379/0
python scripts/stress_test.py --backend qdrant --backend-opt url=http://localhost:6333
```

Before v0.2, keyword search loaded up to 10,000 rows and rebuilt a Python
BM25 index on **every query** — roughly 1s per resolve at 5,000 memories.
The FTS5 index removed that. `stats()` and `cleanup()` are pure SQL
aggregates with no row cap.

## What we do NOT claim

- **No head-to-head accuracy claim against mem0, Zep, or anyone else.** They
  publish end-to-end QA accuracy (LLM answers, LLM judges); we publish retrieval
  recall, which is the *ceiling* on end-to-end accuracy, not the same quantity.
  Comparing the two would flatter us for free. An end-to-end run is on the
  [roadmap](#roadmap-for-external-benchmarks); until it exists there is no
  comparable number.
- **The backend-comparison Recall@k is not an SDK quality figure.** It exists to
  rank this SDK's own storage backends on a synthetic fixture. Quoting it as "our
  recall" — against a competitor or across releases — is a category error.
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
