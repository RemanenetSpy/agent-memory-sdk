# Stress Testing

This guide documents the reproducible stress-test harness. Its output is a
measurement of the supplied workload on the machine that runs it; it is not a
production SLA, a cross-project comparison, or a prediction for a different
corpus.

## Run the Harness

```bash
# Lexical FTS5 storage latency with the bundled workload and LRU cache disabled.
uv run python scripts/stress_test.py \
  --memories 10000 --queries 500 --no-cache --embeddings off --json

# Local ONNX and sqlite-vec, when the semantic extras are installed.
uv run python scripts/stress_test.py \
  --memories 1000 --queries 200 --no-cache --embeddings on --json

# Bulk-load throughput, including the FTS5 rebuild.
uv run python scripts/stress_test.py \
  --memories 100000 --queries 1 --no-cache --fast-seed --embeddings off --json

# Run the 1M workload on the target machine.
uv run python scripts/stress_test.py \
  --memories 1000000 --queries 500 --no-cache --fast-seed --embeddings off --json
```

| Flag | Default | Purpose |
|---|---|---|
| `--memories N` | 10,000 | Entries to seed |
| `--queries N` | 1,000 | `resolve()` calls to measure after warm-up |
| `--no-cache` | on | Disable the in-process LRU cache |
| `--cache` | off | Enable the in-process LRU cache |
| `--fast-seed` | off | SQLite bulk insert followed by an FTS5 rebuild |
| `--embeddings auto\|on\|off` | `auto` | Select embedding mode explicitly for comparable runs |
| `--data-dir PATH` | temporary directory | Keep the SQLite store after the process exits |
| `--json` | off | Emit the complete result object for archival or CI |

## Workload Scope

The bundled workload repeats the entries in
[`benchmarks/stress/memories.jsonl`](../benchmarks/stress/memories.jsonl), adding
an identifier suffix as the store grows. Queries are drawn from
[`benchmarks/stress/queries.jsonl`](../benchmarks/stress/queries.jsonl). It is a
repeatable synthetic workload, not a representative sample of every production
memory store.

SQLite FTS5 work depends on the query terms and their document frequency. A
larger store can be faster or slower than a smaller one for a given query, so
record the corpus, query set, embedding mode, cache setting, machine, and full
JSON output whenever reporting a result.

The harness reports seed time, seed rate, endpoint RSS, CPU time, action mix,
and latency percentiles. Endpoint RSS is process-local `psutil` data; it is not
a complete system-memory measurement or a comparison with another service.

Charts generated from the benchmark results archived in
[`benchmarks/stress/results/`](../benchmarks/stress/results/) are available here:

- [Latency benchmark by stored-memory count](assets/stress_latency_benchmark.png)
- [Fast-seed throughput benchmark](assets/stress_seeding_benchmark.png)
- [Measured resource and latency benchmark](assets/stress_resource_latency_benchmark.png)
- [Measured decision-action mix benchmark](assets/stress_action_mix_benchmark.png)
- [Measured cache and decision rates benchmark](assets/stress_rates_benchmark.png)
- [Measured CPU and RSS benchmark](assets/stress_resources_benchmark.png)

## Verified Seed Samples

The following lexical FTS5 samples were run on an M-series MacBook with
`--no-cache --fast-seed --embeddings off --queries 1`. They include the FTS5
rebuild and are provided only as reproducibility reference points:

| Entries | Seed time | Seed rate |
|---:|---:|---:|
| 10,000 | 1.28s | 7,832/s |
| 100,000 | 12.40s | 8,065/s |
| 1,000,000 | 134.88s | 7,414/s |

These are machine-specific samples, not a portable performance guarantee. The
1M run measured p50 **72.661ms**, p95 **134.429ms**, and endpoint RSS of
**310.0MiB** at the end of retrieval. Retain the JSON output and workload parameters
when publishing or comparing these numbers.

## Interpreting Results

- A cache-hit latency measures the SDK's in-process cache path, not storage retrieval.
- A no-cache result measures this specific fixture and configuration; it does not
  establish a general latency guarantee.
- Semantic runs include local model and vector-extension overhead. Record whether
  model assets were already installed and loaded.
- The action mix is an outcome of the fixture's data, thresholds, and queries. It
  is not an expected distribution for a user application.
- Do not compare this output with another memory product without the matched-workload
  protocol in [the competitive benchmark RFC](competitive-benchmark-rfc.md).

## CI Guard Example

Use a threshold only after establishing a baseline for the same runner and
machine class. This example is a local regression guard, not a product target:

```bash
uv run python scripts/stress_test.py \
  --memories 10000 --queries 200 --no-cache --embeddings off --json \
  | python3 -c '
import json, sys
result = json.load(sys.stdin)
limit_ms = 250.0
if result["p95_ms"] > limit_ms:
    raise SystemExit(f"REGRESSION: p95={result[\"p95_ms\"]:.1f}ms > {limit_ms:.1f}ms")
print(f"OK: p50={result[\"median_ms\"]:.1f}ms p95={result[\"p95_ms\"]:.1f}ms")
'
```

The repository's scale test uses a similarly generous regression bound rather
than treating one laptop measurement as a portability guarantee.
