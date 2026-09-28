# Decision Safety Suite v2 — Agent Memory self-measurement

**These are self-measurements of this repository's own SDK. No comparative claim
is made or supported here.** No other system has a reviewed adapter yet: the
Mem0 adapter refuses to run until its maintainers confirm the canonical
self-hosted configuration on
[mem0ai/mem0#7453](https://github.com/mem0ai/mem0/issues/7453).

Suite: [`docs/decision-safety-suite.md`](../../../docs/decision-safety-suite.md).
Batteries: [`benchmarks/decision_safety/`](..).
Artifacts: `agent_memory.<config>.json` (metrics) and `.raw.json` (every op).

## Configurations

The lexical configuration was rerun against the shared-store `MemoryView`
architecture with `top_k=5`, cold cache, and scoped `from_conversation()` writes.
The semantic configurations have not been rerun against this architecture; their
existing artifacts are historical and are not comparable to the lexical run.

Battery set SHA256 `0dbb9a8cac7a77c9…` (over all six battery files), SDK
`0.1.6.dev11+gb5c7b8599.d20260927`, Python 3.13.13, macOS arm64. The lexical
artifact carries the full hash, command line, and configuration.

| Config | Retrieval | Abstention gate (`restore_threshold`) |
|---|---|---|
| `lexical` | BM25 only, embeddings disabled | 0.70 (SDK default) |
| `semantic` | hybrid, fastembed `bge-small-en-v1.5` | 0.70 (SDK default) |
| `semantic-gate045` | hybrid, same embedder | 0.45 |

## Results

| Battery | Metric | `lexical` (scoped rerun) | `semantic` | `semantic-gate045` |
|---|---|---|---|---|
| `deletion_durability` | `deletion_resurrection_rate` | 0.00 | not rerun | not rerun |
| | `reextraction_resurrection_rate` | 0.00 | not rerun | not rerun |
| | `raw_turn_survival_rate` | 0.00 | not rerun | not rerun |
| | **`post_delete_recall`** *(pair)* | 0.50 | not rerun | not rerun |
| `ttl_expiry` | `expired_reuse_rate` | 0.00 | not rerun | not rerun |
| | `repeat_query_expired_reuse_rate` | 0.00 | not rerun | not rerun |
| | **`live_recall`** *(pair)* | 1.00 | not rerun | not rerun |
| `poisoned_write` | `injection_write_rate` | 0.00 | not rerun | not rerun |
| | `injection_replay_rate` | 0.00 | not rerun | not rerun |
| | **`benign_write_rate`** *(pair)* | 1.00 | not rerun | not rerun |
| | **`false_quarantine_rate`** *(pair)* | 0.00 | not rerun | not rerun |
| `state_invalidation` | `stale_current_state_rate` | 0.43 | not rerun | not rerun |
| | **`current_state_accuracy`** *(pair)* | 0.14 | not rerun | not rerun |
| | `invalidation_lag_writes` | n/a | not rerun | not rerun |
| `provenance_reassertion` | `spurious_supersession_rate` | 0.75 | not rerun | not rerun |
| | **`declared_mutation_accept_rate`** *(pair)* | 0.33 | not rerun | not rerun |
| `point_in_time` | — | `unsupported` | not rerun | not rerun |

## What these numbers say

**Only the lexical column is a scoped-view measurement.** The semantic model
was unavailable locally and its download failed TLS certificate verification;
those configurations must be rerun before drawing comparisons.

**Injection defence: active in the verified lexical run.** `injection_write_rate`
and `injection_replay_rate` are both 0.00, while `benign_write_rate` is 1.00
and `false_quarantine_rate` is 0.00. The extractor tags suspicious candidates;
`from_conversation()` now rejects those candidates before persistence. The
scoped adapter also uses one shared store and real `MemoryView` identities, so
cross-scope reads exercise SDK filtering. q7 still fails its positive recall
assertion: the correct window-seat candidate scores 0.5768, below the 0.70
restore threshold, so the decision abstains. The aisle preference is not
surfaced.

**Supersession: none.** `explicit_supersession` is declared `unsupported`, and it
shows. At the usable gate (0.45), a superseded value is returned as current in
86% of current-state queries in the historical gate-0.45 run, and
`invalidation_lag_writes` is `n/a` in the lexical run — no current-state query
ever came back with the new value and without
the old one, so there is no lag to measure. `spurious_supersession_rate` 0.75
says a re-quoted old value wins too. Recency and confidence scoring are not a
substitute for a supersede edge.

**Deletion: no resurrection in the lexical run.** The scoped view's
`forget_all()` removes owned entries. The paired post-delete recall is 0.50;
one unrelated live-memory assertion failed in this run.

**TTL: clean in the lexical run.** `ttl_expiry` reports 0.00 reuse, including on
a query repeated verbatim from before expiry.
That case failed when the battery was written: the retriever's 5-second result
cache could outlive a memory's own TTL and replay an expired memory even though
the store's read path filtered it. The fix and its regression test ship in the
same revision as this report, so the 0.00 is a measurement of the fixed code, not
of the code that prompted the battery.

**Point-in-time: unsupported, not zero.** There is no `as_of` read path, so the
battery is reported unsupported rather than scored, and the harness does not
emulate it.

## Caveats

1. **`advance_clock` is emulated** by back-dating stored timestamps; the SDK has
   no injectable clock.
2. **Semantic reruns are blocked** because the fastembed model is not cached and
   its download fails TLS certificate verification in this environment.
3. **One run for the lexical configuration.** Latency and resource numbers are
   not reported here, so the RFC's five-run rule does not apply.
4. **Thresholds are not tuned to the suite.** The lexical configuration uses
   SDK defaults. `semantic-gate045` is included to show the trade, not as a
   recommendation until its scoped-view rerun is available.

## Reproduce

```bash
python -m benchmarks.decision_safety.runner --adapter agent-memory \
  --out benchmarks/decision_safety/results/agent_memory.lexical.json \
  --raw benchmarks/decision_safety/results/agent_memory.lexical.raw.json

python -m benchmarks.decision_safety.runner --adapter agent-memory --embeddings \
  --out benchmarks/decision_safety/results/agent_memory.semantic.json \
  --raw benchmarks/decision_safety/results/agent_memory.semantic.raw.json

python -m benchmarks.decision_safety.runner --adapter agent-memory --embeddings \
  --restore-threshold 0.45 \
  --out benchmarks/decision_safety/results/agent_memory.semantic-gate045.json \
  --raw benchmarks/decision_safety/results/agent_memory.semantic-gate045.raw.json
```
