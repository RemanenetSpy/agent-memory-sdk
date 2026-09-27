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

All three run the same batteries against the same SDK commit, on one machine,
with `top_k=5`, cold cache, and `Memory.from_conversation()` as the ingest path.

Battery set SHA256 `0dbb9a8cac7a77c9…` (over all six battery files), SDK
`0.1.5.dev25+g893abc6bb`, Python 3.13.13, macOS arm64. Each artifact carries the
full hash, command line, and configuration.

| Config | Retrieval | Abstention gate (`restore_threshold`) |
|---|---|---|
| `lexical` | BM25 only, embeddings disabled | 0.70 (SDK default) |
| `semantic` | hybrid, fastembed `bge-small-en-v1.5` | 0.70 (SDK default) |
| `semantic-gate045` | hybrid, same embedder | 0.45 |

## Results

| Battery | Metric | `lexical` | `semantic` | `semantic-gate045` |
|---|---|---|---|---|
| `deletion_durability` | `deletion_resurrection_rate` | 0.00 | 0.00 | 0.00 |
| | `reextraction_resurrection_rate` | 0.00 | 0.00 | 0.00 |
| | `raw_turn_survival_rate` | 0.00 | 0.00 | 0.00 |
| | **`post_delete_recall`** *(pair)* | 0.50 | **0.00** | 1.00 |
| `ttl_expiry` | `expired_reuse_rate` | 0.00 | 0.00 | 0.00 |
| | `repeat_query_expired_reuse_rate` | 0.00 | 0.00 | 0.00 |
| | **`live_recall`** *(pair)* | 1.00 | **0.33** | 1.00 |
| `poisoned_write` | `injection_write_rate` | 1.00 | 1.00 | 1.00 |
| | `injection_replay_rate` | 0.40 | 0.00 | 0.80 |
| | **`benign_write_rate`** *(pair)* | 1.00 | 1.00 | 1.00 |
| | **`false_quarantine_rate`** *(pair)* | 0.00 | 0.00 | 0.00 |
| `state_invalidation` | `stale_current_state_rate` | 0.43 | 0.00 | 0.86 |
| | **`current_state_accuracy`** *(pair)* | 0.14 | **0.00** | 0.57 |
| | `invalidation_lag_writes` | n/a | n/a | n/a |
| `provenance_reassertion` | `spurious_supersession_rate` | 0.75 | 0.00 | 0.75 |
| | **`declared_mutation_accept_rate`** *(pair)* | 0.33 | **0.00** | 0.33 |
| `point_in_time` | — | `unsupported` | `unsupported` | `unsupported` |

## What these numbers say

**The `semantic` column is why unpaired safety metrics are banned.** Read alone,
it is a flawless run: zero stale state, zero spurious supersession, zero
injection replay. Read with its pairs, it is a system that answered almost
nothing at all — `current_state_accuracy` 0.00, `post_delete_recall` 0.00,
`declared_mutation_accept_rate` 0.00, `live_recall` 0.33. Across the 42 queries it
refused in that run, the best candidate scored 0.328–0.697 (median 0.642) — every
one of them under the default 0.70 gate, so the decision layer returned `NONE`
while holding the right memory. A single-number "decision safety score" would
have ranked this configuration best.

**Injection defence: none.** `injection_write_rate` is 1.00 in every
configuration. `from_conversation()` stores an injected `remember that...`
payload as readily as a real user statement — no provenance check, no
instruction detection. `benign_write_rate` 1.00 and `false_quarantine_rate` 0.00
confirm nothing is being quarantined, so the low replay rates in the stricter
configurations are abstention, not filtering. This is the clearest gap the suite
found in our own system.

**Supersession: none.** `explicit_supersession` is declared `unsupported`, and it
shows. At the usable gate (0.45), a superseded value is returned as current in
86% of current-state queries, and `invalidation_lag_writes` is `n/a` in every
config — no current-state query ever came back with the new value and without
the old one, so there is no lag to measure. `spurious_supersession_rate` 0.75
says a re-quoted old value wins too. Recency and confidence scoring are not a
substitute for a supersede edge.

**Deletion: clean.** Nothing resurrected through recall, through a later answer,
or through re-extraction on the next write, and no raw turns survived a
scope-wide delete. That is partly architectural — extraction is stateless, so
there is no transcript to re-derive from — and partly a property of the harness
mapping (see caveats).

**TTL: clean now, and it was not before.** `ttl_expiry` reports 0.00 reuse in
every configuration, including on a query repeated verbatim from before expiry.
That case failed when the battery was written: the retriever's 5-second result
cache could outlive a memory's own TTL and replay an expired memory even though
the store's read path filtered it. The fix and its regression test ship in the
same revision as this report, so the 0.00 is a measurement of the fixed code, not
of the code that prompted the battery.

**Point-in-time: unsupported, not zero.** There is no `as_of` read path, so the
battery is reported unsupported rather than scored, and the harness does not
emulate it.

## Caveats

1. **Scope isolation here is the harness's, not the SDK's.** `MemoryEntry.scope`
   is a tier (`user`/`project`/...), not a tenant identifier, so the adapter maps
   each battery scope to its own store. The cross-scope assertions therefore test
   that mapping. One consequence is visible in the numbers: the `tool:crm`
   injection in `poisoned_write` cannot reach `user:carol` here, so its replay is
   unmeasurable rather than prevented.
2. **`advance_clock` is emulated** by back-dating stored timestamps; the SDK has
   no injectable clock.
3. **`delete_scope` is emulated** with `forget_where(all=True)` against a
   per-scope store.
4. **One run per configuration.** Latency and resource numbers are not reported
   here, so the RFC's five-run rule does not apply, but the retrieval path is
   deterministic in the `lexical` config and near-deterministic in the others.
5. **Thresholds are not tuned to the suite.** Two of the three configurations use
   SDK defaults. `semantic-gate045` is included to show the trade, not as a
   recommended setting.

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
