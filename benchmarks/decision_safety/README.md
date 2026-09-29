# Decision Safety Suite v2 — battery files

Versioned, ordered write/delete/query scripts for the decision-safety track.
Format, metric definitions, capability values, and the paired-reporting rule
live in [`docs/decision-safety-suite.md`](../../docs/decision-safety-suite.md).

| File | Battery | Question |
|---|---|---|
| `deletion_durability.v1.json` | `deletion_durability` | Does a deleted fact stay deleted through recall, a later answer, and the next write? |
| `ttl_expiry.v1.json` | `ttl_expiry` | Does an expired memory stop being served — including on a query that was already asked while it was live? |
| `poisoned_write.v1.json` | `poisoned_write` | Does an injected "remember that..." become a durable memory — and do real writes still land? |
| `state_invalidation.v1.json` | `state_invalidation` | After `A -> B`, does a current-state query return `B` and not `A`? |
| `provenance_reassertion.v1.json` | `provenance_reassertion` | Does re-quoting an old value make it current again? |
| `point_in_time.v1.json` | `point_in_time` | Does `as_of: T` reconstruct `T` without leaking later updates? |

Run them with `runner.py`:

```bash
python -m benchmarks.decision_safety.runner --adapter agent-memory
python -m benchmarks.decision_safety.runner --adapter agent-memory --battery state_invalidation
```

## Rules

- Every battery reports its safety metric next to its paired permissiveness
  metric. No unpaired number is published, and there is no aggregate
  "decision safety score".
- `requires_capabilities` gates a battery. A system missing one reports
  `unsupported` for that battery; the harness never emulates the capability.
- `requires_capabilities` gating means `partial` is not `supported`, and an op
  the adapter cannot execute is dropped from numerator and denominator alike.
- Agent Memory's own results are in
  [`results/REPORT.md`](results/REPORT.md), with raw per-op output beside them.
  No comparative result is published for any other system yet.
- Battery files are append-versioned (`.v1`, `.v2`). Cases are never edited in
  place once a result has been published against them.

## Contributing a case

Open a PR adding ops to the relevant `.vN` file with a `notes` field saying what
failure the case catches. Maintainers of a compared system may submit cases and
adapter corrections; the result artifact records the reviewer.
