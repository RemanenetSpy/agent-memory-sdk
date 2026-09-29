# Decision Safety Suite v2 — Adapters

This directory contains adapter implementations connecting various memory systems to the Decision Safety Suite v2 runner.

Each adapter implements the [`DecisionSafetyAdapter`](base.py) base contract, translating benchmark operations (`write`, `delete`, `delete_scope`, `query`, `query_as_of`, `advance_clock`, `inspect`) into the target system's native API.

## Adapters Overview

| Adapter | System | Deployment | Status |
|---|---|---|---|
| [`agent_memory_adapter.py`](agent_memory_adapter.py) | Agent Memory SDK | Local / Embedded SQLite | Canonical Reference Implementation |
| [`mem0_adapter.py`](mem0_adapter.py) | Mem0 | Self-Hosted | Stub awaiting maintainer configuration confirmation ([mem0#7453](https://github.com/mem0ai/mem0/issues/7453)) |
| [`smriti_adapter.py`](smriti_adapter.py) | Smriti Bi-Temporal Knowledge Engine | Hosted API (`smriti-engine:v1.2.0`) | Reference Implementation for Bi-Temporal Track ([Issue #44](https://github.com/theprodsde/agent-memory-sdk/issues/44)) |

---

## Smriti Adapter Specification

The `SmritiAdapter` integrates Smriti's bi-temporal Subject-Verb-Object (SVO) knowledge graph engine with the decision-safety benchmark tracks.

### 1. Capability Declarations

`SmritiAdapter` declares full native support for all 8 benchmark capability dimensions:

```python
capabilities = {
    "delete_by_id": "supported",          # Deterministic invalidation of active validity interval
    "delete_scope": "supported",          # Scope-wide atomic tombstone invalidation
    "tombstones": "supported",            # Bi-temporal interval termination (valid_to = now())
    "raw_message_store": "none",          # SVO extraction decoupled from transcript hoarding
    "ttl": "supported",                   # Deterministic interval expiry
    "explicit_supersession": "supported", # Supersession edges linking prior and new facts
    "as_of_query": "supported",           # Point-in-time state reconstruction
    "explicit_abstention": "supported",   # Returns empty set when state is refuted or absent
}
```

### 2. Temporal Semantics & Provenance Defense

- **Valid Time vs. Transaction Time:**
  - **Valid Time (`valid_from`, `valid_to`):** Defines the real-world timeline during which an asserted fact is true.
  - **Transaction Time (`created_at`, `superseded_at`):** Physical system commit timestamp recording when the database recorded or modified the record.
- **Quote Defense vs. Genuine Mutation:**
  - *Unauthenticated Echoes:* Re-assertions or quotes in prompt contexts/tool logs carry no write authority and do not alter active validity intervals.
  - *Authenticated Mutations:* Explicit user updates terminate prior intervals (`valid_to = tx_now`) and open a new active interval (`valid_from = tx_now, valid_to = NULL`).
- **Zero Label Leakage:**
  - Benchmark ground-truth labels and evaluation needles are never passed to the engine. The adapter receives only the raw turn content (`human`, `assistant`, `at`).

### 3. Reproducibility & Ephemeral Isolation

- **Isolated Tenant Namespaces:**
  Each battery execution triggers `setup(battery=...)`, provisioning an ephemeral tenant UUID:
  `tenant_id = f"ds-{battery}-{uuid.uuid4().hex[:8]}"`
- **Teardown Cascade:**
  `teardown()` issues `DELETE /v1/tenants/{tenant_id}`, executing an atomic cascade purge and verifying zero cross-battery state leakage.
- **Engine Version:**
  Pinned to `smriti-engine:v1.2.0` in adapter metadata.

### 4. Indexing Architecture & Complexity

- **Storage Index Complexity:**
  Point-in-time interval retrieval executes over a composite B-Tree index:
  ```sql
  CREATE INDEX idx_smriti_temporal ON memory_triplets (tenant_id, scope, entity_id, valid_from, valid_to);
  ```
  This guarantees $O(\log N)$ storage retrieval overhead for historical interval bisection.
- **LLM Working Context Complexity:**
  Working memory returned to the agent context is strictly bounded to $O(1)$ active state; outdated, superseded, and tombstoned facts are excluded before context assembly.

---

## Configuration & Execution

To run unit tests asserting the adapter contract:

```bash
pytest tests/test_smriti_adapter.py -v
```

To run against a live Smriti service instance:

```bash
export SMRITI_API_URL="https://api.smriti.ai"
export SMRITI_API_KEY="your-api-key"
```
