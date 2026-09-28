# CLI & Benchmark Reference

## CLI commands

```bash
agent-memory --help
agent-memory --data-dir PATH --backend sqlite COMMAND
```

### remember

```bash
agent-memory remember "query" "response" \
  --type conversation \   # conversation | fact | workflow | tool_output | code | preference | …
  --scope user \          # user | project | team | global | …
  --ttl 30d               # optional: 30d | 2h | 3600
```

### resolve

```bash
agent-memory resolve "query"           # prints action + response
agent-memory resolve "query" --explain # includes full score breakdown
```

### stats

```bash
agent-memory stats
# Output:
# Total memories: 42
# By state: active=38, archived=3, expired=1
# By type:  fact=15, conversation=12, workflow=8, preference=4, code=3
```

### cleanup

```bash
agent-memory cleanup          # mark expired memories as 'expired'
agent-memory cleanup --delete # permanently delete expired memories
```

### benchmark

```bash
agent-memory benchmark                  # quick run with built-in queries
agent-memory benchmark --seed           # seed from eval datasets first
agent-memory benchmark --seed --repeat 3        # run 3 times, report p95
agent-memory benchmark --baseline-ms 500        # compare against 500ms baseline
```

Sample output:

```
Benchmark Results
=================
Queries:    50
Avg (ms):   4.2
P95 (ms):   11.8
Actions:    replay=28, restore=14, verify=5, none=3
```

### eval

```bash
agent-memory eval                                # run all bundled datasets
agent-memory eval --datasets ./my_datasets/     # custom dataset directory
```

Sample output:

```
Agent Memory Evaluation
=======================
Dataset: coding_agent       Cases: 25  Correct: 25  Precision: 100.0%
Dataset: customer_support   Cases: 25  Correct: 25  Precision: 100.0%

Overall precision: 100.0%
```

---

## Eval dataset format

JSON files in `benchmarks/datasets/` or any directory you point `--datasets` at:

```json
{
  "name": "my_dataset",
  "memories": [
    { "query": "Q", "response": "A", "type": "fact", "tags": ["topic"] }
  ],
  "cases": [
    { "query": "Q?", "expected_action": "replay", "notes": "optional" }
  ]
}
```

`expected_action` accepts: `replay` · `restore` · `verify` · `none`.
Flexible matching: `restore` accepts `replay` or `verify`; `verify` accepts `restore`.

---

## LongMemEval / LoCoMo harness (Python API)

For richer metrics (Recall@k, MRR, content-recall, latency) use the Python harness:

```python
from agent_memory import Memory, BenchmarkHarness, BenchmarkDataset

memory  = Memory(persist_dir=".agent_memory")
harness = BenchmarkHarness(memory)
result  = harness.run_from_file("my_dataset.json")
print(result.format())
```

→ See [examples/benchmark_harness.py](../examples/benchmark_harness.py) for a complete runnable example.

---

## REST API server

```bash
pip install "agent-memory-sdk[api]"
export AGENT_MEMORY_API_KEY="replace-with-a-long-random-secret"
AGENT_MEMORY_DIR=.agent_memory agent-memory-api
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

The REST API requires `AGENT_MEMORY_API_KEY` (or an `api_key` passed to
`create_app`). Send it as `X-API-Key` or `Authorization: Bearer ...`. The server
binds to `127.0.0.1` by default. Set `HOST` explicitly only when you intend to
expose it; use TLS and a high-entropy secret for network deployments. For trusted
local tests only, `create_app(allow_unauthenticated=True)` disables the check.

Endpoints: `POST /memories` · `GET /memories` · `GET /memories/{id}` · `DELETE /memories/{id}` · `POST /memories/{id}/archive` · `POST /resolve` · `GET /stats` · `POST /cleanup` · `POST /consolidate`

---

## Dashboard

```bash
pip install "agent-memory-sdk[dashboard]"
AGENT_MEMORY_DIR=.agent_memory agent-memory-dashboard
# → http://localhost:8501
```

→ See [Dashboard section](../README.md#️-dashboard) in the main README.
