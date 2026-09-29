# Competitive Benchmark Suite

This directory contains the public comparison contract described in
[`docs/competitive-benchmark-rfc.md`](../../docs/competitive-benchmark-rfc.md).

## Layout

- `result.schema.json`: required shape for aggregate metrics and run metadata.
- `adapters/`: one adapter per system, reviewed by that system's maintainers
  where possible.
- `configs/`: versioned, system-specific settings used for a published run.
- `results/`: raw per-case output and aggregate result JSON, never only charts.

## Adapter requirements

An adapter must declare whether it supports retrieval, end-to-end QA, explicit
abstention, TTL, stale-data verification, local deployment, and resource
measurement. Unsupported capabilities are reported as unsupported, never scored
as a silent failure or replaced with an unrelated feature.

Do not add a comparative result until it satisfies the RFC's matched-workload
rules and validates against `result.schema.json`.
