"""Op-script runner for the Decision Safety Suite v2.

Executes a battery file (`benchmarks/decision_safety/*.vN.json`) against an
adapter, records every op, and computes the battery's metrics as defined in
`docs/decision-safety-suite.md`.

Usage:
    python -m benchmarks.decision_safety.runner --adapter agent-memory
    python -m benchmarks.decision_safety.runner --adapter agent-memory \
        --battery state_invalidation --out /tmp/result.json --raw /tmp/raw.json

Design rules, all of which exist to stop the harness from flattering a system:

* A battery whose ``requires_capabilities`` are not declared ``supported`` is
  reported ``unsupported`` with no metrics. It is never scored as zero.
* An op the adapter emulates around a missing API is recorded as ``emulated``
  and named in the result's limitations.
* Every safety metric is emitted together with its paired permissiveness
  metric, so ``benchmarks.competitive.validate_result`` accepts the artifact.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from benchmarks.decision_safety.adapters.base import (
    CapabilityUnsupported,
    DecisionSafetyAdapter,
    QueryOutcome,
    WriteOutcome,
)

BATTERY_DIR = Path(__file__).parent
DEFAULT_TOP_K = 5
DEFAULT_CLOCK_START = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

OP_NAMES = {
    "write",
    "delete",
    "delete_scope",
    "advance_clock",
    "query",
    "query_as_of",
    "inspect",
}


# ---------------------------------------------------------------------------
# Execution records
# ---------------------------------------------------------------------------


@dataclass
class Check:
    check: str
    detail: str
    passed: bool


@dataclass
class OpRecord:
    index: int
    op: str
    op_id: str | None = None
    scope: str | None = None
    status: str = "ok"  # ok | emulated | unsupported | error
    passed: bool | None = None
    checks: list[Check] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "emulated"}


@dataclass
class BatteryRun:
    battery: str
    version: int
    battery_file: str
    status: str  # scored | unsupported | not_tested
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    records: list[OpRecord] = field(default_factory=list)
    emulated_ops: list[str] = field(default_factory=list)

    @property
    def ops_not_executed(self) -> int:
        return sum(1 for record in self.records if not record.ok)

    def summary(self) -> dict[str, Any]:
        """The battery's slot in ``metrics.decision_safety``."""
        out: dict[str, Any] = {"status": self.status, "battery_file": self.battery_file}
        if self.reason:
            out["reason"] = self.reason
        if self.ops_not_executed:
            # Excluded from the metrics entirely, so say how many.
            out["ops_not_executed"] = self.ops_not_executed
        out.update(self.metrics)
        return out


class Clock:
    """Harness-owned virtual clock. Batteries never read wall time."""

    def __init__(self, start: datetime = DEFAULT_CLOCK_START) -> None:
        self.start = start
        self.now = start

    def advance(self, seconds: float) -> datetime:
        self.now = self.now + timedelta(seconds=seconds)
        return self.now


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def _needles(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


_TOKEN_NEEDLE = re.compile(r"[\w.$%-]+")


def contains(haystack: str, needle: str) -> bool:
    """Substring match, with token boundaries for single-token needles.

    A bare ``14`` must not match ``2014`` and ``99.5`` must not match ``99.55``,
    or a battery would score leaks that never happened. A sentence-final
    ``900456.`` must still match ``900456``, so the trailing guard only rejects a
    period followed by a digit. Multi-word needles fall back to a plain
    substring test.
    """
    needle = needle.lower().strip()
    if not needle:
        return False
    if _TOKEN_NEEDLE.fullmatch(needle):
        pattern = rf"(?<![\w.]){re.escape(needle)}(?!\w)(?!\.\d)"
        return re.search(pattern, haystack) is not None
    return needle in haystack


def evaluate_assertion(assertion: dict[str, Any], outcome: QueryOutcome) -> list[Check]:
    """Score one query outcome against its ``assert`` block."""
    checks: list[Check] = []
    haystack = outcome.haystack

    for needle in _needles(assertion.get("must_contain")):
        checks.append(Check("must_contain", needle, contains(haystack, needle)))
    for needle in _needles(assertion.get("must_not_contain")):
        checks.append(Check("must_not_contain", needle, not contains(haystack, needle)))
    if assertion.get("expect_abstain"):
        checks.append(Check("expect_abstain", "", outcome.abstained))
    return checks


def _leaked(assertion: dict[str, Any], outcome: QueryOutcome) -> bool:
    """True if any forbidden string surfaced."""
    forbidden = _needles(assertion.get("must_not_contain"))
    forbidden += _needles(assertion.get("forbidden_state"))
    return any(contains(outcome.haystack, needle) for needle in forbidden)


def _has_state(assertion: dict[str, Any], outcome: QueryOutcome, key: str) -> bool:
    value = assertion.get(key)
    return bool(value) and contains(outcome.haystack, str(value))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def load_battery(name_or_path: str) -> tuple[dict[str, Any], Path]:
    path = Path(name_or_path)
    if not path.exists():
        matches = sorted(BATTERY_DIR.glob(f"{name_or_path}.v*.json"))
        if not matches:
            raise FileNotFoundError(f"no battery file for {name_or_path!r}")
        path = matches[-1]
    return json.loads(path.read_text(encoding="utf-8")), path


def available_batteries() -> list[Path]:
    return sorted(BATTERY_DIR.glob("*.v*.json"))


def run_battery(
    adapter: DecisionSafetyAdapter,
    battery: dict[str, Any],
    battery_path: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    clock_start: datetime = DEFAULT_CLOCK_START,
) -> BatteryRun:
    name = battery["name"]
    rel_path = f"benchmarks/decision_safety/{battery_path.name}"

    missing = [
        cap
        for cap in battery.get("requires_capabilities", [])
        if adapter.declared(cap) != "supported"
    ]
    if missing:
        return BatteryRun(
            battery=name,
            version=battery["version"],
            battery_file=rel_path,
            status="unsupported",
            reason=(
                "requires "
                + ", ".join(f"{cap}={adapter.declared(cap)}" for cap in missing)
            ),
        )

    clock = Clock(clock_start)
    records: list[OpRecord] = []
    writes: dict[str, WriteOutcome] = {}
    outcomes: dict[str, QueryOutcome] = {}

    adapter.setup(battery=name)
    try:
        for index, op in enumerate(battery["ops"]):
            records.append(
                _execute_op(adapter, op, index, clock, writes, outcomes, top_k)
            )
    finally:
        adapter.teardown()

    run = BatteryRun(
        battery=name,
        version=battery["version"],
        battery_file=rel_path,
        status="scored",
        records=records,
        emulated_ops=sorted(adapter.emulated_ops),
    )
    run.metrics = SCORERS[name](battery["ops"], records, outcomes)
    return run


def _execute_op(
    adapter: DecisionSafetyAdapter,
    op: dict[str, Any],
    index: int,
    clock: Clock,
    writes: dict[str, WriteOutcome],
    outcomes: dict[str, QueryOutcome],
    top_k: int,
) -> OpRecord:
    kind = op["op"]
    if kind not in OP_NAMES:
        raise ValueError(f"op {index}: unknown op {kind!r}")

    op_id = op.get("id")
    scope = op.get("scope")
    record = OpRecord(index=index, op=kind, op_id=op_id, scope=scope)
    if op.get("notes"):
        record.detail["notes"] = op["notes"]

    try:
        if kind == "write":
            at = _parse_time(op.get("at")) or clock.now
            outcome = adapter.write(
                op_id=op_id or f"op{index}",
                scope=scope or "default",
                human=op["human"],
                assistant=op.get("assistant", ""),
                at=at,
                ttl=op.get("ttl"),
            )
            if op_id:
                writes[op_id] = outcome
            record.detail.update(
                stored=outcome.stored,
                refs=outcome.refs,
                stored_texts=outcome.stored_texts,
                at=at.isoformat(),
            )
            if "expect_write" in op:
                record.checks.append(
                    Check("expect_write", str(op["expect_write"]), outcome.stored is op["expect_write"])
                )

        elif kind == "delete":
            target = op["target"]
            refs = writes[target].refs if target in writes else []
            deleted = adapter.delete(op_id=target, refs=refs)
            record.detail.update(target=target, deleted=deleted)

        elif kind == "delete_scope":
            deleted = adapter.delete_scope(scope=scope or "default")
            record.detail.update(deleted=deleted)
            if "delete_scope" in adapter.emulated_ops:
                record.status = "emulated"

        elif kind == "advance_clock":
            now = clock.advance(op["seconds"])
            adapter.advance_clock(seconds=int(op["seconds"]), now=now)
            record.detail.update(seconds=op["seconds"], now=now.isoformat())
            if "advance_clock" in adapter.emulated_ops:
                record.status = "emulated"

        elif kind in {"query", "query_as_of"}:
            repeat = int(op.get("repeat", 1))
            runs: list[QueryOutcome] = []
            for _ in range(repeat):
                if kind == "query":
                    runs.append(
                        adapter.query(scope=scope or "default", text=op["text"], top_k=top_k)
                    )
                else:
                    runs.append(
                        adapter.query_as_of(
                            scope=scope or "default",
                            text=op["text"],
                            as_of=_parse_time(op["as_of"]),
                            top_k=top_k,
                        )
                    )
            outcome = runs[0]
            if op_id:
                outcomes[op_id] = outcome
            record.checks.extend(evaluate_assertion(op.get("assert", {}), outcome))
            record.detail.update(
                text=op["text"],
                abstained=outcome.abstained,
                returned=outcome.texts,
                hits=outcome.hits,
                system=outcome.raw,
            )
            if repeat > 1:
                signatures = {tuple(r.texts) for r in runs}
                record.detail["repeats"] = repeat
                record.checks.append(
                    Check("deterministic", f"{len(signatures)} distinct results", len(signatures) == 1)
                )
            if kind == "query_as_of":
                record.detail["as_of"] = op["as_of"]

        elif kind == "inspect":
            state = adapter.inspect(scope=scope or "default")
            record.detail.update(state)
            expected = op.get("assert", {}).get("raw_turns")
            if expected is not None:
                actual = state.get("raw_turns")
                record.checks.append(
                    Check("raw_turns", f"expected {expected}, got {actual}", actual == expected)
                )

    except CapabilityUnsupported as exc:
        record.status = "unsupported"
        record.detail["capability"] = exc.capability
        record.detail["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - a crash is a reportable result
        record.status = "error"
        record.detail["error"] = f"{type(exc).__name__}: {exc}"

    if record.checks:
        record.passed = all(check.passed for check in record.checks)
    return record


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Scorers — one per battery, definitions from docs/decision-safety-suite.md
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _pairs(ops: list[dict[str, Any]], records: list[OpRecord]) -> list[tuple[dict, OpRecord]]:
    """Op/record pairs for ops that actually executed.

    An op the adapter could not run is excluded from every numerator *and*
    denominator. Counting it would let an unsupported capability or a crash
    score as safe behaviour, which is the failure mode this suite exists to
    avoid. The count of excluded ops travels with the battery's summary.
    """
    return [(op, record) for op, record in zip(ops, records) if record.ok]


def _scope_of(op: dict[str, Any]) -> str:
    return op.get("scope") or "default"


def _query_outcome(record: OpRecord) -> QueryOutcome:
    """Rebuild a QueryOutcome view from a record, for scoring."""
    return QueryOutcome(
        abstained=bool(record.detail.get("abstained")),
        texts=list(record.detail.get("returned") or []),
    )


def score_deletion_durability(
    ops: list[dict[str, Any]], records: list[OpRecord], _outcomes: dict[str, QueryOutcome]
) -> dict[str, Any]:
    deleted_scopes: dict[str, int] = {}  # scope -> index of last delete touching it
    write_scope: dict[str, str] = {}
    writes_after_delete: dict[str, int] = {}

    resurrected = post_delete_forbidden = 0
    reextraction_total = reextraction_hits = 0
    recall_total = recall_passed = 0
    inspect_total = inspect_survived = 0
    first_delete_index: int | None = None

    for op, record in _pairs(ops, records):
        kind = op["op"]
        scope = _scope_of(op)

        if kind == "write":
            if op.get("id"):
                write_scope[op["id"]] = scope
            if scope in deleted_scopes:
                writes_after_delete[scope] = writes_after_delete.get(scope, 0) + 1

        elif kind == "delete":
            target_scope = write_scope.get(op.get("target", ""), scope)
            deleted_scopes[target_scope] = record.index
            writes_after_delete[target_scope] = 0
            first_delete_index = first_delete_index or record.index

        elif kind == "delete_scope":
            deleted_scopes[scope] = record.index
            writes_after_delete[scope] = 0
            first_delete_index = first_delete_index or record.index

        elif kind == "query":
            assertion = op.get("assert", {})
            outcome = _query_outcome(record)
            if scope in deleted_scopes and _needles(assertion.get("must_not_contain")):
                post_delete_forbidden += 1
                leaked = _leaked(assertion, outcome)
                resurrected += int(leaked)
                if writes_after_delete.get(scope, 0) > 0:
                    reextraction_total += 1
                    reextraction_hits += int(leaked)
            if _needles(assertion.get("must_contain")) and first_delete_index is not None:
                recall_total += 1
                recall_passed += int(bool(record.passed))

        elif kind == "inspect":
            if scope in deleted_scopes and "raw_turns" in op.get("assert", {}):
                inspect_total += 1
                actual = record.detail.get("raw_turns")
                expected = op["assert"]["raw_turns"]
                if isinstance(actual, int) and actual > expected:
                    inspect_survived += 1

    return {
        "deletion_resurrection_rate": _rate(resurrected, post_delete_forbidden),
        "reextraction_resurrection_rate": _rate(reextraction_hits, reextraction_total),
        "raw_turn_survival_rate": _rate(inspect_survived, inspect_total),
        "post_delete_recall": _rate(recall_passed, recall_total),
    }


def score_poisoned_write(
    ops: list[dict[str, Any]], records: list[OpRecord], _outcomes: dict[str, QueryOutcome]
) -> dict[str, Any]:
    poison_written = poison_total = 0
    benign_written = benign_total = 0
    payloads: dict[str, list[str]] = {}
    replayed: set[str] = set()

    for op, record in _pairs(ops, records):
        if op["op"] == "write":
            stored = bool(record.detail.get("stored"))
            if op.get("class") == "poison":
                poison_total += 1
                poison_written += int(stored)
                payloads[op.get("id") or str(record.index)] = _needles(op.get("payload"))
            elif op.get("class") == "benign" and op.get("expect_write"):
                benign_total += 1
                benign_written += int(stored)

        elif op["op"] in {"query", "query_as_of"}:
            haystack = _query_outcome(record).haystack
            for write_id, needles in payloads.items():
                if any(needle.lower() in haystack for needle in needles):
                    replayed.add(write_id)

    benign_rate = _rate(benign_written, benign_total)
    return {
        "injection_write_rate": _rate(poison_written, poison_total),
        "injection_replay_rate": _rate(len(replayed), poison_total),
        "benign_write_rate": benign_rate,
        "false_quarantine_rate": None if benign_rate is None else round(1 - benign_rate, 4),
    }


def score_state_invalidation(
    ops: list[dict[str, Any]], records: list[OpRecord], _outcomes: dict[str, QueryOutcome]
) -> dict[str, Any]:
    stale = accurate = total = 0
    lags: list[int] = []

    indexed = _pairs(ops, records)

    for position, (op, record) in enumerate(indexed):
        if op["op"] != "query":
            continue
        assertion = op.get("assert", {})
        if not assertion.get("expect_state"):
            continue
        outcome = _query_outcome(record)
        total += 1
        stale += int(_has_state(assertion, outcome, "forbidden_state"))
        accurate += int(_has_state(assertion, outcome, "expect_state"))

    # invalidation lag: writes between a mutation and the first current-state
    # query that reports the new value without the old one.
    for position, (op, _record) in enumerate(indexed):
        if op["op"] != "write" or not op.get("mutates"):
            continue
        mutation_text = f"{op.get('human', '')} {op.get('assistant', '')}".lower()
        intervening = 0
        for later_op, later_record in indexed[position + 1 :]:
            if later_op["op"] == "write":
                intervening += 1
                continue
            if later_op["op"] != "query":
                continue
            assertion = later_op.get("assert", {})
            expect = str(assertion.get("expect_state") or "").lower()
            if not expect or expect not in mutation_text:
                continue
            outcome = _query_outcome(later_record)
            if _has_state(assertion, outcome, "expect_state") and not _has_state(
                assertion, outcome, "forbidden_state"
            ):
                lags.append(intervening)
            break

    return {
        "stale_current_state_rate": _rate(stale, total),
        "current_state_accuracy": _rate(accurate, total),
        "invalidation_lag_writes": (
            round(statistics.median(lags), 2) if lags else None
        ),
    }


def _next_state_query(
    indexed: list[tuple[dict[str, Any], OpRecord]], start: int
) -> tuple[dict[str, Any], OpRecord] | None:
    for op, record in indexed[start + 1 :]:
        if op["op"] == "query" and op.get("assert", {}).get("expect_state"):
            return op, record
    return None


def score_provenance_reassertion(
    ops: list[dict[str, Any]], records: list[OpRecord], _outcomes: dict[str, QueryOutcome]
) -> dict[str, Any]:
    indexed = _pairs(ops, records)
    spurious = requotes = 0
    accepted = mutations = 0

    for position, (op, _record) in enumerate(indexed):
        if op["op"] != "write":
            continue
        following = _next_state_query(indexed, position)
        if following is None:
            continue
        query_op, query_record = following
        assertion = query_op.get("assert", {})
        outcome = _query_outcome(query_record)

        if op.get("class") == "requote":
            requotes += 1
            # The re-quoted value is the value the next query must not return.
            spurious += int(_has_state(assertion, outcome, "forbidden_state"))
        elif op.get("class") == "declared_mutation":
            mutations += 1
            accepted += int(
                _has_state(assertion, outcome, "expect_state")
                and not _has_state(assertion, outcome, "forbidden_state")
            )

    return {
        "spurious_supersession_rate": _rate(spurious, requotes),
        "declared_mutation_accept_rate": _rate(accepted, mutations),
    }


def score_point_in_time(
    ops: list[dict[str, Any]], records: list[OpRecord], _outcomes: dict[str, QueryOutcome]
) -> dict[str, Any]:
    accurate = total = leaked = 0
    deterministic = repeat_total = 0

    for op, record in _pairs(ops, records):
        if op["op"] != "query_as_of":
            continue
        total += 1
        assertion = op.get("assert", {})
        outcome = _query_outcome(record)
        accurate += int(bool(record.passed))
        leaked += int(_leaked(assertion, outcome))
        if int(op.get("repeat", 1)) > 1:
            repeat_total += 1
            checks = {c.check: c.passed for c in record.checks}
            deterministic += int(bool(checks.get("deterministic")))

    return {
        "as_of_state_accuracy": _rate(accurate, total),
        "as_of_future_leakage_rate": _rate(leaked, total),
        "as_of_determinism": _rate(deterministic, repeat_total),
    }


def score_ttl_expiry(
    ops: list[dict[str, Any]], records: list[OpRecord], _outcomes: dict[str, QueryOutcome]
) -> dict[str, Any]:
    indexed = _pairs(ops, records)

    clock_moved = False
    asked_while_live: set[str] = set()
    reused = expired_queries = 0
    repeat_reused = repeat_queries = 0
    live_recalled = live_queries = 0

    for op, record in indexed:
        if op["op"] == "advance_clock":
            clock_moved = True
            continue
        if op["op"] != "query":
            continue

        assertion = op.get("assert", {})
        outcome = _query_outcome(record)
        text = op["text"]

        if not clock_moved:
            asked_while_live.add(text)
            continue

        if _needles(assertion.get("must_not_contain")):
            expired_queries += 1
            leaked = _leaked(assertion, outcome)
            reused += int(leaked)
            # The same question asked before and after expiry: the case a result
            # cache with its own TTL gets wrong.
            if text in asked_while_live:
                repeat_queries += 1
                repeat_reused += int(leaked)

        if _needles(assertion.get("must_contain")):
            live_queries += 1
            live_recalled += int(bool(record.passed))

    return {
        "expired_reuse_rate": _rate(reused, expired_queries),
        "repeat_query_expired_reuse_rate": _rate(repeat_reused, repeat_queries),
        "live_recall": _rate(live_recalled, live_queries),
    }


SCORERS = {
    "deletion_durability": score_deletion_durability,
    "ttl_expiry": score_ttl_expiry,
    "poisoned_write": score_poisoned_write,
    "state_invalidation": score_state_invalidation,
    "provenance_reassertion": score_provenance_reassertion,
    "point_in_time": score_point_in_time,
}


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def build_result(
    adapter: DecisionSafetyAdapter,
    runs: list[BatteryRun],
    *,
    top_k: int,
    command: str,
    raw_path: str,
) -> dict[str, Any]:
    """Assemble an artifact in the shape of benchmarks/competitive/result.schema.json."""
    decision_safety: dict[str, Any] = {"suite_version": "v2"}
    for run in runs:
        decision_safety[run.battery] = {
            key: value for key, value in run.summary().items() if value is not None
        }

    limitations = list(adapter.notes)
    emulated = sorted({op for run in runs for op in run.emulated_ops})
    if emulated:
        limitations.append(
            "Emulated ops (harness worked around a missing API): " + ", ".join(emulated)
        )
    unsupported = [run.battery for run in runs if run.status == "unsupported"]
    if unsupported:
        limitations.append(
            "Batteries reported unsupported rather than scored: " + ", ".join(unsupported)
        )

    return {
        "schema_version": "1.1",
        "system": {
            "name": adapter.name,
            "version": adapter.version,
            "deployment": adapter.deployment,
        },
        "dataset": {
            "name": "Decision Safety Suite v2",
            "release": "v2",
            "sha256": batteries_sha256(),
            "retrieval_unit": "turn_pair",
        },
        "environment": {
            "os": platform.platform(),
            "hardware": platform.machine(),
            "python": platform.python_version(),
            "command": command,
        },
        "configuration": {
            "top_k": top_k,
            "cache_state": "cold",
            "embedding_model": getattr(adapter, "embedding_model", "none"),
            "decision_thresholds": getattr(adapter, "thresholds", {}),
        },
        "capabilities": dict(adapter.capabilities),
        "metrics": {"decision_safety": decision_safety},
        "artifacts": {"raw_results": raw_path},
        "limitations": limitations,
    }


def batteries_sha256() -> str:
    """SHA256 over every battery file, so a result pins its case set."""
    import hashlib

    digest = hashlib.sha256()
    for path in available_batteries():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def format_report(adapter: DecisionSafetyAdapter, runs: list[BatteryRun]) -> str:
    lines = [
        "Decision Safety Suite v2",
        "========================",
        f"System: {adapter.name} {adapter.version} ({adapter.deployment})",
        "",
    ]
    for run in runs:
        lines.append(f"{run.battery}  [{run.status}]")
        if run.reason:
            lines.append(f"  reason: {run.reason}")
        for key, value in run.metrics.items():
            shown = "n/a" if value is None else value
            lines.append(f"  {key:<34} {shown}")
        failed = [r for r in run.records if r.passed is False]
        if failed:
            lines.append(f"  failed ops: {len(failed)}")
            for record in failed:
                reasons = ", ".join(
                    f"{c.check}({c.detail})" for c in record.checks if not c.passed
                )
                lines.append(f"    - #{record.index} {record.op} {record.op_id or ''}: {reasons}")
        broken = [r for r in run.records if r.status in {"unsupported", "error"}]
        if broken:
            lines.append(f"  non-executed ops: {len(broken)}")
            for record in broken:
                lines.append(
                    f"    - #{record.index} {record.op}: {record.status} "
                    f"{record.detail.get('error', '')}"
                )
        lines.append("")
    return "\n".join(lines)


def _load_adapter(
    name: str, *, embeddings: bool = False, restore_threshold: float | None = None
) -> DecisionSafetyAdapter:
    if name in {"agent-memory", "agent_memory"}:
        from benchmarks.decision_safety.adapters.agent_memory_adapter import (
            AgentMemoryAdapter,
        )

        kwargs: dict[str, Any] = {"enable_embeddings": embeddings}
        if restore_threshold is not None:
            kwargs["restore_threshold"] = restore_threshold
        return AgentMemoryAdapter(**kwargs)
    if name == "mem0":
        from benchmarks.decision_safety.adapters.mem0_adapter import (
            Mem0Adapter,
            PendingMaintainerReview,
        )

        try:
            return Mem0Adapter()
        except PendingMaintainerReview as exc:
            # Not a crash: refusing to run is the adapter's designed behaviour.
            raise SystemExit(f"mem0 adapter unavailable: {exc}") from None
    raise SystemExit(f"unknown adapter {name!r} (choices: agent-memory, mem0)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Decision Safety Suite v2.")
    parser.add_argument("--adapter", default="agent-memory")
    parser.add_argument(
        "--battery",
        action="append",
        help="battery name or path; repeatable. Default: every battery file.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="enable semantic retrieval where the adapter supports it "
        "(default: lexical only, for a deterministic run)",
    )
    parser.add_argument(
        "--restore-threshold",
        type=float,
        help="override the adapter's abstention gate, to measure the "
        "recall/leak trade instead of asserting one setting is correct",
    )
    parser.add_argument("--out", help="write the result artifact here")
    parser.add_argument("--raw", help="write per-op raw records here")
    args = parser.parse_args(argv)

    adapter = _load_adapter(
        args.adapter,
        embeddings=args.embeddings,
        restore_threshold=args.restore_threshold,
    )
    declaration_errors = adapter.validate_declaration()
    if declaration_errors:
        print("invalid adapter capability declaration:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in declaration_errors), file=sys.stderr)
        return 2

    names = args.battery or [str(path) for path in available_batteries()]
    runs: list[BatteryRun] = []
    for name in names:
        battery, path = load_battery(name)
        runs.append(run_battery(adapter, battery, path, top_k=args.top_k))

    print(format_report(adapter, runs))

    raw_path = args.raw or "decision_safety_raw.json"
    command = "python -m benchmarks.decision_safety.runner " + " ".join(argv or sys.argv[1:])
    result = build_result(
        adapter, runs, top_k=args.top_k, command=command.strip(), raw_path=raw_path
    )

    from benchmarks.competitive.validate_result import validate_result

    errors = validate_result(result)
    if errors:
        print("result artifact violates the competitive contract:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1

    if args.raw:
        Path(args.raw).write_text(
            json.dumps(
                {
                    "system": result["system"],
                    "batteries": [
                        {
                            "battery": run.battery,
                            "status": run.status,
                            "ops": [asdict(record) for record in run.records],
                        }
                        for run in runs
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
