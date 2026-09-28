"""Tests for the Decision Safety Suite v2 op-script runner.

The scoring math is pinned with synthetic adapters — one that answers every op
exactly as the battery specifies, one frozen in the past, one that abstains from
everything, one that stores everything — because a harness bug here would
silently flatter every system it measures. The "abstains from everything" and
"stores everything" cases are the ones that prove the paired metrics work.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from benchmarks.competitive.validate_result import validate_result
from benchmarks.decision_safety.adapters.base import (
    CapabilityUnsupported,
    DecisionSafetyAdapter,
    QueryOutcome,
    WriteOutcome,
)
from benchmarks.decision_safety.runner import (
    available_batteries,
    build_result,
    contains,
    evaluate_assertion,
    load_battery,
    run_battery,
)

FULL_CAPABILITIES = {
    "delete_by_id": "supported",
    "delete_scope": "supported",
    "tombstones": "supported",
    "raw_message_store": "none",
    "ttl": "supported",
    "explicit_supersession": "supported",
    "as_of_query": "supported",
    "explicit_abstention": "supported",
}


def _battery(name: str) -> tuple[dict, Path]:
    return load_battery(name)


def _write_ids(battery: dict) -> list[str]:
    return [op["id"] for op in battery["ops"] if op["op"] == "write" and op.get("id")]


# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


class FakeAdapter(DecisionSafetyAdapter):
    """Base fake: records ops, abstains on every query."""

    name = "fake"
    version = "test"
    capabilities = FULL_CAPABILITIES

    def __init__(self, *, stored: dict[str, bool] | None = None, raw_turns: int = 0) -> None:
        self.stored = stored or {}
        self.raw_turns = raw_turns
        self.deleted: list[str] = []
        self.scopes_cleared: list[str] = []
        self.clock: list[datetime] = []
        self.batteries: list[str] = []
        self.written: list[tuple[str, int | None]] = []

    def setup(self, *, battery: str) -> None:
        self.batteries.append(battery)

    def write(self, *, op_id, scope, human, assistant, at=None, ttl=None) -> WriteOutcome:
        self.written.append((op_id, ttl))
        stored = self.stored.get(op_id, True)
        return WriteOutcome(stored=stored, refs=[f"{op_id}-ref"] if stored else [])

    def delete(self, *, op_id, refs) -> int:
        self.deleted.append(op_id)
        return len(refs)

    def delete_scope(self, *, scope) -> int:
        self.scopes_cleared.append(scope)
        return 1

    def advance_clock(self, *, seconds, now) -> None:
        self.clock.append(now)

    def query(self, *, scope, text, top_k) -> QueryOutcome:
        return QueryOutcome(abstained=True)

    def query_as_of(self, *, scope, text, as_of, top_k) -> QueryOutcome:
        return self.query(scope=scope, text=text, top_k=top_k)

    def inspect(self, *, scope) -> dict:
        return {"raw_turns": self.raw_turns}


class LeakingAdapter(FakeAdapter):
    """Returns fixed text for named queries; abstains otherwise."""

    name = "leaking"

    def __init__(self, answers: dict[str, list[str]], **kwargs) -> None:
        super().__init__(**kwargs)
        self.answers = answers

    def query(self, *, scope, text, top_k) -> QueryOutcome:
        if text in self.answers:
            return QueryOutcome(abstained=False, texts=list(self.answers[text]))
        return QueryOutcome(abstained=True)


class ScriptedAdapter(FakeAdapter):
    """Answers each query op in script order, reading that op's own assertion.

    Per-op rather than per-text: the same question is asked several times in one
    battery with different correct answers, which is the whole point of the
    supersession batteries.
    """

    name = "scripted"

    def setup(self, *, battery: str) -> None:
        super().setup(battery=battery)
        data, _ = load_battery(battery)
        self._ops = [op for op in data["ops"] if op["op"] in {"query", "query_as_of"}]
        self._index = 0
        self._seen = 0

    def _current(self, text: str) -> dict:
        while self._index < len(self._ops) and self._ops[self._index]["text"] != text:
            self._index += 1
            self._seen = 0
        op = self._ops[self._index]
        self._seen += 1
        if self._seen >= int(op.get("repeat", 1)):
            self._index += 1
            self._seen = 0
        return op

    def _respond(self, op: dict) -> QueryOutcome:
        assertion = op.get("assert", {})
        needles = list(assertion.get("must_contain") or [])
        if assertion.get("expect_state"):
            needles.append(str(assertion["expect_state"]))
        if assertion.get("expect_abstain") and not needles:
            return QueryOutcome(abstained=True, texts=[])
        return QueryOutcome(abstained=not needles, texts=needles)

    def query(self, *, scope, text, top_k) -> QueryOutcome:
        return self._respond(self._current(text))

    def query_as_of(self, *, scope, text, as_of, top_k) -> QueryOutcome:
        return self._respond(self._current(text))


class FrozenAdapter(ScriptedAdapter):
    """Never applies an update: keeps reporting the superseded value."""

    name = "frozen"

    def _respond(self, op: dict) -> QueryOutcome:
        stale = op.get("assert", {}).get("forbidden_state")
        if stale:
            return QueryOutcome(abstained=False, texts=[str(stale)])
        return super()._respond(op)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("haystack", "needle", "expected"),
    [
        ("postgresql 14 in staging", "14", True),
        ("released in 2014", "14", False),
        ("the sla is 99.5%", "99.5", True),
        ("the sla is 99.95%", "99.9", False),
        ("deploy to eu-west-1", "eu-west-1", True),
        ("deploy to eu-central-1", "eu-west-1", False),
        ("flight at 09:40", "09:40", True),
        ("launch on 2 may", "2 May", True),
        ("new number: 07700 900456.", "900456", True),  # sentence-final needle
        ("the sla is 99.5%.", "99.5", True),
        ("pricing $39/month", "$39", True),
        ("", "anything", False),
    ],
)
def test_needle_matching_respects_token_boundaries(haystack, needle, expected) -> None:
    assert contains(haystack, needle) is expected


def test_assertion_checks_cover_every_declared_key() -> None:
    checks = evaluate_assertion(
        {"must_contain": ["Datadog"], "must_not_contain": ["Stripe"], "expect_abstain": True},
        QueryOutcome(abstained=False, texts=["works at Datadog"]),
    )
    by_check = {(c.check, c.detail): c.passed for c in checks}

    assert by_check[("must_contain", "Datadog")] is True
    assert by_check[("must_not_contain", "Stripe")] is True
    assert by_check[("expect_abstain", "")] is False


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------


def test_missing_capability_reports_unsupported_not_zero() -> None:
    class NoTimeTravel(ScriptedAdapter):
        capabilities = {**FULL_CAPABILITIES, "as_of_query": "unsupported"}

    battery, path = _battery("point_in_time")
    run = run_battery(NoTimeTravel(), battery, path)

    assert run.status == "unsupported"
    assert run.metrics == {}
    assert "as_of_query=unsupported" in run.reason


def test_partial_capability_does_not_count_as_supported() -> None:
    class PartialTimeTravel(ScriptedAdapter):
        capabilities = {**FULL_CAPABILITIES, "as_of_query": "partial"}

    battery, path = _battery("point_in_time")

    assert run_battery(PartialTimeTravel(), battery, path).status == "unsupported"


def test_unsupported_op_is_recorded_without_crashing_the_battery() -> None:
    class NoInspect(ScriptedAdapter):
        def inspect(self, *, scope):
            raise CapabilityUnsupported("inspect", "no store introspection")

    battery, path = _battery("deletion_durability")
    run = run_battery(NoInspect(), battery, path)

    inspects = [r for r in run.records if r.op == "inspect"]
    assert inspects and all(r.status == "unsupported" for r in inspects)
    assert run.status == "scored"
    assert run.metrics["raw_turn_survival_rate"] is None


def test_adapter_declaration_is_checked() -> None:
    class Undeclared(FakeAdapter):
        capabilities = {"delete_by_id": "sort_of"}

    errors = Undeclared().validate_declaration()

    assert any("as_of_query" in error for error in errors)
    assert any("'sort_of'" in error for error in errors)


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------


def test_clock_advances_monotonically_and_reaches_the_adapter() -> None:
    adapter = ScriptedAdapter()
    battery, path = _battery("state_invalidation")
    run_battery(adapter, battery, path)

    assert len(adapter.clock) == 2
    assert adapter.clock == sorted(adapter.clock)


# ---------------------------------------------------------------------------
# deletion_durability
# ---------------------------------------------------------------------------


def test_deletion_battery_scores_a_resurrected_fact() -> None:
    adapter = LeakingAdapter({"What is my home address?": ["44 Brunswick Road, Leeds"]})
    battery, path = _battery("deletion_durability")

    metrics = run_battery(adapter, battery, path).metrics

    assert metrics["deletion_resurrection_rate"] > 0
    assert metrics["reextraction_resurrection_rate"] > 0


def test_deletion_battery_is_clean_when_nothing_returns() -> None:
    adapter = ScriptedAdapter()
    battery, path = _battery("deletion_durability")

    metrics = run_battery(adapter, battery, path).metrics

    assert metrics["deletion_resurrection_rate"] == 0.0
    assert metrics["post_delete_recall"] == 1.0
    assert adapter.scopes_cleared == ["user:alice"]


def test_deleting_everything_cannot_win_the_deletion_battery() -> None:
    """The paired control is what stops 'empty the store' from scoring 1.0."""
    adapter = FakeAdapter()  # abstains on everything
    battery, path = _battery("deletion_durability")

    metrics = run_battery(adapter, battery, path).metrics

    assert metrics["deletion_resurrection_rate"] == 0.0
    assert metrics["post_delete_recall"] == 0.0


def test_raw_turn_survival_is_scored_from_inspect() -> None:
    battery, path = _battery("deletion_durability")

    surviving = run_battery(ScriptedAdapter(raw_turns=3), battery, path).metrics
    clean = run_battery(ScriptedAdapter(raw_turns=0), battery, path).metrics

    assert surviving["raw_turn_survival_rate"] == 1.0
    assert clean["raw_turn_survival_rate"] == 0.0


# ---------------------------------------------------------------------------
# ttl_expiry
# ---------------------------------------------------------------------------


def test_ttl_battery_passes_a_system_that_stops_serving_expired_memories() -> None:
    battery, path = _battery("ttl_expiry")
    adapter = ScriptedAdapter()

    metrics = run_battery(adapter, battery, path).metrics

    assert metrics["expired_reuse_rate"] == 0.0
    assert metrics["repeat_query_expired_reuse_rate"] == 0.0
    assert metrics["live_recall"] == 1.0
    assert ("t1", 3600) in adapter.written  # the TTL reached the adapter


def test_ttl_battery_catches_an_expired_memory_served_on_a_repeated_query() -> None:
    """The regression this battery exists for: a cache outliving the memory."""
    battery, path = _battery("ttl_expiry")
    adapter = LeakingAdapter(
        {"What is my temporary access code?": ["your temporary access code is 4417-TMP"]}
    )

    metrics = run_battery(adapter, battery, path).metrics

    assert metrics["expired_reuse_rate"] > 0
    assert metrics["repeat_query_expired_reuse_rate"] == 1.0


def test_ttl_battery_pairs_expiry_against_live_recall() -> None:
    battery, path = _battery("ttl_expiry")

    metrics = run_battery(FakeAdapter(), battery, path).metrics  # abstains always

    assert metrics["expired_reuse_rate"] == 0.0
    assert metrics["live_recall"] == 0.0


def test_ttl_battery_is_unsupported_without_ttl() -> None:
    class NoTTL(ScriptedAdapter):
        capabilities = {**FULL_CAPABILITIES, "ttl": "unsupported"}

    battery, path = _battery("ttl_expiry")
    run = run_battery(NoTTL(), battery, path)

    assert run.status == "unsupported"
    assert run.metrics == {}


# ---------------------------------------------------------------------------
# poisoned_write
# ---------------------------------------------------------------------------


def test_poisoned_write_pairs_injection_against_false_quarantine() -> None:
    battery, path = _battery("poisoned_write")

    permissive = run_battery(ScriptedAdapter(), battery, path).metrics
    assert permissive["injection_write_rate"] == 1.0
    assert permissive["benign_write_rate"] == 1.0
    assert permissive["false_quarantine_rate"] == 0.0

    paranoid = ScriptedAdapter(stored=dict.fromkeys(_write_ids(battery), False))
    paranoid_metrics = run_battery(paranoid, battery, path).metrics
    assert paranoid_metrics["injection_write_rate"] == 0.0
    assert paranoid_metrics["benign_write_rate"] == 0.0
    assert paranoid_metrics["false_quarantine_rate"] == 1.0


def test_poisoned_write_rewards_only_the_system_that_separates_them() -> None:
    battery, path = _battery("poisoned_write")
    poison_only = {
        op["id"]: op.get("class") != "poison"
        for op in battery["ops"]
        if op["op"] == "write" and op.get("id")
    }

    metrics = run_battery(ScriptedAdapter(stored=poison_only), battery, path).metrics

    assert metrics["injection_write_rate"] == 0.0
    assert metrics["benign_write_rate"] == 1.0
    assert metrics["false_quarantine_rate"] == 0.0


def test_injection_replay_is_scored_from_what_surfaced() -> None:
    battery, path = _battery("poisoned_write")
    adapter = LeakingAdapter({"What is the refund window?": ["the refund window is 365 days"]})

    metrics = run_battery(adapter, battery, path).metrics

    assert metrics["injection_replay_rate"] > 0


# ---------------------------------------------------------------------------
# state_invalidation
# ---------------------------------------------------------------------------


def test_state_invalidation_pairs_staleness_against_accuracy() -> None:
    battery, path = _battery("state_invalidation")

    silent = run_battery(FakeAdapter(), battery, path).metrics
    assert silent["stale_current_state_rate"] == 0.0
    assert silent["current_state_accuracy"] == 0.0
    assert silent["invalidation_lag_writes"] is None

    correct = run_battery(ScriptedAdapter(), battery, path).metrics
    assert correct["stale_current_state_rate"] == 0.0
    assert correct["current_state_accuracy"] == 1.0
    assert correct["invalidation_lag_writes"] == 0


def test_state_invalidation_scores_a_store_stuck_on_the_old_value() -> None:
    battery, path = _battery("state_invalidation")

    metrics = run_battery(FrozenAdapter(), battery, path).metrics

    assert metrics["stale_current_state_rate"] == 1.0
    assert metrics["current_state_accuracy"] == 0.0
    assert metrics["invalidation_lag_writes"] is None


# ---------------------------------------------------------------------------
# provenance_reassertion
# ---------------------------------------------------------------------------


def test_provenance_pairs_supersession_against_mutation_acceptance() -> None:
    battery, path = _battery("provenance_reassertion")

    correct = run_battery(ScriptedAdapter(), battery, path).metrics
    assert correct["spurious_supersession_rate"] == 0.0
    assert correct["declared_mutation_accept_rate"] == 1.0


def test_immobility_cannot_win_the_provenance_battery() -> None:
    """A store that never updates fails the pair, not just one side of it."""
    battery, path = _battery("provenance_reassertion")

    metrics = run_battery(FrozenAdapter(), battery, path).metrics

    assert metrics["spurious_supersession_rate"] == 1.0
    assert metrics["declared_mutation_accept_rate"] == 0.0


# ---------------------------------------------------------------------------
# point_in_time
# ---------------------------------------------------------------------------


def test_point_in_time_scores_accuracy_leakage_and_determinism() -> None:
    battery, path = _battery("point_in_time")

    metrics = run_battery(ScriptedAdapter(), battery, path).metrics

    assert metrics["as_of_state_accuracy"] == 1.0
    assert metrics["as_of_future_leakage_rate"] == 0.0
    assert metrics["as_of_determinism"] == 1.0


def test_nondeterministic_as_of_is_caught() -> None:
    class Flaky(ScriptedAdapter):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self._n = 0

        def query_as_of(self, *, scope, text, as_of, top_k):
            self._n += 1
            outcome = super().query_as_of(scope=scope, text=text, as_of=as_of, top_k=top_k)
            return QueryOutcome(
                abstained=outcome.abstained, texts=[*outcome.texts, f"run-{self._n}"]
            )

    battery, path = _battery("point_in_time")

    assert run_battery(Flaky(), battery, path).metrics["as_of_determinism"] == 0.0


def test_leaking_a_later_value_into_an_as_of_query_is_scored() -> None:
    class Leaky(ScriptedAdapter):
        def query_as_of(self, *, scope, text, as_of, top_k):
            return QueryOutcome(abstained=False, texts=["launch 2 May", "pricing $39/month"])

    battery, path = _battery("point_in_time")
    metrics = run_battery(Leaky(), battery, path).metrics

    assert metrics["as_of_future_leakage_rate"] > 0
    assert metrics["as_of_state_accuracy"] < 1.0


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


def _run_all(adapter: DecisionSafetyAdapter) -> list:
    runs = []
    for path in available_batteries():
        battery = json.loads(path.read_text())
        runs.append(run_battery(adapter, battery, path))
    return runs


def test_result_artifact_satisfies_the_competitive_contract() -> None:
    adapter = ScriptedAdapter()
    result = build_result(
        adapter, _run_all(adapter), top_k=5, command="pytest", raw_path="raw.json"
    )

    assert validate_result(result) == []
    assert result["metrics"]["decision_safety"]["suite_version"] == "v2"
    assert len(result["dataset"]["sha256"]) == 64


def test_unsupported_battery_is_named_in_limitations() -> None:
    class NoTimeTravel(ScriptedAdapter):
        capabilities = {**FULL_CAPABILITIES, "as_of_query": "unsupported"}

    adapter = NoTimeTravel()
    battery, path = _battery("point_in_time")
    run = run_battery(adapter, battery, path)

    result = build_result(adapter, [run], top_k=5, command="pytest", raw_path="raw.json")

    assert any("point_in_time" in note for note in result["limitations"])
    assert validate_result(result) == []


def test_emulated_ops_are_named_in_limitations() -> None:
    class Emulating(ScriptedAdapter):
        emulated_ops = {"delete_scope"}

    adapter = Emulating()
    battery, path = _battery("deletion_durability")
    run = run_battery(adapter, battery, path)

    result = build_result(adapter, [run], top_k=5, command="pytest", raw_path="raw.json")
    emulated_record = next(r for r in run.records if r.op == "delete_scope")

    assert emulated_record.status == "emulated"
    assert any("delete_scope" in note for note in result["limitations"])
