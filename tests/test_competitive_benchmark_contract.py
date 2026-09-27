"""Tests for the public competitive benchmark result contract."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.competitive.validate_result import validate_result

FIXTURE = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "competitive"
    / "examples"
    / "agent_memory_longmemeval_s_lexical.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_agent_memory_fixture_satisfies_contract() -> None:
    assert validate_result(_fixture()) == []


def test_contract_requires_reproducibility_metadata() -> None:
    result = _fixture()
    del result["dataset"]["sha256"]
    del result["environment"]["command"]

    errors = validate_result(result)

    assert "dataset.sha256 must be a lowercase 64-character SHA256" in errors
    assert "environment.command must be a non-empty string" in errors


def test_contract_rejects_invalid_comparative_metrics() -> None:
    result = _fixture()
    result["metrics"]["recall_at_5"] = 1.1
    result["configuration"]["cache_state"] = "unknown"

    errors = validate_result(result)

    assert "metrics.recall_at_5 must be a number between 0 and 1" in errors
    assert any(error.startswith("configuration.cache_state") for error in errors)


# --- Decision Safety Suite v2 -----------------------------------------------

BATTERY_DIR = Path(__file__).parents[1] / "benchmarks" / "decision_safety"


def test_contract_rejects_unpaired_safety_metric() -> None:
    """A blocking metric may not be published without its permissiveness pair."""
    result = _fixture()
    result["schema_version"] = "1.1"
    result["metrics"]["decision_safety"] = {
        "poisoned_write": {"status": "scored", "injection_write_rate": 0.0}
    }

    errors = validate_result(result)

    assert (
        "metrics.decision_safety.poisoned_write.injection_write_rate may not be "
        "published without metrics.decision_safety.poisoned_write.benign_write_rate"
    ) in errors


def test_contract_accepts_paired_safety_metrics() -> None:
    result = _fixture()
    result["schema_version"] = "1.1"
    result["metrics"]["decision_safety"] = {
        "suite_version": "v2",
        "poisoned_write": {
            "status": "scored",
            "battery_file": "benchmarks/decision_safety/poisoned_write.v1.json",
            "injection_write_rate": 0.0,
            "injection_replay_rate": 0.0,
            "benign_write_rate": 1.0,
            "false_quarantine_rate": 0.0,
        },
        "point_in_time": {"status": "unsupported"},
    }
    result["capabilities"] = {"delete_by_id": "supported", "as_of_query": "unsupported"}

    assert validate_result(result) == []


def test_contract_rejects_metrics_on_unsupported_battery() -> None:
    result = _fixture()
    result["metrics"]["decision_safety"] = {
        "point_in_time": {"status": "unsupported", "as_of_state_accuracy": 1.0}
    }

    errors = validate_result(result)

    assert (
        "metrics.decision_safety.point_in_time reports metrics but status is not 'scored'"
        in errors
    )


def test_contract_rejects_undeclared_capability_values() -> None:
    result = _fixture()
    result["capabilities"] = {"as_of_query": "sort_of", "time_travel": "supported"}

    errors = validate_result(result)

    assert any(error.startswith("capabilities.as_of_query must be one of") for error in errors)
    assert "capabilities.time_travel is not a declared capability" in errors


def test_every_defined_battery_has_a_versioned_case_file() -> None:
    from benchmarks.competitive.validate_result import _BATTERIES

    for battery in _BATTERIES:
        assert list(BATTERY_DIR.glob(f"{battery}.v*.json")), f"no case file for {battery}"


def test_battery_files_are_well_formed() -> None:
    for path in sorted(BATTERY_DIR.glob("*.v*.json")):
        battery = json.loads(path.read_text())
        assert battery["name"] == path.name.split(".")[0]
        assert isinstance(battery["version"], int)
        assert battery["description"]
        assert isinstance(battery["requires_capabilities"], list)

        ops = battery["ops"]
        assert ops, f"{path.name} has no ops"
        seen: set[str] = set()
        for op in ops:
            assert op["op"] in {
                "write",
                "delete",
                "delete_scope",
                "advance_clock",
                "query",
                "query_as_of",
                "inspect",
            }, f"{path.name}: unknown op {op['op']}"
            op_id = op.get("id")
            if op_id is not None:
                assert op_id not in seen, f"{path.name}: duplicate op id {op_id}"
                seen.add(op_id)
            if op["op"] in {"query", "query_as_of"}:
                assert op["assert"], f"{path.name}: {op_id} has no assertions"
            if op["op"] == "query_as_of":
                assert op.get("as_of"), f"{path.name}: {op_id} has no as_of"
            for ref in ("target", "mutates", "requotes"):
                if ref in op:
                    assert op[ref] in seen, f"{path.name}: {ref} -> unknown op {op[ref]}"


def test_every_battery_has_a_paired_control_case() -> None:
    """No battery may be winnable by refusing or deleting everything."""
    for path in sorted(BATTERY_DIR.glob("*.v*.json")):
        battery = json.loads(path.read_text())
        positives = [
            op
            for op in battery["ops"]
            if op["op"] in {"query", "query_as_of"}
            and op["assert"].get("must_contain")
            and not op["assert"].get("expect_abstain")
        ]
        assert positives, f"{path.name} has no case that punishes over-blocking"
