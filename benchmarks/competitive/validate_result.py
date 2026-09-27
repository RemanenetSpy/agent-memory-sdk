"""Validate competitive benchmark result artifacts without external dependencies.

Usage:
    python -m benchmarks.competitive.validate_result path/to/result.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_SCHEMA_VERSIONS = {"1.0", "1.1"}
_DEPLOYMENTS = {"local", "self_hosted", "hosted"}
_RETRIEVAL_UNITS = {"turn", "turn_pair", "session", "fact", "custom"}
_CACHE_STATES = {"cold", "warm", "disabled"}
_SHA256 = re.compile(r"^[a-f0-9]{64}$")

_SUPPORT = {"supported", "partial", "unsupported"}
_CAPABILITIES = {
    "delete_by_id": _SUPPORT,
    "delete_scope": _SUPPORT,
    "tombstones": _SUPPORT,
    "raw_message_store": {"retained", "retained_configurable", "none"},
    "ttl": _SUPPORT,
    "explicit_supersession": _SUPPORT,
    "as_of_query": _SUPPORT,
    "explicit_abstention": _SUPPORT,
}

_BATTERY_STATUSES = {"scored", "unsupported", "not_tested"}

# Decision Safety Suite v2: metric -> the paired metric it may not be published
# without. See docs/decision-safety-suite.md. A safety number that rewards
# blocking is meaningless without the number that punishes overdoing it.
_BATTERIES: dict[str, dict[str, list[str]]] = {
    "deletion_durability": {
        "deletion_resurrection_rate": ["post_delete_recall"],
        "raw_turn_survival_rate": [],
        "reextraction_resurrection_rate": [],
        "post_delete_recall": [],
    },
    "ttl_expiry": {
        "expired_reuse_rate": ["live_recall"],
        "repeat_query_expired_reuse_rate": ["live_recall"],
        "live_recall": [],
    },
    "poisoned_write": {
        "injection_write_rate": ["benign_write_rate", "false_quarantine_rate"],
        "injection_replay_rate": ["benign_write_rate"],
        "benign_write_rate": [],
        "false_quarantine_rate": [],
    },
    "state_invalidation": {
        "stale_current_state_rate": ["current_state_accuracy"],
        "current_state_accuracy": [],
    },
    "provenance_reassertion": {
        "spurious_supersession_rate": ["declared_mutation_accept_rate"],
        "declared_mutation_accept_rate": [],
    },
    "point_in_time": {
        "as_of_state_accuracy": [],
        "as_of_future_leakage_rate": ["as_of_state_accuracy"],
        "as_of_determinism": [],
    },
}


def _require_string(data: dict[str, Any], key: str, errors: list[str], path: str) -> None:
    if not isinstance(data.get(key), str) or not data[key]:
        errors.append(f"{path}.{key} must be a non-empty string")


def _require_mapping(data: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return {}
    return value


def _validate_capabilities(result: dict[str, Any], errors: list[str]) -> None:
    capabilities = result.get("capabilities")
    if capabilities is None:
        return
    if not isinstance(capabilities, dict):
        errors.append("capabilities must be an object")
        return
    for key, value in capabilities.items():
        allowed = _CAPABILITIES.get(key)
        if allowed is None:
            errors.append(f"capabilities.{key} is not a declared capability")
        elif value not in allowed:
            errors.append(f"capabilities.{key} must be one of {sorted(allowed)}")


def _validate_decision_safety(metrics: dict[str, Any], errors: list[str]) -> None:
    safety = metrics.get("decision_safety")
    if safety is None:
        return
    if not isinstance(safety, dict):
        errors.append("metrics.decision_safety must be an object")
        return

    for battery, pairs in _BATTERIES.items():
        reported = safety.get(battery)
        if reported is None:
            continue
        path = f"metrics.decision_safety.{battery}"
        if not isinstance(reported, dict):
            errors.append(f"{path} must be an object")
            continue
        if reported.get("status") not in _BATTERY_STATUSES:
            errors.append(f"{path}.status must be one of {sorted(_BATTERY_STATUSES)}")

        scored = [key for key in pairs if key in reported]
        if reported.get("status") != "scored" and scored:
            errors.append(f"{path} reports metrics but status is not 'scored'")

        for key in scored:
            value = reported[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{path}.{key} must be a number between 0 and 1")
            elif not 0 <= value <= 1:
                errors.append(f"{path}.{key} must be a number between 0 and 1")
            for required in pairs[key]:
                if required not in reported:
                    errors.append(
                        f"{path}.{key} may not be published without {path}.{required}"
                    )

        lag = reported.get("invalidation_lag_writes")
        if lag is not None and (
            not isinstance(lag, (int, float)) or isinstance(lag, bool) or lag < 0
        ):
            errors.append(f"{path}.invalidation_lag_writes must be null or non-negative")

    unknown = set(safety) - set(_BATTERIES) - {"suite_version"}
    for key in sorted(unknown):
        errors.append(f"metrics.decision_safety.{key} is not a defined battery")


def validate_result(result: dict[str, Any]) -> list[str]:
    """Return contract violations for a competitive benchmark result."""
    errors: list[str] = []
    if result.get("schema_version") not in _SCHEMA_VERSIONS:
        errors.append(f"schema_version must be one of {sorted(_SCHEMA_VERSIONS)}")

    system = _require_mapping(result, "system", errors)
    for key in ("name", "version"):
        _require_string(system, key, errors, "system")
    if system.get("deployment") not in _DEPLOYMENTS:
        errors.append(f"system.deployment must be one of {sorted(_DEPLOYMENTS)}")

    dataset = _require_mapping(result, "dataset", errors)
    for key in ("name", "release"):
        _require_string(dataset, key, errors, "dataset")
    if not isinstance(dataset.get("sha256"), str) or not _SHA256.fullmatch(dataset["sha256"]):
        errors.append("dataset.sha256 must be a lowercase 64-character SHA256")
    if dataset.get("retrieval_unit") not in _RETRIEVAL_UNITS:
        errors.append(f"dataset.retrieval_unit must be one of {sorted(_RETRIEVAL_UNITS)}")

    environment = _require_mapping(result, "environment", errors)
    for key in ("os", "hardware", "python", "command"):
        _require_string(environment, key, errors, "environment")

    configuration = _require_mapping(result, "configuration", errors)
    top_k = configuration.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        errors.append("configuration.top_k must be a positive integer")
    if configuration.get("cache_state") not in _CACHE_STATES:
        errors.append(f"configuration.cache_state must be one of {sorted(_CACHE_STATES)}")

    metrics = _require_mapping(result, "metrics", errors)
    for key in ("recall_at_5", "recall_at_10", "qa_accuracy", "wrong_replay_rate"):
        if key in metrics and (
            not isinstance(metrics[key], (int, float)) or not 0 <= metrics[key] <= 1
        ):
            errors.append(f"metrics.{key} must be a number between 0 and 1")
    for key in ("process_rss_mb", "ingest_seconds", "cost_usd"):
        if key in metrics and (
            not isinstance(metrics[key], (int, float)) or metrics[key] < 0
        ):
            errors.append(f"metrics.{key} must be a non-negative number")
    _validate_decision_safety(metrics, errors)
    _validate_capabilities(result, errors)

    artifacts = _require_mapping(result, "artifacts", errors)
    _require_string(artifacts, "raw_results", errors, "artifacts")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m benchmarks.competitive.validate_result RESULT.json", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2
    errors = validate_result(result)
    if errors:
        print("invalid competitive benchmark result:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"valid competitive benchmark result: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
