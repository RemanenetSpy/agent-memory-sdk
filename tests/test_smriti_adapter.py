"""Unit tests for the Smriti DecisionSafetyAdapter implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from benchmarks.decision_safety.adapters.base import (
    QueryOutcome,
    WriteOutcome,
)
from benchmarks.decision_safety.adapters.smriti_adapter import SmritiAdapter


def test_smriti_adapter_capability_declaration():
    adapter = SmritiAdapter()
    errors = adapter.validate_declaration()
    assert errors == [], f"Declaration errors: {errors}"
    assert adapter.declared("delete_by_id") == "supported"
    assert adapter.declared("delete_scope") == "supported"
    assert adapter.declared("tombstones") == "supported"
    assert adapter.declared("raw_message_store") == "none"
    assert adapter.declared("ttl") == "supported"
    assert adapter.declared("explicit_supersession") == "supported"
    assert adapter.declared("as_of_query") == "supported"
    assert adapter.declared("explicit_abstention") == "supported"


def test_smriti_adapter_lifecycle():
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def mock_transport(
        method: str, path: str, payload: dict[str, Any] | None, headers: dict[str, str]
    ) -> dict[str, Any]:
        calls.append((method, path, payload))
        return {"status": "ok"}

    adapter = SmritiAdapter(transport=mock_transport)
    assert adapter.tenant_id is None

    adapter.setup(battery="state_invalidation")
    assert adapter.tenant_id is not None
    assert adapter.tenant_id.startswith("ds-state_invalidation-")
    assert calls == [("POST", "/v1/tenants", {"tenant_id": adapter.tenant_id})]

    tenant_to_delete = adapter.tenant_id
    adapter.teardown()
    assert adapter.tenant_id is None
    assert calls[1] == ("DELETE", f"/v1/tenants/{tenant_to_delete}", None)


def test_smriti_adapter_write():
    def mock_transport(
        method: str, path: str, payload: dict[str, Any] | None, headers: dict[str, str]
    ) -> dict[str, Any]:
        assert method == "POST"
        assert path == "/v1/write"
        assert payload["op_id"] == "w1"
        assert payload["scope"] == "user:alice"
        assert payload["human"] == "My favorite editor is Vim"
        assert headers["X-Smriti-Tenant"] == "test-tenant"
        return {
            "stored": True,
            "refs": ["fact-101"],
            "stored_texts": ["user:alice favorite_editor Vim"],
        }

    adapter = SmritiAdapter(transport=mock_transport)
    adapter.tenant_id = "test-tenant"

    outcome = adapter.write(
        op_id="w1",
        scope="user:alice",
        human="My favorite editor is Vim",
        assistant="Noted.",
        at=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
        ttl=3600,
    )

    assert isinstance(outcome, WriteOutcome)
    assert outcome.stored is True
    assert outcome.refs == ["fact-101"]
    assert outcome.stored_texts == ["user:alice favorite_editor Vim"]


def test_smriti_adapter_query_and_abstention():
    responses = [
        {
            "facts": [{"text": "favorite_editor is Neovim", "score": 0.98}],
            "abstained": False,
        },
        {
            "facts": [],
            "abstained": True,
        },
    ]

    def mock_transport(
        method: str, path: str, payload: dict[str, Any] | None, headers: dict[str, str]
    ) -> dict[str, Any]:
        return responses.pop(0)

    adapter = SmritiAdapter(transport=mock_transport)
    adapter.tenant_id = "test-tenant"

    # Query with positive result
    q1 = adapter.query(scope="user:alice", text="What is my favorite editor?", top_k=5)
    assert isinstance(q1, QueryOutcome)
    assert q1.abstained is False
    assert q1.texts == ["favorite_editor is Neovim"]
    assert "neovim" in q1.haystack

    # Query with abstention
    q2 = adapter.query(scope="user:alice", text="What is my shoe size?", top_k=5)
    assert isinstance(q2, QueryOutcome)
    assert q2.abstained is True
    assert q2.texts == []


def test_smriti_adapter_point_in_time():
    def mock_transport(
        method: str, path: str, payload: dict[str, Any] | None, headers: dict[str, str]
    ) -> dict[str, Any]:
        assert path == "/v1/query/as_of"
        assert payload["as_of"] == "2026-01-01T10:00:00+00:00"
        return {
            "facts": [{"text": "favorite_editor was Vim", "valid_from": "2026-01-01T09:00:00Z"}],
            "abstained": False,
        }

    adapter = SmritiAdapter(transport=mock_transport)
    adapter.tenant_id = "test-tenant"

    as_of_dt = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    res = adapter.query_as_of(scope="user:alice", text="editor", as_of=as_of_dt, top_k=3)
    assert res.abstained is False
    assert res.texts == ["favorite_editor was Vim"]


def test_smriti_adapter_delete_and_inspect():
    def mock_transport(
        method: str, path: str, payload: dict[str, Any] | None, headers: dict[str, str]
    ) -> dict[str, Any]:
        if path == "/v1/delete":
            return {"deleted_count": 2}
        if path == "/v1/delete_scope":
            return {"deleted_count": 10}
        if path.startswith("/v1/inspect"):
            return {"active_fact_count": 5}
        return {}

    adapter = SmritiAdapter(transport=mock_transport)
    adapter.tenant_id = "test-tenant"

    assert adapter.delete(op_id="d1", refs=["f1", "f2"]) == 2
    assert adapter.delete_scope(scope="user:alice") == 10

    info = adapter.inspect(scope="user:alice")
    assert info["raw_turns"] == 0
    assert info["stored_memories"] == 5
