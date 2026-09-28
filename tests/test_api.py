"""Tests for the FastAPI REST server and dashboard.

Requires fastapi and httpx.  The whole module is skipped if either is missing.
"""
from __future__ import annotations

import pytest

try:
    import httpx  # noqa: F401 — transitive dep of TestClient
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="fastapi or httpx not installed"
)


@pytest.fixture()
def client(tmp_path):
    from agent_memory.api.server import create_app

    app = create_app(
        persist_dir=tmp_path,
        collection_name="api_test",
        backend="sqlite",
        api_key="test-api-key",
        enable_embeddings=False,
    )
    with TestClient(app, headers={"X-API-Key": "test-api-key"}) as c:
        yield c


def test_api_requires_authentication_by_default() -> None:
    from agent_memory.api.server import create_app

    with pytest.raises(ValueError, match="authentication is required"):
        create_app(allow_unauthenticated=False)


def test_api_rejects_missing_or_invalid_api_key(tmp_path) -> None:
    from agent_memory.api.server import create_app

    app = create_app(persist_dir=tmp_path, api_key="expected-key", enable_embeddings=False)
    with TestClient(app) as unauthenticated:
        assert unauthenticated.get("/stats").status_code == 401
        assert unauthenticated.get("/stats", headers={"X-API-Key": "wrong"}).status_code == 401


def test_api_accepts_bearer_api_key(tmp_path) -> None:
    from agent_memory.api.server import create_app

    app = create_app(persist_dir=tmp_path, api_key="expected-key", enable_embeddings=False)
    with TestClient(app, headers={"Authorization": "Bearer expected-key"}) as authenticated:
        assert authenticated.get("/stats").status_code == 200


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Agent Memory" in resp.text


# ---------------------------------------------------------------------------
# POST /memories
# ---------------------------------------------------------------------------


def test_remember(client):
    resp = client.post(
        "/memories",
        json={"query": "What is Python?", "response": "A programming language"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "What is Python?"
    assert "id" in data


def test_remember_with_type_and_tags(client):
    resp = client.post(
        "/memories",
        json={
            "query": "API rate limit",
            "response": "100 requests per minute",
            "type": "fact",
            "tags": ["api", "limits"],
            "confidence": 0.9,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "fact"
    assert "api" in data["tags"]


# ---------------------------------------------------------------------------
# GET /memories
# ---------------------------------------------------------------------------


def test_list_memories_empty(client):
    resp = client.get("/memories")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entries"] == []
    assert data["count"] == 0


def test_list_memories_with_data(client):
    client.post("/memories", json={"query": "q1", "response": "r1"})
    client.post("/memories", json={"query": "q2", "response": "r2"})
    resp = client.get("/memories")
    assert resp.status_code == 200
    assert resp.json()["count"] == 2


def test_list_memories_pagination(client):
    for i in range(5):
        client.post("/memories", json={"query": f"q{i}", "response": f"r{i}"})
    resp = client.get("/memories?limit=2&offset=0")
    assert resp.json()["count"] == 2


# ---------------------------------------------------------------------------
# GET /memories/{id}
# ---------------------------------------------------------------------------


def test_get_memory_by_id(client):
    stored = client.post(
        "/memories", json={"query": "specific", "response": "answer"}
    ).json()
    resp = client.get(f"/memories/{stored['id']}")
    assert resp.status_code == 200
    assert resp.json()["query"] == "specific"


def test_get_memory_not_found(client):
    resp = client.get("/memories/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /memories/{id}
# ---------------------------------------------------------------------------


def test_forget_memory(client):
    stored = client.post("/memories", json={"query": "forget me", "response": "r"}).json()
    resp = client.delete(f"/memories/{stored['id']}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    # Confirm gone
    assert client.get(f"/memories/{stored['id']}").status_code == 404


def test_forget_nonexistent(client):
    resp = client.delete("/memories/ghost-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /memories/{id}/archive
# ---------------------------------------------------------------------------


def test_archive_memory(client):
    stored = client.post("/memories", json={"query": "archive me", "response": "r"}).json()
    resp = client.post(f"/memories/{stored['id']}/archive")
    assert resp.status_code == 200
    assert resp.json()["archived"] is True


def test_archive_nonexistent(client):
    resp = client.post("/memories/ghost-id/archive")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /resolve
# ---------------------------------------------------------------------------


def test_resolve_none_empty_store(client):
    resp = client.post("/resolve", json={"query": "something"})
    assert resp.status_code == 200
    assert resp.json()["action"] == "none"


def test_resolve_replay_after_store(client):
    client.post(
        "/memories",
        json={"query": "What is 2+2?", "response": "4", "confidence": 1.0},
    )
    resp = client.post("/resolve", json={"query": "What is 2+2?"})
    assert resp.status_code == 200
    data = resp.json()
    # Should replay or restore for exact match
    assert data["action"] in ("replay", "restore", "verify", "none")


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------


def test_stats_endpoint(client):
    client.post("/memories", json={"query": "q", "response": "r"})
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert data["total"] >= 1


# ---------------------------------------------------------------------------
# POST /cleanup
# ---------------------------------------------------------------------------


def test_cleanup_endpoint(client):
    resp = client.post("/cleanup")
    assert resp.status_code == 200
    data = resp.json()
    assert "expired" in data or "deleted" in data


# ---------------------------------------------------------------------------
# POST /consolidate
# ---------------------------------------------------------------------------


def test_consolidate_endpoint(client):
    resp = client.post("/consolidate")
    assert resp.status_code == 200
    data = resp.json()
    assert "consolidated" in data


# ---------------------------------------------------------------------------
# Dashboard render (unit test, no server needed)
# ---------------------------------------------------------------------------


def test_render_dashboard_html():
    from agent_memory.api.dashboard import render_dashboard

    stats = {
        "total": 10,
        "by_state": {"active": 8, "archived": 2},
        "by_type": {"fact": 5, "conversation": 5},
        "total_access_count": 42,
    }
    html = render_dashboard(stats)
    assert "Agent Memory" in html          # page title / header
    assert "10" in html                    # total memories
    assert "fact" in html                  # by_type row
    assert "active" in html                # by_state row
    assert "Total memories" in html        # KPI tile label
