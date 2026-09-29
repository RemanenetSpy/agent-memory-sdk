"""FastAPI REST server — use agent-memory-sdk over HTTP.

Install:
    pip install agent-memory-sdk[api]

Start the server:
    agent-memory-api
    # or: AGENT_MEMORY_DIR=.agent_memory agent-memory-api

This example shows how to talk to the REST API from Python using httpx,
and also how to embed the server inside your own FastAPI app.
"""
from __future__ import annotations

import os

# ─────────────────────────────────────────────────────────────────────────────
# PART A — Calling the REST API with httpx (server must already be running)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import httpx

    BASE = "http://localhost:8000"

    with httpx.Client(
        base_url=BASE,
        timeout=10,
        headers={"X-API-Key": os.environ.get("AGENT_MEMORY_API_KEY", "")},
    ) as client:

        # Store a memory
        resp = client.post("/memories", json={
            "query":    "How do I reset my password?",
            "response": "Go to Settings → Security → Reset Password.",
            "type":     "conversation",
            "tags":     ["auth", "faq"],
            "confidence": 1.0,
        })
        resp.raise_for_status()
        memory_id = resp.json()["id"]
        print(f"Stored memory: {memory_id[:16]}…")

        # Resolve a query
        resp = client.post("/resolve", json={"query": "How do I reset my password?"})
        data = resp.json()
        print(f"Action: {data['action']}  confidence: {data['confidence']:.2f}")
        if data["action"] == "replay":
            print(f"Response: {data['response']}")

        # List all memories
        resp = client.get("/memories", params={"limit": 10})
        print(f"Total memories: {resp.json()['count']}")

        # Stats
        stats = client.get("/stats").json()
        print(f"Stats: {stats['total']} total, by_type={stats['by_type']}")

        # Archive then delete
        client.post(f"/memories/{memory_id}/archive")
        client.delete(f"/memories/{memory_id}")
        print("Archived and deleted memory.")

except Exception as exc:
    print(f"Server not reachable: {exc}")
    print("Start it with: agent-memory-api")


# ─────────────────────────────────────────────────────────────────────────────
# PART B — Embed the server inside your own FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI

    from agent_memory.api.server import create_app

    # Mount agent-memory as a sub-application under /memory
    main_app = FastAPI(title="My App")

    memory_app = create_app(
        persist_dir=".agent_memory",
        backend="sqlite",
        api_key=os.environ.get("AGENT_MEMORY_API_KEY"),
    )

    main_app.mount("/memory", memory_app)

    # Your own endpoints sit alongside it
    @main_app.get("/health")
    async def health():
        return {"status": "ok"}

    # Run with: uvicorn examples.rest_api:main_app --reload
    print("\nCreated combined app — memory API mounted at /memory")
    print("Endpoints available: GET /memory/, POST /memory/memories, POST /memory/resolve …")

except ImportError:
    print("fastapi not installed — embedding demo skipped.")
