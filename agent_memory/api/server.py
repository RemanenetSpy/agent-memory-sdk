from __future__ import annotations

import hmac
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent_memory.models import MemoryAction

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Request / response schemas (Pydantic models, usable independently of FastAPI)
# ---------------------------------------------------------------------------


class RememberRequest(BaseModel):
    query: str
    response: str
    content: str | None = None
    type: str = "conversation"
    scope: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    requires_verification: bool = False
    ttl: str | int | float | None = None


class ResolveRequest(BaseModel):
    query: str
    mode: str = "auto"
    top_k: int = Field(default=3, ge=1, le=20)
    scope: list[str] | None = None
    enable_verify: bool = True


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    persist_dir: str | Path = ".agent_memory",
    collection_name: str = "agent_memories",
    backend: str = "sqlite",
    memory: Any | None = None,
    api_key: str | None = None,
    allow_unauthenticated: bool = False,
    **memory_kwargs: Any,
) -> Any:
    """Create and return a FastAPI application backed by a Memory instance.

    Pass a pre-built ``memory`` to skip lifespan initialisation (useful in
    tests and scripts).  Without it, the store is created on startup.

    Requires: ``pip install agent-memory-sdk[api]``
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI server requires fastapi and uvicorn. "
            "Install with: pip install agent-memory-sdk[api]"
        )

    resolved_api_key = (
        api_key if api_key is not None else os.environ.get("AGENT_MEMORY_API_KEY")
    )
    if resolved_api_key is not None and not resolved_api_key.strip():
        raise ValueError("api_key must not be empty")
    if resolved_api_key is None and not allow_unauthenticated:
        raise ValueError(
            "REST API authentication is required. Pass api_key, set "
            "AGENT_MEMORY_API_KEY, or explicitly set allow_unauthenticated=True "
            "for a trusted local development environment."
        )

    from agent_memory.manager import Memory

    # Pre-seed state when a ready-made Memory is provided so the app works
    # without a lifespan context (e.g. plain TestClient, scripts).
    _state: dict[str, Any] = {"memory": memory} if memory is not None else {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if _state.get("memory") is None:
            _state["memory"] = Memory(
                persist_dir=persist_dir,
                collection_name=collection_name,
                backend=backend,
                **memory_kwargs,
            )
        yield
        if memory is None:  # only clear if we owned the store
            _state.clear()

    app = FastAPI(
        title="Agent Memory SDK",
        description=(
            "REST API for the Agent Memory SDK — store, retrieve, and manage "
            "persistent AI agent memory with replay/restore/verify decisions."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    if resolved_api_key is not None:

        @app.middleware("http")
        async def require_api_key(request: Request, call_next: Any) -> Any:
            supplied = request.headers.get("x-api-key", "")
            authorization = request.headers.get("authorization", "")
            if not supplied and authorization.lower().startswith("bearer "):
                supplied = authorization[7:].strip()
            if not hmac.compare_digest(
                supplied.encode("utf-8"), resolved_api_key.encode("utf-8")
            ):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await call_next(request)

    def _mem() -> Memory:
        mem: Memory | None = _state.get("memory")  # type: ignore[assignment]
        if mem is None:  # pragma: no cover
            raise RuntimeError("Memory store not initialised")
        return mem

    # -----------------------------------------------------------------------
    # Dashboard
    # -----------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard() -> str:
        from agent_memory.api.dashboard import render_dashboard

        return render_dashboard(_mem().stats())

    # -----------------------------------------------------------------------
    # Memory CRUD
    # -----------------------------------------------------------------------

    @app.post("/memories", summary="Store a new memory")
    async def remember(req: RememberRequest) -> dict[str, Any]:
        entry = _mem().remember(
            req.query,
            req.response,
            content=req.content,
            type=req.type,
            scope=req.scope,
            metadata=req.metadata,
            tags=req.tags,
            confidence=req.confidence,
            requires_verification=req.requires_verification,
            ttl=req.ttl,
        )
        return entry.to_dict()

    @app.get("/memories", summary="List stored memories")
    async def list_memories(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        scope: list[str] | None = Query(None),
        include_archived: bool = False,
        type: str | None = None,
    ) -> dict[str, Any]:
        entries = _mem().list(
            limit=limit,
            offset=offset,
            scope=scope,
            include_archived=include_archived,
            type=type,
        )
        return {"entries": [e.to_dict() for e in entries], "count": len(entries)}

    @app.get("/memories/{memory_id}", summary="Get a memory by ID")
    async def get_memory(memory_id: str) -> dict[str, Any]:
        entry = _mem().get(memory_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Memory not found")
        return entry.to_dict()

    @app.delete("/memories/{memory_id}", summary="Delete a memory")
    async def forget_memory(memory_id: str) -> dict[str, bool]:
        if not _mem().forget(memory_id):
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"deleted": True}

    @app.post("/memories/{memory_id}/archive", summary="Archive a memory")
    async def archive_memory(memory_id: str) -> dict[str, Any]:
        entry = _mem().archive(memory_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Memory not found")
        return entry.to_dict()

    # -----------------------------------------------------------------------
    # Decision endpoint
    # -----------------------------------------------------------------------

    @app.post("/resolve", summary="Resolve a query against stored memories")
    async def resolve(req: ResolveRequest) -> dict[str, Any]:
        decision = _mem().resolve(
            req.query,
            mode=req.mode,
            top_k=req.top_k,
            scope=req.scope,
            enable_verify=req.enable_verify,
        )
        result: dict[str, Any] = {
            "action": decision.action.value,
            "confidence": decision.confidence,
            "reasons": decision.reasons,
            "scores": decision.scores,
        }
        if decision.action == MemoryAction.REPLAY and decision.memory:
            result["response"] = decision.memory.response
            result["memory_id"] = decision.memory.id
        elif decision.action in (MemoryAction.RESTORE, MemoryAction.VERIFY):
            result["context"] = [r.entry.to_dict() for r in decision.context]
        return result

    # -----------------------------------------------------------------------
    # Utility endpoints
    # -----------------------------------------------------------------------

    @app.get("/stats", summary="Aggregate memory statistics")
    async def stats() -> dict[str, Any]:
        return _mem().stats()

    @app.post("/cleanup", summary="Mark or delete expired memories")
    async def cleanup(delete: bool = False) -> dict[str, int]:
        return _mem().cleanup(delete=delete)

    @app.post("/consolidate", summary="Merge near-duplicate memories")
    async def consolidate(similarity_threshold: float = 0.95) -> dict[str, Any]:
        created = _mem().consolidate(similarity_threshold=similarity_threshold)
        return {
            "consolidated": len(created),
            "entries": [e.to_dict() for e in created],
        }

    return app


def main() -> None:
    """Entry point: ``agent-memory-api``."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "Serving requires uvicorn. Install with: pip install agent-memory-sdk[api]"
        ) from None

    app = create_app(
        persist_dir=os.environ.get("AGENT_MEMORY_DIR", ".agent_memory"),
        collection_name=os.environ.get(
            "AGENT_MEMORY_COLLECTION", "agent_memories"
        ),
        backend=os.environ.get("AGENT_MEMORY_BACKEND", "sqlite"),
        api_key=os.environ.get("AGENT_MEMORY_API_KEY"),
    )
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
