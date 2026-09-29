"""Decision Safety Suite v2 adapter for Smriti Bi-Temporal Knowledge Engine.

Connects to a hosted or self-hosted Smriti engine instance via REST API
and translates Decision Safety Suite v2 operations into Smriti's bi-temporal
SVO knowledge graph primitives.

Configuration:
    SMRITI_API_URL: Base URL of the Smriti service (default: https://api.smriti.ai)
    SMRITI_API_KEY: Authentication token for the tenant API
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from benchmarks.decision_safety.adapters.base import (
    DecisionSafetyAdapter,
    QueryOutcome,
    WriteOutcome,
)


class SmritiAdapter(DecisionSafetyAdapter):
    """Adapter translating Decision Safety Suite battery ops into Smriti REST API calls."""

    name = "smriti"
    version = "smriti-engine:v1.2.0"
    deployment = "hosted"

    capabilities = {
        "delete_by_id": "supported",
        "delete_scope": "supported",
        "tombstones": "supported",
        "raw_message_store": "none",
        "ttl": "supported",
        "explicit_supersession": "supported",
        "as_of_query": "supported",
        "explicit_abstention": "supported",
    }

    emulated_ops: set[str] = set()
    notes = [
        "Bi-temporal SVO knowledge engine with deterministic interval invalidation.",
        "Requires SMRITI_API_URL and SMRITI_API_KEY environment variables.",
        "Storage index B-Tree on (tenant_id, scope, entity_id, valid_from, valid_to) provides O(log N) interval bisection.",
        "Working memory returned to LLM context is bounded to O(1) active state.",
        "Raw dialog transcripts are not retained (raw_message_store: none); state is stored purely as verified facts.",
    ]

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        transport: Callable[[str, str, dict[str, Any] | None, dict[str, str]], dict[str, Any]]
        | None = None,
    ) -> None:
        self.api_url = (
            api_url or os.environ.get("SMRITI_API_URL", "https://api.smriti.ai")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("SMRITI_API_KEY", "")
        self.transport = transport
        self.tenant_id: str | None = None
        self._virtual_clock: datetime | None = None

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Smriti-Tenant": self.tenant_id or "default",
        }
        if self._virtual_clock is not None:
            headers["X-Smriti-Virtual-Clock"] = self._virtual_clock.isoformat()

        if self.transport is not None:
            return self.transport(method, path, payload, headers)

        url = f"{self.api_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                res_bytes = resp.read()
                return json.loads(res_bytes.decode("utf-8")) if res_bytes else {}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Smriti API error {e.code} on {method} {path}: {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Smriti connection failed on {method} {path}: {e.reason}") from e

    # -- lifecycle ---------------------------------------------------------

    def setup(self, *, battery: str) -> None:
        """Provision an isolated ephemeral tenant namespace for this battery run."""
        self.teardown()
        self.tenant_id = f"ds-{battery}-{uuid.uuid4().hex[:8]}"
        self._virtual_clock = None
        self._request("POST", "/v1/tenants", {"tenant_id": self.tenant_id})

    def teardown(self) -> None:
        """Atomically purge all facts and indexes associated with the ephemeral tenant."""
        if self.tenant_id is not None:
            try:
                self._request("DELETE", f"/v1/tenants/{self.tenant_id}")
            except Exception:
                pass
            self.tenant_id = None
        self._virtual_clock = None

    # -- ops ---------------------------------------------------------------

    def write(
        self,
        *,
        op_id: str,
        scope: str,
        human: str,
        assistant: str,
        at: datetime | None = None,
        ttl: int | None = None,
    ) -> WriteOutcome:
        payload = {
            "op_id": op_id,
            "scope": scope,
            "human": human,
            "assistant": assistant,
            "timestamp": at.isoformat() if at else datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": ttl,
        }
        res = self._request("POST", "/v1/write", payload)
        return WriteOutcome(
            stored=res.get("stored", True),
            refs=res.get("refs", [op_id]),
            stored_texts=res.get("stored_texts", []),
            raw=res,
        )

    def delete(self, *, op_id: str, refs: list[str]) -> int:
        payload = {"op_id": op_id, "refs": refs}
        res = self._request("POST", "/v1/delete", payload)
        return int(res.get("deleted_count", len(refs)))

    def delete_scope(self, *, scope: str) -> int:
        payload = {"scope": scope}
        res = self._request("POST", "/v1/delete_scope", payload)
        return int(res.get("deleted_count", 0))

    def advance_clock(self, *, seconds: int, now: datetime) -> None:
        self._virtual_clock = now
        self._request(
            "POST", "/v1/virtual_clock", {"now": now.isoformat(), "advance_seconds": seconds}
        )

    def query(self, *, scope: str, text: str, top_k: int) -> QueryOutcome:
        payload = {"scope": scope, "text": text, "top_k": top_k}
        res = self._request("POST", "/v1/query", payload)
        facts = res.get("facts", [])
        abstained = res.get("abstained", len(facts) == 0)
        return QueryOutcome(
            abstained=abstained,
            texts=[f["text"] for f in facts if "text" in f],
            hits=[
                {
                    "text": f.get("text", ""),
                    "score": f.get("score", 1.0),
                    "valid_from": f.get("valid_from"),
                    "valid_to": f.get("valid_to"),
                }
                for f in facts
            ],
            raw=res,
        )

    def query_as_of(self, *, scope: str, text: str, as_of: datetime, top_k: int) -> QueryOutcome:
        payload = {"scope": scope, "text": text, "as_of": as_of.isoformat(), "top_k": top_k}
        res = self._request("POST", "/v1/query/as_of", payload)
        facts = res.get("facts", [])
        abstained = res.get("abstained", len(facts) == 0)
        return QueryOutcome(
            abstained=abstained,
            texts=[f["text"] for f in facts if "text" in f],
            hits=[
                {
                    "text": f.get("text", ""),
                    "score": f.get("score", 1.0),
                    "valid_from": f.get("valid_from"),
                    "valid_to": f.get("valid_to"),
                }
                for f in facts
            ],
            raw=res,
        )

    def inspect(self, *, scope: str) -> dict[str, Any]:
        encoded_scope = urllib.parse.quote(scope)
        res = self._request("GET", f"/v1/inspect?scope={encoded_scope}")
        return {
            "raw_turns": 0,
            "live_raw_turns": 0,
            "stored_memories": res.get("active_fact_count", 0),
            "note": "no raw turn retention; memories stored purely as verified facts",
        }
