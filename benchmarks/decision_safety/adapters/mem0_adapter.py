"""Decision Safety Suite v2 adapter for self-hosted Mem0 — pending maintainer review.

Deliberately not runnable yet. The capability declaration and the config below
are **our reading** of the OSS API, not Mem0's, and the whole point of
[mem0ai/mem0#7453](https://github.com/mem0ai/mem0/issues/7453) is to have the
maintainers confirm or correct it before any comparative number exists. Setting
``PENDING_MAINTAINER_REVIEW = False`` with unverified values would produce
exactly the marketing-driven comparison the RFC exists to prevent.

What we need confirmed before this runs:

1. The canonical self-hosted configuration: retrieval unit, vector store,
   embedding model, and LLM used at ingest and at query time.
2. Whether ``delete_all`` is the right ``delete_scope`` mapping, and what the
   expected behaviour is for raw session messages after it
   ([#7452](https://github.com/mem0ai/mem0/issues/7452)) — the
   ``deletion_durability`` battery inspects retained raw turns directly.
3. Whether any point-in-time read path exists, or whether ``point_in_time``
   should be declared ``unsupported`` (as it is for our own SDK).
4. Whether ``user_id`` / ``agent_id`` / ``run_id`` is the intended mapping for
   battery scopes, and which of them isolates retrieval rather than only listing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from benchmarks.decision_safety.adapters.base import (
    DecisionSafetyAdapter,
    QueryOutcome,
    WriteOutcome,
)

PENDING_MAINTAINER_REVIEW = True

#: Draft only. Every value here is a question for the maintainers, not a claim.
DRAFT_CONFIG: dict[str, Any] = {
    "deployment": "self_hosted",
    "retrieval_unit": "fact",
    "vector_store": "TBD",
    "embedding_model": "TBD",
    "llm_at_ingest": "TBD",
    "llm_at_query": "TBD",
    "scope_mapping": "user_id (to be confirmed; agent_id/run_id may be more appropriate)",
    "delete_scope_mapping": "delete_all(user_id=...) (to be confirmed)",
}


class PendingMaintainerReview(RuntimeError):
    """Raised instead of producing an unreviewed comparative result."""


class Mem0Adapter(DecisionSafetyAdapter):
    name = "mem0"
    version = "unpinned"
    deployment = "self_hosted"

    #: Placeholders. Not to be published until confirmed on #7453.
    capabilities = {
        "delete_by_id": "supported",
        "delete_scope": "supported",
        "tombstones": "unsupported",
        "raw_message_store": "retained",
        "ttl": "unsupported",
        "explicit_supersession": "partial",
        "as_of_query": "unsupported",
        "explicit_abstention": "unsupported",
    }

    notes = [
        "Capability declaration is unreviewed; see mem0ai/mem0#7453.",
        "No comparative Mem0 result is publishable from this adapter until the "
        "maintainers confirm the self-hosted configuration.",
    ]

    def __init__(self) -> None:
        if PENDING_MAINTAINER_REVIEW:
            raise PendingMaintainerReview(
                "The Mem0 adapter is a stub awaiting maintainer confirmation of the "
                "canonical self-hosted configuration (mem0ai/mem0#7453). Pin a version, "
                "fill DRAFT_CONFIG, implement the ops, and set "
                "PENDING_MAINTAINER_REVIEW = False in the same commit."
            )

    def write(
        self,
        *,
        op_id: str,
        scope: str,
        human: str,
        assistant: str,
        at: datetime | None = None,
    ) -> WriteOutcome:
        raise NotImplementedError("awaiting confirmed Mem0 configuration")

    def delete(self, *, op_id: str, refs: list[str]) -> int:
        raise NotImplementedError("awaiting confirmed Mem0 configuration")

    def query(self, *, scope: str, text: str, top_k: int) -> QueryOutcome:
        raise NotImplementedError("awaiting confirmed Mem0 configuration")
