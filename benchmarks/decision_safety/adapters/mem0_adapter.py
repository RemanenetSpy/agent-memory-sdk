"""Decision Safety Suite v2 adapter for self-hosted Mem0 — pending maintainer review.

Deliberately not runnable, and deliberately declares **no capabilities**.

An earlier draft of this file guessed at Mem0's capability declaration. Checking
those guesses against the source showed at least two were wrong (see
`OBSERVED` below), which is exactly the failure mode
[mem0ai/mem0#7453](https://github.com/mem0ai/mem0/issues/7453) exists to prevent.
So this adapter declares nothing: ``capabilities`` stays empty, the runner's
``validate_declaration()`` refuses it, and no Mem0 number can be produced from
this repository until the maintainers confirm the configuration.

``OBSERVED`` records only things read directly out of Mem0's own source or an
upstream issue, with the location, so a maintainer can correct a specific line
rather than a vibe. They are observations about one revision, not verdicts about
the project, and they are **not** a capability declaration — deciding how each
one maps onto a battery capability is the maintainers' call.
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

#: Read from mem0ai/mem0 on 2026-09-27. Each entry cites where it came from.
OBSERVED: dict[str, dict[str, str]] = {
    "scope_identity": {
        "observation": "delete_all(user_id=None, agent_id=None, run_id=None) — three "
        "independent scope ids, any of which can key a scope-wide delete.",
        "source": "mem0/memory/main.py, def delete_all",
        "question": "Which of the three should a battery scope map to, and which of "
        "them isolates retrieval rather than only listing?",
    },
    "expiration": {
        "observation": "add(..., expiration_date=...) accepts YYYY-MM-DD and expired "
        "memories are hidden, so there is day-granularity expiry.",
        "source": "mem0/memory/main.py, _normalize_expiration_date and add()",
        "question": "The ttl_expiry battery advances a virtual clock by seconds. Is "
        "day-granularity expiry testable here, or should the battery declare a "
        "coarser clock for Mem0?",
    },
    "change_log": {
        "observation": "history(memory_id) exists and add_history() records ADD / "
        "UPDATE / DELETE events per memory.",
        "source": "mem0/memory/main.py, def history and add_history call sites",
        "question": "Does that change log make a deleted memory unreachable from "
        "retrieval (a tombstone), or is it an audit trail only?",
    },
    "raw_messages_after_delete_all": {
        "observation": "delete_all() does not clear the scope's rows in the history "
        "DB's messages table, and those messages are re-sent to the LLM on the next "
        "add(), where they can be re-extracted. Reported with a stubbed-LLM repro.",
        "source": "upstream issue mem0ai/mem0#7452 (open), reported against main @ a39a802b",
        "question": "Is this the expected behaviour pending a fix? The "
        "deletion_durability battery inspects retained raw turns directly, so this "
        "is the difference between a reported capability and a reported defect.",
    },
}

#: Every value here is a question for the maintainers, not a claim.
DRAFT_CONFIG: dict[str, Any] = {
    "deployment": "self_hosted",
    "version": "TBD — pin a release tag or commit SHA",
    "retrieval_unit": "TBD",
    "vector_store": "TBD",
    "embedding_model": "TBD",
    "llm_at_ingest": "TBD",
    "llm_at_query": "TBD",
}


class PendingMaintainerReview(RuntimeError):
    """Raised instead of producing an unreviewed comparative result."""


class Mem0Adapter(DecisionSafetyAdapter):
    name = "mem0"
    version = "unpinned"
    deployment = "self_hosted"

    #: Intentionally empty. See the module docstring: we do not publish a
    #: capability declaration for someone else's project.
    capabilities: dict[str, str] = {}

    notes = [
        "No capability declaration: awaiting maintainer confirmation on mem0ai/mem0#7453.",
        "No comparative Mem0 result is publishable from this adapter.",
    ]

    def __init__(self) -> None:
        if PENDING_MAINTAINER_REVIEW:
            raise PendingMaintainerReview(
                "The Mem0 adapter is a stub awaiting maintainer confirmation of the "
                "canonical self-hosted configuration (mem0ai/mem0#7453). To finish it: "
                "pin a version, fill DRAFT_CONFIG, resolve each question in OBSERVED "
                "into a declared capability, implement the ops, and set "
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
        ttl: int | None = None,
    ) -> WriteOutcome:
        raise NotImplementedError("awaiting confirmed Mem0 configuration")

    def delete(self, *, op_id: str, refs: list[str]) -> int:
        raise NotImplementedError("awaiting confirmed Mem0 configuration")

    def query(self, *, scope: str, text: str, top_k: int) -> QueryOutcome:
        raise NotImplementedError("awaiting confirmed Mem0 configuration")
