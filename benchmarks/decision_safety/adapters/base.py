"""Adapter contract for the Decision Safety Suite v2 runner.

One adapter per memory system. An adapter translates the battery ops
(`docs/decision-safety-suite.md`) into that system's own API and reports what it
cannot do, rather than letting a missing capability score as a pass.

Three rules the runner enforces on adapters:

1. A capability the system does not have is declared ``unsupported``. Raise
   :class:`CapabilityUnsupported` from the op instead of faking a result.
2. An op the adapter fulfils by working around a missing API is declared in
   :attr:`DecisionSafetyAdapter.emulated_ops`. Emulation is published with the
   result, because an emulated capability is a harness result, not a system
   result.
3. ``write`` reports whether the turn became a durable memory. That answer is
   the system's, never the harness's guess.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Capability keys and values mirror benchmarks/competitive/result.schema.json.
SUPPORT_VALUES = {"supported", "partial", "unsupported"}
RAW_STORE_VALUES = {"retained", "retained_configurable", "none"}


class CapabilityUnsupported(RuntimeError):
    """Raised by an adapter op the system under test cannot perform."""

    def __init__(self, capability: str, detail: str = "") -> None:
        self.capability = capability
        self.detail = detail
        super().__init__(f"{capability} unsupported" + (f": {detail}" if detail else ""))


@dataclass
class WriteOutcome:
    """Result of offering one conversation turn to the system.

    ``stored`` is the system's decision, not the harness's: ``False`` means the
    turn produced no durable memory.
    """

    stored: bool
    refs: list[str] = field(default_factory=list)
    stored_texts: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryOutcome:
    """Result of one recall or answer.

    ``abstained`` must be the system's own explicit "nothing to answer with"
    signal where it has one. Adapters for systems without explicit abstention
    declare ``explicit_abstention: unsupported`` and fall back to "no results".
    """

    abstained: bool
    texts: list[str] = field(default_factory=list)
    hits: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def haystack(self) -> str:
        return "\n".join(self.texts).lower()


class DecisionSafetyAdapter(ABC):
    """Base class for a system under test."""

    #: Adapter-visible system identity, copied into the result artifact.
    name: str = "unnamed"
    version: str = "unknown"
    deployment: str = "local"

    #: See docs/decision-safety-suite.md. Every key is declared explicitly;
    #: an omitted key is treated as "not declared" and fails validation.
    capabilities: dict[str, str] = {}

    #: Ops fulfilled by working around a missing API, e.g. a scope-wide delete
    #: implemented as an id loop. Published with the result.
    emulated_ops: set[str] = set()

    #: Free-text caveats carried into result.limitations.
    notes: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def setup(self, *, battery: str) -> None:
        """Create a clean store. Called once per battery, before its first op."""

    def teardown(self) -> None:
        """Release resources. Called even if the battery raised."""

    # -- ops ---------------------------------------------------------------

    @abstractmethod
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
        """Offer one conversation turn for ingestion.

        *ttl* is a lifetime in seconds. An adapter whose system has no TTL
        declares ``ttl: unsupported``; the ``ttl_expiry`` battery is then
        reported unsupported rather than scored.
        """

    @abstractmethod
    def delete(self, *, op_id: str, refs: list[str]) -> int:
        """Delete the memories derived from a single earlier write."""

    def delete_scope(self, *, scope: str) -> int:
        """Delete everything in one scope (`delete_all` / `forget` equivalent)."""
        raise CapabilityUnsupported("delete_scope")

    def advance_clock(self, *, seconds: int, now: datetime) -> None:
        """Move the system's notion of time to *now*."""
        raise CapabilityUnsupported("clock_control")

    @abstractmethod
    def query(self, *, scope: str, text: str, top_k: int) -> QueryOutcome:
        """Answer or recall against the live state."""

    def query_as_of(
        self, *, scope: str, text: str, as_of: datetime, top_k: int
    ) -> QueryOutcome:
        """Answer or recall as of a past timestamp."""
        raise CapabilityUnsupported("as_of_query")

    def inspect(self, *, scope: str) -> dict[str, Any]:
        """Report out-of-band stored state, e.g. retained raw turn count."""
        raise CapabilityUnsupported("inspect")

    # -- declaration -------------------------------------------------------

    def declared(self, capability: str) -> str:
        return self.capabilities.get(capability, "unsupported")

    def validate_declaration(self) -> list[str]:
        """Return problems with this adapter's capability declaration."""
        errors: list[str] = []
        required = {
            "delete_by_id",
            "delete_scope",
            "tombstones",
            "raw_message_store",
            "ttl",
            "explicit_supersession",
            "as_of_query",
            "explicit_abstention",
        }
        for key in sorted(required - set(self.capabilities)):
            errors.append(f"{self.name}: capability {key!r} is not declared")
        for key, value in self.capabilities.items():
            allowed = RAW_STORE_VALUES if key == "raw_message_store" else SUPPORT_VALUES
            if value not in allowed:
                errors.append(
                    f"{self.name}: capability {key!r}={value!r} must be one of {sorted(allowed)}"
                )
        return errors
