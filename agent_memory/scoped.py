"""Per-user and per-session views over one shared :class:`~agent_memory.manager.Memory`.

``MemoryEntry.scope`` is a *tier* (``session``/``user``/``project``/...), not a
tenant identifier, so it cannot keep 100 users apart. A :class:`MemoryView`
supplies the missing identity: it stamps ``user_id``/``session_id`` on every
write and applies the matching visibility filter on every read, before scoring.

Visibility is hierarchical, narrow to wide:

* ``scoped(user_id="alice", session_id="s3")`` sees that session, plus alice's
  session-less memories, plus the shared tier.
* ``scoped(user_id="alice")`` sees every one of alice's sessions, plus shared.
* ``scoped(shared=True)`` sees only the shared tier.

Writes never widen: a session view writes into that session, and only a
``shared=True`` view writes shared memories.

Memories stored without a ``user_id`` (for example, written directly through the
underlying ``Memory``) are invisible to a user view unless it is created with
``include_unscoped=True``. Defaulting the other way would leak one process-wide
store into every tenant's view.

Usage::

    memory = Memory(persist_dir=".agent_memory")

    alice_s3 = memory.scoped(user_id="alice", session_id="s3")
    alice_s3.remember("Which seat do I prefer?", "Window seat")

    alice = memory.scoped(user_id="alice")
    alice.resolve("seat preference?")       # sees the s3 write
    memory.scoped(user_id="bob").resolve("seat preference?")  # NONE

    memory.scoped(shared=True).remember("What is the refund window?", "30 days")

    alice.forget_all()                      # delete everything for one user
"""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryDecision, MemoryEntry

if TYPE_CHECKING:
    from agent_memory.manager import Memory

log = get_logger(__name__)

USER_KEY = "user_id"
SESSION_KEY = "session_id"
SHARED_KEY = "shared"


@dataclass(frozen=True)
class MemoryView:
    """A tenant-scoped or session-scoped view over a shared store.

    Build one with :meth:`agent_memory.manager.Memory.scoped`.
    """

    memory: Memory
    user_id: str | None = None
    session_id: str | None = None
    shared: bool = False
    include_unscoped: bool = False

    def __post_init__(self) -> None:
        if self.session_id is not None and self.user_id is None:
            raise ValueError(
                "session_id requires a user_id: a session belongs to a user. "
                "Use Memory.scoped(user_id=..., session_id=...)."
            )
        if self.user_id is None and not self.shared and not self.include_unscoped:
            raise ValueError(
                "Memory.scoped() needs user_id, shared=True, or include_unscoped=True."
            )

    # -- identity ----------------------------------------------------------

    @property
    def write_metadata(self) -> dict[str, Any]:
        """Identity stamped onto everything this view writes."""
        if self.shared:
            return {SHARED_KEY: True}
        metadata: dict[str, Any] = {USER_KEY: self.user_id}
        if self.session_id is not None:
            metadata[SESSION_KEY] = self.session_id
        return metadata

    def _stamp_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        stamped = {
            key: value
            for key, value in (metadata or {}).items()
            if key not in {USER_KEY, SESSION_KEY, SHARED_KEY}
        }
        stamped.update(self.write_metadata)
        return stamped

    def visible(self, entry: MemoryEntry) -> bool:
        """True if *entry* may be read through this view."""
        metadata = entry.metadata or {}
        if metadata.get(SHARED_KEY):
            return True

        entry_user = metadata.get(USER_KEY)
        if entry_user is None:
            # Never written through a view. Only visible on explicit opt-in.
            return self.include_unscoped
        if self.shared:
            return False
        if entry_user != self.user_id:
            return False
        if self.session_id is None:
            return True  # user-wide view: every session of this user
        entry_session = metadata.get(SESSION_KEY)
        return entry_session is None or entry_session == self.session_id

    def _where(
        self, extra: Callable[[MemoryEntry], bool] | None = None
    ) -> Callable[[MemoryEntry], bool]:
        if extra is None:
            return self.visible
        return lambda entry: self.visible(entry) and extra(entry)

    # -- writes ------------------------------------------------------------

    def remember(self, query: str, response: str, **kwargs: Any) -> MemoryEntry:
        """Store a memory owned by this view."""
        metadata = self._stamp_metadata(kwargs.pop("metadata", None))
        return self.memory.remember(query, response, metadata=metadata, **kwargs)

    async def aremember(self, query: str, response: str, **kwargs: Any) -> MemoryEntry:
        """Async version of remember()."""
        return await asyncio.to_thread(self.remember, query, response, **kwargs)

    def from_conversation(
        self, human: str, assistant: str, **kwargs: Any
    ) -> builtins.list[MemoryEntry]:
        """Extract memories from one turn, stamped with this view's identity."""
        metadata = self._stamp_metadata(kwargs.pop("metadata", None))
        return self.memory.from_conversation(human, assistant, metadata=metadata, **kwargs)

    # -- reads -------------------------------------------------------------

    def resolve(self, query: str, **kwargs: Any) -> MemoryDecision:
        """Resolve *query* against only what this view can see.

        The filter is applied during retrieval, before scoring: a decision
        computed over another tenant's memories would leak it through
        ``decision.response`` even if the entry were stripped afterwards.
        """
        kwargs["where"] = self._where(kwargs.get("where"))
        return self.memory.resolve(query, **kwargs)

    async def aresolve(self, query: str, **kwargs: Any) -> MemoryDecision:
        """Async version of resolve()."""
        return await asyncio.to_thread(self.resolve, query, **kwargs)

    def list(
        self, limit: int = 100, offset: int = 0, **kwargs: Any
    ) -> builtins.list[MemoryEntry]:
        """List memories visible to this view, newest-first as the store returns."""
        entries = self.memory.list(limit=10_000, **kwargs)
        visible = [entry for entry in entries if self.visible(entry)]
        return visible[offset : offset + limit]

    def get(self, memory_id: str) -> MemoryEntry | None:
        entry = self.memory.get(memory_id)
        if entry is None or not self.visible(entry):
            return None
        return entry

    # -- deletes -----------------------------------------------------------

    def forget(self, memory_id: str) -> bool:
        """Delete one memory only if this view owns it."""
        entry = self.get(memory_id)
        if entry is None or not self.owns(entry):
            return False
        return self.memory.forget(memory_id)

    def forget_all(self) -> int:
        """Delete everything this view *owns* — not everything it can read.

        A session view deletes that session. A user view deletes every session
        of that user. Neither touches the shared tier or another user, and a
        shared view deletes only shared memories.
        """
        deleted = self.memory.forget_where(where=self.owns)
        log.debug(
            "forget_all  user=%s session=%s shared=%s deleted=%d",
            self.user_id, self.session_id, self.shared, deleted,
        )
        return deleted

    async def aforget_all(self) -> int:
        """Async version of forget_all()."""
        return await asyncio.to_thread(self.forget_all)

    def owns(self, entry: MemoryEntry) -> bool:
        """True if *entry* was written through this view's identity."""
        metadata = entry.metadata or {}
        if self.shared:
            return bool(metadata.get(SHARED_KEY))
        if metadata.get(SHARED_KEY) or metadata.get(USER_KEY) != self.user_id:
            return False
        if self.session_id is None:
            return True
        return metadata.get(SESSION_KEY) == self.session_id

    # -- navigation --------------------------------------------------------

    def session(self, session_id: str) -> MemoryView:
        """Narrow a user view to one of its sessions."""
        if self.shared or self.user_id is None:
            raise ValueError("session() needs a user-scoped view")
        return MemoryView(
            memory=self.memory,
            user_id=self.user_id,
            session_id=session_id,
            include_unscoped=self.include_unscoped,
        )

    def sessions(self) -> builtins.list[str]:
        """Session ids this view can see, sorted."""
        return sorted(
            {
                str(entry.metadata[SESSION_KEY])
                for entry in self.list(limit=10_000)
                if entry.metadata.get(SESSION_KEY) is not None
            }
        )
