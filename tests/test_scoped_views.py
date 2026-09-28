"""Per-user and per-session views: `Memory.scoped()`.

The store keeps many tenants, so the question every test here asks is the one
that matters for isolation: does the *decision path* see what `list()` sees, and
nothing more?
"""

from __future__ import annotations

import pytest

from agent_memory.manager import Memory
from agent_memory.models import MemoryAction
from agent_memory.scoped import MemoryView


@pytest.fixture
def memory(tmp_path) -> Memory:
    return Memory(persist_dir=tmp_path / "store", enable_embeddings=False)


@pytest.fixture
def populated(memory: Memory) -> Memory:
    memory.scoped(user_id="alice", session_id="s1").remember(
        "Which seat do I prefer?", "Window seat"
    )
    memory.scoped(user_id="alice", session_id="s2").remember(
        "What is my hotel?", "The Queens, Leeds"
    )
    memory.scoped(user_id="alice").remember("What is my home city?", "Leeds")
    memory.scoped(user_id="bob").remember("Which seat do I prefer?", "Aisle seat")
    memory.scoped(shared=True).remember("What is the refund window?", "30 days")
    return memory


def _surfaced(decision) -> str:
    return " ".join(
        [decision.response or "", *(result.entry.response for result in decision.context)]
    )


# --- construction ------------------------------------------------------------


def test_session_without_user_is_rejected(memory: Memory) -> None:
    with pytest.raises(ValueError, match="session_id requires a user_id"):
        memory.scoped(session_id="s1")


def test_empty_view_is_rejected(memory: Memory) -> None:
    with pytest.raises(ValueError, match="needs user_id"):
        memory.scoped()


def test_scoped_returns_a_view(memory: Memory) -> None:
    assert isinstance(memory.scoped(user_id="alice"), MemoryView)


# --- isolation between users -------------------------------------------------


def test_one_user_cannot_resolve_anothers_memory(populated: Memory) -> None:
    bob = populated.scoped(user_id="bob")

    assert "Aisle" in _surfaced(bob.resolve("Which seat do I prefer?"))
    assert "Window" not in _surfaced(bob.resolve("Which seat do I prefer?"))
    assert bob.resolve("What is my hotel?").action == MemoryAction.NONE


def test_many_users_stay_separate_in_one_store(memory: Memory) -> None:
    for index in range(100):
        memory.scoped(user_id=f"user{index}").remember(
            "What is my account number?", f"ACCT-{index:04d}"
        )

    for index in (0, 42, 99):
        surfaced = _surfaced(
            memory.scoped(user_id=f"user{index}").resolve("What is my account number?")
        )
        assert f"ACCT-{index:04d}" in surfaced
        others = [f"ACCT-{other:04d}" for other in range(100) if other != index]
        assert not any(account in surfaced for account in others)

    assert len(memory.list(limit=200)) == 100


# --- hierarchy ---------------------------------------------------------------


def test_session_view_sees_its_session_and_the_user_wide_memory(populated: Memory) -> None:
    s1 = populated.scoped(user_id="alice", session_id="s1")

    assert "Window" in _surfaced(s1.resolve("Which seat do I prefer?"))
    assert "Leeds" in _surfaced(s1.resolve("What is my home city?"))


def test_session_view_does_not_see_a_sibling_session(populated: Memory) -> None:
    s1 = populated.scoped(user_id="alice", session_id="s1")

    assert s1.resolve("What is my hotel?").action == MemoryAction.NONE
    assert "Queens" not in _surfaced(s1.resolve("What is my hotel?"))


def test_user_view_sees_every_session(populated: Memory) -> None:
    alice = populated.scoped(user_id="alice")

    assert "Window" in _surfaced(alice.resolve("Which seat do I prefer?"))
    assert "Queens" in _surfaced(alice.resolve("What is my hotel?"))
    assert alice.sessions() == ["s1", "s2"]


def test_shared_tier_is_readable_by_every_view(populated: Memory) -> None:
    for view in (
        populated.scoped(user_id="alice", session_id="s1"),
        populated.scoped(user_id="bob"),
        populated.scoped(shared=True),
    ):
        assert "30 days" in _surfaced(view.resolve("What is the refund window?"))


def test_shared_view_sees_only_shared_memories(populated: Memory) -> None:
    shared = populated.scoped(shared=True)

    assert shared.resolve("Which seat do I prefer?").action == MemoryAction.NONE
    assert [entry.response for entry in shared.list()] == ["30 days"]


def test_session_helper_narrows_a_user_view(populated: Memory) -> None:
    s2 = populated.scoped(user_id="alice").session("s2")

    assert "Queens" in _surfaced(s2.resolve("What is my hotel?"))
    assert s2.resolve("Which seat do I prefer?").action == MemoryAction.NONE


# --- unscoped memories ------------------------------------------------------


def test_unscoped_memories_are_hidden_from_user_views(memory: Memory) -> None:
    memory.remember("What is the API rate limit?", "1000 requests/minute")

    assert memory.scoped(user_id="alice").resolve("What is the API rate limit?").action == (
        MemoryAction.NONE
    )


def test_unscoped_memories_are_visible_on_opt_in(memory: Memory) -> None:
    memory.remember("What is the API rate limit?", "1000 requests/minute")

    view = memory.scoped(user_id="alice", include_unscoped=True)

    assert "1000" in _surfaced(view.resolve("What is the API rate limit?"))


# --- writes ------------------------------------------------------------------


def test_writes_carry_the_view_identity(memory: Memory) -> None:
    entry = memory.scoped(user_id="alice", session_id="s1").remember("Q", "A")

    assert entry.metadata["user_id"] == "alice"
    assert entry.metadata["session_id"] == "s1"
    assert "shared" not in entry.metadata


def test_shared_writes_are_not_owned_by_a_user(memory: Memory) -> None:
    entry = memory.scoped(shared=True).remember("Q", "A")

    assert entry.metadata == {"shared": True}


def test_caller_metadata_is_preserved(memory: Memory) -> None:
    entry = memory.scoped(user_id="alice").remember("Q", "A", metadata={"source": "crm"})

    assert entry.metadata["source"] == "crm"
    assert entry.metadata["user_id"] == "alice"


def test_user_view_cannot_promote_write_to_shared(memory: Memory) -> None:
    entry = memory.scoped(user_id="alice").remember(
        "Private question", "Private answer", metadata={"shared": True}
    )

    assert entry.metadata == {"user_id": "alice"}
    assert memory.scoped(user_id="bob").resolve("Private question").action == MemoryAction.NONE
    assert memory.scoped(shared=True).resolve("Private question").action == MemoryAction.NONE


def test_extracted_memories_carry_the_view_identity(memory: Memory) -> None:
    entries = memory.scoped(user_id="alice", session_id="s1").from_conversation(
        "My home address is 44 Brunswick Road, Leeds.", "Saved your home address."
    )

    assert entries
    assert all(entry.metadata["user_id"] == "alice" for entry in entries)
    assert all(entry.metadata["session_id"] == "s1" for entry in entries)
    assert memory.scoped(user_id="bob").resolve("What is my home address?").action == (
        MemoryAction.NONE
    )


# --- deletion ---------------------------------------------------------------


def test_user_delete_removes_every_session_and_nothing_else(populated: Memory) -> None:
    alice = populated.scoped(user_id="alice")

    deleted = alice.forget_all()

    assert deleted == 3
    # The view still reads the shared tier; what it *owns* is gone.
    assert [entry for entry in alice.list() if alice.owns(entry)] == []
    assert "Aisle" in _surfaced(
        populated.scoped(user_id="bob").resolve("Which seat do I prefer?")
    )
    assert "30 days" in _surfaced(
        populated.scoped(shared=True).resolve("What is the refund window?")
    )


def test_session_delete_leaves_the_rest_of_the_user(populated: Memory) -> None:
    deleted = populated.scoped(user_id="alice", session_id="s1").forget_all()
    alice = populated.scoped(user_id="alice")

    assert deleted == 1
    assert alice.resolve("Which seat do I prefer?").action == MemoryAction.NONE
    assert "Queens" in _surfaced(alice.resolve("What is my hotel?"))
    assert "Leeds" in _surfaced(alice.resolve("What is my home city?"))


def test_deleting_a_user_does_not_touch_the_shared_tier(populated: Memory) -> None:
    populated.scoped(user_id="alice").forget_all()
    populated.scoped(user_id="bob").forget_all()

    assert [entry.response for entry in populated.scoped(shared=True).list()] == ["30 days"]


def test_shared_delete_removes_only_shared_memories(populated: Memory) -> None:
    deleted = populated.scoped(shared=True).forget_all()

    assert deleted == 1
    assert "30 days" not in _surfaced(
        populated.scoped(user_id="alice").resolve("What is the refund window?")
    )
    assert "Window" in _surfaced(
        populated.scoped(user_id="alice").resolve("Which seat do I prefer?")
    )


def test_a_view_cannot_delete_an_invisible_memory(populated: Memory) -> None:
    bob_entry = next(
        entry for entry in populated.list(limit=100) if entry.metadata.get("user_id") == "bob"
    )

    alice = populated.scoped(user_id="alice")

    assert alice.get(bob_entry.id) is None
    assert alice.forget(bob_entry.id) is False
    assert populated.get(bob_entry.id) is not None


def test_user_view_cannot_delete_shared_memory(populated: Memory) -> None:
    shared_entry = next(
        entry
        for entry in populated.list(limit=100)
        if entry.metadata.get("shared")
    )

    assert populated.scoped(user_id="alice").get(shared_entry.id) is not None
    assert populated.scoped(user_id="alice").forget(shared_entry.id) is False
    assert populated.scoped(shared=True).get(shared_entry.id) is not None


def test_a_session_view_cannot_delete_a_sibling_session_memory(populated: Memory) -> None:
    s2_entry = next(
        entry for entry in populated.list(limit=100) if entry.metadata.get("session_id") == "s2"
    )

    assert populated.scoped(user_id="alice", session_id="s1").forget(s2_entry.id) is False
    assert populated.get(s2_entry.id) is not None
