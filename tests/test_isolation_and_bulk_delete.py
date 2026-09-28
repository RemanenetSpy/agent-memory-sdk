"""Read-path isolation, expiry, and bulk deletion.

Every test here is a regression test for a read path that disagreed with
``list()``: ``list()`` hid an entry while ``resolve()`` served it. These were
found by the Decision Safety Suite v2 work, and the shape of the bug is the
same each time — the filter lived in the listing path, not in the path the
decision layer uses.
"""

from __future__ import annotations

import time

import pytest

from agent_memory.manager import Memory
from agent_memory.models import MemoryAction, MemoryScope
from agent_memory.multiagent import IsolationMode, MultiAgentMemory


@pytest.fixture
def memory(tmp_path) -> Memory:
    return Memory(persist_dir=tmp_path / "store", enable_embeddings=False)


# --- cross-agent isolation on the decision path ------------------------------


def test_isolated_agent_cannot_resolve_another_agents_memory(memory: Memory) -> None:
    alice = MultiAgentMemory(memory, agent_id="alice", isolation=IsolationMode.ISOLATED)
    bob = MultiAgentMemory(memory, agent_id="bob", isolation=IsolationMode.ISOLATED)
    alice.remember("What is my home address?", "44 Brunswick Road, Leeds")

    decision = bob.resolve("What is my home address?")

    assert decision.action == MemoryAction.NONE
    assert decision.response is None
    assert decision.context == []
    assert bob.list() == []


def test_namespaced_agent_sees_global_but_not_private_memories(memory: Memory) -> None:
    alice = MultiAgentMemory(memory, agent_id="alice", isolation=IsolationMode.NAMESPACED)
    bob = MultiAgentMemory(memory, agent_id="bob", isolation=IsolationMode.NAMESPACED)
    alice.remember("What is my salary?", "£95,000")
    alice.broadcast("What is the company name?", "Acme Inc.", type="fact")

    private = bob.resolve("What is my salary?")
    shared = bob.resolve("What is the company name?")

    assert private.action == MemoryAction.NONE
    assert "95,000" not in (private.response or "")
    assert shared.action != MemoryAction.NONE
    assert "Acme" in (shared.response or "") or any(
        "Acme" in r.entry.response for r in shared.context
    )


def test_shared_mode_still_sees_everything(memory: Memory) -> None:
    alice = MultiAgentMemory(memory, agent_id="alice", isolation=IsolationMode.SHARED)
    bob = MultiAgentMemory(memory, agent_id="bob", isolation=IsolationMode.SHARED)
    alice.remember("What is the deploy target?", "eu-central-1")

    assert bob.resolve("What is the deploy target?").action != MemoryAction.NONE


def test_isolation_survives_a_prior_query_by_the_owning_agent(memory: Memory) -> None:
    """The retriever caches by query text; a filtered result must not be reused."""
    alice = MultiAgentMemory(memory, agent_id="alice", isolation=IsolationMode.ISOLATED)
    bob = MultiAgentMemory(memory, agent_id="bob", isolation=IsolationMode.ISOLATED)
    bob.remember("Which card is on file?", "card 4471")

    assert bob.resolve("Which card is on file?").action != MemoryAction.NONE
    assert alice.resolve("Which card is on file?").action == MemoryAction.NONE
    # ...and the owner is not locked out by the other agent's empty result.
    assert bob.resolve("Which card is on file?").action != MemoryAction.NONE


def test_unfiltered_resolve_is_not_served_a_filtered_cached_result(memory: Memory) -> None:
    alice = MultiAgentMemory(memory, agent_id="alice", isolation=IsolationMode.ISOLATED)
    memory.remember("What is the rate limit?", "1000 requests/minute")

    assert alice.resolve("What is the rate limit?").action == MemoryAction.NONE
    assert memory.resolve("What is the rate limit?").action != MemoryAction.NONE


def test_where_filter_is_applied_before_scoring(memory: Memory) -> None:
    memory.remember("What is the rate limit?", "1000/min", metadata={"tenant": "acme"})
    memory.remember("What is the rate limit?", "50/min", metadata={"tenant": "globex"})

    decision = memory.resolve(
        "What is the rate limit?",
        where=lambda entry: entry.metadata.get("tenant") == "globex",
    )

    surfaced = (decision.response or "") + " ".join(r.entry.response for r in decision.context)
    assert "50/min" in surfaced
    assert "1000/min" not in surfaced


# --- expiry on the decision path --------------------------------------------


def test_expired_memory_is_not_replayed_on_a_repeated_query(memory: Memory) -> None:
    """A TTL shorter than the retriever's cache TTL used to survive expiry."""
    memory.remember("What is the temp API key?", "sk-temp-123", ttl=1)
    assert memory.resolve("What is the temp API key?").action != MemoryAction.NONE

    time.sleep(1.1)
    decision = memory.resolve("What is the temp API key?")

    assert decision.action == MemoryAction.NONE
    assert "sk-temp-123" not in (decision.response or "")
    assert all("sk-temp-123" not in r.entry.response for r in decision.context)


def test_live_memories_are_still_cached(memory: Memory) -> None:
    memory.remember("What is the API rate limit?", "1000 requests/minute")

    first = memory.resolve("What is the API rate limit?")
    second = memory.resolve("What is the API rate limit?")

    assert first.action == second.action != MemoryAction.NONE


# --- bulk deletion -----------------------------------------------------------


def test_forget_where_deletes_by_metadata(memory: Memory) -> None:
    memory.remember("What is my address?", "44 Brunswick Road", metadata={"user_id": "alice"})
    memory.remember("What is my address?", "8 Grafton Street", metadata={"user_id": "bob"})

    deleted = memory.forget_where(metadata={"user_id": "alice"})

    assert deleted == 1
    assert memory.resolve("What is my address?").action != MemoryAction.NONE
    remaining = memory.list()
    assert [entry.metadata["user_id"] for entry in remaining] == ["bob"]


def test_forget_where_deletes_by_scope_and_tags(memory: Memory) -> None:
    memory.remember("Draft note", "temporary", scope=MemoryScope.SESSION, tags=["scratch"])
    memory.remember("Team policy", "durable", scope=MemoryScope.TEAM, tags=["policy"])

    assert memory.forget_where(scope=["session"]) == 1
    assert memory.forget_where(tags=["policy"]) == 1
    assert memory.list() == []


def test_forget_where_removes_archived_and_expired_copies(memory: Memory) -> None:
    keep = memory.remember("Keep me", "yes", metadata={"user_id": "bob"})
    archived = memory.remember("Archive me", "no", metadata={"user_id": "alice"})
    memory.archive(archived.id)
    memory.remember("Expire me", "no", metadata={"user_id": "alice"}, ttl=1)
    time.sleep(1.1)

    deleted = memory.forget_where(metadata={"user_id": "alice"})

    assert deleted == 2
    assert [entry.id for entry in memory.list(include_archived=True)] == [keep.id]


def test_forget_where_requires_an_explicit_filter(memory: Memory) -> None:
    memory.remember("Keep me", "yes")

    with pytest.raises(ValueError, match="at least one of"):
        memory.forget_where()

    assert len(memory.list()) == 1


def test_forget_where_all_wipes_the_store(memory: Memory) -> None:
    memory.remember("One", "1")
    memory.remember("Two", "2")

    assert memory.forget_where(all=True) == 2
    assert memory.list() == []
    assert memory.resolve("One").action == MemoryAction.NONE


def test_agent_forget_all_leaves_other_agents_intact(memory: Memory) -> None:
    alice = MultiAgentMemory(memory, agent_id="alice", isolation=IsolationMode.NAMESPACED)
    bob = MultiAgentMemory(memory, agent_id="bob", isolation=IsolationMode.NAMESPACED)
    alice.remember("What is my address?", "44 Brunswick Road")
    alice.broadcast("What is the company name?", "Acme Inc.", type="fact")
    bob.remember("What is my address?", "8 Grafton Street")

    deleted = alice.forget_all()

    assert deleted == 2  # its private memory and the global one it authored
    assert alice.resolve("What is my address?").action == MemoryAction.NONE
    assert bob.resolve("What is my address?").action != MemoryAction.NONE
