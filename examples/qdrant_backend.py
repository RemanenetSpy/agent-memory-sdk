"""Qdrant backend — agent-memory-sdk on a purpose-built vector database.

Install:
    pip install agent-memory-sdk[qdrant,semantic]

Requires a running Qdrant server:
    docker compose -f docker-compose.dev.yml up -d qdrant
    # or: docker run -p 6333:6333 qdrant/qdrant

Use this backend when the collection outgrows SQLite or Postgres: Qdrant keeps a
HNSW graph over the whole corpus and answers KNN in single-digit milliseconds at
100M+ points. Retrieval stays hybrid — dense KNN inside Qdrant, BM25 over
candidates narrowed by Qdrant's full-text index — and the decision layer fuses
the two with RRF exactly as it does on every other backend.
"""
from __future__ import annotations

from agent_memory import Memory, MemoryAction, MemoryType

# ── Connect to Qdrant ─────────────────────────────────────────────────────────
# Option A: URL string (recommended)
memory = Memory(
    backend="qdrant",
    url="http://localhost:6333",
    collection_name="myapp_memory",
    # Embeddings score a paraphrase lower than a lexical exact match does
    # (bge-small-en-v1.5 gives ~0.52–0.65 for "related but differently worded"),
    # so the default 0.70 — tuned for the lexical backend — would return NONE for
    # queries vector search genuinely found. Lower it whenever KNN is the primary
    # retrieval path.
    restore_threshold=0.55,
)

# Option B: host/port
# memory = Memory(backend="qdrant", host="localhost", port=6333)

# Option C: Qdrant Cloud
# memory = Memory(backend="qdrant", url="https://xyz.cloud.qdrant.io", api_key="...")

# Option D: embedded, no server — stores the collection on local disk
# memory = Memory(backend="qdrant", path="./qdrant_data")

# search() is real vector KNN only when an embedding model is installed
# ([semantic] extra). Without one it degrades to BM25, same as everywhere else.
print(f"Semantic search: {memory.store.semantic_search_enabled}")  # type: ignore[union-attr]

# ── Store memories ────────────────────────────────────────────────────────────
memory.remember(
    "How do I reset my password?",
    "Go to Settings → Security → Reset Password. Link expires in 30 minutes.",
    type=MemoryType.CONVERSATION,
    tags=["auth", "password"],
)

memory.remember(
    "What is the API rate limit?",
    "Free: 100 req/min. Pro: 1000 req/min. Enterprise: unlimited.",
    type=MemoryType.FACT,
    tags=["api", "limits"],
    requires_verification=True,  # will return VERIFY, not REPLAY
)

memory.remember(
    "How do I invite team members?",
    "Settings → Team → Invite. Invitations expire after 7 days.",
    type=MemoryType.WORKFLOW,
    tags=["team", "onboarding"],
)

print(f"Stored {memory.store.count} memories in Qdrant")

# ── Resolve ───────────────────────────────────────────────────────────────────
test_queries = [
    ("How do I reset my password?",              "→ exact match, expect REPLAY"),
    ("I can't get into my account anymore",      "→ paraphrase with no shared words"),
    ("What are the rate limits for the API?",    "→ similar, expect VERIFY (flagged fact)"),
    ("How do I add a new user to my account?",   "→ close paraphrase of invite workflow"),
    ("What is the capital of France?",           "→ out-of-domain, expect NONE"),
]

print()
for query, note in test_queries:
    d = memory.resolve(query)
    badge = {"replay": "✅ REPLAY", "restore": "📋 RESTORE",
             "verify": "⚠️  VERIFY", "none": "❌ NONE"}.get(d.action.value, d.action.value)
    print(f"{badge}  conf={d.confidence:.2f}  {note}")
    print(f"  Q: {query}")
    if d.action == MemoryAction.REPLAY and d.memory:
        print(f"  A: {d.memory.response}")
    print()

# ── Stats ─────────────────────────────────────────────────────────────────────
stats = memory.stats()
print(f"Stats: total={stats['total']}  by_type={stats['by_type']}")
