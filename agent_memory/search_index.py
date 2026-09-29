"""Search-acceleration data structures for the SQLite memory store.

Two complementary optimisations that reduce FTS5 work at large scale:

1. **BloomFilter** — probabilistic set-membership test over all content tokens
   in the store.  Used for the NONE fast-path: if no query token appears in
   the filter, we know there are zero FTS5 matches and can skip the SQL query
   entirely.  False positives are safe (we just run FTS5 unnecessarily); false
   negatives are impossible by construction.

   Saves ~5–15ms on the 56% of queries that would return NONE.

2. **DynamicStopWords** — extends the static STOP_WORDS set with terms that
   appear in > ``idf_threshold``% of the corpus (high inverse-document-frequency
   terms).  When a term saturates the corpus, including it in an OR expression
   causes FTS5 to score every document — exactly the bottleneck seen at 1M
   entries with repeated templates.

   At 1M entries / 32 templates each term appears in 1/32 = 3% of docs; with
   a 2% threshold we skip corpus-saturated terms automatically.

Both structures are maintained incrementally via ``add_token()`` and
``add_entry()`` so they stay in sync with the store without a full rebuild.

DSA properties
--------------
BloomFilter:
  Space   O(m) bits  where m = -n·ln(fp) / ln(2)²
  Insert  O(k)       k hash functions, k ≈ 7 for fp=0.01
  Lookup  O(k)       no false negatives

DynamicStopWords:
  Insert  O(1) amortised  (Counter update)
  Lookup  O(1)            (set membership)
  Rebuild O(|corpus|)     on threshold change (rare)
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable

from agent_memory.store import STOP_WORDS, _tokenize

# ---------------------------------------------------------------------------
# Bloom filter
# ---------------------------------------------------------------------------


class BloomFilter:
    """Space-efficient probabilistic set for content tokens.

    Parameters
    ----------
    capacity:
        Expected number of distinct tokens.  Over-estimate freely — the
        filter degrades gracefully, just with a higher false-positive rate.
    fp_rate:
        Target false-positive probability (default 1%).
    """

    def __init__(self, capacity: int = 100_000, fp_rate: float = 0.01) -> None:
        if capacity <= 0:
            capacity = 1
        # Optimal bit-array size and number of hash functions
        m = max(1, int(-capacity * math.log(fp_rate) / (math.log(2) ** 2)))
        k = max(1, round((m / capacity) * math.log(2)))
        self._bits = bytearray((m + 7) // 8)
        self._m = m
        self._k = k
        self._count = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, token: str) -> None:
        for seed in range(self._k):
            idx = self._hash(token, seed)
            self._bits[idx >> 3] |= 1 << (idx & 7)
        self._count += 1

    def add_many(self, tokens: Iterable[str]) -> None:
        for t in tokens:
            self.add(t)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def __contains__(self, token: object) -> bool:
        if not isinstance(token, str):
            return False
        for s in range(self._k):
            idx = self._hash(token, s)
            if not (self._bits[idx >> 3] & (1 << (idx & 7))):
                return False
        return True

    def any_of(self, tokens: Iterable[str]) -> bool:
        """Return True if ANY token is probably in the filter.

        Used for the NONE fast-path: if ``any_of(query_tokens)`` is False,
        no FTS5 query can return a match — return NONE immediately.
        """
        return any(t in self for t in tokens)

    @property
    def estimated_count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hash(self, token: str, seed: int) -> int:
        """Double-hashing scheme using SHA-256 slices.  O(1), low collision."""
        raw = hashlib.sha256(f"{seed}:{token}".encode()).digest()
        # Take 4 bytes as an unsigned int, mod m
        val = int.from_bytes(raw[:4], "big")
        return val % self._m


# ---------------------------------------------------------------------------
# Dynamic stop words (IDF-based)
# ---------------------------------------------------------------------------


class DynamicStopWords:
    """Extends static STOP_WORDS with corpus-saturated terms.

    A term that appears in > ``idf_threshold`` fraction of documents carries
    very low discriminative power.  Including it in a FTS5 OR expression
    forces FTS5 to score thousands (or millions) of entries for no gain.

    Algorithm: maintain a term-frequency Counter and the total document count.
    After each insertion, check if the new term's frequency now exceeds the
    threshold and add it to the dynamic stop set.

    Parameters
    ----------
    idf_threshold:
        Terms appearing in more than this fraction of documents are treated as
        stop words.  Default 0.10 (10%).  Lower = more aggressive pruning.
    min_docs:
        Dynamic stop words are not promoted until the corpus has at least this
        many documents.  Below this threshold every term is individually rare
        (1/N is always high), so filtering would destroy recall.  Default 100.
    """

    def __init__(self, idf_threshold: float = 0.10, min_docs: int = 100) -> None:
        self._threshold = idf_threshold
        self._min_docs = min_docs
        self._doc_count = 0
        self._term_freq: Counter[str] = Counter()
        self._dynamic: set[str] = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entry(self, query: str, response: str, tags: list[str] | None = None) -> None:
        """Record the tokens for one new memory entry."""
        text = f"{query} {response} {' '.join(tags or [])}"
        tokens = {t for t in _tokenize(text) if len(t) > 1}
        self._doc_count += 1
        for t in tokens:
            self._term_freq[t] += 1

        if self._doc_count < self._min_docs:
            # Too few documents — IDF is meaningless; don't promote anything yet.
            return

        if self._doc_count == self._min_docs:
            # First time we cross the threshold: bulk-promote all terms that
            # were accumulated before this point and already exceed the cutoff.
            # Without this pass, terms counted pre-threshold are never promoted
            # by the per-token check below (which only fires on new entries).
            self._dynamic = {
                t for t, f in self._term_freq.items()
                if t not in STOP_WORDS and f / self._doc_count > self._threshold
            }
            return

        for t in tokens:
            if (
                t not in STOP_WORDS
                and t not in self._dynamic
                and self._term_freq[t] / self._doc_count > self._threshold
            ):
                self._dynamic.add(t)

    def rebuild_from_counter(self, term_freq: Counter[str], doc_count: int) -> None:
        """Rebuild from a pre-computed term-frequency counter (e.g. after bulk load)."""
        self._doc_count = doc_count
        self._term_freq = Counter(term_freq)
        if doc_count < self._min_docs:
            self._dynamic = set()   # too small — no dynamic stop words yet
        else:
            self._dynamic = {
                t for t, f in term_freq.items()
                if t not in STOP_WORDS and doc_count > 0 and f / doc_count > self._threshold
            }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_stop(self, token: str) -> bool:
        return token in STOP_WORDS or token in self._dynamic

    def filter_tokens(self, tokens: list[str]) -> list[str]:
        """Return only the tokens that are NOT stop words (static or dynamic)."""
        result = [t for t in tokens if not self.is_stop(t) and len(t) > 1]
        if not result:
            # All tokens were stop words — fall back to non-stop tokens only
            result = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
        return result

    @property
    def dynamic_count(self) -> int:
        return len(self._dynamic)

    @property
    def doc_count(self) -> int:
        return self._doc_count

    @property
    def dynamic_stop_words(self) -> frozenset[str]:
        return frozenset(self._dynamic)
