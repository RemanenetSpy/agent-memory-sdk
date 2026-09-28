"""Entity extractor — auto-extract structured memories from raw conversation text.

Eliminates the need to call ``memory.remember()`` manually after every turn.
Works offline with zero external dependencies; spaCy NER is used when
available for better entity coverage.

Usage::

    from agent_memory import Memory, EntityExtractor

    extractor = EntityExtractor()
    memory = Memory(persist_dir=".agent_memory")

    # Extract + store from a raw conversation turn
    entries = memory.from_conversation(
        human="My name is Karan and I prefer Python over Go.",
        assistant="Got it, I'll remember that preference.",
    )
    # → stores preference + fact memories automatically

    # Or extract without storing
    candidates = extractor.extract("The API rate limit is 1000 req/min.")
    for c in candidates:
        print(c.query, c.response, c.memory_type)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent_memory.logging_config import get_logger
from agent_memory.models import MemoryEntry, MemoryScope, MemoryType

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExtractedMemory:
    """A candidate memory extracted from free-form text."""

    query: str
    response: str
    memory_type: MemoryType
    confidence: float
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    requires_verification: bool = False


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Each pattern: (regex, memory_type, confidence, tag, query_template)
# The regex must have a named group ``value`` for the extracted value and
# optionally ``subject`` for what the fact is about.

_PATTERNS: list[tuple[re.Pattern[str], MemoryType, float, str, str]] = [
    # Preferences
    (re.compile(r"\bI (?:prefer|like|love|enjoy|favour|favor)\s+(.+?)(?:\.|,|$)", re.I),
     MemoryType.PREFERENCE, 0.90, "preference",
     "What does the user prefer?"),
    (re.compile(r"\bI (?:don'?t|do not) (?:like|prefer|want|use)\s+(.+?)(?:\.|,|$)", re.I),
     MemoryType.PREFERENCE, 0.88, "preference",
     "What does the user dislike?"),
    (re.compile(r"\bMy (?:favourite|favorite|preferred)\s+\w+\s+is\s+(.+?)(?:\.|,|$)", re.I),
     MemoryType.PREFERENCE, 0.88, "preference",
     "What is the user's preference?"),

    # Identity
    (re.compile(r"\bMy name is\s+([A-Z][a-zA-Z\s]+?)(?:\.|,|$)", re.I),
     MemoryType.FACT, 0.95, "person",
     "What is the user's name?"),
    (re.compile(r"\bI(?:'m| am)\s+([A-Z][a-zA-Z\s]+?)(?:,|\.|$)", re.I),
     MemoryType.FACT, 0.80, "person",
     "Who is the user?"),
    (re.compile(r"\bI work(?:ed)? (?:at|for)\s+(.+?)(?:\.|,|$)", re.I),
     MemoryType.FACT, 0.88, "work,organization",
     "Where does the user work?"),
    (re.compile(r"\bI(?:'m| am) (?:based|located) in\s+(.+?)(?:\.|,|$)", re.I),
     MemoryType.FACT, 0.85, "location",
     "Where is the user located?"),
    (re.compile(r"\bMy (?:email|phone|number|address) (?:is|:)\s+(.+?)(?:\.|,|$)", re.I),
     MemoryType.FACT, 0.92, "contact",
     "What is the user's contact info?"),

    # Facts / technical
    (re.compile(
        r"\b(?:The )?(?:API |rate )?limit (?:is|:)\s*(.+?)(?:\.|,|$)", re.I),
     MemoryType.FACT, 0.88, "api,limits",
     "What is the rate limit?"),
    (re.compile(r"\b(\w[\w\s]+?) (?:is|are)\s+(?:deprecated|removed|no longer supported)",
                re.I),
     MemoryType.FACT, 0.85, "deprecated",
     "What is deprecated?"),
    (re.compile(r"\bThe (?:default|current) (?:\w+ )?is\s+(.+?)(?:\.|,|$)", re.I),
     MemoryType.FACT, 0.78, "default",
     "What is the current default?"),

    # Deadlines / dates
    (re.compile(
        r"\b(?:The )?(?:deadline|due date|release|launch) (?:is|:)\s*(.+?)(?:\.|,|$)", re.I),
     MemoryType.FACT, 0.85, "deadline",
     "When is the deadline?"),

    # Workflows / procedures
    (re.compile(r"\bTo\s+(\w[\w\s]+?),?\s+(?:you|one) (?:should|must|need to|can)\s+(.+?)(?:\.|$)",
                re.I),
     MemoryType.WORKFLOW, 0.80, "workflow",
     "How do you accomplish a task?"),
]

# Simple sentence splitter
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Injection detection patterns
# ---------------------------------------------------------------------------

# Patterns that indicate an injected instruction rather than a genuine user statement.
# These are matched against the human turn text. If matched, the extraction is
# either rejected or flagged with requires_verification=True and low confidence.
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Direct instruction to the assistant/agent
    re.compile(r"\b(?:NOTE TO|INSTRUCTION TO|SYSTEM:|ASSISTANT:|AGENT:)\s*(?:remember|store|save|note)\b", re.I),
    # "remember that..." when not in a natural user context
    re.compile(r"\b(?:please\s+)?remember\s+that\s+(?:the\s+)?(?:refund|password|address|account|policy|window|limit|rate|key|secret|token)\b", re.I),
    # Disregard/ignore previous instructions
    re.compile(r"\b(?:disregard|ignore|forget)\s+(?:previous|prior|earlier|all)\s+(?:instructions?|preferences?|policies?|memories?)\b", re.I),
    # Credential exfiltration attempts
    re.compile(r"\b(?:admin|root|system|master)\s+(?:password|key|secret|token)\s+(?:is|:)\s*\w+", re.I),
    # Third-party assertion of first-party facts
    re.compile(r"\b(?:colleague|friend|someone|they|he|she)\s+(?:said|sent|told)\s+(?:me\s+)?(?:to\s+)?(?:remember|note)\s+that\s+(?:my|your|their)\b", re.I),
    # Third-party assertion with "this:" or similar intro
    re.compile(r"\b(?:colleague|friend|someone|they|he|she)\s+(?:said|sent|told)\s+(?:me\s+)?(?:this|that)\s*:\s*.*\b(?:please\s+)?remember\s+that\s+(?:my|your|their|[A-Z][a-z]+'s)\b", re.I),
    # Tool output injection markers
    re.compile(r"\[(?:tool\s+output|system|assistant)\]\s*.*\bremember\s+that\b", re.I),
    # Summarization/request injection
    re.compile(r"\b(?:summari[sz]e|summary)\s+(?:this|the)\s+(?:thread|conversation|chat)\b.*\bremember\s+that\b", re.I),
    # "Disregard and remember" pattern
    re.compile(r"\bdisregard\s+(?:previous|prior)\s+(?:preferences?|instructions?)\s+and\s+remember\b", re.I),
]


def _looks_like_injection(text: str) -> bool:
    """Check if text contains patterns typical of prompt injection attempts."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """Extract structured memories from free-form text.

    Uses regex patterns as a baseline; spaCy NER is used automatically
    when installed (``pip install spacy && python -m spacy download en_core_web_sm``).
    """

    def __init__(self, use_spacy: bool = True) -> None:
        self._nlp: Any = None
        if use_spacy:
            try:
                import spacy  # noqa: F401
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                pass  # spaCy not installed or model not downloaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[ExtractedMemory]:
        """Extract candidate memories from *text* (a sentence, paragraph, or turn).

        Injection attempts (e.g., "NOTE TO ASSISTANT: remember that...") are
        detected and either rejected or flagged with requires_verification=True
        and reduced confidence.
        """
        # Check for injection patterns in the input text
        is_injection = _looks_like_injection(text)

        candidates: list[ExtractedMemory] = []
        for pattern, mtype, conf, tag_str, query_tmpl in _PATTERNS:
            for m in pattern.finditer(text):
                value = m.group(1).strip().rstrip(".,;")
                if len(value) < 2:
                    continue
                query = query_tmpl
                response = value
                tags = [t.strip() for t in tag_str.split(",") if t.strip()]

                # If injection detected, flag for verification and reduce confidence
                requires_v = (mtype == MemoryType.FACT and conf < 0.85)
                final_conf = conf
                if is_injection:
                    requires_v = True
                    final_conf = min(conf, 0.50)  # Cap confidence for suspicious content
                    tags.append("injection-suspected")

                candidates.append(
                    ExtractedMemory(
                        query=query,
                        response=response,
                        memory_type=mtype,
                        confidence=final_conf,
                        tags=tags,
                        requires_verification=requires_v,
                    )
                )

        if self._nlp:
            candidates.extend(self._spacy_extract(text))

        return self._deduplicate(candidates)

    def extract_from_turn(
        self,
        human: str,
        assistant: str,
        *,
        scope: MemoryScope = MemoryScope.USER,
    ) -> list[ExtractedMemory]:
        """Extract from a single conversation turn (human + assistant text).

        Injection detection runs on the human turn, since that's where
        prompt injections typically arrive (user input, tool output, retrieved docs).
        """
        # Check human turn for injection patterns
        human_injection = _looks_like_injection(human)

        # Mine the human turn for facts/preferences about the user
        human_candidates = self.extract(human)

        # If injection detected in human turn, flag all human-derived candidates
        if human_injection:
            for c in human_candidates:
                c.requires_verification = True
                c.confidence = min(c.confidence, 0.50)
                if "injection-suspected" not in c.tags:
                    c.tags.append("injection-suspected")

        # Mine the assistant turn for facts / answers it stated
        assistant_candidates = self._extract_facts_from_answer(human, assistant, human_injection=human_injection)
        return self._deduplicate(human_candidates + assistant_candidates)

    def extract_from_conversation(
        self,
        turns: list[tuple[str, str]],
        *,
        scope: MemoryScope = MemoryScope.USER,
    ) -> list[ExtractedMemory]:
        """Extract memories from a list of ``(human, assistant)`` turn pairs."""
        all_candidates: list[ExtractedMemory] = []
        for human, assistant in turns:
            all_candidates.extend(self.extract_from_turn(human, assistant, scope=scope))
        return self._deduplicate(all_candidates)

    def to_memory_entries(
        self,
        candidates: list[ExtractedMemory],
        *,
        scope: MemoryScope = MemoryScope.USER,
    ) -> list[MemoryEntry]:
        """Convert *candidates* to ``MemoryEntry`` objects (not yet persisted)."""
        return [
            MemoryEntry(
                query=c.query,
                response=c.response,
                content=c.response,
                type=c.memory_type,
                scope=scope,
                tags=c.tags,
                confidence=c.confidence,
                requires_verification=c.requires_verification,
                metadata=c.metadata,
            )
            for c in candidates
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_facts_from_answer(
        self, question: str, answer: str, *, human_injection: bool = False
    ) -> list[ExtractedMemory]:
        """Treat the Q→A pair as a single memorable fact."""
        q = question.strip().rstrip("?")
        if not q or not answer.strip():
            return []
        confidence = 0.82
        mtype = MemoryType.CONVERSATION
        # Elevate to FACT for assertive answers
        if re.search(r"\b(?:is|are|was|were|costs?|limit|rate|deadline|version)\b",
                     answer, re.I):
            mtype = MemoryType.FACT
            confidence = 0.85
        # Mark stale-prone facts for verification
        requires_v = mtype == MemoryType.FACT and bool(
            re.search(r"\b(?:limit|rate|price|cost|version|deadline)\b", answer, re.I)
        )
        # If injection detected in human turn, flag this candidate too
        if human_injection:
            requires_v = True
            confidence = min(confidence, 0.50)
        return [ExtractedMemory(
            query=question.strip(),
            response=answer.strip(),
            memory_type=mtype,
            confidence=confidence,
            requires_verification=requires_v,
            tags=["injection-suspected"] if human_injection else [],
        )]

    def _spacy_extract(self, text: str) -> list[ExtractedMemory]:
        """Use spaCy NER to extract named-entity facts."""
        if self._nlp is None:
            return []
        doc = self._nlp(text)
        results: list[ExtractedMemory] = []
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "PERCENT"):
                tag = {
                    "PERSON": "person", "ORG": "organization",
                    "GPE": "location", "LOC": "location",
                    "DATE": "date", "MONEY": "money", "PERCENT": "percentage",
                }.get(ent.label_, "entity")
                results.append(ExtractedMemory(
                    query=f"What is {ent.text}?",
                    response=ent.text,
                    memory_type=MemoryType.FACT,
                    confidence=0.75,
                    tags=[tag, "spacy-ner"],
                    metadata={"ner_label": ent.label_},
                ))
        return results

    @staticmethod
    def _deduplicate(
        candidates: list[ExtractedMemory],
    ) -> list[ExtractedMemory]:
        """Remove near-duplicate extracted memories (same query+response)."""
        seen: set[tuple[str, str]] = set()
        out: list[ExtractedMemory] = []
        for c in candidates:
            key = (c.query.lower()[:60], c.response.lower()[:60])
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out
