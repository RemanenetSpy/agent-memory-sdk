"""Probe score distributions: abstention vs answerable questions.

Measures, for the top retrieval hit of each question:
  * hybrid semantic relevance (the discriminating component)
  * final composite score
  * margin between top-1 and top-2 final scores
so we can calibrate a relevance gate and replay margin on evidence.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_retrieval import DATA, pair_turns  # noqa: E402

from agent_memory import Memory  # noqa: E402


def hybrid(s: float, k: float) -> float:
    return max(0.7 * s + 0.3 * k, 0.7 * k + 0.3 * s)


def probe(q: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        memory = Memory(persist_dir=tmp, enable_embeddings=False)
        for sid, session in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            for u, a in pair_turns(session):
                memory.remember(u, a, metadata={"session_id": sid})
        results = memory.retriever.retrieve(q["question"], top_k=5)
    if not results:
        return {"empty": True}
    top = results[0]
    evidence = set(q["answer_session_ids"])
    top_sid = (top.entry.metadata or {}).get("session_id")
    return {
        "hybrid": round(hybrid(top.semantic_score, top.keyword_score), 4),
        "final": round(top.final_score, 4),
        "margin": round(
            top.final_score - results[1].final_score if len(results) > 1 else 1.0, 4
        ),
        "top_is_evidence": top_sid in evidence,
    }


def main() -> None:
    questions = json.loads(DATA.read_text())
    abstention = [q for q in questions if q["question_id"].endswith("_abs")]
    answerable = [q for q in questions if not q["question_id"].endswith("_abs")][:60]

    for label, qs in [("ABSTENTION", abstention), ("ANSWERABLE", answerable)]:
        rows = [probe(q) for q in qs]
        rows = [r for r in rows if not r.get("empty")]
        for field in ("hybrid", "final", "margin"):
            vals = sorted(r[field] for r in rows)
            print(
                f"{label:10s} {field:7s} "
                f"min={vals[0]:.3f} p25={vals[len(vals)//4]:.3f} "
                f"med={statistics.median(vals):.3f} "
                f"p75={vals[3*len(vals)//4]:.3f} max={vals[-1]:.3f}"
            )
        if label == "ANSWERABLE":
            ev = [r for r in rows if r["top_is_evidence"]]
            print(f"{label}: top hit is evidence session in {len(ev)}/{len(rows)}")
        print()


if __name__ == "__main__":
    main()
