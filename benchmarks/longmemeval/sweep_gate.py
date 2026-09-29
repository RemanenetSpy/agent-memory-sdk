"""Calibrate the flat-crowd gate: collect (relevance, margin) for every
question's top hit, then grid-search thresholds offline.

Goal: maximise abstention catch subject to false-abstention <= target.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_retrieval import DATA, pair_turns  # noqa: E402

from agent_memory import Memory  # noqa: E402


def signals(q: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        memory = Memory(persist_dir=tmp, enable_embeddings=False)
        for sid, session in zip(q["haystack_session_ids"], q["haystack_sessions"]):
            for u, a in pair_turns(session):
                memory.remember(u, a, metadata={"session_id": sid})
        # match decide()'s view: top_k=3
        results = memory.retriever.retrieve(q["question"], top_k=3)
    if not results:
        return {"relevance": 0.0, "margin": 1.0}
    best = results[0]
    relevance = max(
        0.7 * best.semantic_score + 0.3 * best.keyword_score,
        0.7 * best.keyword_score + 0.3 * best.semantic_score,
    )
    best_resp = (best.entry.response or "").strip().lower()
    margin = 1.0
    for other in results[1:]:
        if (other.entry.response or "").strip().lower() != best_resp:
            margin = best.final_score - other.final_score
            break
    return {"relevance": round(relevance, 4), "margin": round(margin, 4)}


def main() -> None:
    questions = json.loads(DATA.read_text())
    rows = []
    for i, q in enumerate(questions, 1):
        s = signals(q)
        s["is_abs"] = q["question_id"].endswith("_abs")
        rows.append(s)
        if i % 50 == 0:
            print(f"[{i}/{len(questions)}]", flush=True)

    Path(__file__).with_name("gate_signals.json").write_text(json.dumps(rows))

    abst = [r for r in rows if r["is_abs"]]
    answ = [r for r in rows if not r["is_abs"]]
    print(f"\n{'rel<':>6} {'mrg<':>7} {'abst_catch':>10} {'false_abst':>10}")
    best = None
    for rel in (0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.85, 1.01):
        for mrg in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
            catch = sum(1 for r in abst if r["relevance"] < rel and r["margin"] < mrg) / len(abst)
            false = sum(1 for r in answ if r["relevance"] < rel and r["margin"] < mrg) / len(answ)
            print(f"{rel:>6} {mrg:>7} {catch:>10.3f} {false:>10.3f}")
            if false <= 0.05 and (best is None or catch > best[2]):
                best = (rel, mrg, catch, false)
    print("\nBest with false_abstention <= 5%:", best)


if __name__ == "__main__":
    main()
