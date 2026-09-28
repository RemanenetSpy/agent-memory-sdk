"""Generate README charts from the latest LongMemEval results.

Usage: uv run python benchmarks/longmemeval/make_charts.py
Writes docs/assets/longmemeval_recall.png and longmemeval_vs_baselines.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS_S = Path(__file__).parent / "results_semantic_500q_final.json"
RESULTS_M = Path(__file__).parent / "results_lexical_M_500q.json"
ASSETS = ROOT / "docs" / "assets"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
BLUE = "#2a78d6"   # series 1 — Recall@5
ORANGE = "#eb6834"  # series 2 — Coverage@5
GRAY = "#c3c2b7"   # de-emphasised baseline bars


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_recall(summary: dict) -> None:
    per = {t: v for t, v in summary["per_type"].items() if t != "abstention"}
    order = sorted(per, key=lambda t: per[t]["recall@5"])
    labels = [t.replace("-", " ") for t in order]
    recall = [per[t]["recall@5"] * 100 for t in order]
    coverage = [per[t]["coverage@5"] * 100 for t in order]

    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    y = range(len(order))
    h = 0.34
    b1 = ax.barh([i + h / 2 + 0.03 for i in y], recall, height=h, color=BLUE,
                 label="Session Recall@5", edgecolor=SURFACE, linewidth=1)
    b2 = ax.barh([i - h / 2 - 0.03 for i in y], coverage, height=h, color=ORANGE,
                 label="Evidence coverage@5", edgecolor=SURFACE, linewidth=1)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.0f}%", va="center", ha="left",
                    fontsize=9, color=INK)

    ax.set_yticks(list(y), labels)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    # Title and legend live in figure space, well clear of the axes.
    fig.suptitle("LongMemEval_S retrieval by question type — 500 questions, semantic (local ONNX)",
                 fontsize=12, color=INK, x=0.02, y=0.985, ha="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=2,
              frameon=False, fontsize=9, labelcolor=INK_2, borderaxespad=0,
              handlelength=1.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(ASSETS / "longmemeval_recall.png", facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def chart_baselines(summary_m: dict) -> None:
    # Published values use the original release and session indexing; our result
    # uses the cleaned release and turn-pair indexing, so this is context only.
    systems = [
        ("BM25, best config (paper)", 68.3, GRAY),
        ("Stella V5 1.5B, best config (paper)", 73.2, GRAY),
        ("Contriever, best config (paper)", 76.2, GRAY),
        ("agent-memory-sdk (lexical)", summary_m["overall"]["recall@5"] * 100, BLUE),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    names = [s[0] for s in systems]
    vals = [s[1] for s in systems]
    colors = [s[2] for s in systems]
    bars = ax.barh(range(len(systems)), vals, height=0.55, color=colors,
                   edgecolor=SURFACE, linewidth=1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(range(len(systems)), names)
    ax.set_ylim(-0.6, len(systems) - 0.4)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    fig.suptitle("LongMemEval_M session Recall@5 — non-comparable context",
                 fontsize=12, color=INK, x=0.02, y=0.97, ha="left")
    fig.text(0.02, 0.015,
             "500-session haystacks, ~2,500 memories per question. Baselines: Wu et al. (ICLR 2025) Table 9, each\n"
             "system's best key design, original _M release; ours measured on the cleaned 2025-09 re-release.",
             fontsize=8, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.92))
    fig.savefig(ASSETS / "longmemeval_vs_baselines.png", facecolor=SURFACE,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> None:
    summary_s = json.loads(RESULTS_S.read_text())["summary"]
    summary_m = json.loads(RESULTS_M.read_text())["summary"]
    ASSETS.mkdir(parents=True, exist_ok=True)
    chart_recall(summary_s)
    chart_baselines(summary_m)
    print("wrote", ASSETS / "longmemeval_recall.png")
    print("wrote", ASSETS / "longmemeval_vs_baselines.png")


if __name__ == "__main__":
    main()
