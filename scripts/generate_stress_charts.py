#!/usr/bin/env python
"""Generate stress charts from benchmark JSON results.

Usage:
    uv run python scripts/generate_stress_charts.py
    uv run python scripts/generate_stress_charts.py --results-dir benchmarks/stress/results

The script deliberately has no embedded benchmark values. Every plotted point
must come from an archived JSON result emitted by scripts/stress_test.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_results(results_dir: Path) -> list[dict]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["_source"] = path.name
        results.append(result)
    if not results:
        raise SystemExit(f"No benchmark JSON files found in {results_dir}")
    return results


def _plot(results: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Chart generation requires matplotlib. Install it with "
            "'uv pip install matplotlib' or add it to the local development environment."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_note = "Bundled synthetic stress fixture; each point comes from archived JSON"

    latency = [result for result in results if result.get("n_queries", 0) >= 2]
    if latency:
        latency.sort(key=lambda result: result["n_memories"])
        sizes = [result["n_memories"] for result in latency]
        p50 = [result["median_ms"] for result in latency]
        p75 = [result["p75_ms"] for result in latency]
        p90 = [result["p90_ms"] for result in latency]
        p95 = [result["p95_ms"] for result in latency]
        p99 = [result["p99_ms"] for result in latency]

        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(sizes, p50, "o-", label="p50")
        axis.plot(sizes, p75, "d-", label="p75")
        axis.plot(sizes, p90, "x-", label="p90")
        axis.plot(sizes, p95, "s-", label="p95")
        axis.plot(sizes, p99, "^-", label="p99")
        axis.set_xscale("log")
        axis.set_xlabel("Stored memories")
        axis.set_ylabel("resolve() latency (ms)")
        axis.set_title("Stress latency by stored-memory count")
        axis.text(
            0.02,
            0.02,
            fixture_note,
            transform=axis.transAxes,
            fontsize=8,
            color="dimgray",
        )
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "stress_latency_benchmark.png", dpi=160)
        plt.close(fig)

    seeded = [result for result in results if result.get("seed_rate") is not None]
    if seeded:
        seeded.sort(key=lambda result: result["n_memories"])
        sizes = [result["n_memories"] for result in seeded]
        rates = [result["seed_rate"] for result in seeded]

        fig, axis = plt.subplots(figsize=(9, 5))
        axis.bar([f"{size:,}" for size in sizes], rates)
        axis.set_xlabel("Entries seeded")
        axis.set_ylabel("Seed rate (entries/s)")
        axis.set_title("Fast-seed throughput benchmark")
        axis.text(
            0.02,
            0.02,
            fixture_note,
            transform=axis.transAxes,
            fontsize=8,
            color="dimgray",
        )
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "stress_seeding_benchmark.png", dpi=160)
        plt.close(fig)

    resource_profiles = [
        result
        for result in results
        if result.get("query_rss_end_mb") is not None and result.get("p95_ms") is not None
    ]
    if resource_profiles:
        fig, axis = plt.subplots(figsize=(9, 5))
        for result in resource_profiles:
            axis.scatter(result["query_rss_end_mb"], result["p95_ms"], s=80)
            axis.annotate(
                f"{result['n_memories']:,}",
                (result["query_rss_end_mb"], result["p95_ms"]),
                xytext=(6, 6),
                textcoords="offset points",
            )
        axis.set_xlabel("Endpoint RSS after retrieval (MiB)")
        axis.set_ylabel("resolve() p95 latency (ms)")
        axis.set_title("Measured resource and latency profile")
        axis.text(
            0.02,
            0.02,
            "Not a Pareto frontier; no subjective effort or projected backend data",
            transform=axis.transAxes,
            fontsize=8,
            color="dimgray",
        )
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "stress_resource_latency_benchmark.png", dpi=160)
        plt.close(fig)

    action_results = [result for result in results if result.get("actions")]
    if action_results:
        action_results.sort(key=lambda result: result["n_memories"])
        sizes = [f"{result['n_memories']:,}" for result in action_results]
        actions = sorted({action for result in action_results for action in result["actions"]})
        bottoms = [0.0] * len(action_results)
        fig, axis = plt.subplots(figsize=(9, 5))
        for action in actions:
            values = [
                100 * result["actions"].get(action, 0) / result["n_queries"]
                for result in action_results
            ]
            axis.bar(sizes, values, bottom=bottoms, label=action.upper())
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        axis.set_xlabel("Stored memories")
        axis.set_ylabel("Queries (%)")
        axis.set_ylim(0, 100)
        axis.set_title("Measured decision-action mix")
        axis.text(
            0.02,
            0.02,
            fixture_note,
            transform=axis.transAxes,
            fontsize=8,
            color="dimgray",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=4)
        fig.tight_layout()
        fig.savefig(output_dir / "stress_action_mix_benchmark.png", dpi=160)
        plt.close(fig)

    rate_results = [
        result
        for result in results
        if result.get("cache_hit_pct") is not None and result.get("hit_rate_pct") is not None
    ]
    if rate_results:
        rate_results.sort(key=lambda result: result["n_memories"])
        sizes = [f"{result['n_memories']:,}" for result in rate_results]
        x_positions = range(len(rate_results))
        fig, axis = plt.subplots(figsize=(9, 5))
        width = 0.36
        axis.bar(
            [position - width / 2 for position in x_positions],
            [result["cache_hit_pct"] for result in rate_results],
            width,
            label="Cache-hit heuristic",
        )
        axis.bar(
            [position + width / 2 for position in x_positions],
            [result["hit_rate_pct"] for result in rate_results],
            width,
            label="Decision hit rate",
        )
        axis.set_xticks(list(x_positions), sizes)
        axis.set_xlabel("Stored memories")
        axis.set_ylabel("Queries (%)")
        axis.set_ylim(0, 100)
        axis.set_title("Measured cache and decision rates")
        axis.text(
            0.02,
            0.02,
            "Rates belong to this fixture and query set; they are not production expectations",
            transform=axis.transAxes,
            fontsize=8,
            color="dimgray",
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "stress_rates_benchmark.png", dpi=160)
        plt.close(fig)

    resource_results = [
        result
        for result in results
        if result.get("seed_cpu_s") is not None and result.get("query_cpu_s") is not None
    ]
    if resource_results:
        resource_results.sort(key=lambda result: result["n_memories"])
        sizes = [f"{result['n_memories']:,}" for result in resource_results]
        positions = range(len(resource_results))
        fig, (cpu_axis, rss_axis) = plt.subplots(1, 2, figsize=(12, 5))
        width = 0.36
        cpu_axis.bar(
            [position - width / 2 for position in positions],
            [result["seed_cpu_s"] for result in resource_results],
            width,
            label="Seed CPU",
        )
        cpu_axis.bar(
            [position + width / 2 for position in positions],
            [result["query_cpu_s"] for result in resource_results],
            width,
            label="Query CPU",
        )
        cpu_axis.set_title("CPU time")
        cpu_axis.set_ylabel("User CPU seconds")
        cpu_axis.set_xticks(list(positions), sizes)
        cpu_axis.grid(axis="y", alpha=0.25)
        cpu_axis.legend(fontsize=8)

        rss_axis.bar(
            [position - width / 2 for position in positions],
            [result["seed_rss_end_mb"] for result in resource_results],
            width,
            label="Seed endpoint RSS",
        )
        rss_axis.bar(
            [position + width / 2 for position in positions],
            [result["query_rss_end_mb"] for result in resource_results],
            width,
            label="Query endpoint RSS",
        )
        rss_axis.set_title("Endpoint RSS")
        rss_axis.set_ylabel("MiB")
        rss_axis.set_xticks(list(positions), sizes)
        rss_axis.grid(axis="y", alpha=0.25)
        rss_axis.legend(fontsize=8)
        fig.suptitle("Measured CPU and RSS profiles")
        fig.text(0.5, 0.01, fixture_note, ha="center", fontsize=8, color="dimgray")
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(output_dir / "stress_resources_benchmark.png", dpi=160)
        plt.close(fig)

    if not latency and not seeded:
        raise SystemExit("Benchmark results contain neither latency nor seed metrics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/stress/results"),
        help="directory containing stress_test.py JSON output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets"),
        help="directory for generated PNG charts",
    )
    args = parser.parse_args()
    results = _load_results(args.results_dir)
    _plot(results, args.output_dir)
    print(f"Generated benchmark charts from {len(results)} result(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
