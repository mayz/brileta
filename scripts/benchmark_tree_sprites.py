#!/usr/bin/env python3
"""Benchmark tree sprite generation performance.

Measures per-archetype generation cost and breaks down time by internal
phase (trunk, branching, canopy stamping, post-processing) to identify
hotspots worth moving to native code.

Usage:
    # Full benchmark (default: 200 trees per archetype)
    uv run python -m scripts.benchmark_tree_sprites

    # Quick run
    uv run python -m scripts.benchmark_tree_sprites --iterations 50

    # Save / compare baselines
    uv run python -m scripts.benchmark_tree_sprites --save baseline.json
    uv run python -m scripts.benchmark_tree_sprites --compare baseline.json
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from io import StringIO
from pathlib import Path

import numpy as np

from brileta.game.actors.tree_sprites import (
    TreeArchetype,
    generate_tree_sprite,
)


def _benchmark_archetype(
    archetype: TreeArchetype,
    iterations: int,
    base_size: int,
) -> dict[str, float]:
    """Generate *iterations* sprites of one archetype and return timing stats.

    Returns a dict with total_ms, avg_ms, min_ms, max_ms, and
    sprites_per_second.
    """
    times: list[float] = []

    for i in range(iterations):
        seed = (0xBEEF * (i + 1)) ^ (hash(archetype.value) & 0xFFFFFFFF)
        start = time.perf_counter()
        generate_tree_sprite(seed, archetype, base_size)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total = sum(times)
    avg = total / iterations
    return {
        "total_ms": total * 1000.0,
        "avg_ms": avg * 1000.0,
        "min_ms": min(times) * 1000.0,
        "max_ms": max(times) * 1000.0,
        "sprites_per_second": iterations / total if total > 0 else 0.0,
    }


def _profile_archetype(
    archetype: TreeArchetype,
    iterations: int,
    base_size: int,
    top_n: int = 20,
) -> str:
    """cProfile a batch of sprite generations and return a formatted report."""
    profiler = cProfile.Profile()

    profiler.enable()
    for i in range(iterations):
        seed = (0xBEEF * (i + 1)) ^ (hash(archetype.value) & 0xFFFFFFFF)
        generate_tree_sprite(seed, archetype, base_size)
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)
    return stream.getvalue()


class TreeSpriteBenchmark:
    """Benchmark runner for tree sprite generation."""

    def __init__(self, iterations: int, base_size: int) -> None:
        self.iterations = iterations
        self.base_size = base_size
        self.results: dict[str, dict[str, float]] = {}

    def run(self) -> None:
        """Run timing benchmarks for all archetypes."""
        print("Tree Sprite Generation Benchmark")
        print("=" * 60)
        print(f"Iterations per archetype: {self.iterations}")
        print(f"Base sprite size: {self.base_size}px")
        print()

        # Warmup: generate a few sprites to let numpy JIT / caches settle.
        for archetype in TreeArchetype:
            generate_tree_sprite(0xDEAD, archetype, self.base_size)

        print(
            f"{'Archetype':>12} {'Avg (ms)':>10} {'Min (ms)':>10} "
            f"{'Max (ms)':>10} {'Sprites/s':>12}"
        )
        print("-" * 60)

        total_sprites = 0
        total_time_ms = 0.0

        for archetype in TreeArchetype:
            stats = _benchmark_archetype(archetype, self.iterations, self.base_size)
            self.results[archetype.value] = stats
            total_sprites += self.iterations
            total_time_ms += stats["total_ms"]

            print(
                f"{archetype.value:>12} {stats['avg_ms']:10.3f} "
                f"{stats['min_ms']:10.3f} {stats['max_ms']:10.3f} "
                f"{stats['sprites_per_second']:12.0f}"
            )

        print("-" * 60)
        overall_avg = total_time_ms / total_sprites
        overall_rate = total_sprites / (total_time_ms / 1000.0)
        print(
            f"{'OVERALL':>12} {overall_avg:10.3f} "
            f"{'':>10} {'':>10} {overall_rate:12.0f}"
        )

        # Batch benchmark: generate a full grid like the game does at startup.
        print()
        self._run_batch_benchmark()

    def _run_batch_benchmark(self) -> None:
        """Simulate generating a full map's worth of trees in one batch."""
        # A typical settlement might have 200-400 trees on a 150x100 map.
        batch_sizes = [100, 250, 500]
        archetypes = list(TreeArchetype)

        print("Batch Generation (simulating map startup)")
        print(f"{'Batch size':>12} {'Total (ms)':>12} {'Per tree (ms)':>14}")
        print("-" * 42)

        for n in batch_sizes:
            rng = np.random.default_rng(42)
            seeds = rng.integers(0, 0xFFFFFFFF, size=n, dtype=np.uint64)
            arch_indices = rng.integers(0, len(archetypes), size=n)

            start = time.perf_counter()
            for j in range(n):
                generate_tree_sprite(
                    int(seeds[j]),
                    archetypes[int(arch_indices[j])],
                    self.base_size,
                )
            elapsed = (time.perf_counter() - start) * 1000.0

            per_tree = elapsed / n
            self.results[f"batch_{n}"] = {
                "total_ms": elapsed,
                "per_tree_ms": per_tree,
            }
            print(f"{n:>12} {elapsed:12.1f} {per_tree:14.3f}")

    def profile(self) -> None:
        """Run cProfile on the heaviest archetype and print the top callers."""
        # Deciduous is typically the most expensive (most canopy stamps).
        print()
        print("cProfile Hotspot Analysis (deciduous)")
        print("=" * 60)
        report = _profile_archetype(
            TreeArchetype.DECIDUOUS, self.iterations, self.base_size
        )
        print(report)

    def save_results(self, filename: str) -> None:
        """Save benchmark results to a JSON file."""
        with Path(filename).open("w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved results to {filename}")

    def compare_with_baseline(self, baseline_file: str) -> None:
        """Compare current run with a saved baseline."""
        try:
            with Path(baseline_file).open() as f:
                baseline: dict[str, dict[str, float]] = json.load(f)
        except FileNotFoundError:
            print(f"\nBaseline file not found: {baseline_file}")
            return

        print(f"\nComparison vs baseline: {baseline_file}")
        print("=" * 64)

        for key, current in self.results.items():
            if key not in baseline:
                continue
            old_avg = baseline[key].get("avg_ms") or baseline[key].get("per_tree_ms")
            new_avg = current.get("avg_ms") or current.get("per_tree_ms")
            if old_avg is None or new_avg is None or old_avg <= 0:
                continue

            ratio = old_avg / new_avg if new_avg > 0 else 0.0
            delta_pct = ((new_avg - old_avg) / old_avg) * 100.0
            trend = "faster" if ratio > 1.0 else "slower"
            print(
                f"{key:>12}: {new_avg:.3f}ms vs {old_avg:.3f}ms | "
                f"{ratio:.2f}x {trend} ({delta_pct:+.1f}%)"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark tree sprite generation")
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Sprites to generate per archetype (default: 200)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=20,
        help="Base sprite size in px (default: 20)",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip cProfile hotspot analysis",
    )
    parser.add_argument("--save", type=str, help="Save results to JSON")
    parser.add_argument("--compare", type=str, help="Compare against baseline JSON")
    args = parser.parse_args(argv)

    benchmark = TreeSpriteBenchmark(iterations=args.iterations, base_size=args.size)
    benchmark.run()

    if not args.no_profile:
        benchmark.profile()

    if args.save:
        benchmark.save_results(args.save)

    if args.compare:
        benchmark.compare_with_baseline(args.compare)


if __name__ == "__main__":
    main()
