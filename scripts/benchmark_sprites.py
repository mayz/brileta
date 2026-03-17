#!/usr/bin/env python3
"""Benchmark tree and boulder sprite generation performance.

Measures per-archetype generation cost for both trees and boulders, then
simulates realistic map startup batches to produce a single "Xms for a
typical map" number.

Usage:
    # Full benchmark (default: 200 sprites per archetype)
    uv run python -m scripts.benchmark_sprites

    # Quick run
    uv run python -m scripts.benchmark_sprites --iterations 50

    # Save / compare baselines
    uv run python -m scripts.benchmark_sprites --save baseline.json
    uv run python -m scripts.benchmark_sprites --compare baseline.json
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import numpy as np

from brileta.sprites.boulders import (
    BoulderArchetype,
    generate_boulder_sprite,
)
from brileta.sprites.trees import (
    TreeArchetype,
    generate_tree_sprite,
)


def _benchmark_archetype(
    archetype: TreeArchetype | BoulderArchetype,
    generate_fn: Callable[..., np.ndarray],
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
        generate_fn(seed, archetype, base_size)
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


def _profile_sprite_generation(
    iterations: int,
    tree_base_size: int,
    boulder_base_size: int,
    top_n: int = 20,
) -> str:
    """cProfile a mixed batch of sprite generations and return a report.

    Profiles the heaviest archetype from each category: deciduous trees
    (most canopy stamps) and blocky boulders (most lobe work).
    """
    profiler = cProfile.Profile()

    profiler.enable()
    for i in range(iterations):
        tree_seed = (0xBEEF * (i + 1)) ^ (
            hash(TreeArchetype.DECIDUOUS.value) & 0xFFFFFFFF
        )
        generate_tree_sprite(tree_seed, TreeArchetype.DECIDUOUS, tree_base_size)
        boulder_seed = (0xCAFE * (i + 1)) ^ (
            hash(BoulderArchetype.BLOCKY.value) & 0xFFFFFFFF
        )
        generate_boulder_sprite(
            boulder_seed, BoulderArchetype.BLOCKY, boulder_base_size
        )
    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)
    return stream.getvalue()


class SpriteBenchmark:
    """Benchmark runner for tree and boulder sprite generation."""

    TREE_BASE_SIZE = 20
    BOULDER_BASE_SIZE = 16

    def __init__(self, iterations: int) -> None:
        self.iterations = iterations
        self.results: dict[str, dict[str, float]] = {}

    def run(self) -> None:
        """Run timing benchmarks for all archetypes and batch simulation."""
        print("Sprite Generation Benchmark")
        print("=" * 60)
        print(f"Iterations per archetype: {self.iterations}")
        print()

        self._warmup()
        self._run_archetype_section(
            "Trees",
            TreeArchetype,
            generate_tree_sprite,
            self.TREE_BASE_SIZE,
            "tree",
        )
        print()
        self._run_archetype_section(
            "Boulders",
            BoulderArchetype,
            generate_boulder_sprite,
            self.BOULDER_BASE_SIZE,
            "boulder",
        )
        print()
        self._run_map_simulation()

    def _warmup(self) -> None:
        """Generate a few sprites to let numpy caches settle."""
        for archetype in TreeArchetype:
            generate_tree_sprite(0xDEAD, archetype, self.TREE_BASE_SIZE)
        for archetype in BoulderArchetype:
            generate_boulder_sprite(0xDEAD, archetype, self.BOULDER_BASE_SIZE)

    def _run_archetype_section(
        self,
        label: str,
        archetypes: type[TreeArchetype | BoulderArchetype],
        generate_fn: Callable[..., np.ndarray],
        base_size: int,
        result_prefix: str,
    ) -> None:
        """Run per-archetype timing for one sprite category."""
        print(f"{label} (base size: {base_size}px)")
        print(
            f"{'Archetype':>12} {'Avg (ms)':>10} {'Min (ms)':>10} "
            f"{'Max (ms)':>10} {'Sprites/s':>12}"
        )
        print("-" * 60)

        total_sprites = 0
        total_time_ms = 0.0

        for archetype in archetypes:
            stats = _benchmark_archetype(
                archetype, generate_fn, self.iterations, base_size
            )
            key = f"{result_prefix}_{archetype.value}"
            self.results[key] = stats
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

    def _run_map_simulation(self) -> None:
        """Simulate generating a full map's worth of sprites.

        Uses realistic tree/boulder ratios matching what controller.py
        produces for typical settlements on a 150x100 map.
        """
        # Map sizes: (label, trees, boulders)
        map_configs = [
            ("small", 150, 40),
            ("medium", 300, 80),
            ("large", 500, 130),
        ]
        tree_archetypes = list(TreeArchetype)
        boulder_archetypes = list(BoulderArchetype)

        print("Map Simulation (trees + boulders, simulating startup)")
        print(
            f"{'Map':>8} {'Trees':>6} {'Boulders':>9} "
            f"{'Total (ms)':>12} {'Per sprite (ms)':>16}"
        )
        print("-" * 56)

        for name, n_trees, n_boulders in map_configs:
            rng = np.random.default_rng(42)
            tree_seeds = rng.integers(0, 0xFFFFFFFF, size=n_trees, dtype=np.uint64)
            tree_arch_idx = rng.integers(0, len(tree_archetypes), size=n_trees)
            boulder_seeds = rng.integers(
                0, 0xFFFFFFFF, size=n_boulders, dtype=np.uint64
            )
            boulder_arch_idx = rng.integers(0, len(boulder_archetypes), size=n_boulders)

            start = time.perf_counter()
            for j in range(n_trees):
                generate_tree_sprite(
                    int(tree_seeds[j]),
                    tree_archetypes[int(tree_arch_idx[j])],
                    self.TREE_BASE_SIZE,
                )
            for j in range(n_boulders):
                generate_boulder_sprite(
                    int(boulder_seeds[j]),
                    boulder_archetypes[int(boulder_arch_idx[j])],
                    self.BOULDER_BASE_SIZE,
                )
            elapsed = (time.perf_counter() - start) * 1000.0

            total_sprites = n_trees + n_boulders
            per_sprite = elapsed / total_sprites
            self.results[f"map_{name}"] = {
                "trees": n_trees,
                "boulders": n_boulders,
                "total_ms": elapsed,
                "per_sprite_ms": per_sprite,
            }
            print(
                f"{name:>8} {n_trees:>6} {n_boulders:>9} "
                f"{elapsed:12.1f} {per_sprite:16.3f}"
            )

    def profile(self) -> None:
        """Run cProfile on the heaviest archetypes and print the top callers."""
        print()
        print("cProfile Hotspot Analysis (deciduous trees + blocky boulders)")
        print("=" * 60)
        report = _profile_sprite_generation(
            self.iterations, self.TREE_BASE_SIZE, self.BOULDER_BASE_SIZE
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
            old_avg = baseline[key].get("avg_ms") or baseline[key].get("per_sprite_ms")
            new_avg = current.get("avg_ms") or current.get("per_sprite_ms")
            if old_avg is None or new_avg is None or old_avg <= 0:
                continue

            ratio = old_avg / new_avg if new_avg > 0 else 0.0
            delta_pct = ((new_avg - old_avg) / old_avg) * 100.0
            trend = "faster" if ratio > 1.0 else "slower"
            print(
                f"{key:>20}: {new_avg:.3f}ms vs {old_avg:.3f}ms | "
                f"{ratio:.2f}x {trend} ({delta_pct:+.1f}%)"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tree and boulder sprite generation"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Sprites to generate per archetype (default: 200)",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip cProfile hotspot analysis",
    )
    parser.add_argument("--save", type=str, help="Save results to JSON")
    parser.add_argument("--compare", type=str, help="Compare against baseline JSON")
    args = parser.parse_args(argv)

    benchmark = SpriteBenchmark(iterations=args.iterations)
    benchmark.run()

    if not args.no_profile:
        benchmark.profile()

    if args.save:
        benchmark.save_results(args.save)

    if args.compare:
        benchmark.compare_with_baseline(args.compare)


if __name__ == "__main__":
    main()
