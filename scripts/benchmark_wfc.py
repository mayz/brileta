#!/usr/bin/env python3
"""Benchmark native Wave Function Collapse solver performance."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from catley.environment.generators.pipeline.layers.terrain import (
    create_terrain_patterns,
)
from catley.environment.generators.wfc_solver import WFCSolver

GRID_SIZES: tuple[tuple[int, int], ...] = (
    (30, 30),
    (60, 60),
    (100, 80),
    (150, 150),
    (200, 200),
)


class WFCBenchmark:
    """Benchmark runner for native WFC implementation."""

    def __init__(self, iterations: int) -> None:
        self.iterations = iterations
        self.patterns = create_terrain_patterns()
        self.results: dict[str, dict[str, float]] = {}

    def _run_case(self, width: int, height: int) -> float:
        """Run one benchmark case and return average solve time in milliseconds."""
        elapsed_total = 0.0

        for i in range(self.iterations):
            rng = random.Random((width * 1_000_000) + (height * 1_000) + i)
            solver = WFCSolver(width, height, self.patterns, rng)

            start = time.perf_counter()
            solver.solve()
            elapsed_total += time.perf_counter() - start

        return (elapsed_total / self.iterations) * 1000.0

    def run(self) -> None:
        """Run all configured grid-size benchmarks."""
        print("WFC Benchmark (Native)")
        print("=" * 42)
        print(f"Iterations per size: {self.iterations}")
        print()
        print(f"{'Size':>12} {'Native (ms)':>14}")
        print("-" * 42)

        for width, height in GRID_SIZES:
            native_ms = self._run_case(width, height)

            size_key = f"{width}x{height}"
            self.results[size_key] = {
                "native_ms": native_ms,
            }

            print(f"{size_key:>12} {native_ms:14.2f}")

    def save_results(self, filename: str) -> None:
        """Save benchmark output to a JSON file."""
        with Path(filename).open("w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nSaved benchmark results to {filename}")

    def compare_with_baseline(self, baseline_file: str) -> None:
        """Compare current run with a saved baseline JSON file."""
        try:
            with Path(baseline_file).open() as f:
                baseline: dict[str, dict[str, float]] = json.load(f)
        except FileNotFoundError:
            print(f"\nBaseline file not found: {baseline_file}")
            return

        print(f"\nComparison vs baseline: {baseline_file}")
        print("=" * 64)

        for size_key, current in self.results.items():
            if size_key not in baseline:
                continue

            old_native = baseline[size_key].get("native_ms", 0.0)
            new_native = current["native_ms"]
            if old_native <= 0:
                continue

            delta_pct = ((new_native - old_native) / old_native) * 100.0
            speed_ratio = old_native / new_native if new_native > 0 else 0.0
            trend = "faster" if speed_ratio > 1.0 else "slower"

            print(
                f"{size_key:>12}: {new_native:8.2f}ms "
                f"vs {old_native:8.2f}ms | {speed_ratio:5.2f}x {trend} "
                f"({delta_pct:+6.1f}%)"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark native WFC")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of runs per grid size (default: 5)",
    )
    parser.add_argument("--save", type=str, help="Save current results to JSON")
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare current results against baseline JSON",
    )
    args = parser.parse_args(argv)

    benchmark = WFCBenchmark(iterations=args.iterations)
    benchmark.run()

    if args.save:
        benchmark.save_results(args.save)

    if args.compare:
        benchmark.compare_with_baseline(args.compare)


if __name__ == "__main__":
    main()
