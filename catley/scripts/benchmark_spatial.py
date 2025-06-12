#!/usr/bin/env python3
"""Standalone benchmark for :class:`SpatialHashGrid` performance."""

# ruff: noqa: E402  # Allow path setup before importing project modules

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add the project root to Python path so running as a script works when executed
# directly via ``python catley/scripts/benchmark_spatial.py``
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from catley.util.performance import (
    disable_performance_tracking,
    enable_performance_tracking,
    perf_tracker,
)
from catley.util.spatial import SpatialHashGrid


@dataclass
class MockActor:
    """Simple object with ``x``/``y`` used for benchmarking."""

    x: int
    y: int
    name: str = "test_actor"


class SpatialBenchmark:
    """Benchmark suite for :class:`SpatialHashGrid`."""

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def run_basic_benchmark(
        self, *, num_actors: int = 100, num_frames: int = 1000
    ) -> dict[str, float]:
        """Run a benchmark simulating typical frame usage."""

        print(f"Running basic benchmark: {num_actors} actors, {num_frames} frames...")
        grid = SpatialHashGrid(cell_size=16)
        actors: list[MockActor] = []

        for i in range(num_actors):
            actor = MockActor(
                x=random.randint(0, 79),
                y=random.randint(0, 42),
                name=f"actor_{i}",
            )
            actors.append(actor)
            grid.add(actor)

        start_time = time.perf_counter()

        for _ in range(num_frames):
            moving_actors = random.sample(actors, k=max(1, num_actors // 5))
            for actor in moving_actors:
                actor.x = max(0, min(79, actor.x + random.randint(-1, 1)))
                actor.y = max(0, min(42, actor.y + random.randint(-1, 1)))
                grid.update(actor)

            for _ in range(5):
                grid.get_in_radius(
                    random.randint(0, 79), random.randint(0, 42), radius=8
                )

            for _ in range(10):
                grid.get_in_radius(
                    random.randint(0, 79), random.randint(0, 42), radius=15
                )

            for _ in range(20):
                grid.get_at_point(random.randint(0, 79), random.randint(0, 42))

        total_time = time.perf_counter() - start_time

        results = {
            "total_time": total_time,
            "avg_frame_time": total_time / num_frames,
            "fps": num_frames / total_time,
            "operations_per_second": (num_frames * 35) / total_time,
            "time_per_operation": total_time / (num_frames * 35),
        }

        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg frame time: {results['avg_frame_time'] * 1000:.3f}ms")
        print(f"  Simulated FPS: {results['fps']:.1f}")
        print(f"  Operations/sec: {results['operations_per_second']:.0f}")

        return results

    def run_stress_test(
        self, *, num_actors: int = 1000, num_operations: int = 10000
    ) -> dict[str, float]:
        """Run a stress test with many actors and operations."""

        print(
            f"Running stress test: {num_actors} actors, {num_operations} operations..."
        )
        grid = SpatialHashGrid(cell_size=16)
        actors: list[MockActor] = []

        for i in range(num_actors):
            actor = MockActor(
                x=random.randint(0, 200),
                y=random.randint(0, 200),
                name=f"stress_actor_{i}",
            )
            actors.append(actor)
            grid.add(actor)

        op_times: dict[str, float] = {}

        start_time = time.perf_counter()
        for _ in range(num_operations // 3):
            grid.get_in_radius(
                random.randint(0, 200), random.randint(0, 200), radius=10
            )
        op_times["radius_queries"] = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for _ in range(num_operations // 3):
            x1, y1 = random.randint(0, 180), random.randint(0, 180)
            grid.get_in_bounds(x1, y1, x1 + 20, y1 + 20)
        op_times["bounds_queries"] = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for _ in range(num_operations // 3):
            actor = random.choice(actors)
            actor.x = random.randint(0, 200)
            actor.y = random.randint(0, 200)
            grid.update(actor)
        op_times["updates"] = time.perf_counter() - start_time

        total_time = sum(op_times.values())

        results = {
            "total_time": total_time,
            "operations_per_second": num_operations / total_time,
            "radius_query_time": op_times["radius_queries"],
            "bounds_query_time": op_times["bounds_queries"],
            "update_time": op_times["updates"],
            "avg_radius_query": op_times["radius_queries"] / (num_operations // 3),
            "avg_bounds_query": op_times["bounds_queries"] / (num_operations // 3),
            "avg_update": op_times["updates"] / (num_operations // 3),
        }

        print(f"  Total time: {total_time:.3f}s")
        print(f"  Operations/sec: {results['operations_per_second']:.0f}")
        print(
            f"  Avg radius query: {results['avg_radius_query'] * 1_000_000:.1f}\u03bcs"
        )
        print(
            f"  Avg bounds query: {results['avg_bounds_query'] * 1_000_000:.1f}\u03bcs"
        )
        print(f"  Avg update: {results['avg_update'] * 1_000_000:.1f}\u03bcs")

        return results

    def run_detailed_profiling(self) -> dict[str, Any]:
        """Run a smaller benchmark with performance tracking enabled."""

        print("Running detailed profiling...")
        perf_tracker.reset()
        enable_performance_tracking()

        try:
            results: dict[str, Any] = self.run_basic_benchmark(
                num_actors=200, num_frames=500
            )
            detailed_results: dict[str, Any] = {}
            for name, stats in perf_tracker.stats.items():
                if "spatial" in name:
                    detailed_results[name] = {
                        "call_count": stats.call_count,
                        "total_time": stats.total_time,
                        "avg_time": stats.avg_time,
                        "min_time": stats.min_time,
                        "max_time": stats.max_time,
                    }
            results["detailed_profiling"] = detailed_results

            print("  Detailed profiling results:")
            for name, data in detailed_results.items():
                print(
                    f"    {name}: {data['call_count']} calls, "
                    f"{data['avg_time'] * 1_000_000:.1f}\u03bcs avg, "
                    f"{data['total_time'] * 1000:.1f}ms total"
                )
            return results
        finally:
            disable_performance_tracking()

    def save_results(self, filename: str) -> None:
        """Save collected benchmark results to ``filename``."""

        with Path(filename).open("w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")

    def compare_with_baseline(self, baseline_file: str) -> None:
        """Compare current results with a baseline JSON file."""

        try:
            with Path(baseline_file).open() as f:
                baseline = json.load(f)
        except FileNotFoundError:
            print(f"Baseline file {baseline_file} not found.")
            return
        except Exception as exc:  # pragma: no cover - exercised by runtime only
            print(f"Error reading baseline {baseline_file}: {exc}")
            return

        print(f"\nComparison with baseline ({baseline_file}):")
        print("=" * 60)

        for test_name, current in self.results.items():
            if test_name in baseline and "total_time" in current:
                current_time = current["total_time"]
                baseline_time = baseline[test_name]["total_time"]
                if baseline_time <= 0:
                    continue
                speedup = baseline_time / current_time
                percentage = ((current_time - baseline_time) / baseline_time) * 100

                print(f"{test_name}:")
                print(f"  Current: {current_time:.3f}s")
                print(f"  Baseline: {baseline_time:.3f}s")
                if speedup > 1:
                    print(f"  Speedup: {speedup:.2f}x faster")
                else:
                    print(f"  Slowdown: {1 / speedup:.2f}x slower")
                print(f"  Change: {percentage:+.1f}%\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Spatial hash grid benchmark")
    parser.add_argument("--stress", action="store_true", help="Run stress test")
    parser.add_argument("--profile", action="store_true", help="Run detailed profiling")
    parser.add_argument("--save", type=str, help="Save results to file")
    parser.add_argument("--compare", type=str, help="Compare with baseline file")
    parser.add_argument("--actors", type=int, default=100, help="Number of actors")
    parser.add_argument("--frames", type=int, default=1000, help="Number of frames")
    args = parser.parse_args(argv)

    benchmark = SpatialBenchmark()

    print("Spatial Hash Grid Benchmark")
    print("=" * 50)

    benchmark.results["basic"] = benchmark.run_basic_benchmark(
        num_actors=args.actors, num_frames=args.frames
    )
    print()

    if args.stress:
        benchmark.results["stress"] = benchmark.run_stress_test()
        print()

    if args.profile:
        benchmark.results["profiling"] = benchmark.run_detailed_profiling()
        print()

    if args.save:
        benchmark.save_results(args.save)

    if args.compare:
        benchmark.compare_with_baseline(args.compare)

    print("Benchmark complete!")


if __name__ == "__main__":
    main()
