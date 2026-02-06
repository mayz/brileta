#!/usr/bin/env python3
"""Comprehensive benchmark for :class:`SpatialHashGrid` performance.

Runs both realistic game simulation and stress tests automatically.
Provides detailed per-operation timing and recommendations for optimization.

Usage:
    # Full benchmark
    python catley/scripts/benchmark_spatial.py

    # Save baseline
    python catley/scripts/benchmark_spatial.py --save before.json

    # Compare with baseline
    python catley/scripts/benchmark_spatial.py --compare before.json
"""

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
    get_performance_report,
    perf_tracker,
)
from scripts.instrumented_spatial import InstrumentedSpatialHashGrid as SpatialHashGrid


@dataclass
class MockActor:
    """Mock actor for benchmarking."""

    x: int
    y: int
    name: str = "test_actor"

    def __hash__(self) -> int:
        # Hash based on name since it should be unique per actor
        return hash(self.name)

    def __eq__(self, other) -> bool:
        # Two actors are equal if they have the same name
        if not isinstance(other, MockActor):
            return False
        return self.name == other.name


class SpatialBenchmark:
    """Benchmark suite for :class:`SpatialHashGrid`."""

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def run_basic_benchmark(
        self, *, num_actors: int = 100, num_frames: int = 1000
    ) -> dict[str, Any]:
        """Run a benchmark simulating typical frame usage."""

        print(f"Running basic benchmark: {num_actors} actors, {num_frames} frames...")

        # Enable performance tracking for detailed per-operation timing
        enable_performance_tracking()
        perf_tracker.reset()

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

        # Get detailed spatial operation breakdown
        spatial_stats = perf_tracker.get_measurements_by_prefix("spatial_")
        spatial_total_time = sum(stats.total_time for stats in spatial_stats.values())

        # Compute metrics as local variables for type safety
        avg_frame_time = total_time / num_frames
        fps = num_frames / total_time
        operations_per_second = (num_frames * 35) / total_time
        time_per_operation = total_time / (num_frames * 35)
        spatial_percentage = (
            (spatial_total_time / total_time) * 100 if total_time > 0 else 0
        )

        results = {
            "total_time": total_time,
            "avg_frame_time": avg_frame_time,
            "fps": fps,
            "operations_per_second": operations_per_second,
            "time_per_operation": time_per_operation,
            "spatial_total_time": spatial_total_time,
            "spatial_percentage": spatial_percentage,
            "spatial_breakdown": {
                name: {
                    "avg_time": stats.avg_time,
                    "total_time": stats.total_time,
                    "call_count": stats.call_count,
                }
                for name, stats in spatial_stats.items()
            },
        }

        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg frame time: {avg_frame_time * 1000:.3f}ms")
        print(f"  Simulated FPS: {fps:.1f}")
        print(f"  Operations/sec: {operations_per_second:.0f}")
        print(f"  Spatial operations: {spatial_percentage:.1f}% of total time")

        print("  Detailed operation breakdown:")
        for name, stats in spatial_stats.items():
            print(
                f"    {name}: {stats.avg_time * 1_000_000:.1f}"
                f"µs avg, {stats.call_count} calls"
            )

        print("\n  Full Performance Report:")
        print(get_performance_report(filter_prefix="spatial_"))

        disable_performance_tracking()
        return results

    def run_stress_test(
        self, *, num_actors: int = 1000, num_operations: int = 10000
    ) -> dict[str, Any]:
        """Run a stress test with many actors and operations."""

        print(
            f"Running stress test: {num_actors} actors, {num_operations} operations..."
        )

        # Enable performance tracking for stress test too
        enable_performance_tracking()
        perf_tracker.reset()

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

        # Time different operation types separately (but still use instrumented grid)
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

        # Get instrumented breakdown
        spatial_stats = perf_tracker.get_measurements_by_prefix("spatial_")

        # Compute metrics as local variables for type safety
        ops_per_sec = num_operations / total_time if total_time > 0 else 0
        ops_divisor = num_operations // 3
        avg_radius_query = (
            op_times["radius_queries"] / ops_divisor if num_operations > 0 else 0
        )
        avg_bounds_query = (
            op_times["bounds_queries"] / ops_divisor if num_operations > 0 else 0
        )
        avg_update = op_times["updates"] / ops_divisor if num_operations > 0 else 0

        results = {
            "total_time": total_time,
            "operations_per_second": ops_per_sec,
            "radius_query_time": op_times["radius_queries"],
            "bounds_query_time": op_times["bounds_queries"],
            "update_time": op_times["updates"],
            "avg_radius_query": avg_radius_query,
            "avg_bounds_query": avg_bounds_query,
            "avg_update": avg_update,
            "instrumented_breakdown": {
                name: {
                    "avg_time": stats.avg_time,
                    "total_time": stats.total_time,
                    "call_count": stats.call_count,
                }
                for name, stats in spatial_stats.items()
            },
        }

        print(f"  Total time: {total_time:.3f}s")
        print(f"  Operations/sec: {ops_per_sec:.0f}")
        print(f"  Avg radius query: {avg_radius_query * 1_000_000:.1f}µs")
        print(f"  Avg bounds query: {avg_bounds_query * 1_000_000:.1f}µs")
        print(f"  Avg update: {avg_update * 1_000_000:.1f}µs")

        print("  Instrumented operation breakdown:")
        for name, stats in spatial_stats.items():
            print(
                f"    {name}: {stats.avg_time * 1_000_000:.1f}"
                f"µs avg, {stats.call_count} calls"
            )

        disable_performance_tracking()
        return results

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
                print(f"  Change: {percentage:+.1f}%")

                # Compare spatial breakdown if available
                if (
                    "spatial_breakdown" in current
                    and "spatial_breakdown" in baseline[test_name]
                ):
                    print("  Spatial operation changes:")
                    current_spatial = current["spatial_breakdown"]
                    baseline_spatial = baseline[test_name]["spatial_breakdown"]

                    for op_name in current_spatial:
                        if op_name in baseline_spatial:
                            curr_time = current_spatial[op_name]["avg_time"]
                            base_time = baseline_spatial[op_name]["avg_time"]
                            if base_time > 0:
                                op_speedup = base_time / curr_time
                                print(
                                    f"    {op_name}: {op_speedup:.2f}x "
                                    f"{'faster' if op_speedup > 1 else 'slower'}"
                                )
                print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive spatial hash grid benchmark"
    )
    parser.add_argument("--save", type=str, help="Save results to file")
    parser.add_argument("--compare", type=str, help="Compare with baseline file")
    parser.add_argument(
        "--actors", type=int, default=100, help="Number of actors for basic test"
    )
    parser.add_argument(
        "--frames", type=int, default=1000, help="Number of frames for basic test"
    )
    args = parser.parse_args(argv)

    benchmark = SpatialBenchmark()

    print("Comprehensive Spatial Hash Grid Benchmark")
    print("=" * 55)
    print("Running both realistic game simulation and stress tests...")
    print()

    # Always run basic benchmark (realistic game conditions)
    print("=== REALISTIC GAME SIMULATION ===")
    benchmark.results["basic"] = benchmark.run_basic_benchmark(
        num_actors=args.actors, num_frames=args.frames
    )
    print()

    # Always run stress test (performance limits)
    print("=== STRESS TEST (Performance Limits) ===")
    benchmark.results["stress"] = benchmark.run_stress_test()
    print()

    # Always show comprehensive summary
    print("=== SUMMARY ===")
    basic = benchmark.results["basic"]
    stress = benchmark.results["stress"]

    print("Realistic game performance:")
    print(f"  Frame time: {basic['avg_frame_time'] * 1000:.3f}ms")
    print(f"  Spatial overhead: {basic['spatial_percentage']:.1f}% of frame time")
    print(f"  Max sustainable FPS: {basic['fps']:.0f}")

    print("\nStress test performance:")
    print(f"  Operations per second: {stress['operations_per_second']:.0f}")

    # Safe handling of min/max operations
    instrumented_breakdown = stress.get("instrumented_breakdown", {})
    if instrumented_breakdown:
        avg_times = [op["avg_time"] for op in instrumented_breakdown.values()]
        if avg_times:  # Check if list is not empty
            print(f"  Fastest operation: {min(avg_times) * 1_000_000:.1f}µs")
            print(f"  Slowest operation: {max(avg_times) * 1_000_000:.1f}µs")
    else:
        print("  No instrumented operation data available")

    # Recommendation based on results
    spatial_percentage = basic.get("spatial_percentage", 0)
    print(
        f"\n💡 RECOMMENDATION: Spatial operations use only "
        f"{spatial_percentage:.1f}% of frame time."
    )

    if args.save:
        benchmark.save_results(args.save)

    if args.compare:
        benchmark.compare_with_baseline(args.compare)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
