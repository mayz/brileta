#!/usr/bin/env python3
"""
Realistic WGPU Performance Benchmark

This script measures performance in actual gameplay scenarios with a visible window,
providing realistic baseline metrics for performance tuning.

Usage:
    uv run python scripts/benchmark_realistic_performance.py [--duration SECONDS]
"""

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from catley import config
from catley.app import AppConfig
from catley.backends.glfw.app import GlfwApp
from catley.game.lights import DirectionalLight, DynamicLight
from catley.types import DeltaTime


@dataclass
class RealisticBenchmarkResult:
    """Results from a realistic performance test."""

    test_name: str
    light_count: int
    shadow_enabled: bool
    duration_seconds: float
    total_frames: int
    avg_fps: float
    min_fps: float
    max_fps: float
    p95_fps: float  # 95th percentile
    p99_fps: float  # 99th percentile
    frame_time_ms_avg: float
    frame_time_ms_p95: float
    frame_time_ms_p99: float
    memory_mb: float

    # Real-world metrics
    input_lag_ms: float
    lighting_overhead_percent: float
    frame_drops: int  # Frames that took >16.67ms (60fps target)


class RealisticPerformanceBenchmark:
    """Realistic performance benchmark with visible window and full game simulation."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[RealisticBenchmarkResult] = []

        # Test configuration - realistic settings
        self.viewport_size = (80, 50)
        self.target_fps = 60  # Realistic target
        self.frame_time_target_ms = 1000.0 / self.target_fps  # 16.67ms

        # Create VISIBLE window for realistic testing
        app_config = AppConfig(
            width=config.SCREEN_WIDTH,
            height=config.SCREEN_HEIGHT,
            title="Realistic Performance Benchmark - Close window to exit",
            vsync=True,  # Use vsync for realistic conditions
            resizable=True,
            maximized=False,
            fullscreen=False,
        )

        self.app = GlfwApp(app_config)
        assert self.app.controller is not None
        self.controller = self.app.controller

        if self.verbose:
            print("Initialized realistic performance benchmark")
            print("Close the game window when benchmark completes")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if hasattr(resource, "getrusage"):
                ru = resource.getrusage(resource.RUSAGE_SELF)
                if sys.platform == "darwin":
                    return ru.ru_maxrss / 1024 / 1024  # bytes to MB
                return ru.ru_maxrss / 1024  # KB to MB
            return 0.0
        except Exception:
            return 0.0

    def setup_realistic_scenario(self, light_count: int, enable_shadows: bool) -> None:
        """Set up a realistic game scenario with the specified lighting."""
        # Clear existing lights
        self.controller.gw.lights.clear()

        # Add player torch
        player_torch = DynamicLight.create_player_torch(self.controller.gw.player)
        self.controller.gw.add_light(player_torch)

        # Add sun if enabled (uses config defaults)
        if config.SUN_ENABLED:
            self.controller.gw.add_light(DirectionalLight.create_sun())

        # Add realistic lights around the player's vicinity
        player_x, player_y = self.controller.gw.player.x, self.controller.gw.player.y
        view_radius = 25  # Lights within view

        # Create random number generator
        rng = np.random.default_rng()

        lights_added = 0
        for _ in range(light_count):
            # Place lights around the player (realistic scenario)
            angle = rng.uniform(0, 2 * np.pi)
            distance = rng.uniform(3, view_radius)
            x = int(player_x + distance * np.cos(angle))
            y = int(player_y + distance * np.sin(angle))

            # Ensure within map bounds
            x = max(1, min(self.controller.gw.game_map.width - 2, x))
            y = max(1, min(self.controller.gw.game_map.height - 2, y))

            # Create varied lights
            if lights_added % 4 == 0:
                light = DynamicLight(
                    position=(x, y),
                    radius=6,
                    color=(255, 200, 100),
                    flicker_enabled=True,
                    flicker_speed=2.0,
                    min_brightness=0.8,
                    max_brightness=1.0,
                )
            elif lights_added % 4 == 1:
                light = DynamicLight(
                    position=(x, y),
                    radius=10,
                    color=(255, 255, 200),
                    flicker_enabled=False,
                )
            elif lights_added % 4 == 2:
                light = DynamicLight(
                    position=(x, y),
                    radius=8,
                    color=(100, 255, 100),
                    flicker_enabled=True,
                    flicker_speed=1.5,
                    min_brightness=0.7,
                    max_brightness=1.0,
                )
            else:
                light = DynamicLight(
                    position=(x, y),
                    radius=7,
                    color=(100, 150, 255),
                    flicker_enabled=False,
                )

            self.controller.gw.add_light(light)
            lights_added += 1

        # Configure shadows
        config.SHADOWS_ENABLED = enable_shadows
        if self.controller.gw.lighting_system:
            self.controller.gw.lighting_system.on_global_light_changed()

        if self.verbose:
            print(
                f"  Set up realistic scenario: {lights_added} lights, "
                f"shadows: {enable_shadows}"
            )

    def run_realistic_test(
        self,
        test_name: str,
        light_count: int,
        enable_shadows: bool,
        duration_seconds: float,
    ) -> RealisticBenchmarkResult:
        """Run a realistic performance test with actual game loop."""
        if self.verbose:
            print(f"Running {test_name} for {duration_seconds}s...")
            print("  (Game window is visible - this is measuring real performance)")

        # Setup scenario
        self.setup_realistic_scenario(light_count, enable_shadows)

        # Collect metrics
        frame_times = []
        fps_values = []
        input_latencies = []

        start_time = time.perf_counter()
        last_frame_time = start_time
        total_frames = 0

        # Run realistic game loop
        while time.perf_counter() - start_time < duration_seconds:
            frame_start = time.perf_counter()

            # Simulate input processing latency
            input_start = time.perf_counter()

            # Process events (this is what adds real latency)
            import glfw

            glfw.poll_events()

            # Check for window close
            if glfw.window_should_close(self.app.window):
                if self.verbose:
                    print("  Window closed - ending test early")
                break

            input_end = time.perf_counter()
            input_latencies.append((input_end - input_start) * 1000)

            # Update game logic using realistic timing
            dt = DeltaTime(frame_start - last_frame_time)
            self.app.update_game_logic(dt)

            # Render with real presentation to screen
            self.app.render_frame()

            frame_end = time.perf_counter()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time * 1000)  # Convert to ms
            fps_values.append(1.0 / frame_time if frame_time > 0 else 0)

            last_frame_time = frame_start
            total_frames += 1

        # Calculate statistics
        actual_duration = time.perf_counter() - start_time
        avg_fps = total_frames / actual_duration

        # Calculate percentiles
        fps_sorted = sorted(fps_values)
        p95_fps = float(np.percentile(fps_sorted, 5))  # 5th percentile (95% above this)
        p99_fps = float(np.percentile(fps_sorted, 1))  # 1st percentile (99% above this)

        frame_times_sorted = sorted(frame_times)
        p95_frame_time = float(np.percentile(frame_times_sorted, 95))
        p99_frame_time = float(np.percentile(frame_times_sorted, 99))

        # Count frame drops (>16.67ms for 60fps)
        frame_drops = sum(1 for ft in frame_times if ft > self.frame_time_target_ms)

        # Calculate lighting overhead (estimate)
        lighting_overhead = 0.0  # Would need profiling to measure accurately

        result = RealisticBenchmarkResult(
            test_name=test_name,
            light_count=light_count,
            shadow_enabled=enable_shadows,
            duration_seconds=actual_duration,
            total_frames=total_frames,
            avg_fps=avg_fps,
            min_fps=min(fps_values) if fps_values else 0,
            max_fps=max(fps_values) if fps_values else 0,
            p95_fps=p95_fps,
            p99_fps=p99_fps,
            frame_time_ms_avg=float(np.mean(frame_times)),
            frame_time_ms_p95=p95_frame_time,
            frame_time_ms_p99=p99_frame_time,
            memory_mb=self.get_memory_usage(),
            input_lag_ms=float(np.mean(input_latencies)),
            lighting_overhead_percent=lighting_overhead,
            frame_drops=frame_drops,
        )

        self.results.append(result)

        if self.verbose:
            print(f"  Completed: {avg_fps:.1f} FPS avg, {frame_drops} frame drops")

        return result

    def run_all_tests(self, duration_per_test: float) -> None:
        """Run realistic performance tests."""
        print("Starting Realistic WGPU Performance Benchmark")
        print("=" * 60)
        print("This benchmark uses a VISIBLE window for realistic measurements")
        print(f"Each test runs for {duration_per_test}s")
        print("Close the window to exit early")
        print()

        # Realistic test scenarios
        scenarios = [
            (10, False, "10 lights (no shadows)"),
            (10, True, "10 lights (with shadows)"),
            (25, False, "25 lights (no shadows)"),
            (25, True, "25 lights (with shadows)"),
            (50, False, "50 lights (no shadows)"),
            (50, True, "50 lights (with shadows)"),
        ]

        for light_count, shadows, name in scenarios:
            try:
                self.run_realistic_test(name, light_count, shadows, duration_per_test)

                # Check if window was closed
                import glfw

                if glfw.window_should_close(self.app.window):
                    break

            except Exception as e:
                print(f"ERROR in {name}: {e}")
                if self.verbose:
                    import traceback

                    traceback.print_exc()

    def print_results(self) -> None:
        """Print realistic benchmark results."""
        if not self.results:
            print("No results to display.")
            return

        print("\n" + "=" * 80)
        print("REALISTIC WGPU PERFORMANCE RESULTS")
        print("=" * 80)

        # Print summary table
        header = (
            f"{'Test':<25} {'Avg FPS':<8} {'P95 FPS':<8} "
            f"{'Frame Drops':<11} {'Mem MB':<8}"
        )
        print(header)
        print("-" * len(header))

        for result in self.results:
            print(
                f"{result.test_name:<25} "
                f"{result.avg_fps:>6.1f}  "
                f"{result.p95_fps:>6.1f}  "
                f"{result.frame_drops:>9}  "
                f"{result.memory_mb:>6.1f}"
            )

        print()
        print("DETAILED METRICS")
        print("-" * 40)

        for result in self.results:
            print(f"\n{result.test_name}:")
            print(f"  Average FPS: {result.avg_fps:.1f}")
            print(f"  P95 FPS: {result.p95_fps:.1f} (95% of frames above this)")
            print(f"  P99 FPS: {result.p99_fps:.1f} (99% of frames above this)")
            print(f"  Frame Time P95: {result.frame_time_ms_p95:.2f}ms")
            print(f"  Frame Time P99: {result.frame_time_ms_p99:.2f}ms")
            print(
                f"  Frame Drops: {result.frame_drops} "
                f"(>{self.frame_time_target_ms:.1f}ms)"
            )
            print(f"  Input Latency: {result.input_lag_ms:.2f}ms")
            print(f"  Memory: {result.memory_mb:.1f}MB")

    def export_results(self, filename: str) -> None:
        """Export results to JSON."""
        output_path = Path(filename)

        # WGPU backend - GPU info would need wgpu adapter queries
        gpu_backend = "WGPU (Metal/Vulkan/D3D12)"
        gpu_renderer = "See wgpu adapter info"

        export_data = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_type": "realistic_performance",
                "target_fps": self.target_fps,
                "vsync_enabled": True,
                "visible_window": True,
                "gpu_backend": gpu_backend,
                "gpu_renderer": gpu_renderer,
            },
            "results": [asdict(result) for result in self.results],
        }

        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Realistic WGPU performance benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration for each test in seconds (default: 10.0)",
    )
    parser.add_argument("--export-json", type=str, help="Export results to JSON file")

    args = parser.parse_args()

    try:
        benchmark = RealisticPerformanceBenchmark(verbose=args.verbose)
        benchmark.run_all_tests(args.duration)
        benchmark.print_results()

        if args.export_json:
            benchmark.export_results(args.export_json)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
