#!/usr/bin/env python3
"""
Text Backend Performance Benchmark

Compares the performance of different text rendering backends in real-world scenarios.
Designed to be easily extensible for new backends.

Usage:
    python benchmark_text_backends.py [--iterations N] [--backend BACKEND] [--verbose]
"""

import argparse
import gc

# Add the project root to Python path so we can import catley modules
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import tcod
import tcod.context
from tcod.console import Console

from catley import colors
from catley.view.render.backends.pillow.canvas import PillowImageCanvas
from catley.view.render.backends.tcod.canvas import TCODConsoleCanvas
from catley.view.render.backends.tcod.renderer import TCODRenderer
from catley.view.render.renderer import Renderer


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""

    backend_name: str
    test_name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    ops_per_second: float
    memory_usage_mb: float | None = None


class BackendBenchmark(ABC):
    """Abstract base class for backend-specific benchmarks."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def setup(self, renderer: Renderer) -> Any:
        """Set up the backend for testing. Return backend instance."""
        pass

    @abstractmethod
    def cleanup(self, backend: Any) -> None:
        """Clean up backend resources."""
        pass

    @abstractmethod
    def configure_backend(
        self, backend: Any, width: int, height: int, tile_height: int
    ) -> None:
        """Configure backend dimensions and scaling."""
        pass

    @abstractmethod
    def draw_text_operation(
        self, backend: Any, x: int, y: int, text: str, color: colors.Color
    ) -> None:
        """Perform a single text draw operation."""
        pass

    @abstractmethod
    def frame_cycle(self, backend: Any) -> Any:
        """Perform begin_frame/end_frame cycle. Return any produced texture."""
        pass


class TCODBackendBenchmark(BackendBenchmark):
    """Benchmark for TCOD text backend."""

    def __init__(self):
        super().__init__("TCOD")

    def setup(self, renderer: Renderer) -> TCODConsoleCanvas:
        return TCODConsoleCanvas(renderer)

    def cleanup(self, backend: TCODConsoleCanvas) -> None:
        # TCOD backend doesn't need explicit cleanup
        pass

    def configure_backend(
        self, backend: TCODConsoleCanvas, width: int, height: int, tile_height: int
    ) -> None:
        backend.configure_dimensions(width, height)
        backend.configure_scaling(tile_height)

    def draw_text_operation(
        self, backend: TCODConsoleCanvas, x: int, y: int, text: str, color: colors.Color
    ) -> None:
        backend.draw_text(x, y, text, color)

    def frame_cycle(self, backend: TCODConsoleCanvas) -> Any:
        backend.begin_frame()
        return backend.end_frame()


class PillowBackendBenchmark(BackendBenchmark):
    """Benchmark for Pillow text backend."""

    def __init__(self):
        super().__init__("Pillow")

    def setup(self, renderer: Renderer) -> PillowImageCanvas:
        return PillowImageCanvas(renderer)

    def cleanup(self, backend: PillowImageCanvas) -> None:
        # Clean up any remaining PIL resources
        if hasattr(backend, "image") and backend.image is not None:
            img = backend.image
            assert img is not None
            img.close()

    def configure_backend(
        self, backend: PillowImageCanvas, width: int, height: int, tile_height: int
    ) -> None:
        backend.configure_dimensions(width, height)
        backend.configure_scaling(tile_height)

    def draw_text_operation(
        self, backend: PillowImageCanvas, x: int, y: int, text: str, color: colors.Color
    ) -> None:
        backend.draw_text(x, y, text, color)

    def frame_cycle(self, backend: PillowImageCanvas) -> Any:
        backend.begin_frame()
        return backend.end_frame()


# Registry of available benchmarks - easy to extend
AVAILABLE_BENCHMARKS = {
    "tcod": TCODBackendBenchmark(),
    "pillow": PillowBackendBenchmark(),
}


class TextBackendBenchmarkSuite:
    """Main benchmark suite that runs various tests on text backends."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[BenchmarkResult] = []

        # Initialize memory tracing if available
        try:
            import tracemalloc

            if not tracemalloc.is_tracing():
                tracemalloc.start()
        except ImportError:
            pass

        # Initialize TCOD context for testing
        self.console = Console(80, 50, order="F")
        self.context = tcod.context.new(
            console=self.console,
            width=800,
            height=600,
            title="Text Backend Benchmark",
            sdl_window_flags=tcod.context.SDL_WINDOW_HIDDEN,
        )
        self.renderer = TCODRenderer(self.context, self.console, (10, 12))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up benchmark resources."""
        if hasattr(self, "context"):
            self.context.close()

    @contextmanager
    def timer(self):
        """Context manager for precise timing."""
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.last_time = end - start

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB (approximate)."""
        try:
            # Try built-in resource module first (Unix-like systems)
            import resource

            return (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            )  # KB to MB on Linux
        except (ImportError, AttributeError):
            try:
                # Fall back to tracemalloc for Python memory tracking
                import tracemalloc

                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                current, peak = tracemalloc.get_traced_memory()
                return current / 1024 / 1024  # Bytes to MB
            except ImportError:
                # If all else fails, return 0
                return 0

    def run_benchmark_test(
        self, benchmark: BackendBenchmark, test_name: str, test_func, iterations: int
    ) -> BenchmarkResult:
        """Run a specific benchmark test and collect results."""
        if self.verbose:
            print(f"  Running {test_name}...")

        # Setup
        backend = benchmark.setup(self.renderer)
        times = []

        # Warmup
        for _ in range(min(5, iterations // 10)):
            test_func(backend)

        # Clear any warmup effects
        gc.collect()

        start_memory = self.get_memory_usage()

        # Actual benchmark
        for _ in range(iterations):
            with self.timer():
                test_func(backend)
            times.append(self.last_time)

        end_memory = self.get_memory_usage()
        memory_delta = (
            end_memory - start_memory if start_memory and end_memory else None
        )

        # Cleanup
        benchmark.cleanup(backend)
        gc.collect()

        # Calculate statistics
        total_time = sum(times)
        avg_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)
        ops_per_second = 1.0 / avg_time if avg_time > 0 else 0

        result = BenchmarkResult(
            backend_name=benchmark.name,
            test_name=test_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            ops_per_second=ops_per_second,
            memory_usage_mb=memory_delta,
        )

        self.results.append(result)
        return result

    def test_single_text_draw(
        self, backend_benchmark: BackendBenchmark, iterations: int
    ):
        """Test drawing a single piece of text (like FPS display)."""

        def test_func(backend):
            backend_benchmark.configure_backend(backend, 200, 100, 12)
            backend_benchmark.frame_cycle(backend)
            backend_benchmark.draw_text_operation(
                backend, 10, 10, "FPS: 60.0", colors.YELLOW
            )
            backend_benchmark.frame_cycle(backend)

        return self.run_benchmark_test(
            backend_benchmark, "Single Text Draw", test_func, iterations
        )

    def test_multiple_text_draws(
        self, backend_benchmark: BackendBenchmark, iterations: int
    ):
        """Test drawing multiple pieces of text (like inventory menu)."""

        def test_func(backend):
            backend_benchmark.configure_backend(backend, 400, 300, 12)
            backend_benchmark.frame_cycle(backend)

            # Simulate inventory-like text
            texts = [
                "a) Sword +1",
                "b) Health Potion",
                "c) Magic Ring of Power",
                "d) Leather Armor",
                "e) Bow of Accuracy",
                "f) Scroll of Fireball",
                "g) Gold Coins (50)",
                "h) Rope (10 feet)",
                "i) Torch",
                "j) Rations (3 days)",
            ]

            for i, text in enumerate(texts):
                backend_benchmark.draw_text_operation(
                    backend, 20, 20 + i * 15, text, colors.WHITE
                )

            backend_benchmark.frame_cycle(backend)

        return self.run_benchmark_test(
            backend_benchmark, "Multiple Text Draws", test_func, iterations
        )

    def test_long_text_draw(self, backend_benchmark: BackendBenchmark, iterations: int):
        """Test drawing long text (like message log)."""

        def test_func(backend):
            backend_benchmark.configure_backend(backend, 600, 200, 12)
            backend_benchmark.frame_cycle(backend)

            long_text = (
                "On offering to help the blind man, the man who then stole "
                "his car, had not, at that precise moment, had any evil "
                "intention, quite the contrary, what he did was nothing more "
                "than obey those feelings of generosity and altruism"
            )
            backend_benchmark.draw_text_operation(
                backend, 10, 10, long_text, colors.WHITE
            )

            backend_benchmark.frame_cycle(backend)

        return self.run_benchmark_test(
            backend_benchmark, "Long Text Draw", test_func, iterations
        )

    def test_frequent_resize(
        self, backend_benchmark: BackendBenchmark, iterations: int
    ):
        """Test performance with frequent resizing (window resize scenarios)."""
        # Counter to cycle through different sizes
        counter = [0]  # Use list so it's mutable in closure

        def test_func(backend):
            # Simulate window resize by changing dimensions
            sizes = [(200, 150), (400, 300), (300, 200), (500, 400)]
            size = sizes[counter[0] % len(sizes)]
            counter[0] += 1

            backend_benchmark.configure_backend(backend, size[0], size[1], 12)
            backend_benchmark.frame_cycle(backend)
            backend_benchmark.draw_text_operation(
                backend, 10, 10, f"Size: {size[0]}x{size[1]}", colors.CYAN
            )
            backend_benchmark.frame_cycle(backend)

        return self.run_benchmark_test(
            backend_benchmark, "Frequent Resize", test_func, iterations
        )

    def test_frame_cycle_overhead(
        self, backend_benchmark: BackendBenchmark, iterations: int
    ):
        """Test just the frame cycle overhead without text drawing."""

        def test_func(backend):
            backend_benchmark.configure_backend(backend, 300, 200, 12)
            texture = backend_benchmark.frame_cycle(backend)
            # Clean up texture if created
            if texture and hasattr(texture, "destroy"):
                texture.destroy()

        return self.run_benchmark_test(
            backend_benchmark, "Frame Cycle Overhead", test_func, iterations
        )

    def run_all_tests(self, backends: list[str], iterations: int):
        """Run all benchmark tests on specified backends."""
        print(f"Running text backend benchmarks with {iterations} iterations...")
        print("=" * 60)

        tests = [
            self.test_single_text_draw,
            self.test_multiple_text_draws,
            self.test_long_text_draw,
            self.test_frequent_resize,
            self.test_frame_cycle_overhead,
        ]

        for backend_name in backends:
            if backend_name not in AVAILABLE_BENCHMARKS:
                print(f"Warning: Unknown backend '{backend_name}', skipping.")
                continue

            print(f"\nTesting {backend_name.upper()} backend:")
            benchmark = AVAILABLE_BENCHMARKS[backend_name]

            for test in tests:
                try:
                    test(benchmark, iterations)
                except Exception as e:
                    print(f"  ERROR in {test.__name__}: {e}")
                    if self.verbose:
                        import traceback

                        traceback.print_exc()

    def print_results(self):
        """Print formatted benchmark results."""
        if not self.results:
            print("No results to display.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        # Group results by test
        by_test = {}
        for result in self.results:
            if result.test_name not in by_test:
                by_test[result.test_name] = []
            by_test[result.test_name].append(result)

        for test_name, test_results in by_test.items():
            print(f"\n{test_name}:")
            print("-" * 60)
            print(f"{'Backend':<12} {'Avg Time':<12} {'Ops/Sec':<12} {'Memory':<10}")
            print("-" * 60)

            for result in test_results:
                memory_str = (
                    f"{result.memory_usage_mb:.1f}MB"
                    if result.memory_usage_mb
                    else "N/A"
                )
                print(
                    f"{result.backend_name:<12} "
                    f"{result.avg_time * 1000:.3f}ms{'':<4} "
                    f"{result.ops_per_second:.1f}{'':<6} "
                    f"{memory_str:<10}"
                )

            # Show performance comparison
            if len(test_results) > 1:
                fastest = min(test_results, key=lambda r: r.avg_time)
                print(f"\nFastest: {fastest.backend_name}")
                for result in test_results:
                    if result != fastest:
                        ratio = result.avg_time / fastest.avg_time
                        print(
                            f"{result.backend_name} is {ratio:.1f}x slower "
                            f"than {fastest.backend_name}"
                        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark text rendering backends")
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=1000,
        help="Number of iterations per test (default: 1000)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        action="append",
        choices=list(AVAILABLE_BENCHMARKS.keys()),
        help="Backend to test (can be specified multiple times)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--export-csv", type=str, help="Export results to CSV file")

    args = parser.parse_args()

    # Default to all backends if none specified
    backends_to_test = args.backend or list(AVAILABLE_BENCHMARKS.keys())

    try:
        with TextBackendBenchmarkSuite(verbose=args.verbose) as suite:
            suite.run_all_tests(backends_to_test, args.iterations)
            suite.print_results()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
