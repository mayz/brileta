"""
Performance measurement and profiling utilities.

Provides decorators and context managers for measuring execution time
of functions and code blocks, with aggregation and reporting capabilities.

This is a generic measurement system that can be used across any part of
the codebase. It tracks timing statistics and provides reporting without
knowledge of specific subsystems.

Usage Examples:
    # Enable tracking
    enable_performance_tracking()

    # Method decoration
    @measure("database_query")
    def query_database():
        pass

    # Code block measurement
    with measure_block("complex_calculation"):
        # expensive operations here
        pass

    # Frame-based tracking
    start_frame()
    # ... frame operations ...
    end_frame()

    # Get results
    report = get_performance_report()
    print(report)
"""

import functools
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class PerformanceStats:
    """Statistics for a measured operation."""

    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_measurement(self, duration: float) -> None:
        """Add a new timing measurement."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)

    @property
    def avg_time(self) -> float:
        """Average time per call across all measurements."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def recent_avg_time(self) -> float:
        """Average of recent measurements (last 100 calls)."""
        if not self.recent_times:
            return 0.0
        return sum(self.recent_times) / len(self.recent_times)

    @property
    def calls_per_second(self) -> float:
        """Estimated calls per second based on recent average time."""
        if not self.recent_times or len(self.recent_times) < 2:
            return 0.0
        return 1.0 / self.recent_avg_time if self.recent_avg_time > 0 else 0.0


class PerformanceTracker:
    """Core performance measurement system.

    Tracks timing statistics for named operations and provides
    aggregation and reporting capabilities. Supports both function
    decoration and code block measurement.
    """

    def __init__(self) -> None:
        self.stats: dict[str, PerformanceStats] = {}
        self.enabled = False
        self.frame_start_time: float | None = None
        self.frame_stats: dict[str, float] = defaultdict(float)

    def enable(self) -> None:
        """Enable performance tracking.

        When disabled, decorators and context managers have minimal overhead.
        """
        self.enabled = True

    def disable(self) -> None:
        """Disable performance tracking."""
        self.enabled = False

    def reset(self) -> None:
        """Reset all collected statistics."""
        self.stats.clear()
        self.frame_stats.clear()

    def start_frame(self) -> None:
        """Mark the start of a new frame for per-frame statistics.

        Call this at the beginning of each frame/iteration to enable
        frame-level performance tracking.
        """
        self.frame_start_time = time.perf_counter()
        self.frame_stats.clear()

    def end_frame(self) -> None:
        """Mark the end of a frame and update frame statistics.

        Automatically tracks total frame time and calculates frame-level
        performance metrics.
        """
        if self.frame_start_time is None:
            return

        frame_duration = time.perf_counter() - self.frame_start_time

        # Add frame-level measurement
        if "frame_total" not in self.stats:
            self.stats["frame_total"] = PerformanceStats("frame_total")
        self.stats["frame_total"].add_measurement(frame_duration)

    def measure(self, name: str):
        """Decorator to measure function execution time.

        Tracks timing statistics for the decorated function under the given name.
        Statistics include total time, call count, average time, min/max times,
        and recent performance trends.

        Args:
            name: Unique identifier for this measurement. Use descriptive names
                  that clearly indicate what's being measured. Examples:
                  - "spatial_get_in_radius"
                  - "render_particles"
                  - "pathfind_a_star"
                  - "database_query"

        Returns:
            Decorated function that automatically tracks execution time.

        Usage:
            @measure("my_function")
            def my_function():
                # function code here
                pass

            # Or with class methods:
            class MyClass:
                @measure("myclass_process")
                def process(self):
                    # method code here
                    pass

        Note:
            Only measures when performance tracking is enabled via
            enable_performance_tracking(). Has minimal overhead when disabled.

            If the same name is used for multiple functions, their statistics
            will be combined, which can be useful for measuring total time
            spent in a category of operations.
        """

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                start_time = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time

                    # Update global stats
                    if name not in self.stats:
                        self.stats[name] = PerformanceStats(name)
                    self.stats[name].add_measurement(duration)

                    # Update frame stats
                    self.frame_stats[name] += duration

            return wrapper

        return decorator

    @contextmanager
    def measure_block(self, name: str):
        """Context manager to measure a block of code.

        Tracks timing statistics for the code block under the given name.
        Useful for measuring performance of code sections that can't be
        easily decorated.

        Args:
            name: Unique identifier for this measurement. Use descriptive
                  names that clearly indicate what's being measured.

        Usage:
            with measure_block("complex_calculation"):
                # expensive operations here
                result = complex_calculation()

            with measure_block("file_io"):
                data = load_file("large_file.dat")
                process_data(data)

        Note:
            Only measures when performance tracking is enabled. Has minimal
            overhead when disabled.
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time

            if name not in self.stats:
                self.stats[name] = PerformanceStats(name)
            self.stats[name].add_measurement(duration)
            self.frame_stats[name] += duration

    def get_stats(self, name: str) -> PerformanceStats | None:
        """Get statistics for a specific measurement.

        Args:
            name: The measurement name to look up.

        Returns:
            PerformanceStats object with timing data, or None if not found.
        """
        return self.stats.get(name)

    def get_frame_time(self, name: str) -> float:
        """Get total time spent in a measurement during the current frame.

        Args:
            name: The measurement name to look up.

        Returns:
            Total time in seconds spent in this measurement during the
            current frame. Returns 0.0 if no measurements found.
        """
        return self.frame_stats.get(name, 0.0)

    def get_measurements_by_prefix(self, prefix: str) -> dict[str, PerformanceStats]:
        """Get all measurements whose names start with the given prefix.

        Useful for analyzing related operations, e.g., all "spatial_" operations
        or all "render_" operations.

        Args:
            prefix: The prefix to filter by.

        Returns:
            Dictionary mapping measurement names to their PerformanceStats.
        """
        return {
            name: stats for name, stats in self.stats.items() if name.startswith(prefix)
        }

    def get_frame_time_by_prefix(self, prefix: str) -> float:
        """Get total frame time for all measurements with the given prefix.

        Args:
            prefix: The prefix to filter by.

        Returns:
            Total time in seconds for all matching measurements in current frame.
        """
        return sum(
            time for name, time in self.frame_stats.items() if name.startswith(prefix)
        )

    def get_report(self, sort_by: str = "total_time", filter_prefix: str = "") -> str:
        """Generate a formatted performance report.

        Args:
            sort_by: Field to sort by. Options: 'total_time', 'avg_time',
                    'call_count', 'name'. Defaults to 'total_time'.
            filter_prefix: If provided, only include measurements whose names
                          start with this prefix.

        Returns:
            Formatted string report showing performance statistics.
        """
        if not self.stats:
            return "No performance data collected."

        # Filter statistics if prefix provided
        if filter_prefix:
            stats_to_show = [
                stats
                for name, stats in self.stats.items()
                if name.startswith(filter_prefix)
            ]
            title = f"Performance Report (filtered by '{filter_prefix}')"
        else:
            stats_to_show = list(self.stats.values())
            title = "Performance Report"

        if not stats_to_show:
            return f"No performance data found for prefix '{filter_prefix}'."

        # Sort statistics
        try:
            sorted_stats = sorted(
                stats_to_show, key=lambda s: getattr(s, sort_by, 0), reverse=True
            )
        except AttributeError:
            sorted_stats = sorted(
                stats_to_show, key=lambda s: s.total_time, reverse=True
            )

        lines = [title, "=" * len(title)]
        lines.append(
            f"{'Name':<25} {'Calls':<8} {'Total(ms)':<10} "
            f"{'Avg(µs)':<10} {'Recent(µs)':<12}"
        )
        lines.append("-" * 75)
        lines.extend(
            f"{stat.name:<25} "
            f"{stat.call_count:<8} "
            f"{stat.total_time * 1000:<10.2f} "
            f"{stat.avg_time * 1000000:<10.1f} "
            f"{stat.recent_avg_time * 1000000:<12.1f}"
            for stat in sorted_stats
        )

        return "\n".join(lines)


# Global performance tracker instance
perf_tracker = PerformanceTracker()


# Convenience functions for common usage patterns
def enable_performance_tracking() -> None:
    """Enable global performance tracking."""
    perf_tracker.enable()


def disable_performance_tracking() -> None:
    """Disable global performance tracking."""
    perf_tracker.disable()


def reset_performance_data() -> None:
    """Reset all collected performance data."""
    perf_tracker.reset()


def measure(name: str):
    """Decorator for measuring function performance.

    See PerformanceTracker.measure() for detailed documentation.
    """
    return perf_tracker.measure(name)


def measure_block(name: str):
    """Context manager for measuring code block performance.

    See PerformanceTracker.measure_block() for detailed documentation.
    """
    return perf_tracker.measure_block(name)


def start_frame() -> None:
    """Mark the start of a new frame for frame-level tracking."""
    perf_tracker.start_frame()


def end_frame() -> None:
    """Mark the end of a frame and update frame statistics."""
    perf_tracker.end_frame()


def get_performance_report(sort_by: str = "total_time", filter_prefix: str = "") -> str:
    """Get a formatted performance report.

    Args:
        sort_by: Field to sort by ('total_time', 'avg_time', 'call_count', 'name').
        filter_prefix: Only include measurements starting with this prefix.

    Returns:
        Formatted performance report string.
    """
    return perf_tracker.get_report(sort_by, filter_prefix)


def get_stats(name: str) -> PerformanceStats | None:
    """Get statistics for a specific measurement."""
    return perf_tracker.get_stats(name)


def get_frame_time(name: str) -> float:
    """Get time spent in a measurement during current frame."""
    return perf_tracker.get_frame_time(name)


def get_measurements_by_prefix(prefix: str) -> dict[str, PerformanceStats]:
    """Get all measurements whose names start with the given prefix."""
    return perf_tracker.get_measurements_by_prefix(prefix)


def get_frame_time_by_prefix(prefix: str) -> float:
    """Get total frame time for measurements with the given prefix."""
    return perf_tracker.get_frame_time_by_prefix(prefix)
