from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

from .metrics import MostRecentNVar, StatsVar


@dataclass
class LiveVariable:
    """A variable exposed for live inspection and modification."""

    name: str
    description: str
    getter: Callable[[], Any]
    setter: Callable[[Any], None] | None = None
    formatter: Callable[[Any], str] | None = None
    stats_var: StatsVar | None = None

    def get_value(self) -> Any:
        """Return the current value using the getter."""
        return self.getter()

    def set_value(self, value: Any) -> bool:
        """Set the variable if writable.

        Returns ``True`` if the variable was set, or ``False`` if read-only.
        """
        if self.setter is None:
            return False
        self.setter(value)

        # If we're tracking stats over time, record the value
        if self.stats_var is not None and isinstance(value, int | float):
            self.stats_var.record(float(value))

        return True

    def record_value(self, value: float) -> None:
        """Record a value to the statistics tracker without setting the variable.

        Useful for metrics that are computed rather than set directly.
        """
        if self.stats_var is not None:
            self.stats_var.record(value)

    def get_stats_summary(self) -> str | None:
        """Get a formatted summary of statistics if available."""
        if self.stats_var is None:
            return None
        return self.stats_var.get_percentiles_string()

    def has_stats(self) -> bool:
        """Return True if this variable tracks statistics."""
        return self.stats_var is not None


T = TypeVar("T", bound="MostRecentNVar")


class LiveVariableRegistry:
    """Registry for all ``LiveVariable`` instances."""

    def __init__(self) -> None:
        self._variables: dict[str, LiveVariable] = {}
        self._watched_variables: set[str] = set()

    def register(
        self,
        name: str,
        getter: Callable[[], Any],
        setter: Callable[[Any], None] | None = None,
        *,
        description: str = "",
        formatter: Callable[[Any], str] | None = None,
        stats_var: StatsVar | None = None,
    ) -> None:
        """Register a new live variable."""
        if name in self._variables:
            raise ValueError(f"Live variable '{name}' already registered")
        self._variables[name] = LiveVariable(
            name=name,
            description=description,
            getter=getter,
            setter=setter,
            formatter=formatter,
            stats_var=stats_var,
        )

    def register_metric(
        self,
        name: str,
        description: str = "",
        stats_type: type[T] = MostRecentNVar,  # type: ignore[assignment]
        num_samples: int = 1000,
        formatter: Callable[[Any], str] | None = None,
    ) -> LiveVariable:
        """Register a metric variable that only tracks statistics.

        This creates a write-only variable focused on statistical tracking.
        Perfect for metrics like frame times, memory usage, etc.

        Args:
            name: Variable name
            description: Human-readable description
            stats_type: Type of StatsVar to use (MostRecentNVar or CumulativeVar)
            num_samples: Number of samples to track
            formatter: Optional custom formatter for display

        Returns:
            The created LiveVariable
        """
        if name in self._variables:
            raise ValueError(f"Live variable '{name}' already registered")

        stats_var = stats_type(num_samples)

        def getter() -> str:
            if stats_var.sample_count == 0:
                return "No samples"
            return stats_var.get_percentiles_string()

        def setter(value: float) -> None:
            stats_var.record(value)

        live_var = LiveVariable(
            name=name,
            description=description,
            getter=getter,
            setter=setter,
            formatter=formatter,
            stats_var=stats_var,
        )

        self._variables[name] = live_var
        return live_var

    def get_variable(self, name: str) -> LiveVariable | None:
        """Retrieve a registered ``LiveVariable`` by name."""
        return self._variables.get(name)

    def get_all_variables(self) -> list[LiveVariable]:
        """Return all registered variables sorted alphabetically."""
        return [self._variables[name] for name in sorted(self._variables)]

    def get_stats_variables(self) -> list[LiveVariable]:
        """Return all variables that track statistics."""
        return [var for var in self.get_all_variables() if var.has_stats()]

    def record_metric(self, name: str, value: float) -> bool:
        """Record a value to a metric variable.

        Returns True if the metric was found and recorded, False otherwise.
        """
        var = self.get_variable(name)
        if var is None or var.stats_var is None:
            return False
        var.record_value(value)
        return True

    # ------------------------------------------------------------------
    # Watch Management
    # ------------------------------------------------------------------
    def watch(self, name: str) -> None:
        """Add ``name`` to the watched set if the variable exists."""
        if name in self._variables:
            self._watched_variables.add(name)

    def unwatch(self, name: str) -> None:
        """Remove ``name`` from the watched set if present."""
        self._watched_variables.discard(name)

    def toggle_watch(self, name: str) -> None:
        """Toggle the watched state of ``name``."""
        if name in self._watched_variables:
            self._watched_variables.remove(name)
        elif name in self._variables:
            self._watched_variables.add(name)

    def is_watched(self, name: str) -> bool:
        """Return ``True`` if ``name`` is currently being watched."""
        return name in self._watched_variables

    def get_watched_variables(self) -> list[LiveVariable]:
        """Return watched variables sorted alphabetically by name."""
        names = sorted(self._watched_variables)
        return [self._variables[n] for n in names if n in self._variables]


# Global registry instance used throughout the application
live_variable_registry = LiveVariableRegistry()


# Simple timing helper
@contextmanager
def record_time_live_variable(metric_name: str):
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        live_variable_registry.record_metric(metric_name, elapsed_ms)
