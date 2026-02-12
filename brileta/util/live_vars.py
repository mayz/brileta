from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Any, NamedTuple, TypeVar

from brileta.types import FloatRange

from .metrics import MostRecentNVar, StatsVar


class MetricSpec(NamedTuple):
    """Definition for a metric to register in batch."""

    name: str
    description: str
    num_samples: int = 100


@dataclass
class LiveVariable:
    """A variable exposed for live inspection and modification."""

    name: str
    description: str
    getter: Callable[[], Any]
    setter: Callable[[Any], None] | None = None
    formatter: Callable[[Any], str] | None = None
    display_decimals: int | None = None
    stats_var: StatsVar | None = None
    metric: bool = False
    value_range: FloatRange | None = None

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

    def supports_slider(self) -> bool:
        """Return ``True`` when this variable can be controlled by a slider."""
        return (
            self.setter is not None and self.value_range is not None and not self.metric
        )


T = TypeVar("T", bound="MostRecentNVar")


class LiveVariableRegistry:
    """Registry for all ``LiveVariable`` instances.

    When ``strict`` is ``True`` (the default), attempting to record a metric
    that has not been registered will raise immediately.  Test fixtures that
    clear the registry should set ``strict = False`` to avoid crashes from
    timing decorators that fire during teardown or unrelated test code.
    """

    def __init__(self) -> None:
        self._variables: dict[str, LiveVariable] = {}
        self._watched_variables: set[str] = set()
        self.strict: bool = True

    def register(
        self,
        name: str,
        getter: Callable[[], Any],
        setter: Callable[[Any], None] | None = None,
        *,
        description: str = "",
        formatter: Callable[[Any], str] | None = None,
        display_decimals: int | None = None,
        stats_var: StatsVar | None = None,
        metric: bool = False,
        value_range: FloatRange | None = None,
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
            display_decimals=display_decimals,
            stats_var=stats_var,
            metric=metric,
            value_range=value_range,
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
            metric=True,
        )

        self._variables[name] = live_var
        return live_var

    def register_metrics(self, specs: Sequence[MetricSpec]) -> None:
        """Register multiple metrics from a list of ``MetricSpec`` entries."""
        for spec in specs:
            self.register_metric(
                spec.name, description=spec.description, num_samples=spec.num_samples
            )

    def get_variable(self, name: str) -> LiveVariable | None:
        """Retrieve a registered ``LiveVariable`` by name."""
        return self._variables.get(name)

    def get_all_variables(self) -> list[LiveVariable]:
        """Return all registered variables, vars first then metrics, each group sorted."""
        return sorted(self._variables.values(), key=lambda v: (v.metric, v.name))

    def get_stats_variables(self) -> list[LiveVariable]:
        """Return all variables that track statistics."""
        return [var for var in self.get_all_variables() if var.has_stats()]

    def record_metric(self, name: str, value: float) -> None:
        """Record a value to a metric variable.

        Raises:
            KeyError: If the metric name is not registered or has no stats tracker.
        """
        var = self.get_variable(name)
        if var is None:
            raise KeyError(f"Metric '{name}' is not registered")
        if var.stats_var is None:
            raise KeyError(f"Variable '{name}' is not a metric (has no stats tracker)")
        var.record_value(value)

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


# Timing helper for recording wall-clock time to a metric.
# Works as both a context manager and a decorator:
#   with record_time_live_variable("time.render.cpu_ms"): ...
#   @record_time_live_variable("time.render.cpu_ms")
@contextmanager
def record_time_live_variable(metric_name: str):
    """Record elapsed wall-clock time (ms) to the named metric.

    In strict mode (the default), raises ``KeyError`` immediately if the metric
    is not registered.  This catches mistakes like forgetting to add a
    ``MetricSpec`` entry - the game will crash on the first frame rather than
    silently dropping timing data.

    When the registry has ``strict`` set to ``False`` (e.g. in test fixtures
    that clear the registry between tests), unregistered metrics are silently
    skipped so that timing decorators in production code don't break tests.
    """
    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start) * 1000
        ctx = nullcontext() if live_variable_registry.strict else suppress(KeyError)
        with ctx:
            live_variable_registry.record_metric(metric_name, elapsed_ms)
