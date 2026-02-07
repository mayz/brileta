"""Periodic live-metric dumping helpers.

This module provides a small runtime utility for appending selected live
variable values to a text file on a fixed cadence while the game runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from catley.util.live_vars import live_variable_registry


def _utc_now() -> datetime:
    """Return timezone-aware UTC time for on-disk timestamps."""
    return datetime.now(UTC)


@dataclass
class PeriodicMetricLogger:
    """Append selected live-variable values to a file at fixed intervals."""

    metric_names: tuple[str, ...]
    output_path: Path
    interval_seconds: float = 5.0
    reset_file_on_start: bool = True
    monotonic_now: Callable[[], float] = perf_counter
    timestamp_now: Callable[[], datetime] = _utc_now
    _next_write_time_s: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be > 0")
        if not self.metric_names:
            raise ValueError("metric_names must not be empty")
        if self.reset_file_on_start:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Start from a fresh file each run, then append periodic rows.
            self.output_path.write_text("", encoding="utf-8")

    def maybe_write(self) -> bool:
        """Write one sample row when the interval has elapsed.

        Returns ``True`` if a row was written in this call.
        """
        now_s = self.monotonic_now()
        if self._next_write_time_s is None:
            # Arm the periodic timer on first tick so the first write happens only
            # after one full interval of runtime samples.
            self._next_write_time_s = now_s + self.interval_seconds
            return False

        if now_s < self._next_write_time_s:
            return False

        self._next_write_time_s = now_s + self.interval_seconds
        self._append_row()
        return True

    def _append_row(self) -> None:
        """Append one CSV-like log row to the output file."""
        timestamp = self.timestamp_now().isoformat()
        fields = [timestamp]
        for name in self.metric_names:
            var = live_variable_registry.get_variable(name)
            if var is None:
                fields.append(f"{name}=<missing>")
                continue
            fields.append(f"{name}={var.get_value()}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as out:
            out.write(", ".join(fields))
            out.write("\n")
