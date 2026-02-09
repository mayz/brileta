from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brileta.__main__ import _parse_metric_names
from brileta.util.live_vars import live_variable_registry
from brileta.util.metric_dump import PeriodicMetricLogger


def test_parse_metric_names() -> None:
    assert _parse_metric_names(None) == ()
    assert _parse_metric_names("") == ()
    assert _parse_metric_names("time.render_ms,time.total_ms") == (
        "time.render_ms",
        "time.total_ms",
    )
    assert _parse_metric_names(" a, , b ") == ("a", "b")


def test_periodic_metric_logger_writes_row(tmp_path: Path) -> None:
    live_variable_registry.register("metric.a", getter=lambda: "v1")
    output_path = tmp_path / "logs" / "metrics.txt"
    now_s = 10.0

    logger = PeriodicMetricLogger(
        metric_names=("metric.a", "metric.missing"),
        output_path=output_path,
        interval_seconds=5.0,
        monotonic_now=lambda: now_s,
        timestamp_now=lambda: datetime(2026, 2, 7, 12, 0, tzinfo=UTC),
    )

    assert logger.maybe_write() is False
    assert output_path.exists() is True
    assert output_path.read_text(encoding="utf-8") == ""
    now_s = 15.0
    assert logger.maybe_write() is True
    text = output_path.read_text(encoding="utf-8").strip()
    assert text == ("2026-02-07T12:00:00+00:00, metric.a=v1, metric.missing=<missing>")


def test_periodic_metric_logger_respects_interval(tmp_path: Path) -> None:
    live_variable_registry.register("metric.tick", getter=lambda: "ok")
    output_path = tmp_path / "metrics.txt"

    now_s = 0.0

    def monotonic_now() -> float:
        return now_s

    logger = PeriodicMetricLogger(
        metric_names=("metric.tick",),
        output_path=output_path,
        interval_seconds=5.0,
        monotonic_now=monotonic_now,
        timestamp_now=lambda: datetime(2026, 2, 7, 12, 0, tzinfo=UTC),
    )

    assert logger.maybe_write() is False
    now_s = 1.0
    assert logger.maybe_write() is False
    now_s = 5.0
    assert logger.maybe_write() is True
    now_s = 9.9
    assert logger.maybe_write() is False
    now_s = 10.0
    assert logger.maybe_write() is True

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_periodic_metric_logger_resets_existing_file(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.txt"
    output_path.write_text("old\ncontent\n", encoding="utf-8")

    _ = PeriodicMetricLogger(
        metric_names=("metric.any",),
        output_path=output_path,
        interval_seconds=5.0,
        monotonic_now=lambda: 0.0,
        timestamp_now=lambda: datetime(2026, 2, 7, 12, 0, tzinfo=UTC),
    )

    assert output_path.read_text(encoding="utf-8") == ""
