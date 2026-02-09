from collections.abc import Iterator
from unittest.mock import patch

import pytest

from brileta.util import performance


@pytest.fixture(autouse=True)
def cleanup() -> Iterator[None]:
    """Ensure tracker is reset between tests."""
    performance.disable_performance_tracking()
    performance.reset_performance_data()
    yield
    performance.disable_performance_tracking()
    performance.reset_performance_data()


def test_measure_decorator_collects_stats() -> None:
    times = iter([0.0, 0.1])
    with patch("time.perf_counter", side_effect=lambda: next(times)):
        performance.enable_performance_tracking()

        @performance.measure("decorated")
        def do_work() -> None:
            pass

        do_work()

    stats = performance.get_stats("decorated")
    assert stats is not None
    assert stats.call_count == 1
    assert stats.total_time == pytest.approx(0.1)


def test_measure_block_and_frame_tracking() -> None:
    times = iter([0.0, 0.05, 0.1, 0.5])
    with patch("time.perf_counter", side_effect=lambda: next(times)):
        performance.enable_performance_tracking()
        performance.start_frame()
        with performance.measure_block("block"):
            pass
        performance.end_frame()

    block = performance.get_stats("block")
    assert block is not None
    assert block.call_count == 1
    assert block.total_time == pytest.approx(0.05)
    assert performance.get_frame_time("block") == pytest.approx(0.05)

    frame_total = performance.get_stats("frame_total")
    assert frame_total is not None
    assert frame_total.call_count == 1
    assert frame_total.total_time == pytest.approx(0.5)

    prefixed = performance.get_measurements_by_prefix("bl")
    assert set(prefixed) == {"block"}
    assert performance.get_frame_time_by_prefix("bl") == pytest.approx(0.05)
