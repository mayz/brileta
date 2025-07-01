from unittest.mock import patch

import pytest

from catley.util.clock import Clock


def test_clock_sync_calculates_delta_and_fps() -> None:
    times = iter([0.0, 0.1, 0.3])
    with patch("time.perf_counter", side_effect=lambda: next(times)):
        clock = Clock()
        dt1 = clock.sync()
        dt2 = clock.sync()

    assert dt1 == pytest.approx(0.1)
    assert dt2 == pytest.approx(0.2)
    assert clock.last_fps == pytest.approx(1 / dt2)
    assert clock.mean_fps == pytest.approx(1 / ((dt1 + dt2) / 2))
