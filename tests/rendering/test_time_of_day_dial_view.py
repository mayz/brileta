"""Tests for the time-of-day HUD dial helpers."""

from __future__ import annotations

import pytest

from brileta import config
from brileta.game.clock import format_clock_time
from brileta.view.views.time_of_day_dial_view import (
    TimeOfDayDialView,
)


@pytest.mark.parametrize(
    ("time_of_day", "expected"),
    [
        (0.0, "12:00 am"),
        (0.25, "6:00 am"),
        (0.5, "12:00 pm"),
        (0.75, "6:00 pm"),
        (1.0, "12:00 am"),
    ],
)
def test_format_time_of_day_uses_am_pm_clock(time_of_day: float, expected: str) -> None:
    assert format_clock_time(time_of_day, use_24_hour=False) == expected


def test_dial_positions_put_noon_above_midnight() -> None:
    center = (100.0, 100.0)
    radius = 40.0

    noon_x, noon_y = TimeOfDayDialView._dial_position(0.5, center, radius)
    midnight_x, midnight_y = TimeOfDayDialView._dial_position(0.0, center, radius)
    sunrise_x, _sunrise_y = TimeOfDayDialView._dial_position(0.25, center, radius)
    sunset_x, _sunset_y = TimeOfDayDialView._dial_position(0.75, center, radius)

    assert noon_x == pytest.approx(center[0])
    assert noon_y < center[1]
    assert midnight_x == pytest.approx(center[0])
    assert midnight_y > center[1]
    assert sunrise_x < center[0]
    assert sunset_x > center[0]


def test_rotated_dial_keeps_current_time_at_top() -> None:
    center = (100.0, 100.0)
    radius = 40.0

    for current_time in (0.0, 0.25, 0.5, 0.75):
        current_x, current_y = TimeOfDayDialView._rotated_dial_position(
            current_time, current_time, center, radius
        )

        assert current_x == pytest.approx(center[0])
        assert current_y < center[1]


def test_icon_transition_crossfades_at_sunrise_and_sunset() -> None:
    half_day = config.DAY_FRACTION / 2.0
    sunrise = 0.5 - half_day
    sunset = 0.5 + half_day

    sunrise_transition = TimeOfDayDialView._current_icon_transition(sunrise)
    sunset_transition = TimeOfDayDialView._current_icon_transition(sunset)

    assert sunrise_transition == ("moon", "sun", pytest.approx(0.5))
    assert sunset_transition == ("sun", "moon", pytest.approx(0.5))


def test_icon_transition_is_limited_to_horizon_band() -> None:
    half_day = config.DAY_FRACTION / 2.0
    sunrise = 0.5 - half_day
    span = TimeOfDayDialView._icon_transition_half_span()

    assert TimeOfDayDialView._current_icon_transition(sunrise - span - 0.001) is None
    assert TimeOfDayDialView._current_icon_transition(sunrise + span + 0.001) is None
