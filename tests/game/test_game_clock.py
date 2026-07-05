"""Tests for the day/night game clock and sky-lighting derivation."""

from __future__ import annotations

import math

from brileta import config
from brileta.game.clock import (
    GameClock,
    compute_low_sun_time_fraction,
    compute_sky_lighting,
    format_clock_time,
)
from brileta.types import DeltaTime


def test_clock_wraps_after_one_day() -> None:
    clock = GameClock(time_of_day=0.0, day_length_seconds=100.0)
    clock.advance(DeltaTime(150.0))
    assert math.isclose(clock.time_of_day, 0.5)


def test_format_clock_time_uses_am_pm_clock() -> None:
    assert format_clock_time(0.0, use_24_hour=False) == "12:00 am"
    assert format_clock_time(0.25, use_24_hour=False) == "6:00 am"
    assert format_clock_time(0.5, use_24_hour=False) == "12:00 pm"
    assert format_clock_time(1.0, use_24_hour=False) == "12:00 am"


def test_format_clock_time_uses_24_hour_clock_in_24_hour_locales() -> None:
    assert format_clock_time(0.0, use_24_hour=True) == "00:00"
    assert format_clock_time(0.25, use_24_hour=True) == "06:00"
    assert format_clock_time(0.5, use_24_hour=True) == "12:00"
    assert format_clock_time(0.75, use_24_hour=True) == "18:00"


def test_low_sun_fraction_comes_from_shadow_fade_threshold() -> None:
    low_sun_fraction = compute_low_sun_time_fraction()
    half_day = config.DAY_FRACTION / 2.0

    assert 0.0 < low_sun_fraction < half_day


def test_noon_is_brightest_and_due_south() -> None:
    noon = compute_sky_lighting(0.5)
    assert noon.elevation_degrees == config.SUN_NOON_ELEVATION_DEGREES
    assert math.isclose(noon.azimuth_degrees, 180.0)  # due south
    assert math.isclose(noon.ambient_light, config.AMBIENT_LIGHT_LEVEL)
    assert noon.sun_intensity == config.SUN_INTENSITY


def test_night_is_dark_but_navigable() -> None:
    midnight = compute_sky_lighting(0.0)
    assert midnight.sun_intensity == 0.0
    assert midnight.ambient_light == config.NIGHT_AMBIENT_LIGHT_LEVEL


def test_ambient_never_drops_below_night_floor() -> None:
    floor = config.NIGHT_AMBIENT_LIGHT_LEVEL
    assert all(
        compute_sky_lighting(t / 1000.0).ambient_light >= floor for t in range(1000)
    )


def test_sun_sweeps_east_to_west_across_the_day() -> None:
    # Sunrise sits east of south; sunset sits west of it.
    half_day = config.DAY_FRACTION / 2.0
    morning = compute_sky_lighting(0.5 - half_day / 2.0)
    evening = compute_sky_lighting(0.5 + half_day / 2.0)
    assert morning.azimuth_degrees < 180.0 < evening.azimuth_degrees


def test_low_sun_is_warmer_than_noon_sun() -> None:
    # Redder (higher R relative to B) near dawn than at noon.
    half_day = config.DAY_FRACTION / 2.0
    dawn = compute_sky_lighting(0.5 - half_day * 0.95)
    noon = compute_sky_lighting(0.5)
    dawn_warmth = dawn.sun_color[0] - dawn.sun_color[2]
    noon_warmth = noon.sun_color[0] - noon.sun_color[2]
    assert dawn_warmth > noon_warmth
