"""Tests for the day/night game clock and sky-lighting derivation."""

from __future__ import annotations

import math

from brileta import config
from brileta.game.clock import GameClock, compute_sky_lighting
from brileta.types import DeltaTime


def test_clock_wraps_after_one_day() -> None:
    clock = GameClock(time_of_day=0.0, day_length_seconds=100.0)
    clock.advance(DeltaTime(150.0))
    assert math.isclose(clock.time_of_day, 0.5)


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
