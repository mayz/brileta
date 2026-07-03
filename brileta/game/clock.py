"""Game clock and day/night lighting derivation.

`GameClock` owns ``time_of_day`` (0.0-1.0, wrapping) and is ticked from the
real-time logic loop. ``compute_sky_lighting`` maps a time of day to the sun's
angles plus the ambient/intensity/color ramps that make dawn, day, dusk, and a
navigable moonlit night. Both are pure data with no rendering knowledge; the
controller reads them and pushes the values onto the sun light and lighting
system each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from brileta import colors, config
from brileta.types import DeltaTime


class GameClock:
    """A wrapping clock advancing ``time_of_day`` from real elapsed time.

    ``time_of_day`` runs 0.0 (midnight) through 0.5 (noon) back toward 1.0
    (midnight again). It advances by ``dt / DAY_LENGTH_SECONDS`` each tick, so
    a full cycle takes ``config.DAY_LENGTH_SECONDS`` of unpaused real time.
    """

    def __init__(
        self,
        time_of_day: float = config.DAY_START_TIME_OF_DAY,
        day_length_seconds: float = config.DAY_LENGTH_SECONDS,
    ) -> None:
        self.time_of_day = time_of_day % 1.0
        self.day_length_seconds = day_length_seconds

    def advance(self, dt: DeltaTime) -> None:
        """Advance the clock by ``dt`` real seconds, wrapping at a full day."""
        self.time_of_day = (self.time_of_day + dt / self.day_length_seconds) % 1.0


@dataclass(frozen=True)
class SkyLighting:
    """Derived global-lighting state for a given time of day."""

    azimuth_degrees: float
    elevation_degrees: float
    sun_intensity: float
    sun_color: colors.Color
    ambient_light: float


def compute_sky_lighting(time_of_day: float) -> SkyLighting:
    """Map ``time_of_day`` (0.0-1.0) to sun angles and ambient/color ramps.

    The sun is above the horizon for a ``DAY_FRACTION``-wide window centered on
    noon (0.5). Within it, elevation follows a cosine arc (0 at the horizon,
    peak at noon) and azimuth sweeps east -> south -> west. Sun intensity and
    ambient fill track how high the sun sits, and the sun color warms toward
    ``SUN_DAWN_COLOR`` as it nears the horizon. Outside the window it is night:
    the sun contributes nothing and ambient sits at the moonlight floor.
    """
    half_day = config.DAY_FRACTION / 2.0
    offset_from_noon = time_of_day - 0.5

    if abs(offset_from_noon) >= half_day:
        # Night: sun below the horizon, ambient clamped to a navigable floor.
        return SkyLighting(
            azimuth_degrees=270.0,
            elevation_degrees=0.0,
            sun_intensity=0.0,
            sun_color=config.SUN_COLOR,
            ambient_light=config.NIGHT_AMBIENT_LIGHT_LEVEL,
        )

    # frac: -1 at sunrise, 0 at noon, +1 at sunset.
    frac = offset_from_noon / half_day
    # altitude: 0 at the horizon, 1 at noon. Drives every daylight ramp.
    altitude = math.cos(frac * (math.pi / 2.0))

    elevation = config.SUN_NOON_ELEVATION_DEGREES * altitude
    azimuth = 180.0 + frac * 90.0  # 90=E (sunrise) -> 180=S (noon) -> 270=W (sunset)
    intensity = config.SUN_INTENSITY * altitude
    # Warm the sun as it drops; the exponent keeps midday neutral.
    warmth = (1.0 - altitude) ** 1.5
    sun_color = colors.lerp_color(config.SUN_COLOR, config.SUN_DAWN_COLOR, warmth)
    ambient = colors.lerp(
        config.NIGHT_AMBIENT_LIGHT_LEVEL, config.AMBIENT_LIGHT_LEVEL, altitude
    )

    return SkyLighting(
        azimuth_degrees=azimuth,
        elevation_degrees=elevation,
        sun_intensity=intensity,
        sun_color=sun_color,
        ambient_light=ambient,
    )
