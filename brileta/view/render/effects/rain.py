"""Rain effect configuration and animation state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from brileta import colors, config
from brileta.types import DeltaTime
from brileta.util import rng


class _UniformRNG(Protocol):
    """Minimal RNG interface required for rain wind sampling."""

    def uniform(self, a: float, b: float) -> float: ...


def _clamp_angle(angle_rad: float) -> float:
    """Clamp rain angle to a physically plausible downward tilt."""
    max_abs = float(config.RAIN_ANGLE_MAX_ABS_RAD)
    return max(-max_abs, min(max_abs, float(angle_rad)))


def _normalized_spacing(value: float, lo: float, hi: float) -> float:
    """Map spacing to 0..1 where smaller spacing means denser rain."""
    clamped = max(lo, min(hi, float(value)))
    if hi <= lo:
        return 0.0
    return (hi - clamped) / (hi - lo)


def _gust_envelope(progress: float) -> float:
    """Asymmetric attack/decay envelope for gust events."""
    p = max(0.0, min(1.0, progress))
    attack = 0.22
    if p <= attack:
        return p / attack
    decay_t = (p - attack) / max(1e-6, 1.0 - attack)
    return 1.0 - decay_t


@dataclass
class RainConfig:
    """Runtime-tunable rain settings consumed by world rendering."""

    enabled: bool = False
    intensity: float = 0.4
    angle: float = 0.15
    drop_length: float = 0.8
    drop_speed: float = 25.0
    drop_spacing: float = 1.35
    stream_spacing: float = 0.33
    color: colors.Color = (180, 200, 220)

    @classmethod
    def from_config(cls) -> RainConfig:
        """Build rain settings from global config defaults."""
        return cls(
            enabled=bool(config.RAIN_ENABLED),
            intensity=float(config.RAIN_INTENSITY),
            angle=float(config.RAIN_ANGLE),
            drop_length=float(config.RAIN_DROP_LENGTH),
            drop_speed=float(config.RAIN_DROP_SPEED),
            drop_spacing=float(config.RAIN_DROP_SPACING),
            stream_spacing=float(config.RAIN_STREAM_SPACING),
            color=config.RAIN_COLOR,
        )


@dataclass
class RainAnimationState:
    """Real-time rain animation clock and wind-shape evolution."""

    time: float = 0.0
    render_angle: float = 0.0
    _initialized: bool = False
    _micro_phase_a: float = 0.0
    _micro_phase_b: float = 0.0
    _micro_freq_a_hz: float = 0.0
    _micro_freq_b_hz: float = 0.0
    _gust_active: bool = False
    _gust_elapsed_s: float = 0.0
    _gust_duration_s: float = 0.0
    _gust_amplitude_rad: float = 0.0
    _gust_sign: float = 1.0
    _gust_cooldown_s: float = 0.0
    _wind_rng: _UniformRNG = field(
        default_factory=lambda: rng.get("effects.rain.wind"),
        repr=False,
    )

    def _sample_gust_interval_s(self) -> float:
        """Sample seconds until next gust event."""
        lo, hi = config.RAIN_WIND_GUST_INTERVAL_SEC_RANGE
        return float(self._wind_rng.uniform(float(lo), float(hi)))

    def _sample_gust_duration_s(self) -> float:
        """Sample gust event duration."""
        lo, hi = config.RAIN_WIND_GUST_DURATION_SEC_RANGE
        return float(self._wind_rng.uniform(float(lo), float(hi)))

    def _density(self, rain_config: RainConfig) -> float:
        """Estimate rain density from spacing knobs."""
        density_from_stream = _normalized_spacing(
            rain_config.stream_spacing,
            lo=0.01,
            hi=0.6,
        )
        density_from_drop = _normalized_spacing(
            rain_config.drop_spacing,
            lo=0.1,
            hi=4.0,
        )
        return 0.5 * (density_from_stream + density_from_drop)

    def _reset_wind_state(self, baseline_angle: float) -> None:
        """Reset wind-driven state while preserving animation clock."""
        self._initialized = False
        self.render_angle = baseline_angle
        self._gust_active = False
        self._gust_elapsed_s = 0.0
        self._gust_duration_s = 0.0
        self._gust_amplitude_rad = 0.0
        self._gust_sign = 1.0
        self._gust_cooldown_s = 0.0

    def _ensure_initialized(self, baseline_angle: float) -> None:
        """Initialize oscillator/gust state on first enabled frame."""
        if self._initialized:
            return

        self._initialized = True
        self.render_angle = baseline_angle

        self._micro_freq_a_hz = float(self._wind_rng.uniform(0.05, 0.16))
        self._micro_freq_b_hz = float(self._wind_rng.uniform(0.11, 0.28))
        self._micro_phase_a = float(self._wind_rng.uniform(0.0, 2.0 * math.pi))
        self._micro_phase_b = float(self._wind_rng.uniform(0.0, 2.0 * math.pi))

        self._gust_active = False
        self._gust_cooldown_s = self._sample_gust_interval_s()

    def update(
        self,
        delta_time: DeltaTime,
        rain_config: RainConfig | None = None,
    ) -> None:
        """Advance rain animation and evolve rain angle toward natural motion."""
        dt = max(0.0, float(delta_time))
        self.time = (self.time + dt) % 10000.0

        if rain_config is None:
            return

        baseline_angle = _clamp_angle(rain_config.angle)
        # Preserve bounded user input in the runtime config itself.
        rain_config.angle = baseline_angle

        if not rain_config.enabled:
            self._reset_wind_state(baseline_angle)
            return

        self._ensure_initialized(baseline_angle)

        density = self._density(rain_config)
        angle_headroom = max(
            0.0, float(config.RAIN_ANGLE_MAX_ABS_RAD) - abs(baseline_angle)
        )

        # Always-on subtle variation keeps rain alive between gust events.
        micro_cap = float(config.RAIN_WIND_MICRO_MAX_ABS_RAD)
        micro_amp = min(angle_headroom * 0.4, micro_cap * (0.5 + 0.5 * density))
        self._micro_phase_a += dt * self._micro_freq_a_hz * 2.0 * math.pi
        self._micro_phase_b += dt * self._micro_freq_b_hz * 2.0 * math.pi
        micro_offset = micro_amp * (
            0.65 * math.sin(self._micro_phase_a) + 0.35 * math.sin(self._micro_phase_b)
        )

        # Gust events: brief push away from baseline, then decay back.
        gust_max = (
            float(config.RAIN_WIND_DRIZZLE_MAX_ABS_RAD)
            + (
                float(config.RAIN_WIND_DOWNPOUR_MAX_ABS_RAD)
                - float(config.RAIN_WIND_DRIZZLE_MAX_ABS_RAD)
            )
            * density
        )
        gust_max = min(gust_max, angle_headroom)

        self._gust_cooldown_s -= dt
        if (
            (not self._gust_active)
            and self._gust_cooldown_s <= 0.0
            and gust_max > 0.001
        ):
            self._gust_active = True
            self._gust_elapsed_s = 0.0
            self._gust_duration_s = self._sample_gust_duration_s()
            self._gust_amplitude_rad = float(
                self._wind_rng.uniform(0.35 * gust_max, gust_max)
            )
            self._gust_sign = (
                -1.0 if float(self._wind_rng.uniform(0.0, 1.0)) < 0.5 else 1.0
            )

        gust_offset = 0.0
        if self._gust_active:
            self._gust_elapsed_s += dt
            progress = (
                self._gust_elapsed_s / self._gust_duration_s
                if self._gust_duration_s > 0.0
                else 1.0
            )
            gust_offset = (
                self._gust_sign * self._gust_amplitude_rad * _gust_envelope(progress)
            )
            if progress >= 1.0:
                self._gust_active = False
                self._gust_elapsed_s = 0.0
                self._gust_duration_s = 0.0
                self._gust_amplitude_rad = 0.0
                self._gust_sign = 1.0
                self._gust_cooldown_s += self._sample_gust_interval_s()

        target_angle = _clamp_angle(baseline_angle + micro_offset + gust_offset)

        # Smooth response avoids abrupt angle jumps while preserving gust shape.
        response_rate = max(0.0, float(config.RAIN_WIND_ANGLE_RESPONSE_RATE))
        alpha = 1.0 - math.exp(-response_rate * dt)
        self.render_angle += (target_angle - self.render_angle) * alpha
        self.render_angle = _clamp_angle(self.render_angle)
