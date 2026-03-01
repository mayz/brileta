"""Rain effect configuration and animation state.

Design note - snow as a future sibling effect:

Snow shares most of rain's architecture. When implementing snow:

- Generalize WGPURainRenderer (backends/wgpu/rain_renderer.py) to accept a
  shader path instead of hardcoding rain.wgsl. One renderer class, two
  instances with different shaders. The fullscreen quad setup, bind group
  layout, pipeline creation, and render call are identical.
- Extract the wind animation system (gust events, micro-wobble, angle
  tracking) from RainAnimationState into a shared WindAnimation class that
  both rain and snow compose.
- Write a separate snow.wgsl shader - circular drifting flakes rather than
  oriented line segments. Different enough to warrant its own shader.
- Ground effect: an atmospheric layer with blend_mode="lighten" (inverse of
  rain's "darken" layer in atmospheric.py). Same system, different config.
- Ambient light shift: brighter and cooler (inverse of rain's dimming).
  Same controller toggle pattern, different parameters.
- Rain and snow should be mutually exclusive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from brileta import colors, config
from brileta.types import DeltaTime, saturate
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


def compute_rain_density(rain_config: RainConfig) -> float:
    """Estimate rain density from spacing controls using preset bounds.

    Returns 0..1 where 0 = drizzle (sparse) and 1 = downpour (dense).
    Uses drizzle/downpour preset bounds when available, falling back to
    reasonable hardcoded defaults.
    """
    drizzle = config.RAIN_PRESETS.get("drizzle")
    downpour = config.RAIN_PRESETS.get("downpour")
    if drizzle is not None and downpour is not None:
        stream_lo = min(downpour.stream_spacing)
        stream_hi = max(drizzle.stream_spacing)
        drop_lo = min(downpour.drop_spacing)
        drop_hi = max(drizzle.drop_spacing)
    else:
        stream_lo, stream_hi = 0.01, 0.6
        drop_lo, drop_hi = 0.1, 4.0

    stream_density = _normalized_spacing(
        rain_config.stream_spacing, lo=stream_lo, hi=stream_hi
    )
    drop_density = _normalized_spacing(rain_config.drop_spacing, lo=drop_lo, hi=drop_hi)
    return 0.5 * (stream_density + drop_density)


def _gust_envelope(progress: float) -> float:
    """Asymmetric attack/decay envelope with eased endpoints."""
    p = saturate(progress)
    attack = 0.22
    if p <= attack:
        t = p / max(1e-6, attack)
        # smoothstep: zero slope at gust start avoids visible snapping.
        return t * t * (3.0 - 2.0 * t)
    decay_t = (p - attack) / max(1e-6, 1.0 - attack)
    # Mirrored smoothstep for a softer return-to-baseline phase.
    return 1.0 - (decay_t * decay_t * (3.0 - 2.0 * decay_t))


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
    _gust_prev_sign: float = 1.0
    _gust_has_prev_sign: bool = False
    _gust_cooldown_s: float = 0.0
    _gust_render_offset: float = 0.0
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

    def _sample_gust_sign(self) -> float:
        """Sample gust direction with persistence to avoid rapid flip-flopping."""
        keep_prob = saturate(float(config.RAIN_WIND_GUST_KEEP_DIRECTION_PROB))
        if self._gust_has_prev_sign:
            keep_direction = float(self._wind_rng.uniform(0.0, 1.0)) < keep_prob
            return self._gust_prev_sign if keep_direction else -self._gust_prev_sign
        return -1.0 if float(self._wind_rng.uniform(0.0, 1.0)) < 0.5 else 1.0

    def _clear_active_gust(self) -> None:
        """Clear active gust event state while preserving persistence history."""
        self._gust_active = False
        self._gust_elapsed_s = 0.0
        self._gust_duration_s = 0.0
        self._gust_amplitude_rad = 0.0
        self._gust_sign = 1.0

    def _density(self, rain_config: RainConfig) -> float:
        """Estimate rain density from spacing knobs."""
        return compute_rain_density(rain_config)

    def _reset_wind_state(self, baseline_angle: float) -> None:
        """Reset wind-driven state while preserving animation clock."""
        self._initialized = False
        self.render_angle = baseline_angle
        self._gust_active = False
        self._gust_elapsed_s = 0.0
        self._gust_duration_s = 0.0
        self._gust_amplitude_rad = 0.0
        self._gust_sign = 1.0
        self._gust_prev_sign = 1.0
        self._gust_has_prev_sign = False
        self._gust_cooldown_s = 0.0
        self._gust_render_offset = 0.0

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
        micro_scale_drizzle, micro_scale_downpour = (
            config.RAIN_WIND_MICRO_DENSITY_SCALE_RANGE
        )
        micro_density_scale = (
            float(micro_scale_drizzle)
            + (float(micro_scale_downpour) - float(micro_scale_drizzle)) * density
        )
        micro_density_scale = max(0.0, micro_density_scale)
        micro_amp = min(angle_headroom * 0.4, micro_cap * micro_density_scale)
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

        gusts_enabled = bool(config.RAIN_WIND_GUSTS_ENABLED)
        if gusts_enabled:
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
                self._gust_sign = self._sample_gust_sign()
                self._gust_prev_sign = self._gust_sign
                self._gust_has_prev_sign = True
        else:
            self._clear_active_gust()
            self._gust_cooldown_s = 0.0

        raw_gust_offset = 0.0
        if self._gust_active:
            self._gust_elapsed_s += dt
            progress = (
                self._gust_elapsed_s / self._gust_duration_s
                if self._gust_duration_s > 0.0
                else 1.0
            )
            raw_gust_offset = (
                self._gust_sign * self._gust_amplitude_rad * _gust_envelope(progress)
            )
            if progress >= 1.0:
                self._clear_active_gust()
                self._gust_cooldown_s += self._sample_gust_interval_s()

        # Gust raw offset can still change sharply frame-to-frame due to
        # randomized event timing/sign. Smooth it separately so transitions read
        # as physical easing rather than abrupt direction snaps.
        gust_response_rate = max(0.0, float(config.RAIN_WIND_GUST_OFFSET_RESPONSE_RATE))
        if gust_response_rate <= 0.0:
            self._gust_render_offset = raw_gust_offset
        else:
            gust_alpha = 1.0 - math.exp(-gust_response_rate * dt)
            self._gust_render_offset += (
                raw_gust_offset - self._gust_render_offset
            ) * gust_alpha

        target_angle = _clamp_angle(
            baseline_angle + micro_offset + self._gust_render_offset
        )

        # Smooth response avoids abrupt angle jumps while preserving gust shape.
        response_rate = max(0.0, float(config.RAIN_WIND_ANGLE_RESPONSE_RATE))
        alpha = 1.0 - math.exp(-response_rate * dt)
        self.render_angle += (target_angle - self.render_angle) * alpha
        self.render_angle = _clamp_angle(self.render_angle)
