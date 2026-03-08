import struct

import pytest

from brileta import config
from brileta.view.render.effects.rain import (
    RainAnimationState,
    RainConfig,
    _gust_envelope,
)
from tests.helpers import dt


class _UpperBoundRNG:
    """Deterministic RNG that always returns the upper bound."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def uniform(self, a: float, b: float) -> float:
        self.calls.append((a, b))
        return b


class _MidpointRNG:
    """Deterministic RNG that always returns midpoint."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float]] = []

    def uniform(self, a: float, b: float) -> float:
        self.calls.append((a, b))
        return (a + b) * 0.5


def test_rain_config_from_config_uses_defaults() -> None:
    rain = RainConfig.from_config()

    assert rain.enabled == config.RAIN_ENABLED
    assert rain.intensity == config.RAIN_INTENSITY
    assert rain.angle == config.RAIN_ANGLE
    assert rain.drop_length == config.RAIN_DROP_LENGTH
    assert rain.drop_speed == config.RAIN_DROP_SPEED
    assert rain.drop_spacing == config.RAIN_DROP_SPACING
    assert rain.stream_spacing == config.RAIN_STREAM_SPACING
    assert rain.color == config.RAIN_COLOR


def test_rain_animation_accumulates_advection_incrementally() -> None:
    """Advection should accumulate each frame using current angle and speed."""
    state = RainAnimationState(_wind_rng=_MidpointRNG())
    rain = RainConfig(enabled=True, angle=0.0, drop_speed=10.0)

    state.update(dt(1.0), rain)

    # angle=0 means slant_dir = (sin(0), cos(0)) = (0, 1).
    # After 1 second at speed 10, advection should be roughly (0, 10).
    assert state.advection[0] == pytest.approx(0.0, abs=0.01)
    assert state.advection[1] == pytest.approx(10.0, abs=0.5)


def test_rain_advection_continuous_across_speed_change() -> None:
    """Changing speed should not cause advection to jump discontinuously."""
    state = RainAnimationState(_wind_rng=_MidpointRNG())
    rain = RainConfig(enabled=True, angle=0.0, drop_speed=10.0)

    # Run for a while at the old speed.
    for _ in range(60):
        state.update(dt(1 / 60), rain)
    advection_before = state.advection

    # Switch to a faster speed (like changing presets).
    rain.drop_speed = 25.0
    state.update(dt(1 / 60), rain)
    advection_after = state.advection

    # The delta from a single frame should be bounded by 25 * (1/60) ~ 0.42,
    # not a huge jump proportional to accumulated time.
    dx = advection_after[0] - advection_before[0]
    dy = advection_after[1] - advection_before[1]
    frame_delta = (dx**2 + dy**2) ** 0.5
    assert frame_delta < 1.0


def test_rain_animation_clamps_manual_baseline_angle() -> None:
    state = RainAnimationState(_wind_rng=_UpperBoundRNG())
    rain = RainConfig(enabled=True, angle=2.0)

    state.update(dt(0.1), rain)

    assert rain.angle == pytest.approx(config.RAIN_ANGLE_MAX_ABS_RAD)
    assert state.render_angle == pytest.approx(config.RAIN_ANGLE_MAX_ABS_RAD)


def test_rain_animation_gust_pushes_then_settles_toward_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Disable micro variation to isolate gust behavior.
    monkeypatch.setattr(config, "RAIN_WIND_GUSTS_ENABLED", True)
    monkeypatch.setattr(config, "RAIN_WIND_MICRO_MAX_ABS_RAD", 0.0)
    monkeypatch.setattr(config, "RAIN_WIND_GUST_INTERVAL_SEC_RANGE", (100.0, 100.0))
    monkeypatch.setattr(config, "RAIN_WIND_GUST_DURATION_SEC_RANGE", (1.0, 1.0))
    monkeypatch.setattr(config, "RAIN_WIND_ANGLE_RESPONSE_RATE", 12.0)
    monkeypatch.setattr(config, "RAIN_WIND_DRIZZLE_MAX_ABS_RAD", 0.3)
    monkeypatch.setattr(config, "RAIN_WIND_DOWNPOUR_MAX_ABS_RAD", 0.3)

    state = RainAnimationState(_wind_rng=_UpperBoundRNG())
    rain = RainConfig(enabled=True, angle=0.0, stream_spacing=0.01, drop_spacing=0.1)

    # Initialize wind state and then force a gust start in a controlled frame.
    state.update(dt(0.0), rain)
    state._gust_cooldown_s = 0.0

    state.update(dt(0.4), rain)
    during_gust_abs = abs(state.render_angle)
    assert during_gust_abs > 0.0

    state.update(dt(1.6), rain)
    after_gust_abs = abs(state.render_angle)
    assert after_gust_abs < during_gust_abs


def test_rain_animation_resets_to_baseline_when_disabled() -> None:
    state = RainAnimationState(_wind_rng=_MidpointRNG())
    rain = RainConfig(enabled=True, angle=0.2)

    state.update(dt(0.1), rain)
    assert state.render_angle == pytest.approx(0.2, abs=0.01)

    rain.enabled = False
    rain.angle = -0.1
    state.update(dt(0.1), rain)

    assert state.render_angle == pytest.approx(-0.1)


def test_gust_envelope_has_eased_endpoints() -> None:
    """The gust envelope should start/end with near-zero slope."""
    assert _gust_envelope(0.0) == pytest.approx(0.0)
    assert _gust_envelope(1.0) == pytest.approx(0.0)
    # Early progress rises gently from zero (not a hard linear ramp).
    assert _gust_envelope(0.01) < 0.08
    # Late progress settles gently back toward zero.
    assert _gust_envelope(0.99) < 0.08


def test_rain_animation_smooths_gust_offset_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render gust offset should ease toward raw gust offset, not jump instantly."""
    monkeypatch.setattr(config, "RAIN_WIND_GUSTS_ENABLED", True)
    monkeypatch.setattr(config, "RAIN_WIND_MICRO_MAX_ABS_RAD", 0.0)
    monkeypatch.setattr(config, "RAIN_WIND_GUST_INTERVAL_SEC_RANGE", (100.0, 100.0))
    monkeypatch.setattr(config, "RAIN_WIND_GUST_DURATION_SEC_RANGE", (1.0, 1.0))
    monkeypatch.setattr(config, "RAIN_WIND_ANGLE_RESPONSE_RATE", 100.0)
    monkeypatch.setattr(config, "RAIN_WIND_GUST_OFFSET_RESPONSE_RATE", 2.0)
    monkeypatch.setattr(config, "RAIN_WIND_DRIZZLE_MAX_ABS_RAD", 0.3)
    monkeypatch.setattr(config, "RAIN_WIND_DOWNPOUR_MAX_ABS_RAD", 0.3)

    state = RainAnimationState(_wind_rng=_UpperBoundRNG())
    rain = RainConfig(enabled=True, angle=0.0, stream_spacing=0.01, drop_spacing=0.1)

    state.update(dt(0.0), rain)
    state._gust_cooldown_s = 0.0
    state.update(dt(0.1), rain)

    progress = (
        state._gust_elapsed_s / state._gust_duration_s
        if state._gust_duration_s > 0.0
        else 1.0
    )
    raw_offset = state._gust_sign * state._gust_amplitude_rad * _gust_envelope(progress)
    assert abs(state._gust_render_offset) < abs(raw_offset)


def test_sample_gust_sign_keeps_previous_direction_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "RAIN_WIND_GUST_KEEP_DIRECTION_PROB", 1.0)
    state = RainAnimationState(_wind_rng=_MidpointRNG())
    state._gust_has_prev_sign = True
    state._gust_prev_sign = -1.0

    assert state._sample_gust_sign() == -1.0


def test_sample_gust_sign_can_flip_previous_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "RAIN_WIND_GUST_KEEP_DIRECTION_PROB", 0.0)
    state = RainAnimationState(_wind_rng=_MidpointRNG())
    state._gust_has_prev_sign = True
    state._gust_prev_sign = -1.0

    assert state._sample_gust_sign() == 1.0


def test_rain_animation_has_no_gust_push_when_gusts_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "RAIN_WIND_GUSTS_ENABLED", False)
    monkeypatch.setattr(config, "RAIN_WIND_MICRO_MAX_ABS_RAD", 0.0)
    monkeypatch.setattr(config, "RAIN_WIND_GUST_OFFSET_RESPONSE_RATE", 100.0)
    monkeypatch.setattr(config, "RAIN_WIND_ANGLE_RESPONSE_RATE", 100.0)
    monkeypatch.setattr(config, "RAIN_WIND_DRIZZLE_MAX_ABS_RAD", 0.3)
    monkeypatch.setattr(config, "RAIN_WIND_DOWNPOUR_MAX_ABS_RAD", 0.3)

    state = RainAnimationState(_wind_rng=_UpperBoundRNG())
    rain = RainConfig(enabled=True, angle=0.0, stream_spacing=0.01, drop_spacing=0.1)

    state.update(dt(0.0), rain)
    state._gust_cooldown_s = 0.0
    state.update(dt(0.5), rain)

    assert state._gust_active is False
    assert state._gust_render_offset == pytest.approx(0.0)
    assert state.render_angle == pytest.approx(0.0)


def test_rain_animation_micro_variation_reduces_with_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "RAIN_WIND_GUSTS_ENABLED", False)
    monkeypatch.setattr(config, "RAIN_WIND_MICRO_MAX_ABS_RAD", 0.08)
    monkeypatch.setattr(config, "RAIN_WIND_MICRO_DENSITY_SCALE_RANGE", (1.0, 0.2))
    monkeypatch.setattr(config, "RAIN_WIND_ANGLE_RESPONSE_RATE", 100.0)

    drizzle_state = RainAnimationState(_wind_rng=_UpperBoundRNG())
    downpour_state = RainAnimationState(_wind_rng=_UpperBoundRNG())

    drizzle = RainConfig(
        enabled=True,
        angle=0.0,
        stream_spacing=0.6,
        drop_spacing=4.0,
    )
    downpour = RainConfig(
        enabled=True,
        angle=0.0,
        stream_spacing=0.16,
        drop_spacing=0.75,
    )

    drizzle_state.update(dt(0.1), drizzle)
    downpour_state.update(dt(0.1), downpour)

    assert abs(downpour_state.render_angle) < abs(drizzle_state.render_angle)


def test_rain_uniform_packing_matches_shader_layout() -> None:
    """Verify _build_uniforms packs 5 vec4s in the order the shader expects.

    Shader struct RainUniforms:
      vec4 0: letterbox (offset_x, offset_y, scaled_w, scaled_h)
      vec4 1: viewport  (offset_x, offset_y, size_x, size_y)
      vec4 2: rain_params (intensity, angle, drop_length, advection_x)
      vec4 3: anim_data   (color_r, color_g, color_b, advection_y)
      vec4 4: spacing_data (tile_w_px, tile_h_px, drop_spacing, stream_spacing)
    """
    from brileta.backends.wgpu.rain_renderer import WGPURainRenderer

    data = WGPURainRenderer._build_uniforms(
        None,  # type: ignore[arg-type]  # static method-like, self unused
        letterbox_geometry=(10, 20, 800, 600),
        viewport_offset=(5, 15),
        viewport_size=(30, 20),
        tile_dimensions=(16, 24),
        intensity=0.5,
        angle=0.3,
        drop_length=0.9,
        advection=(100.5, 42.5),
        drop_spacing=1.35,
        stream_spacing=0.33,
        rain_color=(180, 200, 220),
    )

    assert len(data) == 20 * 4  # 20 floats, 4 bytes each
    floats = struct.unpack("20f", data)

    # vec4 0: letterbox
    assert floats[0:4] == (10.0, 20.0, 800.0, 600.0)
    # vec4 1: viewport
    assert floats[4:8] == (5.0, 15.0, 30.0, 20.0)
    # vec4 2: rain_params (intensity, angle, drop_length, advection_x)
    assert floats[8:12] == pytest.approx((0.5, 0.3, 0.9, 100.5))
    # vec4 3: anim_data (color_r, color_g, color_b, advection_y)
    assert floats[12:16] == pytest.approx((180.0, 200.0, 220.0, 42.5))
    # vec4 4: spacing_data (tile_w_px, tile_h_px, drop_spacing, stream_spacing)
    assert floats[16:20] == pytest.approx((16.0, 24.0, 1.35, 0.33))
