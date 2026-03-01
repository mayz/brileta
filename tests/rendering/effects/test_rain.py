import struct

import pytest

from brileta import config
from brileta.view.render.effects.rain import RainAnimationState, RainConfig
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


def test_rain_animation_accumulates_time() -> None:
    state = RainAnimationState()

    state.update(dt(0.25))
    state.update(dt(0.75))

    assert state.time == 1.0


def test_rain_animation_wraps_time_at_10000_seconds() -> None:
    state = RainAnimationState(time=9999.9)

    state.update(dt(0.3))

    assert state.time == pytest.approx(0.2)


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


def test_rain_uniform_packing_matches_shader_layout() -> None:
    """Verify _build_uniforms packs 5 vec4s in the order the shader expects.

    Shader struct RainUniforms:
      vec4 0: letterbox (offset_x, offset_y, scaled_w, scaled_h)
      vec4 1: viewport  (offset_x, offset_y, size_x, size_y)
      vec4 2: rain_params (intensity, angle, drop_length, drop_speed)
      vec4 3: anim_data   (color_r, color_g, color_b, time)
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
        drop_speed=25.0,
        drop_spacing=1.35,
        stream_spacing=0.33,
        rain_color=(180, 200, 220),
        time=42.5,
    )

    assert len(data) == 20 * 4  # 20 floats, 4 bytes each
    floats = struct.unpack("20f", data)

    # vec4 0: letterbox
    assert floats[0:4] == (10.0, 20.0, 800.0, 600.0)
    # vec4 1: viewport
    assert floats[4:8] == (5.0, 15.0, 30.0, 20.0)
    # vec4 2: rain_params (intensity, angle, drop_length, drop_speed)
    assert floats[8:12] == pytest.approx((0.5, 0.3, 0.9, 25.0))
    # vec4 3: anim_data (color_r, color_g, color_b, time)
    assert floats[12:16] == pytest.approx((180.0, 200.0, 220.0, 42.5))
    # vec4 4: spacing_data (tile_w_px, tile_h_px, drop_spacing, stream_spacing)
    assert floats[16:20] == pytest.approx((16.0, 24.0, 1.35, 0.33))
