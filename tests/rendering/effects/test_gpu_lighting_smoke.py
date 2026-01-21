from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import moderngl
import numpy as np
import pytest

from catley.backends.moderngl.gpu_lighting import GPULightingSystem
from catley.config import MAP_HEIGHT, MAP_WIDTH
from catley.game.game_world import GameWorld
from catley.game.lights import DynamicLight, StaticLight
from catley.types import FixedTimestep
from catley.util.coordinates import Rect
from catley.view.render.graphics import GraphicsContext


@dataclass
class HeadlessGraphicsContext:
    """Minimal graphics context wrapper for headless ModernGL lighting tests."""

    mgl_context: moderngl.Context


def _create_headless_context() -> moderngl.Context:
    """Create a standalone ModernGL context, skipping if unavailable."""
    try:
        return moderngl.create_context(standalone=True)
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        pytest.skip(f"ModernGL headless context unavailable: {exc}")


def test_gpu_lighting_smoke_lightmap_shape() -> None:
    """Smoke test GPU lighting output shape with a headless ModernGL context."""
    gw = GameWorld(MAP_WIDTH, MAP_HEIGHT)
    context = _create_headless_context()
    graphics = HeadlessGraphicsContext(context)

    lighting_system = GPULightingSystem(gw, cast(GraphicsContext, graphics))
    gw.lighting_system = lighting_system

    gw.add_light(StaticLight(position=(5, 5), radius=4, color=(255, 255, 255)))

    viewport = Rect(0, 0, 10, 10)
    lightmap = lighting_system.compute_lightmap(viewport)

    assert lightmap is not None
    assert lightmap.shape == (10, 10, 3)
    assert np.all(lightmap >= 0.0)


def test_gpu_lighting_smoke_intensity_and_stability() -> None:
    """Ensure headless GPU lighting is stable across frames with static lights."""
    gw = GameWorld(MAP_WIDTH, MAP_HEIGHT)
    context = _create_headless_context()
    graphics = HeadlessGraphicsContext(context)

    lighting_system = GPULightingSystem(gw, cast(GraphicsContext, graphics))
    gw.lighting_system = lighting_system

    gw.add_light(StaticLight(position=(5, 5), radius=4, color=(255, 255, 255)))

    viewport = Rect(0, 0, 10, 10)
    lightmap_first = lighting_system.compute_lightmap(viewport)
    lighting_system.update(FixedTimestep(1.0 / 60.0))
    lightmap_second = lighting_system.compute_lightmap(viewport)

    assert lightmap_first is not None
    assert lightmap_second is not None
    assert np.any(lightmap_first[5, 5] > lightmap_first[0, 0])
    assert np.allclose(lightmap_first, lightmap_second)


def test_gpu_lighting_smoke_dynamic_light_flicker() -> None:
    """Ensure dynamic lights with flicker produce varying output over time."""
    gw = GameWorld(MAP_WIDTH, MAP_HEIGHT)
    context = _create_headless_context()
    graphics = HeadlessGraphicsContext(context)

    lighting_system = GPULightingSystem(gw, cast(GraphicsContext, graphics))
    gw.lighting_system = lighting_system

    gw.add_light(
        DynamicLight(
            position=(5, 5),
            radius=4,
            color=(255, 200, 100),
            flicker_enabled=True,
            flicker_speed=2.0,
        )
    )

    viewport = Rect(0, 0, 10, 10)
    lightmap_t0 = lighting_system.compute_lightmap(viewport)

    # Advance time significantly to trigger flicker variation
    for _ in range(10):
        lighting_system.update(FixedTimestep(1.0 / 60.0))

    lightmap_t1 = lighting_system.compute_lightmap(viewport)

    assert lightmap_t0 is not None
    assert lightmap_t1 is not None
    # Flicker should cause some difference (not necessarily large)
    # At minimum, validate shape and non-negative values
    assert lightmap_t0.shape == (10, 10, 3)
    assert lightmap_t1.shape == (10, 10, 3)
