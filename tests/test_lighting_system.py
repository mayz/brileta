import numpy as np

from catley.game.game_world import GameWorld
from catley.game.lights import DynamicLight, StaticLight
from catley.types import FixedTimestep
from catley.util.coordinates import Rect
from catley.view.render.lighting.cpu import CPULightingSystem


def test_cpu_lighting_system_basic() -> None:
    """Test that CPULightingSystem can be created and compute a basic lightmap."""
    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Add a static light
    static_light = StaticLight(position=(25, 25), radius=5, color=(255, 255, 255))
    gw.add_light(static_light)

    # Compute lightmap for a small area around the light
    viewport_bounds = Rect(20, 20, 10, 10)
    lightmap = lighting_system.compute_lightmap(viewport_bounds)

    # Should return a numpy array
    assert isinstance(lightmap, np.ndarray)
    assert lightmap.shape == (10, 10, 3)

    # Light should be brighter in the center
    center_intensity = lightmap[5, 5]
    corner_intensity = lightmap[0, 0]
    assert np.any(center_intensity > corner_intensity)


def test_cpu_lighting_system_dynamic_light() -> None:
    """Test that dynamic lights can be added and affect the lightmap."""
    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Add a dynamic light
    dynamic_light = DynamicLight(
        position=(25, 25), radius=3, color=(255, 128, 64), flicker_enabled=True
    )
    gw.add_light(dynamic_light)

    # Compute lightmap
    viewport_bounds = Rect(22, 22, 6, 6)
    lightmap = lighting_system.compute_lightmap(viewport_bounds)

    # Should return a numpy array with the right shape
    assert isinstance(lightmap, np.ndarray)
    assert lightmap.shape == (6, 6, 3)


def test_cpu_lighting_system_update() -> None:
    """Test that lighting system can be updated without errors."""
    gw = GameWorld(50, 50)
    lighting_system = CPULightingSystem(gw)

    # Should not raise any exceptions
    lighting_system.update(FixedTimestep(0.016))  # 60 FPS delta time
