import numpy as np

from catley.view.render.effects.lighting import LightingSystem, LightSource


def test_compute_lighting_with_viewport_offset() -> None:
    ls = LightingSystem()
    ls.add_light(LightSource(radius=2, _pos=(10, 10)))

    ambient = ls.config.ambient_light
    # Slice that includes the light when offset is applied.
    with_offset = ls.compute_lighting_with_shadows(5, 5, [], viewport_offset=(8, 8))
    assert with_offset.shape == (5, 5, 3)
    assert np.all(with_offset[2, 2] > ambient)

    # Same slice without offset should not include the light.
    without_offset = ls.compute_lighting_with_shadows(5, 5, [], viewport_offset=(0, 0))
    assert np.allclose(without_offset, ambient)


def test_compute_lighting_for_viewport() -> None:
    ls = LightingSystem()
    ls.add_light(LightSource(radius=2, _pos=(10, 10)))

    ambient = ls.config.ambient_light
    off_map = ls.compute_lighting_for_viewport(0, 0, 5, 5, [])
    assert np.allclose(off_map, ambient)

    on_map = ls.compute_lighting_for_viewport(8, 8, 5, 5, [])
    assert on_map.shape == (5, 5, 3)
    assert np.all(on_map[2, 2] > ambient)
