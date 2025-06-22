import numpy as np

from catley.util.coordinates import Rect
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
    off_map = ls.compute_lighting_for_viewport(Rect(0, 0, 5, 5))
    assert np.allclose(off_map, ambient)

    on_map = ls.compute_lighting_for_viewport(Rect(8, 8, 5, 5))
    assert on_map.shape == (5, 5, 3)
    assert np.all(on_map[2, 2] > ambient)


def test_lighting_cache_hit() -> None:
    ls = LightingSystem()
    ls.add_light(LightSource(radius=2, _pos=(10, 10)))

    result1 = ls.compute_lighting_with_shadows(5, 5, [], viewport_offset=(8, 8))
    misses_after_first = ls._lighting_cache.stats.misses

    result2 = ls.compute_lighting_with_shadows(5, 5, [], viewport_offset=(8, 8))
    assert ls._lighting_cache.stats.hits == 1
    assert ls._lighting_cache.stats.misses == misses_after_first
    assert np.array_equal(result1, result2)


def test_cache_invalidation_on_add_light() -> None:
    ls = LightingSystem()
    light1 = LightSource(radius=2, _pos=(10, 10))
    ls.add_light(light1)
    ls.compute_lighting_with_shadows(5, 5, [], viewport_offset=(8, 8))

    # Cache should have 1 item after first computation
    assert len(ls._lighting_cache) == 1

    ls.add_light(LightSource(radius=1, _pos=(5, 5)))

    # Cache should be cleared after adding a light
    assert len(ls._lighting_cache) == 0

    ls.compute_lighting_with_shadows(5, 5, [], viewport_offset=(8, 8))
    # After adding a light and recomputing, cache should have content again
    assert len(ls._lighting_cache) == 1
