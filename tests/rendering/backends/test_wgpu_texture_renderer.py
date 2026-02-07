"""Unit tests for WGPUTextureRenderer helpers."""

from unittest.mock import Mock

from catley.backends.wgpu.texture_renderer import WGPUTextureRenderer


def test_set_tile_dimensions_clears_cache_on_change() -> None:
    """Changing tile dimensions invalidates the glyph change-detection cache."""
    renderer = object.__new__(WGPUTextureRenderer)
    renderer.tile_dimensions = (20, 20)
    renderer.buffer_cache = Mock()

    renderer.set_tile_dimensions((21, 21))

    assert renderer.tile_dimensions == (21, 21)
    renderer.buffer_cache.clear.assert_called_once()


def test_set_tile_dimensions_is_noop_when_unchanged() -> None:
    """No cache invalidation is needed when tile dimensions are unchanged."""
    renderer = object.__new__(WGPUTextureRenderer)
    renderer.tile_dimensions = (20, 20)
    renderer.buffer_cache = Mock()

    renderer.set_tile_dimensions((20, 20))

    renderer.buffer_cache.clear.assert_not_called()
