"""Unit tests for WGPUGlyphRenderer helpers."""

from unittest.mock import Mock

from brileta.backends.wgpu.glyph_renderer import WGPUGlyphRenderer


def test_set_tile_dimensions_clears_cache_on_change() -> None:
    """Changing tile dimensions invalidates the glyph change-detection cache."""
    renderer = object.__new__(WGPUGlyphRenderer)
    renderer.tile_dimensions = (20, 20)
    renderer.buffer_cache = Mock()

    renderer.set_tile_dimensions((21, 21))

    assert renderer.tile_dimensions == (21, 21)
    renderer.buffer_cache.clear.assert_called_once()


def test_set_tile_dimensions_is_noop_when_unchanged() -> None:
    """No cache invalidation is needed when tile dimensions are unchanged."""
    renderer = object.__new__(WGPUGlyphRenderer)
    renderer.tile_dimensions = (20, 20)
    renderer.buffer_cache = Mock()

    renderer.set_tile_dimensions((20, 20))

    renderer.buffer_cache.clear.assert_not_called()


def test_ensure_vertex_buffer_capacity_grows_shared_buffer() -> None:
    """Renderer should grow its internal VBO when more vertices are needed."""
    renderer = object.__new__(WGPUGlyphRenderer)
    renderer.resource_manager = Mock()
    renderer.resource_manager.get_or_create_buffer.return_value = "bigger-buffer"
    renderer._vertex_buffer_capacity_vertices = 12
    renderer.vertex_buffer = "old-buffer"

    renderer._ensure_vertex_buffer_capacity(24)

    assert renderer._vertex_buffer_capacity_vertices == 24
    assert renderer.vertex_buffer == "bigger-buffer"
    renderer.resource_manager.get_or_create_buffer.assert_called_once()


def test_ensure_vertex_buffer_capacity_is_noop_when_sufficient() -> None:
    """Renderer should reuse the existing shared VBO when capacity is enough."""
    renderer = object.__new__(WGPUGlyphRenderer)
    renderer.resource_manager = Mock()
    renderer._vertex_buffer_capacity_vertices = 24
    renderer.vertex_buffer = "existing-buffer"

    renderer._ensure_vertex_buffer_capacity(12)

    assert renderer._vertex_buffer_capacity_vertices == 24
    assert renderer.vertex_buffer == "existing-buffer"
    renderer.resource_manager.get_or_create_buffer.assert_not_called()
