"""Unit tests for WGPUGlyphRenderer helpers."""

from unittest.mock import Mock

import numpy as np

from brileta.backends.wgpu.glyph_renderer import TEXTURE_VERTEX_DTYPE, WGPUGlyphRenderer
from brileta.util.glyph_buffer import GlyphBuffer


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


def test_build_glyph_vertices_copies_noise_pattern_to_vertex_data() -> None:
    """Per-cell noise_pattern IDs should be broadcast to all six quad vertices."""
    renderer = object.__new__(WGPUGlyphRenderer)
    renderer.tile_dimensions = (8, 8)
    renderer.unicode_to_cp437_map = np.arange(256, dtype=np.uint8)
    renderer.uv_map = np.zeros((256, 4), dtype=np.float32)

    glyph_buffer = GlyphBuffer(2, 1)
    glyph_buffer.data["noise"][1, 0] = np.float32(0.02)
    glyph_buffer.data["noise_pattern"][1, 0] = np.uint8(2)

    cpu_buffer = np.zeros(
        glyph_buffer.width * glyph_buffer.height * 6, dtype=TEXTURE_VERTEX_DTYPE
    )
    vertex_count = renderer._build_glyph_vertices(glyph_buffer, cpu_buffer)

    assert vertex_count == 12
    verts = cpu_buffer[:vertex_count].reshape(
        glyph_buffer.height, glyph_buffer.width, 6
    )

    assert np.all(verts["noise_pattern"][0, 0] == 0)
    assert np.all(verts["noise_pattern"][0, 1] == 2)
    assert np.all(verts["noise_amplitude"][0, 1] == np.float32(0.02))
