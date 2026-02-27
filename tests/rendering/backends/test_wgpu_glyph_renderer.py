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


def test_build_glyph_vertices_encodes_positions_and_colors() -> None:
    """C encoder should produce correct screen positions, UVs, and colours."""
    renderer = object.__new__(WGPUGlyphRenderer)
    tile_w, tile_h = 16, 16
    renderer.tile_dimensions = (tile_w, tile_h)

    # Identity CP437 map (codepoint == index for 0-255).
    renderer.unicode_to_cp437_map = np.arange(256, dtype=np.uint8)

    # UV map: each entry (u1, v1, u2, v2) set to index-derived values so we
    # can verify the lookup.
    uv_map = np.zeros((256, 4), dtype=np.float32)
    uv_map[ord("A")] = [0.1, 0.2, 0.3, 0.4]
    uv_map[ord("B")] = [0.5, 0.6, 0.7, 0.8]
    renderer.uv_map = uv_map

    # 2-wide, 1-tall glyph buffer: cell (0,0) = 'A', cell (1,0) = 'B'.
    gb = GlyphBuffer(2, 1)
    gb.data["ch"][0, 0] = ord("A")
    gb.data["fg"][0, 0] = (255, 0, 0, 255)
    gb.data["bg"][0, 0] = (0, 255, 0, 128)
    gb.data["ch"][1, 0] = ord("B")
    gb.data["fg"][1, 0] = (0, 0, 255, 255)
    gb.data["bg"][1, 0] = (128, 128, 128, 255)

    # Edge neighbor bg for cell (0,0): set neighbor 0 to (51, 102, 153).
    gb.data["edge_neighbor_bg"][0, 0, 0] = (51, 102, 153)
    gb.data["edge_neighbor_mask"][0, 0] = 1
    gb.data["edge_blend"][0, 0] = np.float32(0.5)

    cpu_buffer = np.zeros(2 * 1 * 6, dtype=TEXTURE_VERTEX_DTYPE)
    vertex_count = renderer._build_glyph_vertices(gb, cpu_buffer)

    assert vertex_count == 12
    # Vertices are stored (h, w, 6) = (1, 2, 6).
    verts = cpu_buffer[:vertex_count].reshape(1, 2, 6)

    # -- Cell (0,0) at grid position x=0, y=0 --
    cell_00 = verts[0, 0]  # 6 vertices
    # Vertex 0 (bottom-left): position (0, 0), uv (0.1, 0.2)
    assert cell_00["position"][0][0] == 0.0
    assert cell_00["position"][0][1] == 0.0
    np.testing.assert_allclose(cell_00["uv"][0], [0.1, 0.2], atol=1e-6)
    # Vertex 5 (top-right): position (16, 16), uv (0.3, 0.4)
    assert cell_00["position"][5][0] == float(tile_w)
    assert cell_00["position"][5][1] == float(tile_h)
    np.testing.assert_allclose(cell_00["uv"][5], [0.3, 0.4], atol=1e-6)
    # Foreground color: (255, 0, 0, 255) -> (1.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(cell_00["fg_color"][0], [1.0, 0.0, 0.0, 1.0], atol=0.01)
    # Background color: (0, 255, 0, 128) -> (0.0, 1.0, 0.0, ~0.502)
    np.testing.assert_allclose(cell_00["bg_color"][0][:3], [0.0, 1.0, 0.0], atol=0.01)
    assert abs(cell_00["bg_color"][0][3] - 128.0 / 255.0) < 0.01
    # Edge data
    assert cell_00["edge_neighbor_mask"][0] == 1
    np.testing.assert_allclose(cell_00["edge_blend"][0], 0.5, atol=1e-6)
    # Neighbor bg 0: (51, 102, 153) -> (0.2, 0.4, 0.6)
    np.testing.assert_allclose(
        cell_00["edge_neighbor_bg_0"][0],
        [51 / 255.0, 102 / 255.0, 153 / 255.0],
        atol=0.01,
    )

    # -- Cell (1,0) at grid position x=1, y=0 --
    cell_10 = verts[0, 1]
    # Vertex 0 (bottom-left): position (16, 0)
    assert cell_10["position"][0][0] == float(tile_w)
    assert cell_10["position"][0][1] == 0.0
    np.testing.assert_allclose(cell_10["uv"][0], [0.5, 0.6], atol=1e-6)


def test_build_glyph_vertices_returns_zero_for_empty_buffer() -> None:
    """An empty glyph buffer should produce zero vertices without error."""
    renderer = object.__new__(WGPUGlyphRenderer)
    renderer.tile_dimensions = (16, 16)
    renderer.unicode_to_cp437_map = np.arange(256, dtype=np.uint8)
    renderer.uv_map = np.zeros((256, 4), dtype=np.float32)

    # Create a 0-width buffer by directly constructing data.
    gb = object.__new__(GlyphBuffer)
    gb.width = 0
    gb.height = 0
    from brileta.util.glyph_buffer import GLYPH_DTYPE

    gb.data = np.zeros((0, 0), dtype=GLYPH_DTYPE)

    cpu_buffer = np.zeros(6, dtype=TEXTURE_VERTEX_DTYPE)
    assert renderer._build_glyph_vertices(gb, cpu_buffer) == 0
