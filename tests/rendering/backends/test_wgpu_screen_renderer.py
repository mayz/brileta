"""Unit tests for WGPUScreenRenderer.add_quad_batch."""

from __future__ import annotations

import numpy as np

from brileta.backends.wgpu.screen_renderer import VERTEX_DTYPE, WGPUScreenRenderer


def _make_renderer(max_quads: int = 100) -> WGPUScreenRenderer:
    """Create a minimal screen renderer with only the CPU vertex buffer."""
    sr = object.__new__(WGPUScreenRenderer)
    sr.cpu_vertex_buffer = np.zeros(max_quads * 6, dtype=VERTEX_DTYPE)
    sr.vertex_count = 0
    return sr


class TestAddQuadBatch:
    """Tests for the vectorized add_quad_batch method."""

    def test_single_quad_positions_and_uvs(self) -> None:
        """A single quad should produce 6 vertices with correct positions and UVs."""
        sr = _make_renderer()
        x = np.array([10.0], dtype=np.float32)
        y = np.array([20.0], dtype=np.float32)
        w = np.array([16.0], dtype=np.float32)
        h = np.array([16.0], dtype=np.float32)
        uvs = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32)

        sr.add_quad_batch(x, y, w, h, uvs, colors)

        assert sr.vertex_count == 6
        verts = sr.cpu_vertex_buffer[:6]
        # TL (v0): (10, 20), TR (v1): (26, 20), BL (v2): (10, 36)
        assert verts["position"][0][0] == 10.0
        assert verts["position"][0][1] == 20.0
        assert verts["position"][1][0] == 26.0
        assert verts["position"][1][1] == 20.0
        assert verts["position"][2][0] == 10.0
        assert verts["position"][2][1] == 36.0
        # BR (v5): (26, 36)
        assert verts["position"][5][0] == 26.0
        assert verts["position"][5][1] == 36.0
        # UV: v0=(u1,v1), v5=(u2,v2)
        np.testing.assert_allclose(verts["uv"][0], [0.1, 0.2], atol=1e-6)
        np.testing.assert_allclose(verts["uv"][5], [0.3, 0.4], atol=1e-6)
        # Color broadcast to all 6 vertices.
        for i in range(6):
            np.testing.assert_allclose(verts["color"][i], [1.0, 0.0, 0.0, 1.0])

    def test_optional_defaults(self) -> None:
        """Omitting optional arrays should fill defaults."""
        sr = _make_renderer()
        sr.add_quad_batch(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([8.0], dtype=np.float32),
            np.array([8.0], dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
        )

        verts = sr.cpu_vertex_buffer[:6]
        # Default world_pos is (-1, -1).
        for i in range(6):
            np.testing.assert_allclose(verts["world_pos"][i], [-1.0, -1.0])
        # Default actor_light_scale is 1.0.
        assert verts["actor_light_scale"][0] == 1.0
        # Default flags is 0.
        assert verts["flags"][0] == 0
        # Default tile_bg is (0, 0, 0).
        np.testing.assert_allclose(verts["tile_bg"][0], [0.0, 0.0, 0.0])

    def test_multiple_quads(self) -> None:
        """Multiple quads should be laid out sequentially in the buffer."""
        sr = _make_renderer()
        n = 3
        x = np.array([0.0, 10.0, 20.0], dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        w = np.full(n, 8.0, dtype=np.float32)
        h = np.full(n, 8.0, dtype=np.float32)
        uvs = np.zeros((n, 4), dtype=np.float32)
        colors = np.ones((n, 4), dtype=np.float32)
        world_pos = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

        sr.add_quad_batch(x, y, w, h, uvs, colors, world_pos=world_pos)

        assert sr.vertex_count == 18
        verts = sr.cpu_vertex_buffer[:18].reshape(3, 6)
        # Each quad's first vertex x should match the input.
        assert verts[0]["position"][0][0] == 0.0
        assert verts[1]["position"][0][0] == 10.0
        assert verts[2]["position"][0][0] == 20.0
        # World positions broadcast correctly.
        np.testing.assert_allclose(verts[0]["world_pos"][0], [1.0, 2.0])
        np.testing.assert_allclose(verts[2]["world_pos"][3], [5.0, 6.0])

    def test_truncation_when_buffer_nearly_full(self) -> None:
        """Batch should be silently truncated when buffer capacity is exceeded."""
        # Buffer for only 2 quads (12 vertices).
        sr = _make_renderer(max_quads=2)
        n = 5
        x = np.arange(n, dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        w = np.ones(n, dtype=np.float32)
        h = np.ones(n, dtype=np.float32)
        uvs = np.zeros((n, 4), dtype=np.float32)
        colors = np.ones((n, 4), dtype=np.float32)

        sr.add_quad_batch(x, y, w, h, uvs, colors)

        # Only 2 quads fit.
        assert sr.vertex_count == 12
        verts = sr.cpu_vertex_buffer[:12].reshape(2, 6)
        assert verts[0]["position"][0][0] == 0.0
        assert verts[1]["position"][0][0] == 1.0

    def test_empty_batch_is_noop(self) -> None:
        """A zero-length batch should not modify the vertex buffer."""
        sr = _make_renderer()
        sr.add_quad_batch(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
        assert sr.vertex_count == 0

    def test_batch_appends_after_existing_vertices(self) -> None:
        """Batch should append after previously written vertices."""
        sr = _make_renderer()
        # Write one quad first.
        sr.add_quad_batch(
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([8.0], dtype=np.float32),
            np.array([8.0], dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
        )
        assert sr.vertex_count == 6

        # Write a second quad.
        sr.add_quad_batch(
            np.array([100.0], dtype=np.float32),
            np.array([100.0], dtype=np.float32),
            np.array([8.0], dtype=np.float32),
            np.array([8.0], dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
            np.ones((1, 4), dtype=np.float32),
        )
        assert sr.vertex_count == 12
        # Second quad starts at vertex 6.
        assert sr.cpu_vertex_buffer[6]["position"][0] == 100.0
