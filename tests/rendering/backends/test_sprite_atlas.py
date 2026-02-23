"""Tests for the SpriteAtlas skyline bin-packing and UV coordinate logic.

These tests mock the GPU resource manager so they run without a WGPU device.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from brileta.backends.wgpu.sprite_atlas import PADDING, SpriteAtlas, compute_atlas_size
from brileta.types import SpriteUV


def _make_mock_resource_manager() -> MagicMock:
    """Create a mock WGPUResourceManager that records write_texture calls."""
    rm = MagicMock()
    rm.device.create_texture.return_value = MagicMock(name="gpu_texture")
    return rm


def _make_pixels(width: int, height: int) -> np.ndarray:
    """Create a dummy RGBA pixel array of the given size."""
    return np.zeros((height, width, 4), dtype=np.uint8)


class TestSpriteAtlasAllocation:
    """Core allocation and UV coordinate tests."""

    def test_single_allocation_returns_sprite_uv(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        uv = atlas.allocate(_make_pixels(10, 10))
        assert uv is not None
        assert isinstance(uv, SpriteUV)

    def test_uv_coordinates_are_normalized(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=100, height=100)
        pixels = _make_pixels(20, 10)
        uv = atlas.allocate(pixels)
        assert uv is not None
        # First allocation starts at (0, 0).
        assert uv.u1 == pytest.approx(0.0)
        assert uv.v1 == pytest.approx(0.0)
        assert uv.u2 == pytest.approx(20 / 100)
        assert uv.v2 == pytest.approx(10 / 100)

    def test_multiple_allocations_do_not_overlap(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        uv1 = atlas.allocate(_make_pixels(20, 20))
        uv2 = atlas.allocate(_make_pixels(20, 20))
        assert uv1 is not None
        assert uv2 is not None
        # The two sprites should not share UV space.
        # They pack horizontally first in skyline packing.
        assert uv2.u1 >= uv1.u2 or uv2.v1 >= uv1.v2

    def test_allocated_count_tracks_sprites(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        assert atlas.allocated_count == 0
        atlas.allocate(_make_pixels(10, 10))
        assert atlas.allocated_count == 1
        atlas.allocate(_make_pixels(10, 10))
        assert atlas.allocated_count == 2

    def test_packing_includes_padding(self) -> None:
        """Adjacent sprites should be separated by at least PADDING pixels."""
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        uv1 = atlas.allocate(_make_pixels(20, 20))
        uv2 = atlas.allocate(_make_pixels(20, 20))
        assert uv1 is not None
        assert uv2 is not None
        # The second sprite's u1 should be at least (20 + PADDING) / 128.
        min_gap = (20 + PADDING) / 128
        assert uv2.u1 >= min_gap - 1e-9


class TestSpriteAtlasFull:
    """Tests for atlas-full behavior."""

    def test_returns_none_when_atlas_full(self) -> None:
        # Very small atlas that can hold at most one sprite.
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=32, height=32)
        uv1 = atlas.allocate(_make_pixels(30, 30))
        assert uv1 is not None
        # This should not fit (30 + PADDING > 32 remaining).
        uv2 = atlas.allocate(_make_pixels(10, 10))
        assert uv2 is None

    def test_exact_fit_succeeds(self) -> None:
        # Atlas exactly big enough for one sprite plus padding.
        size = 20 + PADDING
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=size, height=size)
        uv = atlas.allocate(_make_pixels(20, 20))
        assert uv is not None

    def test_too_wide_sprite_returns_none(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=32, height=128)
        uv = atlas.allocate(_make_pixels(33, 10))
        assert uv is None

    def test_too_tall_sprite_returns_none(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=32)
        uv = atlas.allocate(_make_pixels(10, 33))
        assert uv is None


class TestSpriteAtlasDeallocate:
    """Tests for deallocate() and clear()."""

    def test_deallocate_reduces_count(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        uv = atlas.allocate(_make_pixels(10, 10))
        assert uv is not None
        assert atlas.allocated_count == 1
        atlas.deallocate(uv)
        assert atlas.allocated_count == 0

    def test_deallocate_unknown_uv_is_safe(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        # Should not raise.
        atlas.deallocate(SpriteUV(0.0, 0.0, 0.5, 0.5))

    def test_deallocate_allows_reuse(self) -> None:
        """Deallocated space can be reclaimed by subsequent allocations."""
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        # Fill most of the atlas.  Padded size is 61x33, so two won't stack
        # vertically (33*2 = 66 > 64).
        uv1 = atlas.allocate(_make_pixels(60, 32))
        assert uv1 is not None
        # No room for another sprite of the same size.
        assert atlas.allocate(_make_pixels(60, 32)) is None
        # Free the first sprite.
        atlas.deallocate(uv1)
        # Now it fits.
        uv2 = atlas.allocate(_make_pixels(60, 32))
        assert uv2 is not None

    def test_clear_resets_atlas(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        atlas.allocate(_make_pixels(20, 20))
        atlas.allocate(_make_pixels(20, 20))
        assert atlas.allocated_count == 2
        atlas.clear()
        assert atlas.allocated_count == 0
        # After clearing, the atlas can fit sprites again.
        uv = atlas.allocate(_make_pixels(60, 60))
        assert uv is not None


class TestSpriteAtlasGPU:
    """Tests that verify GPU texture creation and upload calls."""

    def test_texture_created_lazily(self) -> None:
        rm = _make_mock_resource_manager()
        atlas = SpriteAtlas(rm, width=64, height=64)
        assert atlas.texture is None
        rm.device.create_texture.assert_not_called()

        atlas.allocate(_make_pixels(10, 10))
        assert atlas.texture is not None
        rm.device.create_texture.assert_called_once()

    def test_upload_called_on_allocate(self) -> None:
        rm = _make_mock_resource_manager()
        atlas = SpriteAtlas(rm, width=64, height=64)
        atlas.allocate(_make_pixels(10, 8))

        # Three write_texture calls: clear, sprite upload, bottom-row edge
        # extension (prevents linear-filter bleed at atlas boundaries).
        assert rm.queue.write_texture.call_count == 3

        # The sprite upload should target origin (0, 0) with size (10, 8).
        sprite_call = rm.queue.write_texture.call_args_list[1]
        dest = sprite_call[0][0]
        size = sprite_call[0][3]
        assert dest["origin"] == (0, 0, 0)
        assert size == (10, 8, 1)

    def test_clear_preserves_texture(self) -> None:
        rm = _make_mock_resource_manager()
        atlas = SpriteAtlas(rm, width=64, height=64)
        atlas.allocate(_make_pixels(10, 10))
        texture_before = atlas.texture

        atlas.clear()
        # Texture still exists after clear (avoids GPU reallocation).
        assert atlas.texture is texture_before


class TestBatchedPackFlush:
    """Tests for the pack() + flush() batched upload path."""

    def test_pack_returns_uv_without_gpu_upload(self) -> None:
        rm = _make_mock_resource_manager()
        atlas = SpriteAtlas(rm, width=64, height=64)
        uv = atlas.pack(_make_pixels(10, 10))
        assert uv is not None
        assert isinstance(uv, SpriteUV)
        # No GPU calls yet - texture not created, nothing uploaded.
        rm.device.create_texture.assert_not_called()
        rm.queue.write_texture.assert_not_called()

    def test_flush_single_write_texture(self) -> None:
        rm = _make_mock_resource_manager()
        atlas = SpriteAtlas(rm, width=64, height=64)
        atlas.pack(_make_pixels(10, 10))
        atlas.pack(_make_pixels(15, 12))
        atlas.pack(_make_pixels(8, 8))
        atlas.flush()
        # One create_texture + one write_texture for the entire atlas.
        rm.device.create_texture.assert_called_once()
        assert rm.queue.write_texture.call_count == 1
        # The upload covers the full atlas dimensions.
        call_size = rm.queue.write_texture.call_args[0][3]
        assert call_size == (64, 64, 1)

    def test_flush_frees_cpu_buffer(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        atlas.pack(_make_pixels(10, 10))
        assert atlas._cpu_buffer is not None
        atlas.flush()
        assert atlas._cpu_buffer is None

    def test_flush_noop_without_packs(self) -> None:
        rm = _make_mock_resource_manager()
        atlas = SpriteAtlas(rm, width=64, height=64)
        atlas.flush()
        rm.device.create_texture.assert_not_called()
        rm.queue.write_texture.assert_not_called()

    def test_pack_and_allocate_produce_same_uvs(self) -> None:
        """pack() and allocate() use the same skyline logic, so identical
        sprite sequences should produce identical UV coordinates."""
        sprites = [_make_pixels(20, 20) for _ in range(10)]

        atlas_pack = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        uvs_pack = [atlas_pack.pack(s) for s in sprites]

        atlas_alloc = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        uvs_alloc = [atlas_alloc.allocate(s) for s in sprites]

        assert uvs_pack == uvs_alloc

    def test_pack_returns_none_when_full(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=32, height=32)
        uv1 = atlas.pack(_make_pixels(30, 30))
        assert uv1 is not None
        uv2 = atlas.pack(_make_pixels(10, 10))
        assert uv2 is None

    def test_pack_tracks_allocated_count(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        assert atlas.allocated_count == 0
        atlas.pack(_make_pixels(10, 10))
        assert atlas.allocated_count == 1
        atlas.pack(_make_pixels(10, 10))
        assert atlas.allocated_count == 2


class TestBulkPackAll:
    """Tests for the pack_all() shelf-packing bulk path."""

    def test_empty_list_returns_empty(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        assert atlas.pack_all([]) == []

    def test_returns_uvs_in_original_order(self) -> None:
        sprites = [_make_pixels(20, 20) for _ in range(5)]
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=256, height=256)
        uvs = atlas.pack_all(sprites)
        assert len(uvs) == 5
        assert all(uv is not None for uv in uvs)

    def test_all_uvs_are_distinct(self) -> None:
        sprites = [_make_pixels(20, 20) for _ in range(10)]
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=256, height=256)
        uvs = atlas.pack_all(sprites)
        non_none = [uv for uv in uvs if uv is not None]
        assert len(set(non_none)) == len(non_none)

    def test_returns_none_when_atlas_too_small(self) -> None:
        # 32x32 atlas can only hold one 30x30 sprite (padded 31x31).
        sprites = [_make_pixels(30, 30), _make_pixels(10, 10)]
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=32, height=32)
        uvs = atlas.pack_all(sprites)
        assert uvs[0] is not None
        assert uvs[1] is None

    def test_allocated_count_matches_placed(self) -> None:
        sprites = [_make_pixels(10, 10) for _ in range(20)]
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        atlas.pack_all(sprites)
        assert atlas.allocated_count == 20

    def test_flush_after_pack_all_single_upload(self) -> None:
        rm = _make_mock_resource_manager()
        atlas = SpriteAtlas(rm, width=128, height=128)
        atlas.pack_all([_make_pixels(10, 10) for _ in range(50)])
        atlas.flush()
        rm.device.create_texture.assert_called_once()
        assert rm.queue.write_texture.call_count == 1

    def test_variable_size_sprites_all_fit(self) -> None:
        sprites = [
            _make_pixels(16, 16),
            _make_pixels(20, 22),
            _make_pixels(18, 14),
            _make_pixels(22, 20),
            _make_pixels(10, 26),
        ]
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=256, height=256)
        uvs = atlas.pack_all(sprites)
        assert all(uv is not None for uv in uvs)

    def test_end_to_end_with_compute_atlas_size(self) -> None:
        """Full pipeline: compute size, pack_all, flush."""
        sprites = [_make_pixels(20, 20) for _ in range(500)]
        size = compute_atlas_size(sprites, gpu_max_texture_dim=8192)
        rm = _make_mock_resource_manager()
        atlas = SpriteAtlas(rm, width=size, height=size)
        uvs = atlas.pack_all(sprites)
        atlas.flush()
        assert all(uv is not None for uv in uvs), (
            f"Some sprites failed to pack into {size}x{size} atlas"
        )
        assert rm.queue.write_texture.call_count == 1


class TestSpriteAtlasValidation:
    """Input validation tests."""

    def test_rejects_non_rgba_array(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        with pytest.raises(ValueError, match="RGBA"):
            atlas.allocate(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_rejects_2d_array(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        with pytest.raises(ValueError, match="RGBA"):
            atlas.allocate(np.zeros((10, 10), dtype=np.uint8))

    def test_rejects_zero_size_sprite(self) -> None:
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        with pytest.raises(ValueError, match="positive"):
            atlas.allocate(np.zeros((0, 10, 4), dtype=np.uint8))


class TestSkylinePacking:
    """Tests for the skyline bin-packing algorithm's spatial behavior."""

    def test_tall_and_short_sprites_pack_efficiently(self) -> None:
        """Skyline packing should fill space next to tall sprites with short ones."""
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=64, height=64)
        # Place a tall narrow sprite.
        uv_tall = atlas.allocate(_make_pixels(10, 40))
        assert uv_tall is not None
        # Place a short wide sprite - should fit next to the tall one.
        uv_short = atlas.allocate(_make_pixels(40, 10))
        assert uv_short is not None
        # The short sprite should start to the right of the tall one.
        assert uv_short.u1 >= uv_tall.u2

    def test_many_small_sprites_pack_densely(self) -> None:
        """Packing many small sprites should use space efficiently."""
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=128, height=128)
        count = 0
        # 128x128 atlas with 10x10 sprites (+ 1px padding = 11x11).
        # Theoretical max: (128 // 11) * (128 // 11) = 11 * 11 = 121 sprites.
        for _ in range(100):
            uv = atlas.allocate(_make_pixels(10, 10))
            if uv is None:
                break
            count += 1
        # Skyline packing should fit at least 80 of the theoretical 121.
        assert count >= 80

    def test_variable_size_sprites(self) -> None:
        """Atlas handles a mix of sprite sizes."""
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=256, height=256)
        sizes = [(16, 16), (20, 20), (22, 22), (18, 14), (20, 20)]
        uvs = []
        for w, h in sizes:
            uv = atlas.allocate(_make_pixels(w, h))
            assert uv is not None, f"Failed to allocate {w}x{h}"
            uvs.append(uv)

        # All UVs should be distinct.
        assert len(set(uvs)) == len(uvs)

    def test_wide_sprite_spans_multiple_skyline_segments(self) -> None:
        """A wide sprite that spans fragmented skyline segments uses the max y."""
        # Use a narrow atlas (80px) so the wide sprite can't fit in the
        # remaining flat space to the right and must span both columns.
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=80, height=80)
        # Create two narrow sprites of different heights to fragment the skyline.
        uv_short = atlas.allocate(_make_pixels(20, 10))
        uv_tall = atlas.allocate(_make_pixels(20, 30))
        assert uv_short is not None
        assert uv_tall is not None

        # The two sprites consume 42px (21 padded each).  The remaining flat
        # space is 80-42 = 38px, too narrow for a 50px sprite (padded 51).
        # The packer must place it spanning both fragmented columns, so its y
        # is at least the taller segment's top (30 + PADDING).
        uv_wide = atlas.allocate(_make_pixels(50, 10))
        assert uv_wide is not None
        min_v1 = (30 + PADDING) / 80
        assert uv_wide.v1 >= min_v1 - 1e-9


class TestComputeAtlasSize:
    """Tests for the compute_atlas_size auto-sizing function."""

    def test_empty_sprite_list_returns_minimum(self) -> None:
        assert compute_atlas_size([], gpu_max_texture_dim=8192) == 256

    def test_returns_power_of_two(self) -> None:
        sprites = [_make_pixels(20, 20) for _ in range(100)]
        size = compute_atlas_size(sprites, gpu_max_texture_dim=8192)
        assert size & (size - 1) == 0, f"{size} is not a power of two"

    def test_small_map_gets_small_atlas(self) -> None:
        # 10 sprites of 20x20 need ~5,460 px^2 with overhead.
        # Should fit in 256x256 (65,536 px^2).
        sprites = [_make_pixels(20, 20) for _ in range(10)]
        size = compute_atlas_size(sprites, gpu_max_texture_dim=8192)
        assert size <= 256

    def test_large_map_gets_larger_atlas(self) -> None:
        # 10,000 sprites of 20x20 need more than 2048x2048.
        # Padded area: 10000 * 21 * 21 * 1.3 = ~5.7M px^2 -> sqrt ~2400 -> 4096.
        sprites = [_make_pixels(20, 20) for _ in range(10_000)]
        size = compute_atlas_size(sprites, gpu_max_texture_dim=8192)
        assert size > 2048

    def test_clamped_to_gpu_max(self) -> None:
        # Even with a huge number of sprites, the atlas cannot exceed the GPU limit.
        sprites = [_make_pixels(20, 20) for _ in range(100_000)]
        size = compute_atlas_size(sprites, gpu_max_texture_dim=4096)
        assert size == 4096

    def test_atlas_at_least_as_wide_as_widest_sprite(self) -> None:
        sprites = [_make_pixels(200, 10)]
        size = compute_atlas_size(sprites, gpu_max_texture_dim=8192)
        # Must accommodate the 200px width + PADDING.
        assert size >= 200 + PADDING

    def test_computed_size_actually_fits_all_sprites(self) -> None:
        """End-to-end: compute the size, create an atlas, and pack all sprites."""
        sprites = [_make_pixels(20, 20) for _ in range(500)]
        size = compute_atlas_size(sprites, gpu_max_texture_dim=8192)
        atlas = SpriteAtlas(_make_mock_resource_manager(), width=size, height=size)
        for sprite in sprites:
            uv = atlas.allocate(sprite)
            assert uv is not None, (
                f"Atlas {size}x{size} could not fit all 500 sprites "
                f"({atlas.allocated_count} allocated)"
            )
