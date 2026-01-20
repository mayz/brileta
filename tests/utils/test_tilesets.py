"""Tests for tileset utilities."""

from __future__ import annotations

import numpy as np

from catley.util.tilesets import derive_outlined_atlas


class TestDeriveOutlinedAtlas:
    """Tests for the derive_outlined_atlas function."""

    def test_output_dimensions_match_input(self) -> None:
        """Output atlas should have same dimensions as input."""
        # 2x2 atlas with 4x4 tiles
        atlas = np.zeros((8, 8, 4), dtype=np.uint8)
        result = derive_outlined_atlas(
            atlas, tile_width=4, tile_height=4, columns=2, rows=2
        )
        assert result.shape == atlas.shape

    def test_empty_tile_produces_no_outline(self) -> None:
        """A tile with no pixels should produce no outline."""
        # Single empty 4x4 tile
        atlas = np.zeros((4, 4, 4), dtype=np.uint8)
        result = derive_outlined_atlas(
            atlas,
            tile_width=4,
            tile_height=4,
            columns=1,
            rows=1,
            color=(255, 0, 0, 255),
        )
        # All pixels should remain zero (no outline)
        assert np.all(result == 0)

    def test_single_pixel_produces_surrounding_outline(self) -> None:
        """A single opaque pixel should produce an outline in all 8 directions."""
        # 5x5 tile with single pixel in center
        atlas = np.zeros((5, 5, 4), dtype=np.uint8)
        atlas[2, 2] = [255, 255, 255, 255]  # Center pixel opaque

        outline_color = (255, 0, 0, 255)
        result = derive_outlined_atlas(
            atlas, tile_width=5, tile_height=5, columns=1, rows=1, color=outline_color
        )

        # Center should be empty (outline excludes original pixels)
        assert np.all(result[2, 2] == 0)

        # All 8 neighbors should have the outline color
        neighbors = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
        for y, x in neighbors:
            assert tuple(result[y, x]) == outline_color, (
                f"Neighbor at ({y}, {x}) missing outline"
            )

        # Corners and edges beyond the outline should be empty
        assert np.all(result[0, 0] == 0)
        assert np.all(result[4, 4] == 0)

    def test_filled_rectangle_produces_perimeter_outline(self) -> None:
        """A filled rectangle should produce an outline only around its perimeter."""
        # 6x6 tile with 2x2 filled rectangle in center
        atlas = np.zeros((6, 6, 4), dtype=np.uint8)
        atlas[2:4, 2:4] = [255, 255, 255, 255]  # 2x2 block

        outline_color = (0, 255, 0, 255)
        result = derive_outlined_atlas(
            atlas, tile_width=6, tile_height=6, columns=1, rows=1, color=outline_color
        )

        # Original rectangle area should be empty (outline excludes original)
        for y in range(2, 4):
            for x in range(2, 4):
                assert np.all(result[y, x] == 0), (
                    f"Original pixel at ({y}, {x}) should be empty"
                )

        # Outline should exist around the rectangle
        # Check a few known outline positions
        outline_positions = [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (3, 1),
            (4, 2),
            (4, 3),
        ]
        for y, x in outline_positions:
            assert tuple(result[y, x]) == outline_color, (
                f"Outline at ({y}, {x}) missing"
            )

    def test_outline_color_is_applied(self) -> None:
        """The specified outline color should be applied to outline pixels."""
        atlas = np.zeros((4, 4, 4), dtype=np.uint8)
        atlas[1, 1] = [255, 255, 255, 255]

        # Use a distinctive color
        outline_color = (123, 45, 67, 200)
        result = derive_outlined_atlas(
            atlas, tile_width=4, tile_height=4, columns=1, rows=1, color=outline_color
        )

        # Check that outline pixels have the exact color
        assert tuple(result[0, 0]) == outline_color  # Top-left neighbor
        assert tuple(result[0, 1]) == outline_color  # Top neighbor
        assert tuple(result[1, 0]) == outline_color  # Left neighbor

    def test_multiple_tiles_processed_independently(self) -> None:
        """Each tile in the atlas should be outlined independently."""
        # 2x1 atlas with 4x4 tiles (8 wide, 4 tall)
        atlas = np.zeros((4, 8, 4), dtype=np.uint8)
        # First tile: pixel at (1, 1)
        atlas[1, 1] = [255, 255, 255, 255]
        # Second tile: pixel at (1, 5) which is (1, 1) in tile coords
        atlas[1, 5] = [255, 255, 255, 255]

        outline_color = (255, 128, 0, 255)
        result = derive_outlined_atlas(
            atlas, tile_width=4, tile_height=4, columns=2, rows=1, color=outline_color
        )

        # First tile should have outline around (1, 1)
        assert tuple(result[0, 0]) == outline_color
        assert tuple(result[0, 1]) == outline_color

        # Second tile should have outline around (1, 5)
        assert tuple(result[0, 4]) == outline_color
        assert tuple(result[0, 5]) == outline_color

    def test_bounds_check_skips_out_of_bounds_tiles(self) -> None:
        """Tiles that would exceed atlas bounds should be skipped without error."""
        # 5x5 atlas but claim 2x2 tiles of 4x4 each (would need 8x8)
        atlas = np.zeros((5, 5, 4), dtype=np.uint8)
        atlas[1, 1] = [255, 255, 255, 255]

        # Should not raise, even though tile grid exceeds atlas size
        result = derive_outlined_atlas(
            atlas,
            tile_width=4,
            tile_height=4,
            columns=2,
            rows=2,
            color=(255, 0, 0, 255),
        )

        # First tile (0,0) should still be processed
        assert tuple(result[0, 0]) == (255, 0, 0, 255)
