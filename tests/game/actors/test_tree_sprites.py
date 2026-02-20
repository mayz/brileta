"""Tests for the procedural tree sprite generator."""

from __future__ import annotations

import numpy as np
import pytest

from brileta.game.actors.tree_sprites import (
    TreeArchetype,
    _stamp_ellipse,
    _stamp_fuzzy_circle,
    generate_preview_sheet,
    generate_tree_sprite,
    generate_tree_sprite_for_position,
    sprite_visual_scale_for_shadow_height,
    tree_sprite_seed,
)


class TestGenerateTreeSprite:
    """Core tests for generate_tree_sprite()."""

    def test_returns_rgba_uint8_array(self) -> None:
        sprite = generate_tree_sprite(42, TreeArchetype.DECIDUOUS)
        assert sprite.ndim == 3
        assert sprite.shape[2] == 4
        assert sprite.dtype == np.uint8

    @pytest.mark.parametrize("archetype", list(TreeArchetype))
    def test_all_archetypes_produce_visible_pixels(
        self, archetype: TreeArchetype
    ) -> None:
        """Every archetype should draw something (non-zero alpha pixels)."""
        sprite = generate_tree_sprite(42, archetype)
        assert sprite[:, :, 3].any(), f"{archetype.value} produced an empty sprite"

    def test_deterministic_from_seed(self) -> None:
        """Same seed + archetype must always produce identical output."""
        a = generate_tree_sprite(42, TreeArchetype.DECIDUOUS)
        b = generate_tree_sprite(42, TreeArchetype.DECIDUOUS)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_sprites(self) -> None:
        a = generate_tree_sprite(42, TreeArchetype.DECIDUOUS)
        b = generate_tree_sprite(43, TreeArchetype.DECIDUOUS)
        assert not np.array_equal(a, b)

    def test_different_archetypes_produce_different_sprites(self) -> None:
        a = generate_tree_sprite(42, TreeArchetype.DECIDUOUS)
        b = generate_tree_sprite(42, TreeArchetype.CONIFER)
        assert not np.array_equal(a, b)

    def test_sapling_is_smaller_than_deciduous(self) -> None:
        """Saplings should generally produce smaller sprites."""
        sapling_sizes = [
            generate_tree_sprite(s, TreeArchetype.SAPLING).shape[0] for s in range(20)
        ]
        deciduous_sizes = [
            generate_tree_sprite(s, TreeArchetype.DECIDUOUS).shape[0] for s in range(20)
        ]
        # On average, saplings should be smaller.
        assert np.mean(sapling_sizes) < np.mean(deciduous_sizes)

    def test_size_variation_across_seeds(self) -> None:
        """Different seeds should produce at least a few different sizes."""
        sizes = {
            generate_tree_sprite(seed, TreeArchetype.DECIDUOUS).shape[0]
            for seed in range(50)
        }
        assert len(sizes) >= 3

    def test_sprite_is_square(self) -> None:
        """Sprites should always be square (height == width)."""
        for archetype in TreeArchetype:
            sprite = generate_tree_sprite(99, archetype)
            assert sprite.shape[0] == sprite.shape[1]

    def test_custom_base_size(self) -> None:
        """base_size parameter should influence output dimensions."""
        small = generate_tree_sprite(42, TreeArchetype.DECIDUOUS, base_size=14)
        large = generate_tree_sprite(42, TreeArchetype.DECIDUOUS, base_size=24)
        assert small.shape[0] < large.shape[0]

    def test_sapling_min_canvas_size(self) -> None:
        sizes = [
            generate_tree_sprite(seed, TreeArchetype.SAPLING).shape[0]
            for seed in range(30)
        ]
        assert min(sizes) >= 15


class TestCanopyHardness:
    """Tests for canopy primitive hardness and ellipse shaping."""

    def test_hard_circle_has_more_opaque_pixels(self) -> None:
        soft = np.zeros((24, 24, 4), dtype=np.uint8)
        hard = np.zeros((24, 24, 4), dtype=np.uint8)

        rgba = (80, 150, 30, 255)
        _stamp_fuzzy_circle(soft, 12.0, 12.0, 6.0, rgba, hardness=0.0)
        _stamp_fuzzy_circle(hard, 12.0, 12.0, 6.0, rgba, hardness=1.0)

        soft_opaque = int(np.count_nonzero(soft[:, :, 3] >= 250))
        hard_opaque = int(np.count_nonzero(hard[:, :, 3] >= 250))
        assert hard_opaque > soft_opaque

    def test_ellipse_wider_than_tall(self) -> None:
        canvas = np.zeros((24, 24, 4), dtype=np.uint8)
        _stamp_ellipse(canvas, 12.0, 12.0, 7.0, 3.0, (65, 145, 50, 255))

        alpha_mask = canvas[:, :, 3] > 0
        ys, xs = np.where(alpha_mask)
        horizontal_span = int(xs.max() - xs.min() + 1)
        vertical_span = int(ys.max() - ys.min() + 1)
        assert horizontal_span > vertical_span


class TestPreviewSheet:
    """Tests for the preview sheet utility."""

    def test_returns_rgba_array(self) -> None:
        sheet = generate_preview_sheet(columns=4, rows=2, base_size=16)
        assert sheet.ndim == 3
        assert sheet.shape[2] == 4
        assert sheet.dtype == np.uint8

    def test_sheet_has_visible_content(self) -> None:
        sheet = generate_preview_sheet(columns=4, rows=2, base_size=16)
        # Should have many non-background pixels.
        non_bg = np.any(sheet[:, :, :3] != 30, axis=2)
        assert non_bg.sum() > 100


class TestPositionSeededGeneration:
    """Tests for position/map-seeded tree generation helpers."""

    def test_tree_sprite_seed_changes_with_map_seed(self) -> None:
        a = tree_sprite_seed(10, 20, map_seed=12345)
        b = tree_sprite_seed(10, 20, map_seed=54321)
        assert a != b

    def test_generate_for_position_deterministic(self) -> None:
        a = generate_tree_sprite_for_position(7, 9, 123, TreeArchetype.DECIDUOUS)
        b = generate_tree_sprite_for_position(7, 9, 123, TreeArchetype.DECIDUOUS)
        np.testing.assert_array_equal(a, b)

    def test_generate_for_position_varies_by_map_seed(self) -> None:
        a = generate_tree_sprite_for_position(7, 9, 123, TreeArchetype.DECIDUOUS)
        b = generate_tree_sprite_for_position(7, 9, 124, TreeArchetype.DECIDUOUS)
        assert not np.array_equal(a, b)


class TestVisualScaleCalibration:
    """Tests for physically calibrated sprite visual scaling."""

    def test_medium_height_full_sprite_maps_to_one_tile(self) -> None:
        sprite = np.zeros((20, 20, 4), dtype=np.uint8)
        sprite[:, :, 3] = 255
        scale = sprite_visual_scale_for_shadow_height(sprite, shadow_height=2)
        assert scale == pytest.approx(1.0)

    def test_larger_shadow_height_requires_larger_scale(self) -> None:
        sprite = np.zeros((20, 20, 4), dtype=np.uint8)
        sprite[:, :, 3] = 255
        small = sprite_visual_scale_for_shadow_height(sprite, shadow_height=1)
        large = sprite_visual_scale_for_shadow_height(sprite, shadow_height=3)
        assert large > small

    def test_short_silhouette_gets_scaled_up(self) -> None:
        sprite = np.zeros((20, 20, 4), dtype=np.uint8)
        sprite[10:20, :, 3] = 255  # 10px tall alpha silhouette in 20px texture.
        scale = sprite_visual_scale_for_shadow_height(sprite, shadow_height=2)
        assert scale == pytest.approx(2.0)
