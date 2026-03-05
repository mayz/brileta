"""Tests for the procedural tree sprite generator."""

from __future__ import annotations

import numpy as np
import pytest

from brileta.sprites.primitives import (
    CanvasStamper,
    draw_tapered_trunk,
    fill_triangle,
    stamp_ellipse,
    stamp_fuzzy_circle,
)
from brileta.sprites.trees import (
    TreeArchetype,
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
        stamp_fuzzy_circle(soft, 12.0, 12.0, 6.0, rgba, hardness=0.0)
        stamp_fuzzy_circle(hard, 12.0, 12.0, 6.0, rgba, hardness=1.0)

        soft_opaque = int(np.count_nonzero(soft[:, :, 3] >= 250))
        hard_opaque = int(np.count_nonzero(hard[:, :, 3] >= 250))
        assert hard_opaque > soft_opaque

    def test_ellipse_wider_than_tall(self) -> None:
        canvas = np.zeros((24, 24, 4), dtype=np.uint8)
        stamp_ellipse(canvas, 12.0, 12.0, 7.0, 3.0, (65, 145, 50, 255))

        alpha_mask = canvas[:, :, 3] > 0
        ys, xs = np.where(alpha_mask)
        horizontal_span = int(xs.max() - xs.min() + 1)
        vertical_span = int(ys.max() - ys.min() + 1)
        assert horizontal_span > vertical_span


class TestCanvasStamperBatch:
    """Tests for batch-stamping equivalence on CanvasStamper."""

    def test_single_ellipse_batch_matches_single_stamp(self) -> None:
        sequential = np.zeros((24, 24, 4), dtype=np.uint8)
        batched = np.zeros((24, 24, 4), dtype=np.uint8)

        rgba = (65, 145, 50, 220)
        ellipse = (12.0, 11.5, 6.0, 4.0)
        cx, cy, rx, ry = ellipse

        sequential_stamper = CanvasStamper(sequential)
        sequential_stamper.stamp_ellipse(
            cx, cy, rx, ry, rgba, falloff=1.6, hardness=0.7
        )

        batched_stamper = CanvasStamper(batched)
        batched_stamper.batch_stamp_ellipses([ellipse], rgba, falloff=1.6, hardness=0.7)

        np.testing.assert_array_equal(sequential, batched)

    def test_non_overlapping_batch_matches_sequential_stamps(self) -> None:
        sequential = np.zeros((32, 32, 4), dtype=np.uint8)
        batched = np.zeros((32, 32, 4), dtype=np.uint8)

        rgba = (75, 150, 45, 210)
        ellipses = [
            (7.0, 7.0, 3.0, 2.0),
            (24.0, 8.0, 4.0, 2.5),
            (16.0, 24.0, 3.5, 3.0),
        ]

        sequential_stamper = CanvasStamper(sequential)
        for cx, cy, rx, ry in ellipses:
            sequential_stamper.stamp_ellipse(
                cx, cy, rx, ry, rgba, falloff=1.5, hardness=0.6
            )

        batched_stamper = CanvasStamper(batched)
        batched_stamper.batch_stamp_ellipses(ellipses, rgba, falloff=1.5, hardness=0.6)

        np.testing.assert_array_equal(sequential, batched)

    def test_overlapping_batch_matches_sequential_stamps(self) -> None:
        """Overlapping ellipses must composite identically to one-by-one.

        The batch path keeps full float32 precision and quantizes to uint8
        once at the end, whereas the sequential path quantizes after each
        stamp.  This means results can differ by ±1-2 due to intermediate
        rounding.  We assert a tight tolerance rather than exact equality.
        """
        sequential = np.zeros((32, 32, 4), dtype=np.uint8)
        batched = np.zeros((32, 32, 4), dtype=np.uint8)

        rgba = (80, 160, 55, 200)
        # Heavily overlapping ellipses centered near the middle.
        ellipses = [
            (14.0, 14.0, 6.0, 5.0),
            (17.0, 15.0, 5.5, 4.5),
            (15.0, 12.0, 4.0, 6.0),
        ]

        sequential_stamper = CanvasStamper(sequential)
        for cx, cy, rx, ry in ellipses:
            sequential_stamper.stamp_ellipse(
                cx, cy, rx, ry, rgba, falloff=1.5, hardness=0.7
            )

        batched_stamper = CanvasStamper(batched)
        batched_stamper.batch_stamp_ellipses(ellipses, rgba, falloff=1.5, hardness=0.7)

        np.testing.assert_allclose(
            sequential.astype(np.int16),
            batched.astype(np.int16),
            atol=2,
        )


class TestDrawTaperedTrunk:
    """Regression tests for the vectorized draw_tapered_trunk."""

    def test_draws_visible_pixels(self) -> None:
        canvas = np.zeros((32, 16, 4), dtype=np.uint8)
        draw_tapered_trunk(
            canvas,
            cx=8.0,
            y_bottom=28,
            y_top=10,
            width_bottom=4.0,
            width_top=2.0,
            rgba=(100, 70, 40, 255),
        )
        assert canvas[:, :, 3].any(), "Trunk should draw visible pixels"

    def test_deterministic(self) -> None:
        a = np.zeros((32, 16, 4), dtype=np.uint8)
        b = np.zeros((32, 16, 4), dtype=np.uint8)
        for canvas in (a, b):
            draw_tapered_trunk(
                canvas,
                cx=8.0,
                y_bottom=28,
                y_top=10,
                width_bottom=4.0,
                width_top=2.0,
                rgba=(100, 70, 40, 255),
            )
        np.testing.assert_array_equal(a, b)

    def test_tapers_narrower_at_top(self) -> None:
        canvas = np.zeros((32, 16, 4), dtype=np.uint8)
        draw_tapered_trunk(
            canvas,
            cx=8.0,
            y_bottom=28,
            y_top=10,
            width_bottom=6.0,
            width_top=2.0,
            rgba=(100, 70, 40, 255),
        )
        # Count non-zero alpha pixels in a top row vs a bottom row.
        top_row_alpha = canvas[12, :, 3]
        bottom_row_alpha = canvas[26, :, 3]
        assert int(np.count_nonzero(bottom_row_alpha)) > int(
            np.count_nonzero(top_row_alpha)
        )

    def test_root_flare_widens_base(self) -> None:
        no_flare = np.zeros((32, 16, 4), dtype=np.uint8)
        flared = np.zeros((32, 16, 4), dtype=np.uint8)
        draw_tapered_trunk(
            no_flare,
            cx=8.0,
            y_bottom=28,
            y_top=10,
            width_bottom=4.0,
            width_top=2.0,
            rgba=(100, 70, 40, 255),
            root_flare=0,
        )
        draw_tapered_trunk(
            flared,
            cx=8.0,
            y_bottom=28,
            y_top=10,
            width_bottom=4.0,
            width_top=2.0,
            rgba=(100, 70, 40, 255),
            root_flare=4,
        )
        # Bottom row should be wider with root flare.
        no_flare_width = int(np.count_nonzero(no_flare[27, :, 3]))
        flared_width = int(np.count_nonzero(flared[27, :, 3]))
        assert flared_width > no_flare_width

    def test_clipped_to_canvas_bounds(self) -> None:
        """Trunk partially off-canvas should not raise."""
        canvas = np.zeros((16, 16, 4), dtype=np.uint8)
        draw_tapered_trunk(
            canvas,
            cx=8.0,
            y_bottom=20,
            y_top=-5,
            width_bottom=4.0,
            width_top=2.0,
            rgba=(100, 70, 40, 255),
        )
        assert canvas[:, :, 3].any()

    def test_inverted_range_is_noop(self) -> None:
        """y_top > y_bottom means zero height, should draw nothing."""
        canvas = np.zeros((16, 16, 4), dtype=np.uint8)
        draw_tapered_trunk(
            canvas,
            cx=8.0,
            y_bottom=5,
            y_top=10,
            width_bottom=4.0,
            width_top=2.0,
            rgba=(100, 70, 40, 255),
        )
        assert not canvas[:, :, 3].any()


class TestFillTriangle:
    """Regression tests for the vectorized fill_triangle."""

    def test_draws_visible_pixels(self) -> None:
        canvas = np.zeros((32, 32, 4), dtype=np.uint8)
        fill_triangle(
            canvas,
            cx=16.0,
            top_y=4,
            base_width=12.0,
            height=20,
            rgba=(50, 120, 40, 255),
        )
        assert canvas[:, :, 3].any(), "Triangle should draw visible pixels"

    def test_deterministic(self) -> None:
        a = np.zeros((32, 32, 4), dtype=np.uint8)
        b = np.zeros((32, 32, 4), dtype=np.uint8)
        for canvas in (a, b):
            fill_triangle(
                canvas,
                cx=16.0,
                top_y=4,
                base_width=12.0,
                height=20,
                rgba=(50, 120, 40, 255),
            )
        np.testing.assert_array_equal(a, b)

    def test_wider_at_base_than_tip(self) -> None:
        canvas = np.zeros((32, 32, 4), dtype=np.uint8)
        fill_triangle(
            canvas,
            cx=16.0,
            top_y=4,
            base_width=14.0,
            height=20,
            rgba=(50, 120, 40, 255),
        )
        tip_row_width = int(np.count_nonzero(canvas[5, :, 3]))
        base_row_width = int(np.count_nonzero(canvas[22, :, 3]))
        assert base_row_width > tip_row_width

    def test_clipped_to_canvas_bounds(self) -> None:
        """Triangle partially off-canvas should not raise."""
        canvas = np.zeros((16, 16, 4), dtype=np.uint8)
        fill_triangle(
            canvas,
            cx=8.0,
            top_y=-5,
            base_width=10.0,
            height=30,
            rgba=(50, 120, 40, 255),
        )
        assert canvas[:, :, 3].any()

    def test_zero_height_is_noop(self) -> None:
        canvas = np.zeros((16, 16, 4), dtype=np.uint8)
        fill_triangle(
            canvas, cx=8.0, top_y=4, base_width=10.0, height=0, rgba=(50, 120, 40, 255)
        )
        assert not canvas[:, :, 3].any()

    def test_edge_antialiasing_produces_partial_alpha(self) -> None:
        """Edge pixels should have sub-255 alpha for smooth silhouettes."""
        canvas = np.zeros((32, 32, 4), dtype=np.uint8)
        fill_triangle(
            canvas,
            cx=16.0,
            top_y=4,
            base_width=14.0,
            height=20,
            rgba=(50, 120, 40, 255),
        )
        alpha = canvas[:, :, 3]
        partial = (alpha > 0) & (alpha < 255)
        assert partial.any(), "Edges should have anti-aliased partial alpha"


class TestBatchStampCircles:
    """Tests for batch_stamp_circles equivalence."""

    def test_batch_circles_matches_sequential(self) -> None:
        sequential = np.zeros((32, 32, 4), dtype=np.uint8)
        batched = np.zeros((32, 32, 4), dtype=np.uint8)

        rgba = (70, 140, 50, 210)
        circles = [
            (8.0, 8.0, 3.5),
            (22.0, 10.0, 4.0),
            (15.0, 24.0, 3.0),
        ]

        seq_stamper = CanvasStamper(sequential)
        for cx, cy, r in circles:
            seq_stamper.stamp_circle(cx, cy, r, rgba, falloff=1.4, hardness=0.5)

        bat_stamper = CanvasStamper(batched)
        bat_stamper.batch_stamp_circles(circles, rgba, falloff=1.4, hardness=0.5)

        # The batch path accumulates alpha in float32 and quantizes once at
        # the end, while sequential stamps quantize after each shape. This
        # can produce +/-1 per channel even for non-overlapping shapes.
        np.testing.assert_allclose(
            sequential.astype(np.int16), batched.astype(np.int16), atol=1
        )

    def test_overlapping_circles_within_tolerance(self) -> None:
        sequential = np.zeros((24, 24, 4), dtype=np.uint8)
        batched = np.zeros((24, 24, 4), dtype=np.uint8)

        rgba = (85, 155, 60, 200)
        circles = [
            (10.0, 10.0, 5.0),
            (13.0, 11.0, 4.5),
            (11.0, 8.0, 3.5),
        ]

        seq_stamper = CanvasStamper(sequential)
        for cx, cy, r in circles:
            seq_stamper.stamp_circle(cx, cy, r, rgba, falloff=1.3, hardness=0.6)

        bat_stamper = CanvasStamper(batched)
        bat_stamper.batch_stamp_circles(circles, rgba, falloff=1.3, hardness=0.6)

        np.testing.assert_allclose(
            sequential.astype(np.int16),
            batched.astype(np.int16),
            atol=2,
        )


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
