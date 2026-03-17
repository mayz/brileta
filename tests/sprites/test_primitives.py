"""Direct tests for reusable sprite drawing primitives."""

from __future__ import annotations

import numpy as np

from brileta import colors
from brileta.sprites.primitives import (
    alpha_blend,
    composite_over,
    draw_line,
    draw_thick_line,
    generate_deciduous_canopy,
    nibble_canopy,
    paste_sprite,
)


def _over_pixel(
    src_rgb: colors.Color, src_a: float, dst_rgba: colors.ColorRGBA
) -> colors.ColorRGBA:
    """Return expected Porter-Duff "over" result for one pixel."""
    dst_a = dst_rgba[3] / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    if out_a <= 0.0:
        return (0, 0, 0, 0)

    inv_src = 1.0 - src_a
    out_r = int((src_rgb[0] * src_a + dst_rgba[0] * dst_a * inv_src) / out_a)
    out_g = int((src_rgb[1] * src_a + dst_rgba[1] * dst_a * inv_src) / out_a)
    out_b = int((src_rgb[2] * src_a + dst_rgba[2] * dst_a * inv_src) / out_a)
    out_alpha = int(out_a * 255)
    return (out_r, out_g, out_b, out_alpha)


class TestAlphaBlend:
    """Unit tests for alpha_blend()."""

    def test_out_of_bounds_is_noop(self) -> None:
        canvas = np.zeros((2, 2, 4), dtype=np.uint8)
        alpha_blend(canvas, x=5, y=5, rgba=(255, 0, 0, 255))
        assert not canvas.any()

    def test_zero_alpha_is_noop(self) -> None:
        canvas = np.zeros((2, 2, 4), dtype=np.uint8)
        alpha_blend(canvas, x=1, y=1, rgba=(255, 0, 0, 0))
        assert not canvas.any()

    def test_blend_over_transparent_sets_pixel(self) -> None:
        canvas = np.zeros((2, 2, 4), dtype=np.uint8)
        alpha_blend(canvas, x=1, y=1, rgba=(20, 40, 60, 128))

        np.testing.assert_array_equal(canvas[1, 1], np.array([20, 40, 60, 128]))

    def test_blend_over_existing_pixel_matches_over_math(self) -> None:
        canvas = np.zeros((2, 2, 4), dtype=np.uint8)
        canvas[0, 0] = np.array([100, 120, 140, 200], dtype=np.uint8)
        src_rgba: colors.ColorRGBA = (20, 60, 200, 128)

        alpha_blend(canvas, x=0, y=0, rgba=src_rgba)

        expected = _over_pixel(src_rgba[:3], src_rgba[3] / 255.0, (100, 120, 140, 200))
        np.testing.assert_array_equal(canvas[0, 0], np.array(expected, dtype=np.uint8))


class TestCompositeOver:
    """Unit tests for composite_over()."""

    def test_composites_region_and_preserves_outside_pixels(self) -> None:
        canvas = np.zeros((3, 3, 4), dtype=np.uint8)
        canvas[1, 1] = np.array([30, 60, 90, 128], dtype=np.uint8)
        canvas[1, 2] = np.array([10, 20, 40, 255], dtype=np.uint8)

        src_a = np.array([[0.5, 1.0]], dtype=np.float32)
        src_rgb: colors.Color = (200, 100, 50)
        composite_over(
            canvas, y_min=1, y_max=1, x_min=1, x_max=2, src_a=src_a, rgb=src_rgb
        )

        expected_11 = _over_pixel(src_rgb, 0.5, (30, 60, 90, 128))
        expected_12 = _over_pixel(src_rgb, 1.0, (10, 20, 40, 255))

        np.testing.assert_array_equal(
            canvas[1, 1], np.array(expected_11, dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            canvas[1, 2], np.array(expected_12, dtype=np.uint8)
        )
        assert not canvas[0, 0].any()


class TestDrawLine:
    """Unit tests for draw_line()."""

    def test_draws_horizontal_segment_including_endpoints(self) -> None:
        canvas = np.zeros((5, 5, 4), dtype=np.uint8)
        draw_line(canvas, x0=1.0, y0=2.0, x1=3.0, y1=2.0, rgba=(255, 0, 0, 255))

        assert int(canvas[2, 1, 3]) == 255
        assert int(canvas[2, 2, 3]) == 255
        assert int(canvas[2, 3, 3]) == 255
        assert int(np.count_nonzero(canvas[:, :, 3])) == 3

    def test_draws_diagonal_segment(self) -> None:
        canvas = np.zeros((5, 5, 4), dtype=np.uint8)
        draw_line(canvas, x0=0.0, y0=0.0, x1=4.0, y1=4.0, rgba=(0, 255, 0, 255))

        assert int(np.count_nonzero(np.diag(canvas[:, :, 3]))) == 5
        assert int(canvas[0, 0, 3]) == 255
        assert int(canvas[4, 4, 3]) == 255


class TestDrawThickLine:
    """Unit tests for draw_thick_line()."""

    def test_thickness_one_matches_draw_line(self) -> None:
        thin = np.zeros((8, 8, 4), dtype=np.uint8)
        thick = np.zeros((8, 8, 4), dtype=np.uint8)

        draw_line(thin, x0=1.0, y0=1.0, x1=6.0, y1=5.0, rgba=(30, 60, 90, 255))
        draw_thick_line(
            thick,
            x0=1.0,
            y0=1.0,
            x1=6.0,
            y1=5.0,
            rgba=(30, 60, 90, 255),
            thickness=1,
        )

        np.testing.assert_array_equal(thin, thick)

    def test_higher_thickness_covers_more_pixels(self) -> None:
        thin = np.zeros((12, 12, 4), dtype=np.uint8)
        thick = np.zeros((12, 12, 4), dtype=np.uint8)

        draw_line(thin, x0=2.0, y0=6.0, x1=9.0, y1=6.0, rgba=(255, 255, 255, 255))
        draw_thick_line(
            thick,
            x0=2.0,
            y0=6.0,
            x1=9.0,
            y1=6.0,
            rgba=(255, 255, 255, 255),
            thickness=3,
        )

        assert int(np.count_nonzero(thick[:, :, 3])) > int(
            np.count_nonzero(thin[:, :, 3])
        )


class TestPasteSprite:
    """Unit tests for paste_sprite()."""

    def test_pastes_opaque_sprite_inside_bounds(self) -> None:
        sheet = np.zeros((4, 4, 4), dtype=np.uint8)
        sprite = np.zeros((2, 2, 4), dtype=np.uint8)
        sprite[:, :] = np.array([10, 20, 30, 255], dtype=np.uint8)

        paste_sprite(sheet, sprite, x0=1, y0=1)

        np.testing.assert_array_equal(sheet[1:3, 1:3], sprite)
        assert int(np.count_nonzero(sheet[:, :, 3])) == 4

    def test_clips_when_pasting_outside_bounds(self) -> None:
        sheet = np.zeros((3, 3, 4), dtype=np.uint8)
        sprite = np.zeros((2, 2, 4), dtype=np.uint8)
        sprite[:, :] = np.array([99, 88, 77, 255], dtype=np.uint8)

        paste_sprite(sheet, sprite, x0=-1, y0=-1)

        np.testing.assert_array_equal(
            sheet[0, 0], np.array([99, 88, 77, 255], dtype=np.uint8)
        )
        assert int(np.count_nonzero(sheet[:, :, 3])) == 1

    def test_alpha_composites_pasted_pixels(self) -> None:
        sheet = np.zeros((2, 2, 4), dtype=np.uint8)
        sheet[0, 0] = np.array([120, 30, 60, 200], dtype=np.uint8)

        sprite = np.zeros((1, 1, 4), dtype=np.uint8)
        sprite[0, 0] = np.array([20, 200, 40, 128], dtype=np.uint8)

        paste_sprite(sheet, sprite, x0=0, y0=0)

        expected = _over_pixel((20, 200, 40), 128 / 255.0, (120, 30, 60, 200))
        np.testing.assert_array_equal(sheet[0, 0], np.array(expected, dtype=np.uint8))


class TestGenerateDeciduousCanopy:
    """Unit tests for the native deciduous canopy hot path."""

    def test_deterministic_output_and_lobe_centers(self) -> None:
        canvas_a = np.zeros((32, 32, 4), dtype=np.uint8)
        canvas_b = np.zeros((32, 32, 4), dtype=np.uint8)
        tips = [(15.0, 17.0), (18.0, 16.0), (13.0, 15.0)]
        lobes_a = generate_deciduous_canopy(
            canvas_a,
            seed=12345,
            size=20,
            canopy_cx=16.0,
            canopy_cy=12.0,
            base_radius=5.5,
            crown_rx_scale=1.1,
            crown_ry_scale=0.9,
            canopy_center_x_offset=0.2,
            tips=tips,
            shadow_rgba=(20, 80, 20, 230),
            mid_rgba=(45, 120, 40, 220),
            highlight_rgba=(90, 160, 65, 200),
        )
        lobes_b = generate_deciduous_canopy(
            canvas_b,
            seed=12345,
            size=20,
            canopy_cx=16.0,
            canopy_cy=12.0,
            base_radius=5.5,
            crown_rx_scale=1.1,
            crown_ry_scale=0.9,
            canopy_center_x_offset=0.2,
            tips=tips,
            shadow_rgba=(20, 80, 20, 230),
            mid_rgba=(45, 120, 40, 220),
            highlight_rgba=(90, 160, 65, 200),
        )

        assert 3 <= len(lobes_a) <= 5
        assert lobes_a == lobes_b
        np.testing.assert_array_equal(canvas_a, canvas_b)

    def test_draws_visible_canopy_pixels(self) -> None:
        canvas = np.zeros((24, 24, 4), dtype=np.uint8)
        lobes = generate_deciduous_canopy(
            canvas,
            seed=7,
            size=18,
            canopy_cx=12.0,
            canopy_cy=9.0,
            base_radius=4.5,
            crown_rx_scale=1.0,
            crown_ry_scale=1.0,
            canopy_center_x_offset=0.0,
            tips=[],
            shadow_rgba=(15, 70, 15, 230),
            mid_rgba=(35, 120, 35, 220),
            highlight_rgba=(75, 155, 55, 200),
        )

        assert 3 <= len(lobes) <= 5
        assert int(np.count_nonzero(canvas[:, :, 3])) > 0


def _make_tree_canvas() -> np.ndarray:
    """Build a small canvas with a wide canopy and a thin trunk.

    Layout (16x16 RGBA):
      Rows 2-8, cols 3-12: wide canopy blob (10px)
      Rows 9-12, cols 7-8:  thin trunk (2px)
    """
    canvas = np.zeros((16, 16, 4), dtype=np.uint8)
    canvas[2:9, 3:13] = [40, 120, 40, 255]  # canopy
    canvas[9:13, 7:9] = [80, 50, 30, 255]  # trunk
    return canvas


class TestNibbleCanopyThinProtection:
    """Thin structures (trunks, branches) must survive canopy nibbling."""

    # Center / radius chosen so the trunk falls inside the canopy envelope.
    CX, CY, RADIUS = 7.5, 5.0, 10.0

    def test_thin_trunk_survives_nibbling(self) -> None:
        """A 2px-wide trunk inside the canopy envelope must not be erased."""
        canvas = _make_tree_canvas()
        trunk_alpha_before = canvas[9:13, 7:9, 3].copy()

        nibble_canopy(
            canvas,
            seed=42,
            center_x=self.CX,
            center_y=self.CY,
            canopy_radius=self.RADIUS,
            nibble_probability=1.0,
            interior_probability=1.0,
        )

        # Every trunk pixel should still be fully opaque.
        np.testing.assert_array_equal(canvas[9:13, 7:9, 3], trunk_alpha_before)

    def test_wide_canopy_is_nibbled(self) -> None:
        """Wide canopy edges (span > 3) should still be eroded normally."""
        canvas = _make_tree_canvas()
        canopy_opaque_before = int(np.count_nonzero(canvas[2:9, 3:13, 3]))

        nibble_canopy(
            canvas,
            seed=42,
            center_x=self.CX,
            center_y=self.CY,
            canopy_radius=self.RADIUS,
            nibble_probability=1.0,
            interior_probability=1.0,
        )

        canopy_opaque_after = int(np.count_nonzero(canvas[2:9, 3:13, 3]))
        assert canopy_opaque_after < canopy_opaque_before


def _make_dumbbell_canvas() -> np.ndarray:
    """Build a canvas with two blobs connected by a single bridge pixel.

    Layout (10x10 RGBA):
      Left blob:   rows 3-6, cols 1-3
      Bridge:      row 5, col 4
      Right blob:  rows 3-6, cols 5-7
    Removing the bridge pixel at (5, 4) would disconnect the two blobs.
    """
    canvas = np.zeros((10, 10, 4), dtype=np.uint8)
    canvas[3:7, 1:4] = [40, 120, 40, 255]  # left blob
    canvas[3:7, 5:8] = [40, 120, 40, 255]  # right blob
    canvas[5, 4] = [40, 120, 40, 255]  # bridge pixel
    return canvas


class TestNibbleCanopyBridgeProtection:
    """Bridge pixels whose removal would disconnect the sprite are preserved."""

    # Center / radius chosen so all pixels fall inside the canopy envelope.
    CX, CY, RADIUS = 4.5, 4.5, 8.0

    def test_bridge_pixel_survives_nibbling(self) -> None:
        """A single-pixel bridge connecting two blobs must not be erased."""
        canvas = _make_dumbbell_canvas()

        nibble_canopy(
            canvas,
            seed=0,
            center_x=self.CX,
            center_y=self.CY,
            canopy_radius=self.RADIUS,
            nibble_probability=1.0,
            interior_probability=1.0,
        )

        # The bridge pixel must remain opaque.
        assert int(canvas[5, 4, 3]) > 128

    def test_bridge_survives_many_seeds(self) -> None:
        """Bridge pixel is protected regardless of PRNG seed."""
        for seed in range(20):
            canvas = _make_dumbbell_canvas()
            nibble_canopy(
                canvas,
                seed=seed,
                center_x=self.CX,
                center_y=self.CY,
                canopy_radius=self.RADIUS,
                nibble_probability=1.0,
                interior_probability=1.0,
            )
            assert int(canvas[5, 4, 3]) > 128, f"Bridge erased with seed={seed}"
