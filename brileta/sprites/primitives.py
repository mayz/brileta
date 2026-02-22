"""Reusable RGBA sprite drawing primitives.

All drawing is performed by native C implementations in
``brileta.util._native`` (see ``_native_sprites.c``).  The functions
here are thin wrappers that unpack color tuples into the positional
arguments the C layer expects.
"""

from __future__ import annotations

import numpy as np

from brileta import colors
from brileta.types import PixelPos
from brileta.util._native import (
    sprite_alpha_blend as _c_alpha_blend,
)
from brileta.util._native import (
    sprite_batch_stamp_circles as _c_batch_stamp_circles,
)
from brileta.util._native import (
    sprite_batch_stamp_ellipses as _c_batch_stamp_ellipses,
)
from brileta.util._native import (
    sprite_composite_over as _c_composite_over,
)
from brileta.util._native import (
    sprite_darken_rim as _c_darken_rim,
)
from brileta.util._native import (
    sprite_draw_line as _c_draw_line,
)
from brileta.util._native import (
    sprite_draw_tapered_trunk as _c_draw_tapered_trunk,
)
from brileta.util._native import (
    sprite_draw_thick_line as _c_draw_thick_line,
)
from brileta.util._native import (
    sprite_fill_triangle as _c_fill_triangle,
)
from brileta.util._native import (
    sprite_generate_deciduous_canopy as _c_generate_deciduous_canopy,
)
from brileta.util._native import (
    sprite_nibble_boulder as _c_nibble_boulder,
)
from brileta.util._native import (
    sprite_nibble_canopy as _c_nibble_canopy,
)
from brileta.util._native import (
    sprite_paste_sprite as _c_paste_sprite,
)
from brileta.util._native import (
    sprite_stamp_ellipse as _c_stamp_ellipse,
)
from brileta.util._native import (
    sprite_stamp_fuzzy_circle as _c_stamp_fuzzy_circle,
)

__all__ = [
    "CanvasStamper",
    "alpha_blend",
    "clamp",
    "composite_over",
    "darken_rim",
    "draw_line",
    "draw_tapered_trunk",
    "draw_thick_line",
    "fill_triangle",
    "generate_deciduous_canopy",
    "nibble_boulder",
    "nibble_canopy",
    "paste_sprite",
    "stamp_ellipse",
    "stamp_fuzzy_circle",
]


def clamp(value: int, lo: int = 0, hi: int = 255) -> int:
    """Clamp an integer to [lo, hi]."""
    return max(lo, min(hi, value))


def alpha_blend(canvas: np.ndarray, x: int, y: int, rgba: colors.ColorRGBA) -> None:
    """Alpha-composite a single RGBA pixel onto the canvas."""
    _c_alpha_blend(canvas, x, y, rgba[0], rgba[1], rgba[2], rgba[3])


def composite_over(
    canvas: np.ndarray,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    src_a: np.ndarray,
    rgb: colors.Color,
) -> None:
    """Composite uniform RGB over a canvas region with per-pixel alpha.

    ``src_a`` is a 2D float array with alpha values in ``[0.0, 1.0]`` whose
    shape matches the region defined by ``y_min:y_max+1, x_min:x_max+1``.
    """
    _c_composite_over(
        canvas,
        y_min,
        y_max,
        x_min,
        x_max,
        np.ascontiguousarray(src_a, dtype=np.float32),
        rgb[0],
        rgb[1],
        rgb[2],
    )


def draw_line(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rgba: colors.ColorRGBA,
) -> None:
    """Draw a 1px Bresenham line between two points."""
    _c_draw_line(canvas, x0, y0, x1, y1, rgba[0], rgba[1], rgba[2], rgba[3])


def draw_thick_line(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rgba: colors.ColorRGBA,
    thickness: int,
) -> None:
    """Draw a line with integer thickness using parallel Bresenham lines."""
    _c_draw_thick_line(
        canvas, x0, y0, x1, y1, rgba[0], rgba[1], rgba[2], rgba[3], thickness
    )


def draw_tapered_trunk(
    canvas: np.ndarray,
    cx: float,
    y_bottom: int,
    y_top: int,
    width_bottom: float,
    width_top: float,
    rgba: colors.ColorRGBA,
    root_flare: int = 0,
) -> None:
    """Draw a trunk as a vertically tapered filled column.

    The trunk tapers linearly from *width_bottom* at *y_bottom* to
    *width_top* at *y_top*. An optional *root_flare* widens the
    bottom 1-2 rows to suggest ground contact.
    """
    _c_draw_tapered_trunk(
        canvas,
        cx,
        y_bottom,
        y_top,
        width_bottom,
        width_top,
        rgba[0],
        rgba[1],
        rgba[2],
        rgba[3],
        root_flare,
    )


def stamp_fuzzy_circle(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    rgba: colors.ColorRGBA,
    falloff: float = 1.5,
    hardness: float = 0.0,
) -> None:
    """Stamp a soft circle with alpha falloff at edges."""
    _c_stamp_fuzzy_circle(
        canvas, cx, cy, radius, rgba[0], rgba[1], rgba[2], rgba[3], falloff, hardness
    )


def stamp_ellipse(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    rgba: colors.ColorRGBA,
    falloff: float = 1.5,
    hardness: float = 0.0,
) -> None:
    """Stamp a soft-to-hard ellipse using normalized ellipse-space distance."""
    _c_stamp_ellipse(
        canvas, cx, cy, rx, ry, rgba[0], rgba[1], rgba[2], rgba[3], falloff, hardness
    )


class CanvasStamper:
    """Thin wrapper that holds a canvas reference for batch stamping.

    All drawing is performed by the native C layer. The class preserves
    the public API expected by tree and boulder sprite generators.
    """

    __slots__ = ("_canvas",)

    def __init__(self, canvas: np.ndarray) -> None:
        self._canvas = canvas

    def stamp_ellipse(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        rgba: colors.ColorRGBA,
        falloff: float = 1.5,
        hardness: float = 0.0,
    ) -> None:
        """Alpha-composite a soft ellipse onto the canvas."""
        _c_stamp_ellipse(
            self._canvas,
            cx,
            cy,
            rx,
            ry,
            rgba[0],
            rgba[1],
            rgba[2],
            rgba[3],
            falloff,
            hardness,
        )

    def batch_stamp_ellipses(
        self,
        ellipses: list[tuple[float, float, float, float]],
        rgba: colors.ColorRGBA,
        falloff: float = 1.5,
        hardness: float = 0.0,
    ) -> None:
        """Batch-stamp multiple same-style ellipses with one composite pass.

        Uses screen-blend alpha accumulation so all ellipses composite in a
        single pass. Results may differ by +/-1 per channel from sequential
        stamping due to float accumulation order.
        """
        if not ellipses:
            return
        _c_batch_stamp_ellipses(
            self._canvas,
            ellipses,
            rgba[0],
            rgba[1],
            rgba[2],
            rgba[3],
            falloff,
            hardness,
        )

    def batch_stamp_circles(
        self,
        circles: list[tuple[float, float, float]],
        rgba: colors.ColorRGBA,
        falloff: float = 1.5,
        hardness: float = 0.0,
    ) -> None:
        """Batch-stamp circles with shared style parameters."""
        if not circles:
            return
        _c_batch_stamp_circles(
            self._canvas,
            circles,
            rgba[0],
            rgba[1],
            rgba[2],
            rgba[3],
            falloff,
            hardness,
        )

    def stamp_circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        rgba: colors.ColorRGBA,
        falloff: float = 1.5,
        hardness: float = 0.0,
    ) -> None:
        """Alpha-composite a soft circle (ellipse with rx = ry = radius)."""
        self.stamp_ellipse(cx, cy, radius, radius, rgba, falloff, hardness)


def fill_triangle(
    canvas: np.ndarray,
    cx: float,
    top_y: int,
    base_width: float,
    height: int,
    rgba: colors.ColorRGBA,
) -> None:
    """Draw a filled isoceles triangle pointing upward.

    The triangle's apex is at *(cx, top_y)* and its base at
    *top_y + height - 1* spans *base_width* pixels.
    """
    _c_fill_triangle(
        canvas, cx, top_y, base_width, height, rgba[0], rgba[1], rgba[2], rgba[3]
    )


def generate_deciduous_canopy(
    canvas: np.ndarray,
    seed: int,
    size: int,
    canopy_cx: float,
    canopy_cy: float,
    base_radius: float,
    crown_rx_scale: float,
    crown_ry_scale: float,
    canopy_center_x_offset: float,
    tips: list[PixelPos],
    shadow_rgba: colors.ColorRGBA,
    mid_rgba: colors.ColorRGBA,
    highlight_rgba: colors.ColorRGBA,
) -> list[PixelPos]:
    """Generate the deciduous canopy lobe/shading passes in native code.

    Returns the generated lobe centers so higher-level Python code can add
    small branch-extension details without rebuilding the hot lobe loop.
    """
    return _c_generate_deciduous_canopy(
        canvas,
        seed,
        size,
        canopy_cx,
        canopy_cy,
        base_radius,
        crown_rx_scale,
        crown_ry_scale,
        canopy_center_x_offset,
        tips,
        shadow_rgba[0],
        shadow_rgba[1],
        shadow_rgba[2],
        shadow_rgba[3],
        mid_rgba[0],
        mid_rgba[1],
        mid_rgba[2],
        mid_rgba[3],
        highlight_rgba[0],
        highlight_rgba[1],
        highlight_rgba[2],
        highlight_rgba[3],
    )


def paste_sprite(
    sheet: np.ndarray,
    sprite: np.ndarray,
    x0: int,
    y0: int,
) -> None:
    """Alpha-composite *sprite* onto *sheet* at pixel offset (x0, y0)."""
    _c_paste_sprite(sheet, sprite, x0, y0)


def darken_rim(
    canvas: np.ndarray,
    darken: tuple[int, int, int] = (30, 30, 20),
) -> None:
    """Darken the 1px silhouette rim of opaque pixels bordering transparency.

    For every opaque pixel (alpha > 128) adjacent to a transparent pixel
    (alpha == 0) in any cardinal direction, subtract *darken* from the RGB
    channels (clamped to 0). Operates in-place on *canvas*.
    """
    _c_darken_rim(canvas, darken[0], darken[1], darken[2])


def nibble_canopy(
    canvas: np.ndarray,
    seed: int,
    center_x: float,
    center_y: float,
    canopy_radius: float,
    nibble_probability: float = 0.25,
    interior_probability: float = 0.10,
) -> None:
    """Carve small random notches in the canopy edge for silhouette variety.

    Only erodes rim pixels within an elliptical canopy envelope centered at
    *(center_x, center_y)* so trunks are not affected. Uses a C-side PRNG
    seeded from *seed* for deterministic results.
    """
    _c_nibble_canopy(
        canvas,
        seed,
        center_x,
        center_y,
        canopy_radius,
        nibble_probability,
        interior_probability,
    )


def nibble_boulder(
    canvas: np.ndarray,
    seed: int,
    nibble_probability: float = 0.10,
) -> None:
    """Remove random edge pixels from the upper half of a boulder silhouette.

    The bottom edge stays solid (ground contact). Uses a C-side PRNG seeded
    from *seed* for deterministic results.
    """
    _c_nibble_boulder(canvas, seed, nibble_probability)
