"""Reusable RGBA sprite drawing primitives."""

from __future__ import annotations

import math

import numpy as np

from brileta import colors

__all__ = [
    "CanvasStamper",
    "alpha_blend",
    "clamp",
    "composite_over",
    "draw_line",
    "draw_tapered_trunk",
    "draw_thick_line",
    "fill_triangle",
    "paste_sprite",
    "stamp_ellipse",
    "stamp_fuzzy_circle",
]


def clamp(value: int, lo: int = 0, hi: int = 255) -> int:
    """Clamp an integer to [lo, hi]."""
    return max(lo, min(hi, value))


def alpha_blend(canvas: np.ndarray, x: int, y: int, rgba: colors.ColorRGBA) -> None:
    """Alpha-composite a single RGBA pixel onto the canvas."""
    h, w = canvas.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return

    src_a = rgba[3] / 255.0
    if src_a <= 0:
        return

    dst_a = canvas[y, x, 3] / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    if out_a <= 0:
        return

    inv_src = 1.0 - src_a
    for c in range(3):
        canvas[y, x, c] = int(
            (rgba[c] * src_a + float(canvas[y, x, c]) * dst_a * inv_src) / out_a
        )
    canvas[y, x, 3] = int(out_a * 255)


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
    region = canvas[y_min : y_max + 1, x_min : x_max + 1]
    dst_a = region[:, :, 3].astype(np.float32) * np.float32(1.0 / 255.0)
    out_a = src_a + dst_a * (1.0 - src_a)
    safe_out_a = np.maximum(out_a, np.float32(1e-6))
    inv_src = 1.0 - src_a

    src_rgb = np.array(rgb, dtype=np.float32)
    dst_rgb = region[:, :, :3].astype(np.float32)
    sa3 = src_a[:, :, np.newaxis]
    da_inv3 = (dst_a * inv_src)[:, :, np.newaxis]
    soa3 = safe_out_a[:, :, np.newaxis]

    region[:, :, :3] = np.clip(
        (src_rgb * sa3 + dst_rgb * da_inv3) / soa3, 0, 255
    ).astype(np.uint8)
    region[:, :, 3] = np.clip(out_a * 255, 0, 255).astype(np.uint8)


def draw_line(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rgba: colors.ColorRGBA,
) -> None:
    """Draw a 1px Bresenham line between two points."""
    ix0, iy0 = round(x0), round(y0)
    ix1, iy1 = round(x1), round(y1)

    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx - dy

    while True:
        alpha_blend(canvas, ix0, iy0, rgba)
        if ix0 == ix1 and iy0 == iy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            ix0 += sx
        if e2 < dx:
            err += dx
            iy0 += sy


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
    if thickness <= 1:
        draw_line(canvas, x0, y0, x1, y1, rgba)
        return

    # Perpendicular direction for offsetting parallel lines.
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.01:
        alpha_blend(canvas, round(x0), round(y0), rgba)
        return

    # Unit perpendicular vector.
    px = -dy / length
    py = dx / length

    half = thickness / 2.0
    for i in range(thickness):
        offset = -half + 0.5 + i
        ox = px * offset
        oy = py * offset
        draw_line(canvas, x0 + ox, y0 + oy, x1 + ox, y1 + oy, rgba)


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
    h, w = canvas.shape[:2]
    height = y_bottom - y_top + 1
    if height <= 0:
        return

    draw_y_min = max(0, y_top)
    draw_y_max = min(h - 1, y_bottom)
    if draw_y_min > draw_y_max:
        return

    rows = np.arange(height, dtype=np.float32)
    ys = y_top + rows

    # Linear interpolation: 0.0 at top, 1.0 at bottom.
    t = rows / np.float32(max(1, height - 1))
    row_w = np.float32(width_top) + np.float32(width_bottom - width_top) * t

    # Root flare: widen bottom rows.
    if root_flare > 0:
        dist_to_bottom = np.float32(y_bottom) - ys
        flare_mask = dist_to_bottom < np.float32(root_flare)
        flare_t = 1.0 - dist_to_bottom / np.float32(root_flare)
        row_w = row_w + np.where(flare_mask, flare_t * np.float32(1.5), np.float32(0.0))

    row_w = row_w.astype(np.float32)
    half_w = row_w * np.float32(0.5)

    local_start = int(draw_y_min - y_top)
    local_end = int(draw_y_max - y_top + 1)
    row_half_w = half_w[local_start:local_end]

    x_min = max(0, math.floor(cx - float(np.max(row_half_w)) - 0.5))
    x_max = min(w - 1, math.ceil(cx + float(np.max(row_half_w)) + 0.5))
    if x_min > x_max:
        return

    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    dist = np.abs(xs[np.newaxis, :] - np.float32(cx))
    half_w_2d = row_half_w[:, np.newaxis]

    inside = dist <= (half_w_2d + np.float32(0.5))
    edge_mask = dist > (half_w_2d - np.float32(0.5))
    edge = np.maximum(np.float32(0.0), 1.0 - (dist - half_w_2d + np.float32(0.5)))

    row_alpha_i = np.where(
        edge_mask,
        np.floor(np.float32(rgba[3]) * edge).astype(np.int16),
        np.int16(rgba[3]),
    )
    row_alpha_i = np.where(inside, row_alpha_i, np.int16(0))
    src_a = row_alpha_i.astype(np.float32) * np.float32(1.0 / 255.0)
    composite_over(canvas, draw_y_min, draw_y_max, x_min, x_max, src_a, rgba[:3])


def stamp_fuzzy_circle(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    rgba: colors.ColorRGBA,
    falloff: float = 1.5,
    hardness: float = 0.0,
) -> None:
    """Stamp a soft circle with alpha falloff at edges.

    Uses vectorized numpy operations for the whole circle region at once.
    The alpha profile transitions from soft to crisp by adjusting the fully
    opaque core radius fraction and edge falloff exponent via *hardness*.
    """
    h, w = canvas.shape[:2]
    r_ceil = math.ceil(radius) + 1

    # Bounding box clipped to canvas.
    y_min = max(0, int(cy) - r_ceil)
    y_max = min(h - 1, int(cy) + r_ceil)
    x_min = max(0, int(cx) - r_ceil)
    x_max = min(w - 1, int(cx) + r_ceil)
    if y_min > y_max or x_min > x_max:
        return

    # Coordinate grids for the bounding box.
    ys = np.arange(y_min, y_max + 1, dtype=np.float32)
    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    # Distance from center.
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Alpha profile: interpolate from soft (legacy) to hard edge.
    hardness_clamped = max(0.0, min(1.0, hardness))
    inner_fraction = 0.3 + (0.85 - 0.3) * hardness_clamped
    effective_falloff = falloff + 2.5 * hardness_clamped
    inner_r = radius * inner_fraction
    outer_r = radius * (1.0 - inner_fraction)
    safe_outer_r = max(outer_r, 1e-6)
    falloff_region = np.maximum((dist - inner_r) / safe_outer_r, np.float32(0.0))
    outer_alpha = np.clip(
        1.0 - np.power(falloff_region, effective_falloff),
        0.0,
        1.0,
    )
    alpha = np.where(dist <= inner_r, np.float32(1.0), outer_alpha)
    alpha = np.where(dist > radius + 0.5, np.float32(0.0), alpha)

    # Source alpha incorporating the RGBA's own alpha channel.
    src_a = (alpha * (rgba[3] / 255.0)).astype(np.float32)

    # Slice out the affected canvas region.
    region = canvas[y_min : y_max + 1, x_min : x_max + 1]
    dst_a = region[:, :, 3].astype(np.float32) / 255.0

    # Porter-Duff "over" compositing.
    out_a = src_a + dst_a * (1.0 - src_a)
    safe_out_a = np.maximum(out_a, np.float32(1e-6))

    inv_src = 1.0 - src_a
    for c in range(3):
        dst_c = region[:, :, c].astype(np.float32)
        blended = (rgba[c] * src_a + dst_c * dst_a * inv_src) / safe_out_a
        region[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)
    region[:, :, 3] = np.clip(out_a * 255, 0, 255).astype(np.uint8)


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
    if rx <= 0.0 or ry <= 0.0:
        return

    h, w = canvas.shape[:2]
    rx_ceil = math.ceil(rx) + 1
    ry_ceil = math.ceil(ry) + 1

    # Bounding box clipped to canvas.
    y_min = max(0, int(cy) - ry_ceil)
    y_max = min(h - 1, int(cy) + ry_ceil)
    x_min = max(0, int(cx) - rx_ceil)
    x_max = min(w - 1, int(cx) + rx_ceil)
    if y_min > y_max or x_min > x_max:
        return

    ys = np.arange(y_min, y_max + 1, dtype=np.float32)
    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    # Unit distance in normalized ellipse space (1.0 is ellipse boundary).
    dx = (xx - cx) / np.float32(rx)
    dy = (yy - cy) / np.float32(ry)
    dist = np.sqrt(dx**2 + dy**2)

    hardness_clamped = max(0.0, min(1.0, hardness))
    inner_fraction = 0.3 + (0.85 - 0.3) * hardness_clamped
    effective_falloff = falloff + 2.5 * hardness_clamped
    outer_fraction = max(1e-6, 1.0 - inner_fraction)

    falloff_region = np.maximum((dist - inner_fraction) / outer_fraction, 0.0)
    outer_alpha = np.clip(
        1.0 - np.power(falloff_region, effective_falloff),
        0.0,
        1.0,
    )
    alpha = np.where(dist <= inner_fraction, np.float32(1.0), outer_alpha)
    edge_pad = 0.5 / max(rx, ry)
    alpha = np.where(dist > 1.0 + edge_pad, np.float32(0.0), alpha)

    src_a = (alpha * (rgba[3] / 255.0)).astype(np.float32)
    region = canvas[y_min : y_max + 1, x_min : x_max + 1]
    dst_a = region[:, :, 3].astype(np.float32) / 255.0

    out_a = src_a + dst_a * (1.0 - src_a)
    safe_out_a = np.maximum(out_a, np.float32(1e-6))

    inv_src = 1.0 - src_a
    for c in range(3):
        dst_c = region[:, :, c].astype(np.float32)
        blended = (rgba[c] * src_a + dst_c * dst_a * inv_src) / safe_out_a
        region[:, :, c] = np.clip(blended, 0, 255).astype(np.uint8)
    region[:, :, 3] = np.clip(out_a * 255, 0, 255).astype(np.uint8)


class CanvasStamper:
    """Pre-allocated coordinate grids for efficient repeated stamping.

    When stamping many ellipses/circles onto one canvas (e.g. 20+ canopy
    lobes in the deciduous generator), constructing numpy meshgrids and
    temporary arrays per stamp dominates the runtime at small canvas sizes.

    This class pre-computes the full-canvas coordinate grid once and reuses
    sliced views for each stamp, and vectorizes the RGB channel compositing
    to eliminate the per-channel Python loop.
    """

    __slots__ = ("_canvas", "_full_xx", "_full_yy", "_h", "_w")

    def __init__(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        self._canvas = canvas
        self._h = h
        self._w = w
        # Full-canvas coordinate grids, allocated once and sliced per stamp.
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        self._full_yy, self._full_xx = np.meshgrid(ys, xs, indexing="ij")

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
        """Alpha-composite a soft ellipse onto the canvas.

        Equivalent to the standalone ``stamp_ellipse`` but reuses
        pre-computed coordinate grids.
        """
        if rx <= 0.0 or ry <= 0.0:
            return

        rx_ceil = math.ceil(rx) + 1
        ry_ceil = math.ceil(ry) + 1

        # Bounding box clipped to canvas.
        y_min = max(0, int(cy) - ry_ceil)
        y_max = min(self._h - 1, int(cy) + ry_ceil)
        x_min = max(0, int(cx) - rx_ceil)
        x_max = min(self._w - 1, int(cx) + rx_ceil)
        if y_min > y_max or x_min > x_max:
            return

        # Sliced views into pre-computed coordinate grids (no allocation).
        yy = self._full_yy[y_min : y_max + 1, x_min : x_max + 1]
        xx = self._full_xx[y_min : y_max + 1, x_min : x_max + 1]

        # Normalized ellipse-space distance (1.0 = ellipse boundary).
        dx = (xx - np.float32(cx)) / np.float32(rx)
        dy = (yy - np.float32(cy)) / np.float32(ry)
        dist = np.sqrt(dx * dx + dy * dy)

        # Alpha profile: interpolate from soft to hard edge via hardness.
        hardness_clamped = max(0.0, min(1.0, hardness))
        inner_fraction = 0.3 + 0.55 * hardness_clamped
        effective_falloff = falloff + 2.5 * hardness_clamped
        outer_fraction = max(1e-6, 1.0 - inner_fraction)

        falloff_region = np.maximum((dist - inner_fraction) / outer_fraction, 0.0)
        alpha = np.where(
            dist <= inner_fraction,
            np.float32(1.0),
            np.clip(1.0 - np.power(falloff_region, effective_falloff), 0.0, 1.0),
        )
        edge_pad = 0.5 / max(rx, ry)
        alpha = np.where(dist > 1.0 + edge_pad, np.float32(0.0), alpha)
        src_a = alpha * np.float32(rgba[3] / 255.0)
        composite_over(self._canvas, y_min, y_max, x_min, x_max, src_a, rgba[:3])

    def batch_stamp_ellipses(
        self,
        ellipses: list[tuple[float, float, float, float]],
        rgba: colors.ColorRGBA,
        falloff: float = 1.5,
        hardness: float = 0.0,
    ) -> None:
        """Batch-stamp multiple same-style ellipses with one composite pass.

        The provided ellipses use ``(cx, cy, rx, ry)`` tuples. Because every
        ellipse shares the same RGB colour, sequential Porter-Duff "over"
        compositing reduces to a single composite whose alpha is the screen
        blend of the individual alphas:
        ``combined = 1 - (1 - a1)(1 - a2)...(1 - aN)``.

        The batch path accumulates source alpha in float32 and quantizes to
        uint8 once at the end, whereas sequential stamping quantizes after
        each ellipse. This makes batch results slightly *more* accurate
        (fewer rounding losses) but can differ by +/-1-2 per channel from the
        sequential path.
        """
        if not ellipses:
            return
        if len(ellipses) == 1:
            cx, cy, rx, ry = ellipses[0]
            self.stamp_ellipse(cx, cy, rx, ry, rgba, falloff, hardness)
            return

        bounds: list[tuple[int, int, int, int, float, float, float, float]] = []
        for cx, cy, rx, ry in ellipses:
            if rx <= 0.0 or ry <= 0.0:
                continue
            rx_ceil = math.ceil(rx) + 1
            ry_ceil = math.ceil(ry) + 1
            y_min = max(0, int(cy) - ry_ceil)
            y_max = min(self._h - 1, int(cy) + ry_ceil)
            x_min = max(0, int(cx) - rx_ceil)
            x_max = min(self._w - 1, int(cx) + rx_ceil)
            if y_min > y_max or x_min > x_max:
                continue
            bounds.append((y_min, y_max, x_min, x_max, cx, cy, rx, ry))

        if not bounds:
            return

        union_y_min = min(y_min for y_min, _, _, _, _, _, _, _ in bounds)
        union_y_max = max(y_max for _, y_max, _, _, _, _, _, _ in bounds)
        union_x_min = min(x_min for _, _, x_min, _, _, _, _, _ in bounds)
        union_x_max = max(x_max for _, _, _, x_max, _, _, _, _ in bounds)

        yy = self._full_yy[union_y_min : union_y_max + 1, union_x_min : union_x_max + 1]
        xx = self._full_xx[union_y_min : union_y_max + 1, union_x_min : union_x_max + 1]
        # Track the product of (1 - src_a_i) across all ellipses so we can
        # derive the screen-blended combined source alpha at the end. The
        # global opacity (rgba[3]/255) is folded into each per-ellipse alpha
        # *before* accumulation, matching sequential Porter-Duff behaviour.
        remaining = np.ones_like(yy, dtype=np.float32)
        opacity = np.float32(rgba[3] / 255.0)

        hardness_clamped = max(0.0, min(1.0, hardness))
        inner_fraction = 0.3 + 0.55 * hardness_clamped
        effective_falloff = falloff + 2.5 * hardness_clamped
        outer_fraction = max(1e-6, 1.0 - inner_fraction)

        for _, _, _, _, cx, cy, rx, ry in bounds:
            dx = (xx - np.float32(cx)) / np.float32(rx)
            dy = (yy - np.float32(cy)) / np.float32(ry)
            dist = np.sqrt(dx * dx + dy * dy)

            falloff_region = np.maximum((dist - inner_fraction) / outer_fraction, 0.0)
            alpha = np.where(
                dist <= inner_fraction,
                np.float32(1.0),
                np.clip(1.0 - np.power(falloff_region, effective_falloff), 0.0, 1.0),
            )
            edge_pad = 0.5 / max(rx, ry)
            alpha = np.where(dist > 1.0 + edge_pad, np.float32(0.0), alpha)
            remaining *= 1.0 - alpha * opacity

        src_a = 1.0 - remaining
        composite_over(
            self._canvas,
            union_y_min,
            union_y_max,
            union_x_min,
            union_x_max,
            src_a,
            rgba[:3],
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
        ellipses = [(cx, cy, r, r) for cx, cy, r in circles]
        self.batch_stamp_ellipses(ellipses, rgba, falloff, hardness)

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
    h, w = canvas.shape[:2]
    if height <= 0:
        return

    bottom_y = top_y + height - 1
    draw_y_min = max(0, top_y)
    draw_y_max = min(h - 1, bottom_y)
    if draw_y_min > draw_y_max:
        return

    rows = np.arange(height, dtype=np.float32)
    t = rows / np.float32(max(1, height - 1))
    row_w = 1.0 + np.float32(base_width - 1.0) * np.power(t, np.float32(0.8))
    half_w = row_w * np.float32(0.5)

    local_start = int(draw_y_min - top_y)
    local_end = int(draw_y_max - top_y + 1)
    row_half_w = half_w[local_start:local_end]

    x_min = max(0, math.floor(cx - float(np.max(row_half_w)) - 0.5))
    x_max = min(w - 1, math.ceil(cx + float(np.max(row_half_w)) + 0.5))
    if x_min > x_max:
        return

    xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    dist = np.abs(xs[np.newaxis, :] - np.float32(cx))
    half_w_2d = row_half_w[:, np.newaxis]

    inside = dist <= (half_w_2d + np.float32(0.5))
    edge_mask = dist > (half_w_2d - np.float32(0.5))
    edge = np.maximum(np.float32(0.0), 1.0 - (dist - half_w_2d + np.float32(0.5)))

    row_alpha_i = np.where(
        edge_mask,
        np.floor(np.float32(rgba[3]) * edge).astype(np.int16),
        np.int16(rgba[3]),
    )
    row_alpha_i = np.where(inside, row_alpha_i, np.int16(0))
    src_a = row_alpha_i.astype(np.float32) * np.float32(1.0 / 255.0)
    composite_over(canvas, draw_y_min, draw_y_max, x_min, x_max, src_a, rgba[:3])


def paste_sprite(
    sheet: np.ndarray,
    sprite: np.ndarray,
    x0: int,
    y0: int,
) -> None:
    """Alpha-composite *sprite* onto *sheet* at pixel offset (x0, y0).

    Uses vectorized numpy operations instead of per-pixel Python loops.
    """
    sh, sw = sprite.shape[:2]
    sheet_h, sheet_w = sheet.shape[:2]

    # Clip the paste region to sheet bounds.
    src_y0 = max(0, -y0)
    src_x0 = max(0, -x0)
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    h = min(sh - src_y0, sheet_h - dst_y0)
    w = min(sw - src_x0, sheet_w - dst_x0)
    if h <= 0 or w <= 0:
        return

    src = sprite[src_y0 : src_y0 + h, src_x0 : src_x0 + w]
    dst = sheet[dst_y0 : dst_y0 + h, dst_x0 : dst_x0 + w]

    src_a = src[:, :, 3].astype(np.float32) / 255.0
    dst_a = dst[:, :, 3].astype(np.float32) / 255.0

    out_a = src_a + dst_a * (1.0 - src_a)
    safe_out_a = np.maximum(out_a, 1e-6)
    inv_src = 1.0 - src_a

    sa3 = src_a[:, :, np.newaxis]
    da_inv3 = (dst_a * inv_src)[:, :, np.newaxis]
    soa3 = safe_out_a[:, :, np.newaxis]

    src_rgb = src[:, :, :3].astype(np.float32)
    dst_rgb = dst[:, :, :3].astype(np.float32)

    dst[:, :, :3] = np.clip((src_rgb * sa3 + dst_rgb * da_inv3) / soa3, 0, 255).astype(
        np.uint8
    )
    dst[:, :, 3] = np.clip(out_a * 255, 0, 255).astype(np.uint8)
