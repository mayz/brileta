"""Procedural tree sprite generator.

Generates RGBA numpy arrays for tree sprites using recursive branching
for trunks and various canopy techniques per archetype. Each tree's
visual is deterministic from its spatial position hash.

Archetypes:
  - DECIDUOUS: Round canopy with trunk. Warm greens, organic branching.
  - CONIFER: Triangular silhouette with stacked tiers. Cool dark greens.
  - DEAD: Bare branching skeleton. Gray-brown palette with optional moss.
  - SAPLING: Small young tree with thin trunk and sparse canopy.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from brileta import colors, config
from brileta.game.enums import CreatureSize
from brileta.types import MapDecorationSeed, PixelPos, SpatialSeed, WorldTileCoord
from brileta.util import rng as brileta_rng
from brileta.util.noise import FractalType, NoiseGenerator, NoiseType

# ---------------------------------------------------------------------------
# Tree archetype enum
# ---------------------------------------------------------------------------


class TreeArchetype(Enum):
    """Tree species archetype that determines visual generation rules."""

    DECIDUOUS = "deciduous"
    CONIFER = "conifer"
    DEAD = "dead"
    SAPLING = "sapling"


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

# Trunk color families - warm to cool browns.
_TRUNK_PALETTES: list[colors.Color] = [
    (90, 58, 32),  # Warm brown (classic)
    (100, 50, 25),  # Reddish brown (oak-like)
    (85, 70, 55),  # Cool gray-brown (ash-like)
    (60, 40, 20),  # Dark brown (walnut-like)
]

# Deciduous canopy base hues - will get per-tree R/G jitter.
_DECIDUOUS_CANOPY_BASES: list[colors.Color] = [
    (65, 145, 50),  # Vivid leafy green
    (55, 135, 45),  # Slightly cooler
    (75, 150, 40),  # Warmer, more yellow
    (50, 140, 55),  # Blue-green tint
    (80, 150, 30),  # Chartreuse
    (90, 145, 25),  # Yellow-olive
    (35, 130, 60),  # Teal-green
    (40, 135, 55),  # Blue-green
    (55, 120, 35),  # Dark olive
    (110, 130, 35),  # Warm olive
    (95, 140, 30),  # Golden-green
]

# Conifer canopy base hues - cooler, distinctly darker than deciduous.
_CONIFER_CANOPY_BASES: list[colors.Color] = [
    (30, 90, 30),  # Classic dark pine
    (25, 85, 35),  # Blue-green pine
    (35, 95, 25),  # Yellow-green pine
    (20, 82, 40),  # Darker blue-green
    (40, 98, 22),  # Warm dark pine
]

# Dead tree trunk and branch base colors.
_DEAD_TRUNK_BASES: list[colors.Color] = [
    (80, 70, 55),  # Warm gray
    (75, 65, 50),  # Darker warm gray
    (90, 80, 65),  # Light ash
]

# Moss/lichen RGBA for dead trees (slightly transparent).
_MOSS_RGBA: colors.ColorRGBA = (50, 70, 40, 180)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------


def _clamp(value: int, lo: int = 0, hi: int = 255) -> int:
    """Clamp an integer to [lo, hi]."""
    return max(lo, min(hi, value))


def _alpha_blend(canvas: np.ndarray, x: int, y: int, rgba: colors.ColorRGBA) -> None:
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


def _composite_over(
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


def _draw_line(
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
        _alpha_blend(canvas, ix0, iy0, rgba)
        if ix0 == ix1 and iy0 == iy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            ix0 += sx
        if e2 < dx:
            err += dx
            iy0 += sy


def _draw_thick_line(
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
        _draw_line(canvas, x0, y0, x1, y1, rgba)
        return

    # Perpendicular direction for offsetting parallel lines.
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.01:
        _alpha_blend(canvas, round(x0), round(y0), rgba)
        return

    # Unit perpendicular vector.
    px = -dy / length
    py = dx / length

    half = thickness / 2.0
    for i in range(thickness):
        offset = -half + 0.5 + i
        ox = px * offset
        oy = py * offset
        _draw_line(canvas, x0 + ox, y0 + oy, x1 + ox, y1 + oy, rgba)


def _draw_tapered_trunk(
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
    _composite_over(canvas, draw_y_min, draw_y_max, x_min, x_max, src_a, rgba[:3])


def _stamp_fuzzy_circle(
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


def _stamp_ellipse(
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


class _CanvasStamper:
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

        Equivalent to the standalone ``_stamp_ellipse`` but reuses
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
        _composite_over(self._canvas, y_min, y_max, x_min, x_max, src_a, rgba[:3])

    def batch_stamp_ellipses(
        self,
        ellipses: list[tuple[float, float, float, float]],
        rgba: colors.ColorRGBA,
        falloff: float = 1.5,
        hardness: float = 0.0,
    ) -> None:
        """Batch-stamp multiple same-style ellipses with one composite pass.

        The provided ellipses use ``(cx, cy, rx, ry)`` tuples.  Because every
        ellipse shares the same RGB colour, sequential Porter-Duff "over"
        compositing reduces to a single composite whose alpha is the screen
        blend of the individual alphas:
        ``combined = 1 - (1 - a1)(1 - a2)...(1 - aN)``.

        The batch path accumulates source alpha in float32 and quantizes to
        uint8 once at the end, whereas sequential stamping quantizes after
        each ellipse.  This makes batch results slightly *more* accurate
        (fewer rounding losses) but can differ by ±1-2 per channel from the
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
        # derive the screen-blended combined source alpha at the end.  The
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
        _composite_over(
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


def _darken_canopy_rim(
    canvas: np.ndarray, darken: tuple[int, int, int] = (30, 30, 20)
) -> None:
    """Darken canopy edge pixels that border transparency.

    For every opaque pixel (alpha > 128) adjacent to a transparent pixel
    (alpha == 0) in any cardinal direction, subtract *darken* from the RGB
    channels (clamped to 0). This creates a 1px dark outline around the
    canopy silhouette so trees pop against any terrain background.

    Operates in-place on *canvas* using vectorized numpy shifts.
    """
    rim = _canopy_rim_mask(canvas[:, :, 3])

    # Darken rim pixels.
    for c in range(3):
        channel = canvas[:, :, c].astype(np.int16)
        channel[rim] -= darken[c]
        canvas[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)


def _canopy_rim_mask(alpha: np.ndarray) -> np.ndarray:
    """Return a mask for opaque canopy edge pixels that border transparency."""
    opaque = alpha > 128
    transparent = alpha == 0

    # Shift transparent mask in four directions; pad with True (border counts).
    border_up = np.pad(transparent[:-1, :], ((1, 0), (0, 0)), constant_values=True)
    border_down = np.pad(transparent[1:, :], ((0, 1), (0, 0)), constant_values=True)
    border_left = np.pad(transparent[:, :-1], ((0, 0), (1, 0)), constant_values=True)
    border_right = np.pad(transparent[:, 1:], ((0, 0), (0, 1)), constant_values=True)
    return opaque & (border_up | border_down | border_left | border_right)


def _nibble_canopy_silhouette(
    canvas: np.ndarray,
    rng: np.random.Generator,
    canopy_center: PixelPos,
    canopy_radius: float,
    nibble_probability: float = 0.25,
    interior_probability: float = 0.10,
) -> None:
    """Carve small random notches in the canopy edge for silhouette variety."""
    alpha = canvas[:, :, 3]
    rim = _canopy_rim_mask(alpha)

    # Limit nibbles to the canopy envelope so trunks are not eroded.
    center_x, center_y = canopy_center
    h, w = alpha.shape
    ys, xs = np.indices((h, w), dtype=np.float32)
    safe_radius = max(canopy_radius, 1e-6)
    dx = (xs - np.float32(center_x)) / safe_radius
    dy = (ys - np.float32(center_y)) / (safe_radius * 0.9)
    canopy_region = (dx * dx + dy * dy) <= 1.6
    rim &= canopy_region

    if not np.any(rim):
        return

    nibble_chance = float(np.clip(nibble_probability, 0.0, 1.0))
    interior_chance = float(np.clip(interior_probability, 0.0, 1.0))

    nibble_mask = rim & (rng.random(alpha.shape) < nibble_chance)
    if not np.any(nibble_mask):
        return
    alpha[nibble_mask] = 0

    if interior_chance <= 0.0:
        return

    interior_mask = nibble_mask & (rng.random(alpha.shape) < interior_chance)
    if not np.any(interior_mask):
        return

    ys_i16 = ys.astype(np.int16)
    xs_i16 = xs.astype(np.int16)
    step_x = np.sign(np.float32(center_x) - xs).astype(np.int16)
    step_y = np.sign(np.float32(center_y) - ys).astype(np.int16)

    inner_x = np.clip(xs_i16 + step_x, 0, w - 1)
    inner_y = np.clip(ys_i16 + step_y, 0, h - 1)
    target_y = inner_y[interior_mask]
    target_x = inner_x[interior_mask]
    target_opaque = alpha[target_y, target_x] > 128
    alpha[target_y[target_opaque], target_x[target_opaque]] = 0


def _fill_triangle(
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
    _composite_over(canvas, draw_y_min, draw_y_max, x_min, x_max, src_a, rgba[:3])


# ---------------------------------------------------------------------------
# Recursive branching
# ---------------------------------------------------------------------------


def _branch(
    canvas: np.ndarray,
    tips: list[PixelPos],
    x: float,
    y: float,
    angle: float,
    length: float,
    thickness: int,
    depth: int,
    rng: np.random.Generator,
    rgba: colors.ColorRGBA,
    spread_base: float = 0.5,
    shrink: float = 0.7,
    asymmetry: float = 0.0,
) -> None:
    """Recursively draw branches and collect branch-tip positions.

    Each recursion level reduces *length* by *shrink*, reduces *thickness*
    by 1, and forks into left/right children at +-*spread_base* radians
    (plus random jitter). Leaf-stamping functions use the collected *tips*
    to place canopy clusters.
    """
    if depth <= 0 or length < 1.5:
        tips.append((x, y))
        return

    # End point of this segment.
    x2 = x + length * math.sin(angle)
    y2 = y - length * math.cos(angle)

    _draw_thick_line(canvas, x, y, x2, y2, rgba, thickness)

    # Fork into two children with jittered spread.
    spread = spread_base + rng.uniform(-0.15, 0.15)
    left_spread = spread + asymmetry * rng.uniform(-0.2, 0.2)
    right_spread = spread + asymmetry * rng.uniform(-0.2, 0.2)
    child_length = length * (shrink + rng.uniform(-0.1, 0.1))
    child_thickness = max(1, thickness - 1)

    # Lighten branches toward the tips.
    lighter: colors.ColorRGBA = (
        min(255, rgba[0] + 8),
        min(255, rgba[1] + 8),
        min(255, rgba[2] + 8),
        rgba[3],
    )

    _branch(
        canvas,
        tips,
        x2,
        y2,
        angle - left_spread,
        child_length,
        child_thickness,
        depth - 1,
        rng,
        lighter,
        spread_base,
        shrink,
        asymmetry,
    )
    _branch(
        canvas,
        tips,
        x2,
        y2,
        angle + right_spread,
        child_length,
        child_thickness,
        depth - 1,
        rng,
        lighter,
        spread_base,
        shrink,
        asymmetry,
    )


# ---------------------------------------------------------------------------
# Archetype generators
# ---------------------------------------------------------------------------


def _generate_deciduous(
    canvas: np.ndarray, size: int, rng: np.random.Generator
) -> None:
    """Generate a deciduous tree with three-step canopy shading.

    The canopy is built from several foliage lobes stamped in three passes
    (shadow, mid-tone, and highlight) so the silhouette has visible
    concavities instead of reading as one smooth ball.
    A final rim-darkening pass outlines the canopy silhouette.
    """
    # --- Pick colors with per-tree jitter ---
    trunk_base = _TRUNK_PALETTES[int(rng.integers(len(_TRUNK_PALETTES)))]
    canopy_base = list(
        _DECIDUOUS_CANOPY_BASES[int(rng.integers(len(_DECIDUOUS_CANOPY_BASES)))]
    )
    # Independent R/G/B channel jitter for hue variation across trees.
    canopy_base[0] = _clamp(canopy_base[0] + int(rng.integers(-20, 21)), 15, 120)
    canopy_base[1] = _clamp(canopy_base[1] + int(rng.integers(-20, 21)), 90, 185)
    canopy_base[2] = _clamp(canopy_base[2] + int(rng.integers(-15, 16)), 10, 85)

    # Three-step color palette: shadow, mid-tone, highlight.
    # Aggressive deltas (50+) so zones read as distinct at 20px.
    shadow_rgba: colors.ColorRGBA = (
        max(0, canopy_base[0] - 55),
        max(0, canopy_base[1] - 50),
        max(0, canopy_base[2] - 30),
        230,
    )
    mid_rgba: colors.ColorRGBA = (
        canopy_base[0],
        canopy_base[1],
        canopy_base[2],
        220,
    )
    # Highlight shifts toward yellow for a sun-warm feel.
    highlight_rgba: colors.ColorRGBA = (
        min(255, canopy_base[0] + 55),
        min(255, canopy_base[1] + 45),
        min(255, canopy_base[2] + 20),
        200,
    )
    trunk_rgba: colors.ColorRGBA = (*trunk_base, 255)

    # Crown silhouette variation controls ellipse aspect and canopy center bias.
    crown_roll = int(rng.integers(0, 100))
    crown_rx_scale = 1.0
    crown_ry_scale = 1.0
    canopy_center_x_offset = 0.0
    if 35 <= crown_roll <= 54:
        crown_rx_scale = 0.75
        crown_ry_scale = 1.25
    elif 55 <= crown_roll <= 84:
        crown_rx_scale = 1.3
        crown_ry_scale = 0.8
    elif crown_roll >= 85:
        canopy_center_x_offset = float(rng.uniform(-size * 0.18, size * 0.18))

    # --- Trunk geometry ---
    lean = float(rng.uniform(-1.5, 1.5))
    cx = size / 2.0 + lean * 0.3
    trunk_bottom = size - 1
    trunk_height = int(size * rng.uniform(0.35, 0.45))
    trunk_top = trunk_bottom - trunk_height
    trunk_w_bot = max(2.0, size * 0.15 + rng.uniform(-0.3, 0.3))
    trunk_w_top = max(1.0, trunk_w_bot * 0.5)

    _draw_tapered_trunk(
        canvas,
        cx,
        trunk_bottom,
        trunk_top,
        trunk_w_bot,
        trunk_w_top,
        trunk_rgba,
        root_flare=2,
    )

    # --- Branch from trunk top (short, mostly hidden by canopy) ---
    tips: list[PixelPos] = []
    branch_x = cx + lean * 0.5
    branch_y = float(trunk_top)

    _branch(
        canvas,
        tips,
        branch_x,
        branch_y,
        angle=lean * 0.08,
        length=size * 0.15,
        thickness=1,
        depth=2,
        rng=rng,
        rgba=trunk_rgba,
        spread_base=0.6,
        shrink=0.65,
    )
    canopy_cx = branch_x + canopy_center_x_offset
    # Center canopy at the mean branch-tip height rather than the trunk
    # junction so stamps (especially the shadow pass) don't extend down
    # and obscure the trunk.
    canopy_cy = (
        float(np.mean([ty for _, ty in tips])) if tips else branch_y - size * 0.1
    )

    # --- Lobe-driven canopy structure ---
    base_radius = float(size * rng.uniform(0.20, 0.28))

    # Prevent the canopy from floating above the trunk. The central fill
    # stamps extend about 0.7 * base_radius below canopy_cy, so place the
    # center low enough that they bridge the trunk junction.
    min_canopy_cy = float(trunk_top) - base_radius * 0.55
    canopy_cy = max(canopy_cy, min_canopy_cy)

    n_lobes = int(rng.integers(3, 6))
    lobe_centers: list[PixelPos] = []
    base_angle_step = 2.0 * math.pi / n_lobes
    lobe_angle_offset = float(rng.uniform(-math.pi, math.pi))
    for i in range(n_lobes):
        angle = lobe_angle_offset + base_angle_step * i + float(rng.uniform(-0.4, 0.4))
        dist = base_radius * float(rng.uniform(0.45, 0.65))
        # Bottom lobes hang a little lower for an irregular lower canopy edge.
        vertical_scale = 0.9 if math.sin(angle) > 0.0 else 0.7
        lx = canopy_cx + math.cos(angle) * dist
        ly = canopy_cy + math.sin(angle) * dist * vertical_scale
        lobe_centers.append((lx, ly))

    stamper = _CanvasStamper(canvas)
    central_fills: list[tuple[float, float, float, float]] = []
    shadows: list[tuple[float, float, float, float]] = []
    mids: list[tuple[float, float, float, float]] = []
    highlights: list[tuple[float, float, float, float]] = []

    # Central fill: a solid core connecting all lobes so the canopy reads
    # as one mass with bumps rather than disconnected amoeba blobs.
    for _ in range(int(rng.integers(1, 3))):
        cr = base_radius * float(rng.uniform(0.55, 0.70))
        central_fills.append(
            (
                canopy_cx + float(rng.uniform(-0.5, 0.5)),
                canopy_cy + float(rng.uniform(-0.5, 0.3)),
                cr * crown_rx_scale,
                cr * crown_ry_scale,
            )
        )

    # Pass 1: Shadow zone.
    for lx, ly in lobe_centers:
        for _ in range(int(rng.integers(1, 3))):
            sx = lx + float(rng.uniform(-size * 0.05, size * 0.05))
            sy = ly + float(rng.uniform(-size * 0.07, size * 0.04))
            r = base_radius * float(rng.uniform(0.62, 0.76))
            shadows.append(
                (
                    sx,
                    sy,
                    r * crown_rx_scale,
                    r * crown_ry_scale,
                )
            )

    n_tip_shadow = min(len(tips), int(rng.integers(1, 3)))
    if n_tip_shadow > 0:
        tip_indices = np.atleast_1d(
            rng.choice(len(tips), size=n_tip_shadow, replace=False)
        )
        for tip_index in tip_indices:
            tx, ty = tips[int(tip_index)]
            r = base_radius * float(rng.uniform(0.50, 0.68))
            shadows.append(
                (
                    tx + canopy_center_x_offset + float(rng.uniform(-0.5, 0.5)),
                    ty - 1.0 + float(rng.uniform(-0.5, 0.3)),
                    r * crown_rx_scale,
                    r * crown_ry_scale,
                )
            )

    # Pass 2: Mid-tone zone.
    for lx, ly in lobe_centers:
        for _ in range(int(rng.integers(1, 3))):
            mx = lx + float(rng.uniform(-size * 0.04, size * 0.04))
            my = ly + float(rng.uniform(-size * 0.05, size * 0.03))
            r = base_radius * float(rng.uniform(0.58, 0.72))
            mids.append(
                (
                    mx,
                    my,
                    r * crown_rx_scale,
                    r * crown_ry_scale,
                )
            )

    n_tip_mid = min(len(tips), int(rng.integers(1, 3)))
    if n_tip_mid > 0:
        tip_indices = np.atleast_1d(
            rng.choice(len(tips), size=n_tip_mid, replace=False)
        )
        for tip_index in tip_indices:
            tx, ty = tips[int(tip_index)]
            r = base_radius * float(rng.uniform(0.46, 0.62))
            mids.append(
                (
                    tx + canopy_center_x_offset + float(rng.uniform(-0.6, 0.6)),
                    ty - 1.1 + float(rng.uniform(-0.6, 0.2)),
                    r * crown_rx_scale,
                    r * crown_ry_scale,
                )
            )

    # Pass 3: Highlight zone.
    for lx, ly in lobe_centers:
        for _ in range(int(rng.integers(1, 3))):
            hx = lx + float(rng.uniform(-size * 0.03, size * 0.03))
            hy = ly - size * 0.05 + float(rng.uniform(-size * 0.03, size * 0.02))
            r = base_radius * float(rng.uniform(0.45, 0.60))
            highlights.append(
                (
                    hx,
                    hy,
                    r * crown_rx_scale,
                    r * crown_ry_scale,
                )
            )

    n_tip_highlight = min(len(tips), int(rng.integers(1, 3)))
    if n_tip_highlight > 0:
        tip_indices = np.atleast_1d(
            rng.choice(len(tips), size=n_tip_highlight, replace=False)
        )
        for tip_index in tip_indices:
            tx, ty = tips[int(tip_index)]
            r = base_radius * float(rng.uniform(0.38, 0.50))
            highlights.append(
                (
                    tx + canopy_center_x_offset + float(rng.uniform(-0.5, 0.5)),
                    ty - 1.3 + float(rng.uniform(-0.4, 0.2)),
                    r * crown_rx_scale,
                    r * crown_ry_scale,
                )
            )

    # Batch by shading group to reduce Python-to-numpy call overhead.
    stamper.batch_stamp_ellipses(
        central_fills,
        shadow_rgba,
        falloff=1.6,
        hardness=0.7,
    )
    stamper.batch_stamp_ellipses(
        shadows,
        shadow_rgba,
        falloff=1.8,
        hardness=0.8,
    )
    stamper.batch_stamp_ellipses(
        mids,
        mid_rgba,
        falloff=1.5,
        hardness=0.7,
    )
    stamper.batch_stamp_ellipses(
        highlights,
        highlight_rgba,
        falloff=1.3,
        hardness=0.6,
    )

    # Add a few tiny branch segments that poke out from lobe edges.
    branch_tip_rgba: colors.ColorRGBA = (
        max(0, shadow_rgba[0] - 10),
        max(0, shadow_rgba[1] - 10),
        max(0, shadow_rgba[2] - 5),
        200,
    )
    n_extensions = min(len(lobe_centers), int(rng.integers(2, 5)))
    if n_extensions > 0:
        extension_indices = np.atleast_1d(
            rng.choice(len(lobe_centers), size=n_extensions, replace=False)
        )
        for lobe_index in extension_indices:
            lx, ly = lobe_centers[int(lobe_index)]
            dx = lx - canopy_cx
            dy = ly - canopy_cy
            direction_length = math.hypot(dx, dy)
            if direction_length <= 1e-5:
                continue

            ux = dx / direction_length
            uy = dy / direction_length
            start_dist = base_radius * float(rng.uniform(0.70, 0.95))
            sx = canopy_cx + ux * start_dist
            sy = canopy_cy + uy * start_dist
            ext_len = float(rng.uniform(2.0, 3.2))
            ex = sx + ux * ext_len
            ey = sy + uy * ext_len
            _draw_line(canvas, sx, sy, ex, ey, branch_tip_rgba)

    _nibble_canopy_silhouette(
        canvas,
        rng,
        (canopy_cx, canopy_cy),
        canopy_radius=base_radius * 2.0,
        nibble_probability=0.25,
        interior_probability=0.10,
    )
    _darken_canopy_rim(canvas)


def _generate_conifer(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a conifer tree: thin trunk with stacked triangular tiers.

    Tiers stack from bottom (widest, darkest) to top (narrowest, lightest)
    with steep taper for an unmistakable pointed silhouette. Distinctly
    darker than deciduous trees so the two archetypes read differently.

    Each tier gets independent x-offset jitter and width variation so
    the silhouette breaks away from a perfectly symmetric triangle.
    A slight overall lean tilts the whole tree for natural variety.
    """
    trunk_base = _TRUNK_PALETTES[int(rng.integers(len(_TRUNK_PALETTES)))]
    canopy_base = list(
        _CONIFER_CANOPY_BASES[int(rng.integers(len(_CONIFER_CANOPY_BASES)))]
    )
    canopy_base[0] = _clamp(canopy_base[0] + int(rng.integers(-15, 16)), 10, 60)
    canopy_base[1] = _clamp(canopy_base[1] + int(rng.integers(-15, 16)), 55, 130)
    canopy_base[2] = _clamp(canopy_base[2] + int(rng.integers(-10, 11)), 10, 60)

    trunk_rgba: colors.ColorRGBA = (*trunk_base, 255)

    # Subtle overall lean so not every conifer is perfectly vertical.
    lean = float(rng.uniform(-0.5, 0.5))
    cx = size / 2.0 + lean * 0.2
    trunk_bottom = size - 1

    _draw_tapered_trunk(
        canvas,
        cx,
        trunk_bottom,
        1,
        width_bottom=max(1.5, size * 0.1),
        width_top=1.0,
        rgba=trunk_rgba,
        root_flare=1,
    )

    # Stacked triangular tiers - taller, narrower, more numerous.
    n_tiers = int(rng.integers(3, 6))
    max_width = float(size * rng.uniform(0.35, 0.55))
    tier_height_base = size * 0.35
    # Start tiers above the trunk's visible base.
    current_bottom = trunk_bottom - int(size * 0.2)

    for i in range(n_tiers):
        t = i / max(1, n_tiers - 1) if n_tiers > 1 else 0.0

        # Base taper with per-tier width jitter (+/-10%) so the silhouette
        # is not a perfectly smooth cone.
        base_tier_w = max_width * (1.0 - t * 0.65)
        width_jitter = float(rng.uniform(0.90, 1.10))
        tier_w = base_tier_w * width_jitter
        tier_h = max(3, int(tier_height_base * (1.0 - t * 0.2)))

        # Per-tier x-offset: shift each tier's center up to 0.7px from trunk.
        # Upper tiers (higher t) also accumulate the lean offset so the
        # whole tree visually leans in one direction.
        tier_cx = cx + float(rng.uniform(-0.7, 0.7)) + lean * t * 0.4

        # Stronger shade progression for visible light-to-dark banding.
        shade = int(t * 35)
        tier_rgba: colors.ColorRGBA = (
            min(255, canopy_base[0] + shade),
            min(255, canopy_base[1] + shade),
            min(255, canopy_base[2] + shade),
            240,
        )

        tier_top = current_bottom - tier_h
        _fill_triangle(canvas, tier_cx, tier_top, tier_w, tier_h, tier_rgba)

        # Next tier starts partway up this one (overlap ~35%).
        current_bottom = tier_top + max(1, int(tier_h * 0.35))

    _darken_canopy_rim(canvas)


def _generate_dead(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a dead/bare tree: exposed branching skeleton, no canopy.

    Uses deeper branching, wider spread, and a thicker trunk than other
    archetypes so the bare skeletal structure is clearly visible and
    immediately distinguishable from green canopy trees.
    """
    trunk_base = _DEAD_TRUNK_BASES[int(rng.integers(len(_DEAD_TRUNK_BASES)))]
    trunk_rgba: colors.ColorRGBA = (
        _clamp(trunk_base[0] + int(rng.integers(-8, 9))),
        _clamp(trunk_base[1] + int(rng.integers(-6, 7))),
        _clamp(trunk_base[2] + int(rng.integers(-6, 7))),
        255,
    )

    # More pronounced lean for dead trees (weathered/weakened).
    lean = float(rng.uniform(-2.0, 2.0))
    cx = size / 2.0 + lean * 0.3
    trunk_bottom = size - 1
    trunk_height = int(size * rng.uniform(0.45, 0.55))
    trunk_top = trunk_bottom - trunk_height

    _draw_tapered_trunk(
        canvas,
        cx,
        trunk_bottom,
        trunk_top,
        width_bottom=max(3.0, size * 0.18),
        width_top=max(1.5, size * 0.08),
        rgba=trunk_rgba,
        root_flare=2,
    )

    # Recursive branching - the main visual for dead trees.
    # Deeper, wider, and thicker than deciduous branching so the bare
    # skeleton reads clearly without any canopy to fill in gaps.
    tips: list[PixelPos] = []
    _branch(
        canvas,
        tips,
        cx + lean * 0.5,
        float(trunk_top),
        angle=lean * 0.06,
        length=size * 0.28,
        thickness=3,
        depth=4,
        rng=rng,
        rgba=trunk_rgba,
        spread_base=0.70,
        shrink=0.65,
        asymmetry=0.4,
    )

    # Moss/lichen at branch junctions - stamped as small fuzzy circles
    # so they're visible at 20px (single pixels disappear).
    n_moss = int(rng.integers(0, 5))
    for _ in range(n_moss):
        if tips:
            idx = int(rng.integers(len(tips)))
            mx, my = tips[idx]
            _stamp_fuzzy_circle(canvas, mx, my, 1.5, _MOSS_RGBA, falloff=1.5)


def _generate_sapling(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a small sapling with simplified three-step canopy shading.

    Uses fewer/smaller canopy lobes than deciduous trees so saplings read as
    miniatures of the same foliage language.
    """
    trunk_base = _TRUNK_PALETTES[int(rng.integers(len(_TRUNK_PALETTES)))]
    canopy_base = list(
        _DECIDUOUS_CANOPY_BASES[int(rng.integers(len(_DECIDUOUS_CANOPY_BASES)))]
    )
    canopy_base[0] = _clamp(canopy_base[0] + int(rng.integers(-10, 11)), 40, 100)
    canopy_base[1] = _clamp(canopy_base[1] + int(rng.integers(-10, 11)), 110, 170)
    canopy_base[2] = _clamp(canopy_base[2] + int(rng.integers(-8, 9)), 20, 65)

    trunk_rgba: colors.ColorRGBA = (*trunk_base, 255)
    # Shadow tone.
    shadow_rgba: colors.ColorRGBA = (
        max(0, canopy_base[0] - 40),
        max(0, canopy_base[1] - 35),
        max(0, canopy_base[2] - 20),
        210,
    )
    # Mid-tone: base color.
    mid_rgba: colors.ColorRGBA = (
        canopy_base[0],
        canopy_base[1],
        canopy_base[2],
        200,
    )
    # Highlight tone shifted yellow.
    highlight_rgba: colors.ColorRGBA = (
        min(255, canopy_base[0] + 40),
        min(255, canopy_base[1] + 30),
        min(255, canopy_base[2] + 15),
        190,
    )

    lean = float(rng.uniform(-1.0, 1.0))
    cx = size / 2.0 + lean * 0.2
    trunk_bottom = size - 1
    trunk_top = int(size * 0.4)

    _draw_tapered_trunk(
        canvas,
        cx,
        trunk_bottom,
        trunk_top,
        width_bottom=1.5,
        width_top=1.0,
        rgba=trunk_rgba,
        root_flare=1,
    )

    canopy_cx = cx + lean * 0.3
    canopy_cy = float(trunk_top) - size * 0.08
    base_radius = float(size * rng.uniform(0.16, 0.22))
    n_lobes = int(rng.integers(2, 4))
    lobe_centers: list[PixelPos] = []
    base_angle_step = 2.0 * math.pi / n_lobes
    lobe_angle_offset = float(rng.uniform(-math.pi, math.pi))
    for i in range(n_lobes):
        angle = lobe_angle_offset + base_angle_step * i + float(rng.uniform(-0.4, 0.4))
        dist = base_radius * float(rng.uniform(0.40, 0.60))
        vertical_scale = 0.9 if math.sin(angle) > 0.0 else 0.7
        lx = canopy_cx + math.cos(angle) * dist
        ly = canopy_cy + math.sin(angle) * dist * vertical_scale
        lobe_centers.append((lx, ly))

    stamper = _CanvasStamper(canvas)
    central_fills: list[tuple[float, float, float]] = []
    shadows: list[tuple[float, float, float]] = []
    mids: list[tuple[float, float, float]] = []
    highlights: list[tuple[float, float, float]] = []

    # Central fill so sapling lobes connect.
    cr = base_radius * float(rng.uniform(0.45, 0.55))
    central_fills.append((canopy_cx, canopy_cy, cr))

    for lx, ly in lobe_centers:
        for _ in range(int(rng.integers(1, 3))):
            sx = lx + float(rng.uniform(-size * 0.05, size * 0.05))
            sy = ly + float(rng.uniform(-size * 0.06, size * 0.04))
            r = base_radius * float(rng.uniform(0.50, 0.68))
            shadows.append((sx, sy, r))

        for _ in range(int(rng.integers(1, 3))):
            mx = lx + float(rng.uniform(-size * 0.04, size * 0.04))
            my = ly + float(rng.uniform(-size * 0.05, size * 0.03))
            r = base_radius * float(rng.uniform(0.44, 0.60))
            mids.append((mx, my, r))

        for _ in range(int(rng.integers(1, 3))):
            hx = lx + float(rng.uniform(-size * 0.03, size * 0.03))
            hy = ly - size * 0.05 + float(rng.uniform(-size * 0.03, size * 0.02))
            r = base_radius * float(rng.uniform(0.32, 0.48))
            highlights.append((hx, hy, r))

    # Apply grouped passes for consistent shading and lower call overhead.
    stamper.batch_stamp_circles(
        central_fills,
        shadow_rgba,
        falloff=1.4,
        hardness=0.5,
    )
    stamper.batch_stamp_circles(
        shadows,
        shadow_rgba,
        falloff=1.5,
        hardness=0.5,
    )
    stamper.batch_stamp_circles(
        mids,
        mid_rgba,
        falloff=1.3,
        hardness=0.5,
    )
    stamper.batch_stamp_circles(
        highlights,
        highlight_rgba,
        falloff=1.2,
        hardness=0.4,
    )

    _nibble_canopy_silhouette(
        canvas,
        rng,
        (canopy_cx, canopy_cy),
        canopy_radius=base_radius * 1.9,
        nibble_probability=0.25,
        interior_probability=0.10,
    )
    _darken_canopy_rim(canvas)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_TREE_SPRITE_SEED_SALT: SpatialSeed = 0x5CEAE
_HEIGHT_JITTER_SALT: SpatialSeed = 0x48544A54

# Mapping from archetype to generator function.
_GENERATORS = {
    TreeArchetype.DECIDUOUS: _generate_deciduous,
    TreeArchetype.CONIFER: _generate_conifer,
    TreeArchetype.DEAD: _generate_dead,
    TreeArchetype.SAPLING: _generate_sapling,
}


def tree_sprite_seed(
    x: WorldTileCoord,
    y: WorldTileCoord,
    map_seed: MapDecorationSeed,
) -> SpatialSeed:
    """Return deterministic seed for a tree sprite at *(x, y)* on one map."""
    return brileta_rng.derive_spatial_seed(
        x,
        y,
        map_seed=map_seed,
        salt=_TREE_SPRITE_SEED_SALT,
    )


def sprite_visual_scale_for_shadow_height(
    sprite: np.ndarray,
    shadow_height: int,
) -> float:
    """Return visual scale so rendered sprite height matches physical shadow height.

    Calibration rule:
    - CreatureSize.MEDIUM (shadow_height=2) is treated as roughly one tile tall.
    - Therefore target height in tiles is ``shadow_height / 2``.

    The function measures the sprite's non-transparent alpha silhouette height
    and solves the scale factor that yields the target on-screen height.
    """
    if shadow_height <= 0:
        return 0.0

    alpha = sprite[:, :, 3]
    occupied_rows = np.where(np.any(alpha > 0, axis=1))[0]
    if occupied_rows.size == 0:
        return 1.0

    silhouette_height = int(occupied_rows[-1] - occupied_rows[0] + 1)
    if silhouette_height <= 0:
        return 1.0

    sprite_height = int(sprite.shape[0])
    medium_shadow_height = float(CreatureSize.MEDIUM.shadow_height)
    target_height_tiles = float(shadow_height) / medium_shadow_height
    scale = target_height_tiles * float(sprite_height) / float(silhouette_height)
    return max(0.3, min(3.0, scale))


def visual_scale_with_height_jitter(
    sprite: np.ndarray,
    shadow_height: int,
    x: WorldTileCoord,
    y: WorldTileCoord,
    map_seed: MapDecorationSeed,
) -> float:
    """Compute visual scale with deterministic per-tree height jitter.

    Wraps ``sprite_visual_scale_for_shadow_height`` and applies a spatially
    deterministic +/-18% multiplier so same-archetype trees at different map
    positions do not all render at identical heights.
    """
    base_scale = sprite_visual_scale_for_shadow_height(sprite, shadow_height)
    jitter_seed = brileta_rng.derive_spatial_seed(
        x,
        y,
        map_seed=map_seed,
        salt=_HEIGHT_JITTER_SALT,
    )
    jitter = 0.82 + (jitter_seed / 0xFFFFFFFF) * 0.36
    return base_scale * jitter


def generate_tree_sprite_for_position(
    x: WorldTileCoord,
    y: WorldTileCoord,
    map_seed: MapDecorationSeed,
    archetype: TreeArchetype,
    base_size: int = 20,
) -> np.ndarray:
    """Generate a deterministic tree sprite from world position and map seed."""
    seed = tree_sprite_seed(x, y, map_seed)
    return generate_tree_sprite(seed, archetype, base_size)


def generate_tree_sprite(
    seed: SpatialSeed,
    archetype: TreeArchetype,
    base_size: int = 20,
) -> np.ndarray:
    """Generate a tree sprite as an RGBA numpy array.

    The output is deterministic for a given (seed, archetype, base_size)
    triple, so the same world position always produces the same tree.

    Args:
        seed: Deterministic seed derived from spatial position hash.
        archetype: Type of tree to generate.
        base_size: Target sprite size in pixels. Actual size varies
            by a few pixels for natural variation.

    Returns:
        uint8 numpy array with shape ``(height, width, 4)``.
    """
    rng = np.random.default_rng(seed)

    # Size jitter: saplings are smaller, others vary ±3 around base_size.
    if archetype == TreeArchetype.SAPLING:
        size = base_size - int(rng.integers(2, 6))
    else:
        size = base_size + int(rng.integers(-3, 4))
    size = max(10, min(26, size))

    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    _GENERATORS[archetype](canvas, size, rng)
    return canvas


# ---------------------------------------------------------------------------
# Shared spatial hash for per-tree deterministic values
# ---------------------------------------------------------------------------

# Base conifer fraction among living trees: 20% conifer / 75% living = ~0.267.
_BASE_CONIFER_FRACTION: float = 4.0 / 15.0

# How much species noise can shift the conifer fraction from the base.
# At max noise (+1), conifer_prob reaches base + spread (clamped to 1.0).
# At min noise (-1), conifer_prob reaches base - spread (clamped to 0.0).
# 0.65 gives ~92% conifer in strong conifer zones and near-0% in deciduous
# zones, which is decisive enough to read as "pine forest" at a glance.
_SPECIES_NOISE_SPREAD: float = 0.65


def _tree_hash(
    x: WorldTileCoord, y: WorldTileCoord, map_seed: MapDecorationSeed
) -> int:
    """Deterministic spatial hash for a tree at (x, y)."""
    return brileta_rng.derive_spatial_seed(x, y, map_seed=map_seed)


def create_species_noise(map_seed: MapDecorationSeed) -> NoiseGenerator:
    """Create the noise generator used for spatial species biome variation.

    The seed is derived from *map_seed* via XOR so it is uncorrelated with
    both the density noise (seeded from the RNG stream) and the decoration
    noise (map_seed itself).
    """
    return NoiseGenerator(
        seed=int(map_seed) ^ config.TREE_SPECIES_SEED_XOR,
        noise_type=NoiseType.OPENSIMPLEX2,
        frequency=config.TREE_SPECIES_NOISE_FREQUENCY,
        fractal_type=FractalType.FBM,
        octaves=config.TREE_SPECIES_NOISE_OCTAVES,
    )


def archetype_for_position(
    x: WorldTileCoord,
    y: WorldTileCoord,
    map_seed: MapDecorationSeed,
    species_noise: NoiseGenerator,
) -> TreeArchetype:
    """Pick tree archetype using spatial hash + species noise.

    Dead trees (15%) and saplings (10%) are assigned at fixed rates
    everywhere. The remaining 75% of trees are split between deciduous
    and conifer based on a species noise field: high-noise areas skew
    conifer ("pine groves"), low-noise areas skew deciduous ("hardwood
    groves"), and neutral areas preserve the base ~27% conifer ratio.

    Args:
        x: World tile x coordinate.
        y: World tile y coordinate.
        map_seed: Map decoration seed for deterministic hashing.
        species_noise: Pre-created noise generator from ``create_species_noise``.
    """
    h = _tree_hash(x, y, map_seed)
    health_r = h % 20

    # Fixed-rate non-living archetypes (same everywhere on the map).
    if health_r >= 18:  # 10% sapling
        return TreeArchetype.SAPLING
    if health_r >= 15:  # 15% dead
        return TreeArchetype.DEAD

    # Living tree: species noise biases the deciduous/conifer split.
    # Sample noise in [-1, 1] and shift the base conifer fraction.
    species_val = species_noise.sample(float(x), float(y))
    conifer_prob = max(
        0.0, min(1.0, _BASE_CONIFER_FRACTION + species_val * _SPECIES_NOISE_SPREAD)
    )

    # Use upper bits of the spatial hash as the per-tree species coin flip,
    # independent of the health_r bucket (which used the lower bits).
    species_frac = ((h >> 8) & 0xFFFF) / 65535.0
    if species_frac < conifer_prob:
        return TreeArchetype.CONIFER
    return TreeArchetype.DECIDUOUS


def _paste_sprite(
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


def generate_preview_sheet(
    columns: int = 16,
    rows: int = 8,
    base_size: int = 20,
    padding: int = 2,
    bg_color: colors.ColorRGBA = (30, 30, 30, 255),
    map_seed: MapDecorationSeed = 42,
) -> np.ndarray:
    """Generate a preview sheet of trees as they would appear in the game.

    Each cell simulates a tree at a unique world coordinate, using the same
    noise-modulated archetype distribution and ``generate_tree_sprite_for_position``
    path that the real game uses.

    Args:
        columns: Number of trees per row.
        rows: Number of rows.
        base_size: Base sprite size in pixels.
        padding: Pixel gap between cells.
        bg_color: Background fill color.
        map_seed: Map decoration seed for deterministic generation.

    Returns:
        uint8 numpy array with shape ``(sheet_h, sheet_w, 4)``.
    """
    cell = base_size + 6 + padding  # Max sprite size + padding.
    sheet_w = columns * cell + padding
    sheet_h = rows * cell + padding

    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :] = bg_color

    species_noise = create_species_noise(map_seed)

    for idx in range(columns * rows):
        col = idx % columns
        row = idx // columns

        # Treat (col, row) as a world tile position so every cell gets
        # a unique, deterministic tree identical to what the game produces.
        x, y = col, row
        archetype = archetype_for_position(x, y, map_seed, species_noise)
        sprite = generate_tree_sprite_for_position(x, y, map_seed, archetype, base_size)
        sh, sw = sprite.shape[:2]

        # Center sprite in cell.
        x0 = col * cell + padding + (cell - padding - sw) // 2
        y0 = row * cell + padding + (cell - padding - sh) // 2
        _paste_sprite(sheet, sprite, x0, y0)

    return sheet


if __name__ == "__main__":
    import argparse
    import sys

    from PIL import Image
    from PIL.Image import Resampling

    parser = argparse.ArgumentParser(
        description="Generate a preview PNG of procedural tree sprites."
    )
    parser.add_argument(
        "-o", "--output", default="tree_preview.png", help="Output PNG path."
    )
    parser.add_argument(
        "-c", "--columns", type=int, default=16, help="Trees per row (default 16)."
    )
    parser.add_argument(
        "-r", "--rows", type=int, default=8, help="Number of rows (default 8)."
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=20,
        help="Base sprite size in px (default 20).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Map decoration seed (default 42)."
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Upscale factor for visibility (default 4).",
    )
    args = parser.parse_args()

    sheet = generate_preview_sheet(
        columns=args.columns,
        rows=args.rows,
        base_size=args.size,
        map_seed=args.seed,
    )

    img = Image.fromarray(sheet, "RGBA")
    if args.scale > 1:
        img = img.resize(
            (img.width * args.scale, img.height * args.scale),
            resample=Resampling.NEAREST,
        )
    img.save(args.output)
    print(f"Saved {img.width}x{img.height} preview to {args.output}")
    sys.exit(0)
