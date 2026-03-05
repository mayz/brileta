"""Procedural boulder sprite generation.

Boulders are generated as hard-edged ellipse clusters in three shading passes:
shadow, body, and highlight. Four shape archetypes provide silhouette variety,
each with distinct proportions and per-archetype shadow heights so shape
differences survive the visual-scale normalization at rendering time.

Color variety comes from a broad palette spanning warm sandstone, cool slate,
neutral granite, and earthy basalt tones with aggressive per-channel jitter.
Surface cracks and lichen spots add fine detail that reads at close zoom.
Light silhouette nibbling removes a few random edge pixels, breaking the smooth
oval into something that reads as fractured stone without creating lumps.

Archetypes:
  - ROUNDED: Wide, low mound - the classic irregular rock. shadow_height=2.
  - TALL: Tapered wedge wider at base, taller presence. shadow_height=3.
  - FLAT: Low-profile slab close to the ground. shadow_height=1.
  - BLOCKY: Asymmetric chunky mass with clustered lobes. shadow_height=2.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from brileta import colors
from brileta.sprites.common import sprite_visual_scale_for_shadow_height
from brileta.sprites.primitives import (
    CanvasStamper,
    clamp,
    darken_rim,
    draw_line,
    nibble_boulder,
    paste_sprite,
    stamp_fuzzy_circle,
)
from brileta.types import MapDecorationSeed, SpatialSeed, WorldTileCoord
from brileta.util import rng as brileta_rng

# ---------------------------------------------------------------------------
# Boulder archetype enum
# ---------------------------------------------------------------------------


class BoulderArchetype(Enum):
    """Boulder shape archetype that determines visual generation rules."""

    ROUNDED = "rounded"
    TALL = "tall"
    FLAT = "flat"
    BLOCKY = "blocky"


# Per-archetype physical shadow heights.  FLAT rocks are low, TALL ones
# are imposing.  The visual-scale system uses shadow_height to normalize
# rendered size, so this is what makes archetypes visibly different sizes
# in-game rather than all converging to the same height.
_ARCHETYPE_SHADOW_HEIGHTS: dict[BoulderArchetype, int] = {
    BoulderArchetype.ROUNDED: 2,
    BoulderArchetype.TALL: 3,
    BoulderArchetype.FLAT: 1,
    BoulderArchetype.BLOCKY: 2,
}


def shadow_height_for_archetype(archetype: BoulderArchetype) -> int:
    """Return the physical shadow height for a boulder archetype."""
    return _ARCHETYPE_SHADOW_HEIGHTS[archetype]


# ---------------------------------------------------------------------------
# Color palettes and helpers
# ---------------------------------------------------------------------------

_BOULDER_SPRITE_SEED_SALT: SpatialSeed = 0xB04D3
_HEIGHT_JITTER_SALT: SpatialSeed = 0xB0444A54

# Broad stone palette: warm, cool, neutral, and earthy tones so that
# boulders across a map never feel like they all came from the same quarry.
_BOULDER_BASE_PALETTES: list[colors.Color] = [
    # Warm stone - sandstone and fieldstone
    (152, 132, 112),  # Warm sandstone
    (140, 120, 100),  # Tan fieldstone
    (130, 114, 94),  # Brown river stone
    # Cool stone - slate and granite
    (120, 128, 140),  # Blue-gray slate
    (110, 120, 134),  # Cool slate
    (128, 130, 138),  # Cool granite
    # Neutral grays
    (136, 137, 141),  # Neutral granite
    (124, 126, 130),  # Mid gray
    (114, 116, 120),  # Dark basalt
    # Dark earthy tones
    (108, 114, 104),  # Mossy basalt (green tint)
    (122, 116, 108),  # Dark weathered brown
]

# Type alias for an ellipse specification: (cx, cy, rx, ry).
_Ellipse = tuple[float, float, float, float]


def _pick_stone_colors(
    rng: np.random.Generator,
) -> tuple[colors.ColorRGBA, colors.ColorRGBA, colors.ColorRGBA]:
    """Pick shadow, body, and highlight RGBA from the stone palette with jitter.

    Uses wide per-channel jitter (brightness +-18, temperature +-8) so that
    even boulders from the same palette family look distinct.  The shadow and
    highlight deltas are aggressive (+-36) so the three-tone shading reads
    clearly at small rendered sizes.

    Returns ``(shadow_rgba, body_rgba, highlight_rgba)``.
    """
    base_rgb = list(
        _BOULDER_BASE_PALETTES[int(rng.integers(len(_BOULDER_BASE_PALETTES)))]
    )
    # Wide brightness and temperature jitter for visible color variety.
    brightness_offset = int(rng.integers(-18, 19))
    temp_offset = int(rng.integers(-8, 9))
    base_rgb[0] = clamp(base_rgb[0] + brightness_offset - temp_offset, 70, 190)
    base_rgb[1] = clamp(base_rgb[1] + brightness_offset, 70, 190)
    base_rgb[2] = clamp(base_rgb[2] + brightness_offset + temp_offset, 70, 195)

    # Aggressive shading deltas so depth reads at game zoom.
    shadow_rgba: colors.ColorRGBA = (
        clamp(base_rgb[0] - 36),
        clamp(base_rgb[1] - 36),
        clamp(base_rgb[2] - 34),
        220,
    )
    body_rgba: colors.ColorRGBA = (
        int(base_rgb[0]),
        int(base_rgb[1]),
        int(base_rgb[2]),
        250,
    )
    highlight_rgba: colors.ColorRGBA = (
        clamp(base_rgb[0] + 36),
        clamp(base_rgb[1] + 36),
        clamp(base_rgb[2] + 38),
        210,
    )
    return shadow_rgba, body_rgba, highlight_rgba


# ---------------------------------------------------------------------------
# Surface detail: cracks and lichen
# ---------------------------------------------------------------------------

# Muted green/yellow lichen tones, semi-transparent so they blend
# with the underlying stone color rather than sitting on top.
_LICHEN_PALETTE: list[colors.ColorRGBA] = [
    (55, 72, 42, 140),  # Dark moss green
    (62, 78, 48, 130),  # Warmer moss
    (48, 65, 38, 120),  # Cool dark lichen
    (70, 80, 50, 110),  # Pale lichen
]


def _add_surface_cracks(
    canvas: np.ndarray,
    rng: np.random.Generator,
    cx: float,
    cy: float,
    base_rx: float,
    base_ry: float,
    shadow_rgba: colors.ColorRGBA,
) -> None:
    """Draw thin dark lines across the boulder to suggest fracture planes.

    Cracks run inward from random points on the upper surface.  They use a
    color darker than the shadow tone so they read as recessed grooves.
    Only drawn when the rng roll allows it (~60% of boulders).
    """
    if rng.random() > 0.60:
        return

    crack_rgba: colors.ColorRGBA = (
        clamp(shadow_rgba[0] - 20),
        clamp(shadow_rgba[1] - 20),
        clamp(shadow_rgba[2] - 18),
        190,
    )

    n_cracks = int(rng.integers(1, 4))
    for _ in range(n_cracks):
        # Start near the surface edge, biased toward the upper half.
        angle = float(rng.uniform(-math.pi, math.pi * 0.3))
        start_r = float(rng.uniform(0.55, 0.85))
        sx = cx + math.cos(angle) * base_rx * start_r
        sy = cy + math.sin(angle) * base_ry * start_r

        # Run inward toward center with slight angular wander.
        crack_angle = angle + math.pi + float(rng.uniform(-0.6, 0.6))
        crack_len = float(rng.uniform(1.5, 3.5))
        ex = sx + math.cos(crack_angle) * crack_len
        ey = sy + math.sin(crack_angle) * crack_len

        draw_line(canvas, sx, sy, ex, ey, crack_rgba)


def _add_lichen_spots(
    canvas: np.ndarray,
    rng: np.random.Generator,
    cx: float,
    cy: float,
    base_rx: float,
    base_ry: float,
) -> None:
    """Scatter small fuzzy lichen spots on the boulder surface.

    Spots are biased toward the upper half of the rock where moisture
    and light accumulate.  Only drawn when the rng roll allows it (~50%).
    """
    if rng.random() > 0.50:
        return

    n_spots = int(rng.integers(2, 5))
    for _ in range(n_spots):
        # Bias toward the upper hemisphere (negative y in image coords).
        spot_angle = float(rng.uniform(-math.pi * 0.8, math.pi * 0.4))
        spot_dist = float(rng.uniform(0.25, 0.70))
        mx = cx + math.cos(spot_angle) * base_rx * spot_dist
        my = cy + math.sin(spot_angle) * base_ry * spot_dist

        spot_radius = float(rng.uniform(0.7, 1.5))
        lichen_rgba = _LICHEN_PALETTE[int(rng.integers(len(_LICHEN_PALETTE)))]
        stamp_fuzzy_circle(
            canvas, mx, my, spot_radius, lichen_rgba, falloff=1.4, hardness=0.15
        )


def _add_surface_detail(
    canvas: np.ndarray,
    rng: np.random.Generator,
    cx: float,
    cy: float,
    base_rx: float,
    base_ry: float,
    shadow_rgba: colors.ColorRGBA,
) -> None:
    """Apply surface cracks and lichen in one call.

    Called after ellipse stamping but before nibble/rim so that cracks get
    properly edge-treated and lichen blends with the stone underneath.
    """
    _add_surface_cracks(canvas, rng, cx, cy, base_rx, base_ry, shadow_rgba)
    _add_lichen_spots(canvas, rng, cx, cy, base_rx, base_ry)


# ---------------------------------------------------------------------------
# Archetype generators
# ---------------------------------------------------------------------------


def _generate_rounded(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a rounded boulder - the classic irregular rock mound.

    A wide, bottom-heavy mass with 2-3 asymmetric lobes.  High stamp hardness
    gives crisp edges where overlapping ellipses meet, breaking the smooth
    oval silhouette into something more angular and rock-like.
    shadow_height=2 (medium presence).
    """
    shadow_rgba, body_rgba, highlight_rgba = _pick_stone_colors(rng)
    stamper = CanvasStamper(canvas)

    cx = size * 0.5 + float(rng.uniform(-0.3, 0.3))
    # Shifted down so that the lowest visible pixels reach near the canvas
    # bottom row.  The rendering engine projects ground shadows from the
    # canvas bottom, so transparent padding below the rock reads as a gap.
    cy = size * 0.72 + float(rng.uniform(-0.08, 0.14))
    base_rx = size * float(rng.uniform(0.28, 0.34))
    base_ry = size * float(rng.uniform(0.20, 0.26))
    lobe_count = int(rng.integers(2, 4))

    body_ellipses: list[_Ellipse] = []
    shadow_ellipses: list[_Ellipse] = []
    highlight_ellipses: list[_Ellipse] = []

    # Core shape: main body (lower/wider) + narrower crown above.
    body_ellipses.append((cx, cy, base_rx, base_ry))
    body_ellipses.append(
        (
            cx + float(rng.uniform(-0.5, 0.5)),
            cy - size * 0.08,
            base_rx * 0.72,
            base_ry * 0.56,
        )
    )
    shadow_ellipses.append((cx, cy + size * 0.04, base_rx * 1.06, base_ry * 1.10))
    # Wide, flat highlight suggesting a planar top face.
    highlight_ellipses.append(
        (cx - size * 0.05, cy - size * 0.10, base_rx * 0.52, base_ry * 0.24)
    )

    # Asymmetric lobes extend sideways and downward only.  Lobe ry is kept
    # small relative to base_ry so lobes widen the silhouette horizontally
    # without poking above the main body's top edge.
    angle_step = 2.0 * math.pi / lobe_count
    offset = float(rng.uniform(0.0, math.pi))
    for i in range(lobe_count):
        angle = offset + i * angle_step + float(rng.uniform(-0.45, 0.45))
        radial_x = base_rx * float(rng.uniform(0.20, 0.42))
        radial_y = base_ry * float(rng.uniform(0.10, 0.30))
        ox = math.cos(angle) * radial_x
        oy = max(0.0, math.sin(angle) * radial_y)

        rx = base_rx * float(rng.uniform(0.70, 1.0))
        ry = base_ry * float(rng.uniform(0.48, 0.68))

        body_ellipses.append((cx + ox, cy + oy, rx, ry))
        shadow_ellipses.append(
            (cx + ox * 0.76, cy + oy * 0.68 + size * 0.03, rx * 1.06, ry * 1.08)
        )
        highlight_ellipses.append(
            (
                cx + ox * 0.36 - size * 0.03,
                cy + oy * 0.30 - size * 0.07,
                rx * 0.30,
                ry * 0.20,
            )
        )

    stamper.batch_stamp_ellipses(
        shadow_ellipses, shadow_rgba, falloff=2.2, hardness=0.88
    )
    stamper.batch_stamp_ellipses(body_ellipses, body_rgba, falloff=2.0, hardness=0.86)
    stamper.batch_stamp_ellipses(
        highlight_ellipses, highlight_rgba, falloff=1.9, hardness=0.80
    )
    _add_surface_detail(canvas, rng, cx, cy, base_rx, base_ry, shadow_rgba)
    nibble_boulder(canvas, seed=int(rng.integers(0, 2**63)))
    darken_rim(canvas, darken=(18, 18, 15))


def _generate_tall(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a tall boulder - a tapered wedge wider at the base.

    Built from overlapping body ellipses stacked vertically: a wide base
    grounds the stone, the main body gives height, and a narrower crown
    tapers the top.  shadow_height=3 so it renders visibly taller than
    other boulders.
    """
    shadow_rgba, body_rgba, highlight_rgba = _pick_stone_colors(rng)
    stamper = CanvasStamper(canvas)

    cx = size * 0.5 + float(rng.uniform(-0.3, 0.3))
    # Pushed down so the wide base and shadow ellipses reach the canvas
    # bottom.  TALL has the biggest gap because its main body has the
    # largest ry among archetypes while the original cy was the highest up.
    cy = size * 0.70 + float(rng.uniform(-0.06, 0.10))
    # Slightly portrait aspect for the main body.
    base_rx = size * float(rng.uniform(0.22, 0.28))
    base_ry = size * float(rng.uniform(0.26, 0.34))
    lobe_count = int(rng.integers(1, 3))

    body_ellipses: list[_Ellipse] = []
    shadow_ellipses: list[_Ellipse] = []
    highlight_ellipses: list[_Ellipse] = []

    # Wide base ellipse anchoring the stone to the ground.
    body_ellipses.append((cx, cy + size * 0.08, base_rx * 1.30, base_ry * 0.58))
    # Main body mass.
    body_ellipses.append((cx, cy, base_rx, base_ry))
    # Narrower crown tapering upward.
    body_ellipses.append(
        (
            cx + float(rng.uniform(-0.5, 0.5)),
            cy - size * 0.12,
            base_rx * 0.65,
            base_ry * 0.48,
        )
    )
    # Shadow wraps around the wider base.
    shadow_ellipses.append((cx, cy + size * 0.09, base_rx * 1.36, base_ry * 0.65))
    # Flat highlight on the upper face.
    highlight_ellipses.append(
        (cx - size * 0.04, cy - size * 0.14, base_rx * 0.56, base_ry * 0.22)
    )

    # Side bulges for irregularity.  Flat lobe ry prevents protrusions
    # above the crown.
    angle_step = 2.0 * math.pi / max(1, lobe_count)
    offset = float(rng.uniform(0.0, math.pi))
    for i in range(lobe_count):
        angle = offset + i * angle_step + float(rng.uniform(-0.4, 0.4))
        radial_x = base_rx * float(rng.uniform(0.28, 0.52))
        radial_y = base_ry * float(rng.uniform(0.10, 0.26))
        ox = math.cos(angle) * radial_x
        oy = max(0.0, math.sin(angle) * radial_y)

        rx = base_rx * float(rng.uniform(0.62, 0.90))
        ry = base_ry * float(rng.uniform(0.42, 0.62))

        body_ellipses.append((cx + ox, cy + oy, rx, ry))
        shadow_ellipses.append(
            (cx + ox * 0.72, cy + oy * 0.68 + size * 0.03, rx * 1.06, ry * 1.06)
        )
        highlight_ellipses.append(
            (
                cx + ox * 0.34 - size * 0.02,
                cy + oy * 0.28 - size * 0.08,
                rx * 0.32,
                ry * 0.20,
            )
        )

    stamper.batch_stamp_ellipses(
        shadow_ellipses, shadow_rgba, falloff=2.2, hardness=0.88
    )
    stamper.batch_stamp_ellipses(body_ellipses, body_rgba, falloff=2.0, hardness=0.86)
    stamper.batch_stamp_ellipses(
        highlight_ellipses, highlight_rgba, falloff=1.9, hardness=0.80
    )
    _add_surface_detail(canvas, rng, cx, cy, base_rx, base_ry, shadow_rgba)
    nibble_boulder(canvas, seed=int(rng.integers(0, 2**63)))
    darken_rim(canvas, darken=(18, 18, 15))


def _generate_flat(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a flat boulder - a low-profile slab close to the ground.

    Wide landscape proportions (rx >> ry) that would be over-scaled with
    shadow_height=2 but render naturally with shadow_height=1, producing a
    rock that's genuinely wider than tall in the game world.
    """
    shadow_rgba, body_rgba, highlight_rgba = _pick_stone_colors(rng)
    stamper = CanvasStamper(canvas)

    cx = size * 0.5 + float(rng.uniform(-0.4, 0.4))
    # Sits low in the canvas - pushed further down so the shadow ellipse
    # bottom reaches near the canvas edge where ground shadows project from.
    cy = size * 0.78 + float(rng.uniform(-0.06, 0.08))
    # Aggressively wide and squat - shadow_height=1 prevents scale blowup.
    base_rx = size * float(rng.uniform(0.36, 0.44))
    base_ry = size * float(rng.uniform(0.13, 0.18))
    lobe_count = int(rng.integers(1, 3))

    body_ellipses: list[_Ellipse] = []
    shadow_ellipses: list[_Ellipse] = []
    highlight_ellipses: list[_Ellipse] = []

    # Core shape: wide slab body.
    body_ellipses.append((cx, cy, base_rx, base_ry))
    # Slight raised ridge near center-top.
    body_ellipses.append(
        (
            cx + float(rng.uniform(-0.5, 0.5)),
            cy - size * 0.04,
            base_rx * 0.80,
            base_ry * 0.55,
        )
    )
    shadow_ellipses.append((cx, cy + size * 0.03, base_rx * 1.06, base_ry * 1.14))
    # Wide highlight across the flat top face.
    highlight_ellipses.append(
        (cx - size * 0.04, cy - size * 0.05, base_rx * 0.58, base_ry * 0.28)
    )

    # Few lobes, biased toward horizontal extension.  Flat lobe ry.
    horizontal_angles = np.array([0.0, math.pi])
    offset = float(rng.choice(horizontal_angles)) + float(rng.uniform(-0.4, 0.4))
    angle_step = 2.0 * math.pi / max(1, lobe_count)
    for i in range(lobe_count):
        angle = offset + i * angle_step + float(rng.uniform(-0.35, 0.35))
        radial_x = base_rx * float(rng.uniform(0.18, 0.38))
        radial_y = base_ry * float(rng.uniform(0.08, 0.22))
        ox = math.cos(angle) * radial_x
        oy = max(0.0, math.sin(angle) * radial_y)

        rx = base_rx * float(rng.uniform(0.65, 0.92))
        ry = base_ry * float(rng.uniform(0.44, 0.64))

        body_ellipses.append((cx + ox, cy + oy, rx, ry))
        shadow_ellipses.append(
            (cx + ox * 0.78, cy + oy * 0.68 + size * 0.02, rx * 1.04, ry * 1.08)
        )
        highlight_ellipses.append(
            (
                cx + ox * 0.38 - size * 0.03,
                cy + oy * 0.28 - size * 0.04,
                rx * 0.34,
                ry * 0.22,
            )
        )

    stamper.batch_stamp_ellipses(
        shadow_ellipses, shadow_rgba, falloff=2.2, hardness=0.88
    )
    stamper.batch_stamp_ellipses(body_ellipses, body_rgba, falloff=2.0, hardness=0.86)
    stamper.batch_stamp_ellipses(
        highlight_ellipses, highlight_rgba, falloff=1.9, hardness=0.80
    )
    _add_surface_detail(canvas, rng, cx, cy, base_rx, base_ry, shadow_rgba)
    nibble_boulder(canvas, seed=int(rng.integers(0, 2**63)), nibble_probability=0.08)
    darken_rim(canvas, darken=(18, 18, 15))


def _generate_blocky(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a blocky boulder - an asymmetric chunky mass.

    Lobes cluster to one side rather than distributing radially, creating an
    irregular silhouette.  The highest hardness values of any archetype give
    the crispest edges, suggesting angular faces and fracture planes.
    shadow_height=2 (medium presence).
    """
    shadow_rgba, body_rgba, highlight_rgba = _pick_stone_colors(rng)
    stamper = CanvasStamper(canvas)

    # Offset center from canvas middle for asymmetry.
    cx = size * 0.5 + float(rng.uniform(-0.6, 0.6))
    # Pushed down so visual mass extends to the canvas bottom where the
    # rendering engine projects ground shadows.
    cy = size * 0.72 + float(rng.uniform(-0.08, 0.12))
    # Roughly square-ish aspect ratio.
    base_rx = size * float(rng.uniform(0.26, 0.32))
    base_ry = size * float(rng.uniform(0.22, 0.28))
    lobe_count = int(rng.integers(2, 4))

    body_ellipses: list[_Ellipse] = []
    shadow_ellipses: list[_Ellipse] = []
    highlight_ellipses: list[_Ellipse] = []

    # Core shape: chunky main body.
    body_ellipses.append((cx, cy, base_rx, base_ry))
    body_ellipses.append(
        (
            cx - size * 0.03,
            cy - size * 0.07,
            base_rx * 0.74,
            base_ry * 0.62,
        )
    )
    shadow_ellipses.append((cx, cy + size * 0.04, base_rx * 1.10, base_ry * 1.10))
    # Flat-edged highlight suggesting an angular face.
    highlight_ellipses.append(
        (cx - size * 0.06, cy - size * 0.10, base_rx * 0.44, base_ry * 0.22)
    )

    # Lobes clustered within a ~120-degree arc for asymmetric mass.
    # Flat lobe ry keeps mass lateral, not vertical.
    cluster_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    for _i in range(lobe_count):
        angle = cluster_angle + float(rng.uniform(-1.05, 1.05))
        radial_x = base_rx * float(rng.uniform(0.26, 0.50))
        radial_y = base_ry * float(rng.uniform(0.16, 0.38))
        ox = math.cos(angle) * radial_x
        oy = max(0.0, math.sin(angle) * radial_y)

        rx = base_rx * float(rng.uniform(0.68, 1.02))
        ry = base_ry * float(rng.uniform(0.48, 0.70))

        body_ellipses.append((cx + ox, cy + oy, rx, ry))
        shadow_ellipses.append(
            (cx + ox * 0.74, cy + oy * 0.66 + size * 0.03, rx * 1.08, ry * 1.10)
        )
        highlight_ellipses.append(
            (
                cx + ox * 0.36 - size * 0.03,
                cy + oy * 0.30 - size * 0.07,
                rx * 0.32,
                ry * 0.22,
            )
        )

    # Highest hardness for the crispest, most angular edges.
    stamper.batch_stamp_ellipses(
        shadow_ellipses, shadow_rgba, falloff=2.4, hardness=0.92
    )
    stamper.batch_stamp_ellipses(body_ellipses, body_rgba, falloff=2.2, hardness=0.90)
    stamper.batch_stamp_ellipses(
        highlight_ellipses, highlight_rgba, falloff=2.0, hardness=0.84
    )
    _add_surface_detail(canvas, rng, cx, cy, base_rx, base_ry, shadow_rgba)
    nibble_boulder(canvas, seed=int(rng.integers(0, 2**63)), nibble_probability=0.14)
    darken_rim(canvas, darken=(18, 18, 15))


# Mapping from archetype to generator function.
_GENERATORS = {
    BoulderArchetype.ROUNDED: _generate_rounded,
    BoulderArchetype.TALL: _generate_tall,
    BoulderArchetype.FLAT: _generate_flat,
    BoulderArchetype.BLOCKY: _generate_blocky,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def boulder_sprite_seed(
    x: WorldTileCoord,
    y: WorldTileCoord,
    map_seed: MapDecorationSeed,
) -> SpatialSeed:
    """Return deterministic seed for a boulder sprite at *(x, y)*."""
    return brileta_rng.derive_spatial_seed(
        x,
        y,
        map_seed=map_seed,
        salt=_BOULDER_SPRITE_SEED_SALT,
    )


def visual_scale_with_height_jitter(
    sprite: np.ndarray,
    shadow_height: int,
    x: WorldTileCoord,
    y: WorldTileCoord,
    map_seed: MapDecorationSeed,
) -> float:
    """Apply deterministic per-boulder height jitter to the base visual scale."""
    base_scale = sprite_visual_scale_for_shadow_height(sprite, shadow_height)
    jitter_seed = brileta_rng.derive_spatial_seed(
        x,
        y,
        map_seed=map_seed,
        salt=_HEIGHT_JITTER_SALT,
    )
    jitter = 0.92 + (jitter_seed / 0xFFFFFFFF) * 0.16
    return base_scale * jitter


# ---------------------------------------------------------------------------
# Archetype selection
# ---------------------------------------------------------------------------


def _boulder_hash(
    x: WorldTileCoord, y: WorldTileCoord, map_seed: MapDecorationSeed
) -> int:
    """Deterministic spatial hash for a boulder at (x, y)."""
    return brileta_rng.derive_spatial_seed(x, y, map_seed=map_seed)


def archetype_for_position(
    x: WorldTileCoord,
    y: WorldTileCoord,
    map_seed: MapDecorationSeed,
) -> BoulderArchetype:
    """Pick boulder archetype using spatial hash with fixed distribution.

    Distribution: 35% rounded, 20% tall, 25% flat, 20% blocky.
    Unlike trees, boulders do not use species noise - the distribution
    is uniform across the map.
    """
    h = _boulder_hash(x, y, map_seed)
    r = h % 20
    if r < 7:
        return BoulderArchetype.ROUNDED
    if r < 11:
        return BoulderArchetype.TALL
    if r < 16:
        return BoulderArchetype.FLAT
    return BoulderArchetype.BLOCKY


# ---------------------------------------------------------------------------
# Sprite generation entry points
# ---------------------------------------------------------------------------


def generate_boulder_sprite_for_position(
    x: WorldTileCoord,
    y: WorldTileCoord,
    map_seed: MapDecorationSeed,
    archetype: BoulderArchetype,
    base_size: int = 16,
) -> np.ndarray:
    """Generate a deterministic boulder sprite from world position and map seed."""
    seed = boulder_sprite_seed(x, y, map_seed)
    return generate_boulder_sprite(seed, archetype, base_size)


def generate_boulder_sprite(
    seed: SpatialSeed,
    archetype: BoulderArchetype,
    base_size: int = 16,
) -> np.ndarray:
    """Generate a procedural boulder sprite as a deterministic RGBA array.

    The output is deterministic for a given (seed, archetype, base_size)
    triple, so the same world position always produces the same boulder.

    Args:
        seed: Deterministic seed derived from spatial position hash.
        archetype: Shape family to generate.
        base_size: Target sprite size in pixels. Actual size varies
            by a few pixels for natural variation.

    Returns:
        uint8 numpy array with shape ``(height, width, 4)``.
    """
    rng = np.random.default_rng(seed)

    size = int(base_size + rng.integers(-2, 3))
    size = max(12, min(22, size))
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    _GENERATORS[archetype](canvas, size, rng)

    # Shift content down so the lowest *visibly solid* pixel sits at the
    # canvas bottom row.  The rendering engine projects ground shadows
    # from the canvas bottom, so any gap between the visual base of the
    # rock and the canvas edge shows terrain through the sprite.
    #
    # A threshold of 128 (half-opaque) ignores the soft anti-aliased
    # fringe from ellipse falloff that would otherwise anchor the shift
    # to near-invisible pixels, leaving the solid body too high.
    opaque_rows = np.where(np.any(canvas[:, :, 3] > 128, axis=1))[0]
    if opaque_rows.size > 0:
        gap = size - 1 - int(opaque_rows[-1])
        if gap > 0:
            canvas[gap:] = canvas[:-gap].copy()
            canvas[:gap] = 0

    return canvas


# ---------------------------------------------------------------------------
# Preview sheet
# ---------------------------------------------------------------------------


def generate_preview_sheet(
    columns: int = 16,
    rows: int = 8,
    base_size: int = 16,
    padding: int = 2,
    bg_color: colors.ColorRGBA = (30, 30, 30, 255),
    map_seed: MapDecorationSeed = 42,
) -> np.ndarray:
    """Generate a preview sheet of deterministic boulder sprites.

    Each cell maps to one world tile coordinate, so this preview uses the
    same position-seeded generation path as runtime map population.

    Returns:
        uint8 numpy array with shape ``(sheet_h, sheet_w, 4)``.
    """
    cell = base_size + 6 + padding
    sheet_w = columns * cell + padding
    sheet_h = rows * cell + padding

    sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.uint8)
    sheet[:, :] = bg_color

    for idx in range(columns * rows):
        col = idx % columns
        row = idx // columns
        x, y = col, row

        archetype = archetype_for_position(x, y, map_seed)
        sprite = generate_boulder_sprite_for_position(
            x, y, map_seed, archetype, base_size
        )
        sh, sw = sprite.shape[:2]

        x0 = col * cell + padding + (cell - padding - sw) // 2
        y0 = row * cell + padding + (cell - padding - sh) // 2
        paste_sprite(sheet, sprite, x0, y0)

    return sheet


if __name__ == "__main__":
    import argparse
    import sys

    from PIL import Image
    from PIL.Image import Resampling

    parser = argparse.ArgumentParser(
        description="Generate a preview PNG of procedural boulder sprites."
    )
    parser.add_argument(
        "-o", "--output", default="boulder_preview.png", help="Output PNG path."
    )
    parser.add_argument(
        "-c", "--columns", type=int, default=16, help="Boulders per row (default 16)."
    )
    parser.add_argument(
        "-r", "--rows", type=int, default=8, help="Number of rows (default 8)."
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=16,
        help="Base sprite size in px (default 16).",
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
