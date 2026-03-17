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
from brileta.sprites.common import sprite_visual_scale_for_shadow_height
from brileta.sprites.primitives import (
    PaletteBrush,
    clamp,
    darken_rim,
    draw_line,
    draw_tapered_trunk,
    draw_thick_line,
    fill_triangle,
    generate_deciduous_canopy,
    nibble_canopy,
    paste_sprite,
    stamp_fuzzy_circle,
)
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

    draw_thick_line(canvas, x, y, x2, y2, rgba, thickness)

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
    canopy_base[0] = clamp(canopy_base[0] + int(rng.integers(-20, 21)), 15, 120)
    canopy_base[1] = clamp(canopy_base[1] + int(rng.integers(-20, 21)), 90, 185)
    canopy_base[2] = clamp(canopy_base[2] + int(rng.integers(-15, 16)), 10, 85)

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

    draw_tapered_trunk(
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

    # Native hot path: lobe placement + ellipse list construction + batch
    # stamping. Returns lobe centers so Python can keep the small branch-tip
    # extension detail logic without reintroducing the hot inner loop.
    lobe_centers = generate_deciduous_canopy(
        canvas=canvas,
        seed=int(rng.integers(0, 2**63)),
        size=size,
        canopy_cx=canopy_cx,
        canopy_cy=canopy_cy,
        base_radius=base_radius,
        crown_rx_scale=crown_rx_scale,
        crown_ry_scale=crown_ry_scale,
        canopy_center_x_offset=canopy_center_x_offset,
        tips=tips,
        shadow_rgba=shadow_rgba,
        mid_rgba=mid_rgba,
        highlight_rgba=highlight_rgba,
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

            # Skip extensions that would reach into the trunk zone.
            if sy + uy * ext_len > trunk_top:
                continue

            ex = sx + ux * ext_len
            ey = sy + uy * ext_len
            draw_line(canvas, sx, sy, ex, ey, branch_tip_rgba)

    nibble_canopy(
        canvas,
        seed=int(rng.integers(0, 2**63)),
        center_x=canopy_cx,
        center_y=canopy_cy,
        canopy_radius=base_radius * 2.0,
        nibble_probability=0.25,
        interior_probability=0.10,
    )
    darken_rim(canvas)


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
    canopy_base[0] = clamp(canopy_base[0] + int(rng.integers(-15, 16)), 10, 60)
    canopy_base[1] = clamp(canopy_base[1] + int(rng.integers(-15, 16)), 55, 130)
    canopy_base[2] = clamp(canopy_base[2] + int(rng.integers(-10, 11)), 10, 60)

    trunk_rgba: colors.ColorRGBA = (*trunk_base, 255)

    # Subtle overall lean so not every conifer is perfectly vertical.
    lean = float(rng.uniform(-0.5, 0.5))
    cx = size / 2.0 + lean * 0.2
    trunk_bottom = size - 1

    # Pre-compute canopy tier geometry so we know where the canopy base
    # lands before drawing the trunk (prevents trunk poking through
    # narrow triangle apexes between or above tiers).
    n_tiers = int(rng.integers(3, 6))
    max_width = float(size * rng.uniform(0.35, 0.55))
    tier_height_base = size * 0.35
    # Start tiers above the trunk's visible base.
    canopy_base_y = trunk_bottom - int(size * 0.2)
    current_bottom = canopy_base_y

    tier_specs: list[
        tuple[float, int, int, float, colors.ColorRGBA]
    ] = []  # (cx, top, height, half_w, rgba)

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
        tier_specs.append((tier_cx, tier_top, tier_h, tier_w, tier_rgba))

        # Next tier starts partway up this one (overlap ~35%).
        current_bottom = tier_top + max(1, int(tier_h * 0.35))

    # Stop the trunk at the canopy base so no trunk pixels leak through
    # narrow triangle apexes between or above tiers.
    draw_tapered_trunk(
        canvas,
        cx,
        trunk_bottom,
        canopy_base_y,
        width_bottom=max(1.5, size * 0.1),
        width_top=1.0,
        rgba=trunk_rgba,
        root_flare=1,
    )

    # Draw canopy tiers over the trunk.
    for tier_cx, tier_top, tier_h, tier_w, tier_rgba in tier_specs:
        fill_triangle(canvas, tier_cx, tier_top, tier_w, tier_h, tier_rgba)

    darken_rim(canvas)


def _generate_dead(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a dead/bare tree: exposed branching skeleton, no canopy.

    Uses deeper branching, wider spread, and a thicker trunk than other
    archetypes so the bare skeletal structure is clearly visible and
    immediately distinguishable from green canopy trees.
    """
    trunk_base = _DEAD_TRUNK_BASES[int(rng.integers(len(_DEAD_TRUNK_BASES)))]
    trunk_rgba: colors.ColorRGBA = (
        clamp(trunk_base[0] + int(rng.integers(-8, 9))),
        clamp(trunk_base[1] + int(rng.integers(-6, 7))),
        clamp(trunk_base[2] + int(rng.integers(-6, 7))),
        255,
    )

    # More pronounced lean for dead trees (weathered/weakened).
    lean = float(rng.uniform(-2.0, 2.0))
    cx = size / 2.0 + lean * 0.3
    trunk_bottom = size - 1
    trunk_height = int(size * rng.uniform(0.45, 0.55))
    trunk_top = trunk_bottom - trunk_height

    draw_tapered_trunk(
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
            stamp_fuzzy_circle(canvas, mx, my, 1.5, _MOSS_RGBA, falloff=1.5)


def _generate_sapling(canvas: np.ndarray, size: int, rng: np.random.Generator) -> None:
    """Generate a small sapling with simplified three-step canopy shading.

    Uses fewer/smaller canopy lobes than deciduous trees so saplings read as
    miniatures of the same foliage language.
    """
    trunk_base = _TRUNK_PALETTES[int(rng.integers(len(_TRUNK_PALETTES)))]
    canopy_base = list(
        _DECIDUOUS_CANOPY_BASES[int(rng.integers(len(_DECIDUOUS_CANOPY_BASES)))]
    )
    canopy_base[0] = clamp(canopy_base[0] + int(rng.integers(-10, 11)), 40, 100)
    canopy_base[1] = clamp(canopy_base[1] + int(rng.integers(-10, 11)), 110, 170)
    canopy_base[2] = clamp(canopy_base[2] + int(rng.integers(-8, 9)), 20, 65)

    trunk_rgba: colors.ColorRGBA = (*trunk_base, 255)

    # Three-tone canopy palette: (shadow, mid, highlight).
    canopy_pal = (
        (
            max(0, canopy_base[0] - 40),
            max(0, canopy_base[1] - 35),
            max(0, canopy_base[2] - 20),
        ),
        (canopy_base[0], canopy_base[1], canopy_base[2]),
        (
            min(255, canopy_base[0] + 40),
            min(255, canopy_base[1] + 30),
            min(255, canopy_base[2] + 15),
        ),
    )

    lean = float(rng.uniform(-1.0, 1.0))
    cx = size / 2.0 + lean * 0.2
    trunk_bottom = size - 1
    trunk_top = int(size * 0.4)

    draw_tapered_trunk(
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

    brush = PaletteBrush(canvas, canopy_pal)
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
    brush.batch_circles(central_fills, tone=0, alpha=210, falloff=1.4, hardness=0.5)
    brush.batch_circles(shadows, tone=0, alpha=210, falloff=1.5, hardness=0.5)
    brush.batch_circles(mids, alpha=200, falloff=1.3, hardness=0.5)
    brush.batch_circles(highlights, tone=2, alpha=190, falloff=1.2, hardness=0.4)

    nibble_canopy(
        canvas,
        seed=int(rng.integers(0, 2**63)),
        center_x=canopy_cx,
        center_y=canopy_cy,
        canopy_radius=base_radius * 1.9,
        nibble_probability=0.25,
        interior_probability=0.10,
    )
    darken_rim(canvas)


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
        paste_sprite(sheet, sprite, x0, y0)

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
