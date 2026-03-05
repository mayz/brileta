"""Procedural humanoid character sprite generation with directional output.

This module provides deterministic sprite generation for humanoid actors. A single
seed rolls one character appearance, then that appearance is rendered in four
directional standing poses (front/back/left/right).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from brileta import colors
from brileta.sprites.primitives import (
    darken_rim,
    draw_tapered_trunk,
    draw_thick_line,
    stamp_ellipse,
    stamp_fuzzy_circle,
)
from brileta.types import Facing, MapDecorationSeed, SpatialSeed
from brileta.util import rng as brileta_rng

# ===========================================================================
# TODO(Character Sprite Roadmap)
# ---------------------------------------------------------------------------
# Keep this module intentionally simple for current gameplay readability.
# Planned visual expansions:
#   - Add a dedicated facial-hair system (e.g., stubble/mustache/beard variants)
#     with directional rendering support.
#   - Expand clothing coverage with many more torso/outerwear archetypes.
#   - Expand hairstyle variety beyond the current baseline set.
# ===========================================================================

# ---------------------------------------------------------------------------
# Glyphs eligible for humanoid character sprites
# ---------------------------------------------------------------------------

HUMANOID_GLYPHS: frozenset[str] = frozenset({"@", "T", "B", "R"})

# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------

_CHARACTER_SPRITE_SEED_SALT: SpatialSeed = 0xC4A2AC


def character_sprite_seed(
    actor_id: int,
    map_seed: MapDecorationSeed,
) -> SpatialSeed:
    """Return a deterministic seed for a character sprite."""
    return brileta_rng.derive_spatial_seed(
        actor_id,
        0,
        map_seed=map_seed,
        salt=_CHARACTER_SPRITE_SEED_SALT,
    )


# ---------------------------------------------------------------------------
# Three-tone shading palettes: (shadow, mid, highlight)
# ---------------------------------------------------------------------------

Palette3 = tuple[colors.Color, colors.Color, colors.Color]

SKIN_PALETTES: list[Palette3] = [
    ((150, 105, 75), (200, 160, 125), (230, 200, 170)),
    ((140, 95, 60), (190, 140, 100), (225, 185, 150)),
    ((115, 70, 40), (165, 110, 65), (200, 150, 105)),
    ((80, 50, 30), (125, 75, 45), (165, 115, 80)),
    ((55, 35, 20), (95, 55, 30), (135, 90, 55)),
]

HAIR_PALETTES: list[Palette3] = [
    ((15, 12, 8), (35, 25, 18), (55, 42, 30)),
    ((50, 30, 15), (85, 55, 25), (125, 80, 40)),
    ((95, 45, 12), (155, 80, 18), (195, 125, 45)),
    ((135, 105, 45), (195, 170, 75), (225, 205, 125)),
    ((90, 90, 95), (145, 145, 150), (190, 190, 195)),
    ((170, 160, 150), (210, 200, 190), (235, 225, 218)),
]

CLOTHING_PALETTES: list[Palette3] = [
    ((55, 25, 25), (110, 45, 45), (155, 75, 75)),
    ((25, 35, 75), (45, 65, 130), (75, 95, 170)),
    ((25, 55, 25), (45, 95, 45), (75, 135, 75)),
    ((75, 65, 45), (125, 110, 80), (165, 150, 115)),
    ((45, 45, 50), (85, 85, 95), (125, 125, 135)),
    ((75, 55, 25), (135, 95, 40), (175, 135, 65)),
    ((45, 25, 55), (85, 45, 105), (125, 75, 145)),
    ((18, 18, 22), (38, 38, 48), (60, 60, 72)),
    ((70, 35, 15), (120, 65, 30), (160, 100, 55)),
    ((25, 50, 55), (45, 90, 100), (70, 130, 140)),
]

PANTS_PALETTES: list[Palette3] = [
    ((25, 25, 30), (50, 50, 60), (75, 75, 85)),
    ((40, 30, 20), (75, 55, 35), (110, 85, 55)),
    ((20, 25, 45), (35, 45, 80), (55, 70, 115)),
    ((18, 18, 18), (35, 35, 35), (55, 55, 55)),
    ((50, 45, 35), (90, 80, 60), (130, 120, 95)),
]


def _luma(color: colors.Color) -> float:
    """Return perceptual luminance for an RGB color."""
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ensure_palette_contrast(
    skin: Palette3,
    hair: Palette3,
    cloth: Palette3,
) -> tuple[Palette3, Palette3]:
    """Apply deterministic contrast guardrails for readability.

    Small sprites lose part separation quickly when mid-tones are too close.
    Keep hair/cloth distinct from skin without adding extra RNG draws.
    """
    skin_mid = skin[1]
    hair_mid = hair[1]
    cloth_mid = cloth[1]

    if abs(_luma(hair_mid) - _luma(skin_mid)) < 18.0:
        hair = max(
            HAIR_PALETTES,
            key=lambda pal: abs(_luma(pal[1]) - _luma(skin_mid)),
        )

    if abs(_luma(cloth_mid) - _luma(skin_mid)) < 20.0:
        cloth = max(
            CLOTHING_PALETTES,
            key=lambda pal: abs(_luma(pal[1]) - _luma(skin_mid)),
        )

    return hair, cloth


# ---------------------------------------------------------------------------
# Pose system
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    """Body part offsets for one animation pose."""

    name: str
    facing: Facing = Facing.SOUTH
    left_leg_dx: float = 0.0
    left_leg_dy: float = 0.0
    right_leg_dx: float = 0.0
    right_leg_dy: float = 0.0
    left_arm_dy: float = 0.0
    right_arm_dy: float = 0.0
    body_dy: float = 0.0


POSE_FRONT_STAND = Pose("front_stand", facing=Facing.SOUTH)
POSE_BACK_STAND = Pose("back_stand", facing=Facing.NORTH)
POSE_LEFT_STAND = Pose("left_stand", facing=Facing.WEST)
POSE_RIGHT_STAND = Pose("right_stand", facing=Facing.EAST)

# Backward-compatible name for existing call sites.
POSE_STAND = POSE_FRONT_STAND

POSES: tuple[Pose, ...] = (
    POSE_FRONT_STAND,
    POSE_BACK_STAND,
    POSE_LEFT_STAND,
    POSE_RIGHT_STAND,
)

CHARACTER_DIRECTIONAL_POSE_COUNT: int = len(POSES)


# ---------------------------------------------------------------------------
# Body build templates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyParams:
    """Core shape knobs for caricature body generation."""

    canvas_size: int
    head_radius: float
    head_top_pad: float
    torso_rx: float
    torso_ry: float
    torso_cy_factor: float
    belly_rx: float
    belly_ry: float
    belly_cy_offset: float
    shoulder_width: float
    arm_thickness: int
    arm_splay: float
    leg_spacing: float
    leg_w1: float
    leg_w2: float
    neck_rx: float
    neck_ry: float
    neck_gap: float
    hand_radius: float
    arm_end_from_belly: bool = False
    leg_from_torso: bool = False


BodyParamRollFn = Callable[[np.random.Generator, int], BodyParams]


@dataclass(frozen=True)
class CharacterAppearance:
    """Pre-rolled appearance data shared across all poses."""

    body_params: BodyParams
    skin_pal: Palette3
    hair_pal: Palette3
    cloth_pal: Palette3
    pants_pal: Palette3
    hair_style_idx: int
    clothing_fn: ClothingDrawFn
    covers_legs: bool
    build_name: str
    clothing_style_idx: int
    hair_seed: int

    @classmethod
    def from_seed(
        cls,
        seed: int,
        size: int,
        *,
        forced_build_idx: int | None = None,
    ) -> CharacterAppearance:
        """Roll appearance data deterministically from ``seed``.

        Build selection intentionally uses a dedicated RNG stream to preserve the
        historical behavior of ``generate_character_sprite(seed)``.
        """
        chooser_rng = np.random.default_rng(seed)
        if forced_build_idx is None:
            build_idx = int(chooser_rng.integers(len(_BODY_PARAM_BUILD_ROLLERS)))
        else:
            build_idx = forced_build_idx

        build_name, roll_fn = _BODY_PARAM_BUILD_ROLLERS[build_idx]

        rng = np.random.default_rng(seed)
        skin_pal, hair_pal, cloth_pal, pants_pal = _roll_palettes(rng)
        body_params = roll_fn(rng, size)
        hair_style_idx = int(rng.integers(len(HAIR_STYLES)))

        clothing_style_idx = int(rng.integers(len(CLOTHING_DEFS)))
        _clothing_name, clothing_fn, covers_legs = CLOTHING_DEFS[clothing_style_idx]

        return cls(
            body_params=body_params,
            skin_pal=skin_pal,
            hair_pal=hair_pal,
            cloth_pal=cloth_pal,
            pants_pal=pants_pal,
            hair_style_idx=hair_style_idx,
            clothing_fn=clothing_fn,
            covers_legs=covers_legs,
            build_name=build_name,
            clothing_style_idx=clothing_style_idx,
            hair_seed=seed + 7777,
        )


# ---------------------------------------------------------------------------
# Hair drawing
# ---------------------------------------------------------------------------

HairDrawFn = Callable[
    [np.ndarray, float, float, float, Palette3, np.random.Generator, Facing],
    None,
]


def _hair_bald(
    _canvas: np.ndarray,
    _hx: float,
    _hy: float,
    _hr: float,
    _pal: Palette3,
    _rng: np.random.Generator,
    _facing: Facing,
) -> None:
    """No visible hair."""


def _hair_short(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    _rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Short crop hairstyle."""
    _shadow, mid, highlight = pal
    cap_cy = hy - hr * 0.45
    if facing == Facing.NORTH:
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.15,
            hr * 1.15,
            hr * 0.95,
            (*mid, 235),
            1.8,
            0.85,
        )
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.3,
            hr * 0.8,
            hr * 0.6,
            (*highlight, 200),
            1.5,
            0.75,
        )
    elif facing in {Facing.EAST, Facing.WEST}:
        # Side profile is authored as left-facing; right-facing is mirrored during pose rendering.
        front_sign = -1.0
        back_sign = 1.0
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.04,
            cap_cy - hr * 0.12,
            hr * 0.9,
            hr * 0.62,
            (*mid, 235),
            1.9,
            0.86,
        )
        # Back-side volume and small front fringe avoid "chef-hat" silhouette.
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.52,
            hy - hr * 0.02,
            hr * 0.25,
            hr * 0.34,
            (*mid, 220),
            1.45,
            0.76,
        )
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.22,
            cap_cy - hr * 0.34,
            hr * 0.36,
            hr * 0.24,
            (*highlight, 198),
            1.35,
            0.72,
        )
    else:
        stamp_ellipse(
            canvas,
            hx,
            cap_cy - hr * 0.12,
            hr * 1.08,
            hr * 0.54,
            (*mid, 235),
            1.9,
            0.86,
        )
        stamp_ellipse(
            canvas,
            hx,
            cap_cy - 0.7,
            hr * 0.62,
            hr * 0.36,
            (*highlight, 200),
            1.5,
            0.75,
        )


def _hair_medium(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Medium hair with side drape."""
    shadow, mid, highlight = pal

    if facing == Facing.NORTH:
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.1,
            hr * 1.25,
            hr * 1.1,
            (*shadow, 235),
            1.6,
            0.82,
        )
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.2,
            hr * 1.15,
            hr * 1.0,
            (*mid, 225),
            1.5,
            0.8,
        )
        for side in (-1, 1):
            sx = hx + side * hr * 0.85
            stamp_ellipse(
                canvas,
                sx,
                hy + hr * 0.3,
                hr * 0.45,
                hr * 0.7,
                (*shadow, 220),
                1.5,
                0.78,
            )
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.4,
            hr * 0.7,
            hr * 0.5,
            (*highlight, 195),
            1.3,
            0.7,
        )
        return

    if facing in {Facing.EAST, Facing.WEST}:
        # Side profile is authored as left-facing; right-facing is mirrored during pose rendering.
        front_sign = -1.0
        back_sign = 1.0
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.06,
            hy - hr * 0.48,
            hr * 1.12,
            hr * 0.7,
            (*shadow, 235),
            1.6,
            0.82,
        )
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.05,
            hy - hr * 0.56,
            hr * 1.0,
            hr * 0.6,
            (*mid, 225),
            1.5,
            0.8,
        )
        # Back side carries more volume than the front fringe in profile.
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.78,
            hy + hr * 0.3,
            hr * 0.4,
            hr * 0.76,
            (*shadow, 220),
            1.5,
            0.78,
        )
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.54,
            hy + hr * 0.02,
            hr * 0.12,
            hr * 0.24,
            (*mid, 214),
            1.35,
            0.74,
        )
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.3,
            hy - hr * 0.76,
            hr * 0.48,
            hr * 0.32,
            (*highlight, 195),
            1.3,
            0.7,
        )
        return

    stamp_ellipse(
        canvas,
        hx,
        hy - hr * 0.46,
        hr * 1.18,
        hr * 0.74,
        (*shadow, 235),
        1.6,
        0.82,
    )
    stamp_ellipse(
        canvas,
        hx,
        hy - hr * 0.56,
        hr * 1.06,
        hr * 0.64,
        (*mid, 225),
        1.5,
        0.8,
    )
    side_scale = 0.72 if _luma(mid) > 155.0 else 1.0
    for side in (-1, 1):
        sx = hx + side * hr * 0.85
        stamp_ellipse(
            canvas,
            sx,
            hy + hr * 0.12,
            hr * 0.4 * side_scale,
            hr * 0.55 * side_scale,
            (*shadow, 220),
            1.5,
            0.78,
        )
    stamp_ellipse(
        canvas,
        hx + float(rng.uniform(-0.4, 0.4)),
        hy - hr * 0.75,
        hr * 0.58,
        hr * 0.38,
        (*highlight, 195),
        1.3,
        0.7,
    )


def _hair_long(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Long hair draping well below the shoulders."""
    shadow, mid, highlight = pal

    if facing == Facing.NORTH:
        stamp_ellipse(canvas, hx, hy, hr * 1.3, hr * 1.1, (*shadow, 235), 1.6, 0.82)
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.1,
            hr * 1.2,
            hr * 1.0,
            (*mid, 225),
            1.5,
            0.8,
        )
        drape = hr * float(rng.uniform(2.2, 3.0))
        stamp_ellipse(
            canvas,
            hx,
            hy + drape * 0.3,
            hr * 0.9,
            drape * 0.45,
            (*shadow, 220),
            1.7,
            0.78,
        )
        stamp_ellipse(
            canvas,
            hx,
            hy + drape * 0.25,
            hr * 0.75,
            drape * 0.38,
            (*mid, 215),
            1.5,
            0.75,
        )
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.3,
            hr * 0.7,
            hr * 0.6,
            (*highlight, 190),
            1.3,
            0.68,
        )
        return

    if facing in {Facing.EAST, Facing.WEST}:
        # Side profile is authored as left-facing; right-facing is mirrored during pose rendering.
        front_sign = -1.0
        back_sign = 1.0
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.1,
            hy - hr * 0.38,
            hr * 1.15,
            hr * 0.8,
            (*shadow, 235),
            1.6,
            0.82,
        )
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.08,
            hy - hr * 0.46,
            hr * 1.02,
            hr * 0.7,
            (*mid, 225),
            1.5,
            0.8,
        )
        back_drape = hr * float(rng.uniform(2.1, 3.0))
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.62,
            hy + back_drape * 0.2,
            hr * 0.32,
            back_drape * 0.3,
            (*shadow, 215),
            1.8,
            0.78,
        )
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.58,
            hy + back_drape * 0.15,
            hr * 0.24,
            back_drape * 0.24,
            (*mid, 210),
            1.6,
            0.75,
        )
        # Keep a smaller front lock so profile still reads as "face side".
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.52,
            hy + hr * 0.12,
            hr * 0.1,
            hr * 0.28,
            (*mid, 208),
            1.35,
            0.72,
        )
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.26,
            hy - hr * 0.7,
            hr * 0.56,
            hr * 0.4,
            (*highlight, 190),
            1.3,
            0.68,
        )
        return

    if facing == Facing.SOUTH:
        # Front long hair keeps clear face read while maintaining shoulder drape.
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.48,
            hr * 1.14,
            hr * 0.72,
            (*shadow, 235),
            1.6,
            0.82,
        )
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.58,
            hr * 1.02,
            hr * 0.62,
            (*mid, 225),
            1.5,
            0.8,
        )
        side_scale = 0.74 if _luma(mid) > 155.0 else 1.0
        for side in (-1, 1):
            sx = hx + side * hr * 0.78
            drape = hr * float(rng.uniform(2.0, 3.0))
            stamp_ellipse(
                canvas,
                sx,
                hy + drape * 0.28,
                hr * 0.3 * side_scale,
                drape * 0.22 * side_scale,
                (*shadow, 215),
                1.8,
                0.78,
            )
            stamp_ellipse(
                canvas,
                sx,
                hy + drape * 0.24,
                hr * 0.21 * side_scale,
                drape * 0.18 * side_scale,
                (*mid, 210),
                1.6,
                0.75,
            )
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.74,
            hr * 0.66,
            hr * 0.42,
            (*highlight, 190),
            1.3,
            0.68,
        )
        return

    stamp_ellipse(
        canvas,
        hx,
        hy - hr * 0.35,
        hr * 1.18,
        hr * 0.82,
        (*shadow, 235),
        1.6,
        0.82,
    )
    stamp_ellipse(
        canvas,
        hx,
        hy - hr * 0.45,
        hr * 1.08,
        hr * 0.72,
        (*mid, 225),
        1.5,
        0.8,
    )
    side_scale = 0.74 if _luma(mid) > 155.0 else 1.0
    for side in (-1, 1):
        sx = hx + side * hr * 0.7
        drape = hr * float(rng.uniform(2.0, 3.0))
        stamp_ellipse(
            canvas,
            sx,
            hy + drape * 0.22,
            hr * 0.36 * side_scale,
            drape * 0.28 * side_scale,
            (*shadow, 215),
            1.8,
            0.78,
        )
        stamp_ellipse(
            canvas,
            sx,
            hy + drape * 0.18,
            hr * 0.24 * side_scale,
            drape * 0.22 * side_scale,
            (*mid, 210),
            1.6,
            0.75,
        )
    stamp_ellipse(
        canvas,
        hx,
        hy - hr * 0.68,
        hr * 0.68,
        hr * 0.44,
        (*highlight, 190),
        1.3,
        0.68,
    )


def _hair_tall(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    _rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Tall upward hair (mohawk / swept up)."""
    shadow, mid, highlight = pal

    if facing == Facing.NORTH:
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 0.1,
            hr * 1.1,
            hr * 1.0,
            (*shadow, 230),
            1.7,
            0.82,
        )
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 1.0,
            hr * 0.65,
            hr * 1.1,
            (*mid, 225),
            1.6,
            0.8,
        )
        stamp_ellipse(
            canvas,
            hx,
            hy - hr * 1.2,
            hr * 0.45,
            hr * 0.8,
            (*highlight, 200),
            1.3,
            0.7,
        )
        return

    if facing in {Facing.EAST, Facing.WEST}:
        # Side profile is authored as left-facing; right-facing is mirrored during pose rendering.
        front_sign = -1.0
        back_sign = 1.0
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.04,
            hy - hr * 1.12,
            hr * 0.66,
            hr * 1.12,
            (*shadow, 230),
            1.8,
            0.82,
        )
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.02,
            hy - hr * 1.24,
            hr * 0.52,
            hr * 0.94,
            (*mid, 220),
            1.5,
            0.78,
        )
        # Side scalp cap keeps the back-of-head hair mass readable.
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.22,
            hy - hr * 0.24,
            hr * 0.34,
            hr * 0.48,
            (*shadow, 220),
            1.45,
            0.74,
        )
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.18,
            hy - hr * 0.28,
            hr * 0.25,
            hr * 0.36,
            (*mid, 210),
            1.35,
            0.7,
        )
        # Back-of-head side mass so profile reads with rear hair coverage.
        stamp_ellipse(
            canvas,
            hx + back_sign * hr * 0.58,
            hy - hr * 0.02,
            hr * 0.3,
            hr * 0.44,
            (*shadow, 214),
            1.4,
            0.74,
        )
        # Small front lock keeps a face-side edge cue.
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.22,
            hy - hr * 0.34,
            hr * 0.2,
            hr * 0.26,
            (*highlight, 198),
            1.3,
            0.7,
        )
        return

    stamp_ellipse(
        canvas,
        hx,
        hy - hr * 1.2,
        hr * 0.7,
        hr * 1.2,
        (*shadow, 230),
        1.8,
        0.82,
    )
    stamp_ellipse(
        canvas,
        hx,
        hy - hr * 1.3,
        hr * 0.55,
        hr * 1.0,
        (*mid, 220),
        1.5,
        0.78,
    )
    stamp_ellipse(
        canvas,
        hx,
        hy - hr * 1.5,
        hr * 0.35,
        hr * 0.7,
        (*highlight, 200),
        1.3,
        0.7,
    )


HAIR_STYLE_NAMES: tuple[str, ...] = (
    "bald",
    "short",
    "medium",
    "long",
    "tall",
)
HAIR_STYLES: tuple[HairDrawFn, ...] = (
    _hair_bald,
    _hair_short,
    _hair_medium,
    _hair_long,
    _hair_tall,
)


# ---------------------------------------------------------------------------
# Clothing drawing
# ---------------------------------------------------------------------------

ClothingDrawFn = Callable[
    [
        np.ndarray,
        float,
        float,
        BodyParams,
        Palette3,
        Palette3,
        np.random.Generator,
        Facing,
    ],
    None,
]


def _draw_torso_mass(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    palette: Palette3,
    facing: Facing,
) -> None:
    """Draw one torso mass with three-tone shading."""
    shadow, mid, highlight = palette
    stamp_ellipse(canvas, cx, cy + 0.3, rx + 0.3, ry + 0.3, (*shadow, 240), 2.0, 0.88)
    stamp_ellipse(canvas, cx, cy, rx, ry, (*mid, 235), 2.0, 0.88)

    if facing == Facing.NORTH:
        highlight_alpha = 185
        highlight_rx = rx * 0.55
        highlight_ry = ry * 0.4
    else:
        highlight_alpha = 210
        highlight_rx = rx * 0.7
        highlight_ry = ry * 0.5

    stamp_ellipse(
        canvas,
        cx,
        cy - ry * 0.25,
        highlight_rx,
        highlight_ry,
        (*highlight, highlight_alpha),
        1.8,
        0.75,
    )


def _draw_belly_mass(
    canvas: np.ndarray,
    cx: float,
    belly_cy: float,
    params: BodyParams,
    cloth_pal: Palette3,
    facing: Facing,
) -> None:
    """Draw the lower-body belly mass used by the belly build."""
    c_shadow, c_mid, c_highlight = cloth_pal
    stamp_ellipse(
        canvas,
        cx,
        belly_cy + 0.3,
        params.belly_rx + 0.3,
        params.belly_ry + 0.3,
        (*c_shadow, 240),
        2.0,
        0.85,
    )
    stamp_ellipse(
        canvas,
        cx,
        belly_cy,
        params.belly_rx,
        params.belly_ry,
        (*c_mid, 235),
        2.0,
        0.85,
    )

    hi_alpha = 175 if facing == Facing.NORTH else 190
    hi_rx = params.belly_rx * (0.4 if facing == Facing.NORTH else 0.5)
    hi_ry = params.belly_ry * (0.3 if facing == Facing.NORTH else 0.4)
    stamp_ellipse(
        canvas,
        cx - params.belly_rx * 0.1,
        belly_cy - params.belly_ry * 0.1,
        hi_rx,
        hi_ry,
        (*c_highlight, hi_alpha),
        1.6,
        0.7,
    )


def _draw_broad_torso(
    canvas: np.ndarray,
    cx: float,
    torso_cy: float,
    params: BodyParams,
    cloth_pal: Palette3,
    facing: Facing,
) -> None:
    """Draw a shoulder-dominant torso silhouette."""
    c_shadow, c_mid, c_highlight = cloth_pal
    shoulder_half_w = params.shoulder_width
    torso_ry = params.torso_ry

    stamp_ellipse(
        canvas,
        cx,
        torso_cy - torso_ry * 0.2,
        shoulder_half_w,
        torso_ry * 0.8,
        (*c_shadow, 240),
        2.2,
        0.9,
    )
    stamp_ellipse(
        canvas,
        cx,
        torso_cy - torso_ry * 0.3,
        shoulder_half_w - 0.3,
        torso_ry * 0.7,
        (*c_mid, 235),
        2.2,
        0.9,
    )
    stamp_ellipse(
        canvas,
        cx,
        torso_cy + torso_ry * 0.3,
        shoulder_half_w - 1.5,
        torso_ry * 0.6,
        (*c_mid, 230),
        2.0,
        0.85,
    )

    hi_alpha = 185 if facing == Facing.NORTH else 200
    stamp_ellipse(
        canvas,
        cx,
        torso_cy - torso_ry * 0.35,
        shoulder_half_w * 0.45,
        torso_ry * 0.4,
        (*c_highlight, hi_alpha),
        1.6,
        0.72,
    )


def _draw_bare_torso(
    canvas: np.ndarray,
    cx: float,
    torso_cy: float,
    params: BodyParams,
    skin_pal: Palette3,
    _cloth_pal: Palette3,
    _rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Draw bare skin torso."""
    _draw_torso_mass(
        canvas,
        cx,
        torso_cy,
        params.torso_rx,
        params.torso_ry,
        skin_pal,
        facing,
    )


def _draw_shirt_torso(
    canvas: np.ndarray,
    cx: float,
    torso_cy: float,
    params: BodyParams,
    _skin_pal: Palette3,
    cloth_pal: Palette3,
    _rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Draw standard shirt/tunic torso."""
    if params.shoulder_width > params.torso_rx:
        _draw_broad_torso(canvas, cx, torso_cy, params, cloth_pal, facing)
        return

    _draw_torso_mass(
        canvas,
        cx,
        torso_cy,
        params.torso_rx,
        params.torso_ry,
        cloth_pal,
        facing,
    )


def _draw_armor_torso(
    canvas: np.ndarray,
    cx: float,
    torso_cy: float,
    params: BodyParams,
    _skin_pal: Palette3,
    cloth_pal: Palette3,
    _rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Draw armor torso with pauldrons."""
    shadow, mid, highlight = cloth_pal
    if facing in {Facing.EAST, Facing.WEST}:
        # Side profile should be narrower than front/back plate width.
        rx = max(1.8, params.torso_rx * 0.78)
    else:
        rx = max(params.torso_rx + 0.4, params.shoulder_width * 0.9)
    ry = max(1.4, params.torso_ry - 0.2)

    stamp_ellipse(
        canvas, cx, torso_cy + 0.3, rx + 0.3, ry + 0.3, (*shadow, 245), 2.2, 0.92
    )
    stamp_ellipse(canvas, cx, torso_cy, rx, ry, (*mid, 240), 2.2, 0.92)

    if facing == Facing.NORTH:
        stamp_ellipse(
            canvas,
            cx,
            torso_cy - ry * 0.25,
            rx * 0.4,
            ry * 0.3,
            (*highlight, 190),
            1.5,
            0.7,
        )
    elif facing == Facing.SOUTH:
        stamp_ellipse(
            canvas,
            cx - rx * 0.15,
            torso_cy - ry * 0.3,
            rx * 0.4,
            ry * 0.35,
            (*highlight, 215),
            1.5,
            0.7,
        )
    else:
        # Side-view plate highlight and shoulder sleeve.
        front_sign = -1.0
        stamp_ellipse(
            canvas,
            cx + front_sign * rx * 0.08,
            torso_cy - ry * 0.26,
            rx * 0.36,
            ry * 0.34,
            (*highlight, 208),
            1.45,
            0.68,
        )
        stamp_fuzzy_circle(
            canvas,
            cx + front_sign * (rx + 0.25),
            torso_cy - ry * 0.42,
            1.15,
            (*mid, 226),
            1.8,
            0.82,
        )
        stamp_ellipse(
            canvas,
            cx + front_sign * (rx * 0.72),
            torso_cy + ry * 0.06,
            0.6,
            0.82,
            (*shadow, 220),
            1.4,
            0.72,
        )
        stamp_ellipse(
            canvas,
            cx + front_sign * (rx * 0.62),
            torso_cy + ry * 0.34,
            0.52,
            0.3,
            (*highlight, 195),
            1.15,
            0.62,
        )
        stamp_fuzzy_circle(
            canvas,
            cx + 0.22,
            torso_cy - ry * 0.38,
            0.65,
            (*mid, 205),
            1.5,
            0.72,
        )

    if facing in {Facing.NORTH, Facing.SOUTH}:
        for side in (-1, 1):
            sx = cx + side * (rx + 0.5)
            stamp_fuzzy_circle(
                canvas,
                sx,
                torso_cy - ry * 0.5,
                1.5,
                (*mid, 230),
                2.0,
                0.88,
            )


def _draw_robe_torso(
    canvas: np.ndarray,
    cx: float,
    torso_cy: float,
    params: BodyParams,
    _skin_pal: Palette3,
    cloth_pal: Palette3,
    _rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Draw robe torso that extends down and covers legs."""
    shadow, mid, highlight = cloth_pal
    ry = params.torso_ry + 2.0

    stamp_ellipse(
        canvas,
        cx,
        torso_cy + 1.0,
        params.torso_rx + 0.8,
        ry + 0.3,
        (*shadow, 240),
        2.0,
        0.85,
    )
    stamp_ellipse(
        canvas,
        cx,
        torso_cy + 0.8,
        params.torso_rx + 0.5,
        ry,
        (*mid, 235),
        2.0,
        0.85,
    )

    hi_alpha = 185 if facing == Facing.NORTH else 210
    hi_rx = params.torso_rx * (0.5 if facing == Facing.NORTH else 0.7)
    hi_ry = ry * (0.3 if facing == Facing.NORTH else 0.4)
    stamp_ellipse(
        canvas,
        cx,
        torso_cy - ry * (0.1 if facing == Facing.NORTH else 0.15),
        hi_rx,
        hi_ry,
        (*highlight, hi_alpha),
        1.6,
        0.7,
    )


CLOTHING_STYLE_NAMES: tuple[str, ...] = (
    "bare",
    "shirt",
    "shirt",
    "shirt",
    "armor",
    "robe",
)
CLOTHING_DEFS: tuple[tuple[str, ClothingDrawFn, bool], ...] = (
    ("bare", _draw_bare_torso, False),
    ("shirt", _draw_shirt_torso, False),
    ("shirt", _draw_shirt_torso, False),
    ("shirt", _draw_shirt_torso, False),
    ("armor", _draw_armor_torso, False),
    ("robe", _draw_robe_torso, True),
)


# ---------------------------------------------------------------------------
# Build parameter rolling
# ---------------------------------------------------------------------------


def _roll_palettes(
    rng: np.random.Generator,
) -> tuple[Palette3, Palette3, Palette3, Palette3]:
    """Roll skin, hair, clothing, and pants palettes from an RNG."""
    skin = SKIN_PALETTES[int(rng.integers(len(SKIN_PALETTES)))]
    hair = HAIR_PALETTES[int(rng.integers(len(HAIR_PALETTES)))]
    cloth = CLOTHING_PALETTES[int(rng.integers(len(CLOTHING_PALETTES)))]
    pants = PANTS_PALETTES[int(rng.integers(len(PANTS_PALETTES)))]
    hair, cloth = _ensure_palette_contrast(skin, hair, cloth)
    return skin, hair, cloth, pants


def _roll_standard_body_params(rng: np.random.Generator, size: int) -> BodyParams:
    torso_ry = 2.0 + float(rng.uniform(0, 0.3))
    return BodyParams(
        canvas_size=size,
        head_radius=4.0 + float(rng.uniform(0, 0.5)),
        head_top_pad=1.5,
        torso_rx=3.5 + float(rng.uniform(0, 0.5)),
        torso_ry=torso_ry,
        torso_cy_factor=0.65,
        belly_rx=0.0,
        belly_ry=0.0,
        belly_cy_offset=0.0,
        shoulder_width=0.0,
        arm_thickness=2,
        arm_splay=0.18,
        leg_spacing=1.5,
        leg_w1=1.3,
        leg_w2=1.1,
        neck_rx=1.2,
        neck_ry=0.8,
        neck_gap=0.3 + torso_ry * 0.07,
        hand_radius=0.8,
    )


def _roll_belly_body_params(rng: np.random.Generator, size: int) -> BodyParams:
    chest_ry = 1.2 + float(rng.uniform(0, 0.3))
    belly_ry = 3.5 + float(rng.uniform(0, 1.0))
    return BodyParams(
        canvas_size=size,
        head_radius=3.0 + float(rng.uniform(0, 0.4)),
        head_top_pad=1.5,
        torso_rx=3.2 + float(rng.uniform(0, 0.4)),
        torso_ry=chest_ry,
        torso_cy_factor=0.4,
        belly_rx=3.8 + float(rng.uniform(0, 0.8)),
        belly_ry=belly_ry,
        belly_cy_offset=chest_ry * 0.4 + belly_ry * 0.55,
        shoulder_width=0.0,
        arm_thickness=2,
        arm_splay=0.28,
        leg_spacing=1.5,
        leg_w1=1.3,
        leg_w2=1.1,
        neck_rx=1.2,
        neck_ry=0.7,
        neck_gap=0.2,
        hand_radius=0.8,
        arm_end_from_belly=True,
    )


def _roll_broad_body_params(rng: np.random.Generator, size: int) -> BodyParams:
    return BodyParams(
        canvas_size=size,
        head_radius=3.2 + float(rng.uniform(0, 0.4)),
        head_top_pad=1.5,
        torso_rx=3.0,
        torso_ry=2.8 + float(rng.uniform(0, 0.5)),
        torso_cy_factor=0.5,
        belly_rx=0.0,
        belly_ry=0.0,
        belly_cy_offset=0.0,
        shoulder_width=5.0 + float(rng.uniform(0, 0.8)),
        arm_thickness=2,
        arm_splay=0.25,
        leg_spacing=1.8,
        leg_w1=1.35,
        leg_w2=1.15,
        neck_rx=1.4,
        neck_ry=0.9,
        neck_gap=0.2,
        hand_radius=1.0,
    )


def _roll_thin_body_params(rng: np.random.Generator, size: int) -> BodyParams:
    return BodyParams(
        canvas_size=size,
        head_radius=3.5 + float(rng.uniform(0, 0.3)),
        head_top_pad=1.5,
        torso_rx=2.0 + float(rng.uniform(0, 0.3)),
        torso_ry=2.2 + float(rng.uniform(0, 0.4)),
        torso_cy_factor=0.55,
        belly_rx=0.0,
        belly_ry=0.0,
        belly_cy_offset=0.0,
        shoulder_width=0.0,
        arm_thickness=1,
        arm_splay=0.14,
        leg_spacing=1.0,
        leg_w1=1.0,
        leg_w2=0.9,
        neck_rx=1.0,
        neck_ry=0.9,
        neck_gap=0.3,
        hand_radius=0.6,
    )


def _roll_child_body_params(rng: np.random.Generator, size: int) -> BodyParams:
    return BodyParams(
        canvas_size=max(8, size - 5),
        head_radius=3.5 + float(rng.uniform(0, 0.3)),
        head_top_pad=1.2,
        torso_rx=2.5,
        torso_ry=1.35,
        torso_cy_factor=0.5,
        belly_rx=0.0,
        belly_ry=0.0,
        belly_cy_offset=0.0,
        shoulder_width=0.0,
        arm_thickness=1,
        arm_splay=0.12,
        leg_spacing=1.2,
        leg_w1=1.15,
        leg_w2=0.95,
        neck_rx=0.7,
        neck_ry=0.55,
        neck_gap=0.2,
        hand_radius=0.45,
        leg_from_torso=True,
    )


_BODY_PARAM_BUILD_ROLLERS: tuple[tuple[str, BodyParamRollFn], ...] = (
    ("standard", _roll_standard_body_params),
    ("belly", _roll_belly_body_params),
    ("broad", _roll_broad_body_params),
    ("thin", _roll_thin_body_params),
    ("child", _roll_child_body_params),
)


# ---------------------------------------------------------------------------
# Unified drawing pipeline
# ---------------------------------------------------------------------------


def _draw_head_and_hair(
    canvas: np.ndarray,
    appearance: CharacterAppearance,
    hx: float,
    hy: float,
    hr: float,
    facing: Facing,
) -> None:
    """Draw head and hair with facing-specific shapes."""
    s_shadow, s_mid, s_highlight = appearance.skin_pal
    h_shadow, h_mid, _h_highlight = appearance.hair_pal

    if facing == Facing.NORTH:
        stamp_fuzzy_circle(canvas, hx, hy + 0.2, hr + 0.2, (*s_shadow, 240), 2.0, 0.88)
        stamp_fuzzy_circle(canvas, hx, hy, hr, (*s_mid, 235), 2.0, 0.88)
    elif facing in {Facing.EAST, Facing.WEST}:
        # Side profile is authored as left-facing; right-facing is mirrored during pose rendering.
        front_sign = -1.0
        back_sign = 1.0
        stamp_ellipse(
            canvas,
            hx,
            hy + 0.2,
            hr * 0.85,
            hr + 0.2,
            (*s_shadow, 240),
            2.0,
            0.88,
        )
        stamp_ellipse(canvas, hx, hy, hr * 0.8, hr, (*s_mid, 235), 2.0, 0.88)
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.14,
            hy - hr * 0.2,
            hr * 0.45,
            hr * 0.55,
            (*s_highlight, 210),
            1.6,
            0.75,
        )
        # Profile cues: brighter face plane on the front side and darker back skull.
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.26,
            hy - hr * 0.12,
            hr * 0.34,
            hr * 0.48,
            (*s_highlight, 205),
            1.5,
            0.72,
        )
        if appearance.hair_style_idx != 4:
            stamp_ellipse(
                canvas,
                hx + back_sign * hr * 0.28,
                hy + hr * 0.03,
                hr * 0.3,
                hr * 0.46,
                (*s_shadow, 195),
                1.5,
                0.72,
            )
        stamp_fuzzy_circle(
            canvas,
            hx + front_sign * hr * 0.63,
            hy - hr * 0.02,
            hr * 0.12,
            (*s_highlight, 190),
            1.4,
            0.7,
        )
    else:
        stamp_fuzzy_circle(canvas, hx, hy + 0.2, hr + 0.2, (*s_shadow, 240), 2.0, 0.88)
        stamp_fuzzy_circle(canvas, hx, hy, hr, (*s_mid, 235), 2.0, 0.88)
        stamp_fuzzy_circle(
            canvas,
            hx,
            hy - hr * 0.22,
            hr * 0.6,
            (*s_highlight, 210),
            1.6,
            0.75,
        )

    hair_fn = HAIR_STYLES[appearance.hair_style_idx]
    hair_rng = np.random.default_rng(appearance.hair_seed)
    hair_fn(canvas, hx, hy, hr, appearance.hair_pal, hair_rng, facing)

    # Keep a readable front-facing face window even with heavy hairstyles.
    if facing == Facing.SOUTH:
        is_long_front_hair = appearance.hair_style_idx == 3
        is_medium_front_hair = appearance.hair_style_idx == 2

        # Hard guardrail: medium/long front hair should never receive a
        # post-hair skin overpaint at the forehead center (can read as bald spot).
        if is_long_front_hair or is_medium_front_hair:
            # Keep a hairline/fringe lock at the center so the forehead region
            # remains hair-colored even after readability touch-ups.
            stamp_ellipse(
                canvas,
                hx,
                hy - hr * (0.26 if is_long_front_hair else 0.22),
                hr * (0.24 if is_long_front_hair else 0.2),
                hr * (0.13 if is_long_front_hair else 0.11),
                (*h_mid, 210),
                1.35,
                0.7,
            )
            if is_long_front_hair:
                stamp_ellipse(
                    canvas,
                    hx,
                    hy - hr * 0.36,
                    hr * 0.23,
                    hr * 0.12,
                    (*h_mid, 205),
                    1.25,
                    0.68,
                )
        else:
            face_rx = hr * 0.44
            face_ry = hr * 0.36
            stamp_ellipse(
                canvas,
                hx,
                hy - hr * 0.05,
                face_rx,
                face_ry,
                (*s_mid, 205),
                1.35,
                0.72,
            )
            stamp_ellipse(
                canvas,
                hx,
                hy - hr * 0.22,
                hr * 0.28,
                hr * 0.2,
                (*s_highlight, 190),
                1.2,
                0.65,
            )
    elif facing in {Facing.EAST, Facing.WEST}:
        # Re-assert profile facial cues after hair so side direction is legible.
        front_sign = -1.0
        back_sign = 1.0
        if appearance.hair_style_idx == 4:
            # Tall hair needs explicit back-of-head fill to avoid skin patches.
            stamp_ellipse(
                canvas,
                hx + back_sign * hr * 0.42,
                hy - hr * 0.08,
                hr * 0.32,
                hr * 0.46,
                (*h_shadow, 224),
                1.4,
                0.76,
            )
            stamp_ellipse(
                canvas,
                hx + back_sign * hr * 0.36,
                hy - hr * 0.12,
                hr * 0.24,
                hr * 0.34,
                (*h_mid, 214),
                1.3,
                0.72,
            )
            # Hard mask over back half of side head to prevent any skin leak.
            stamp_ellipse(
                canvas,
                hx + back_sign * hr * 0.52,
                hy + hr * 0.02,
                hr * 0.34,
                hr * 0.5,
                (*h_shadow, 232),
                1.5,
                0.82,
            )
            stamp_ellipse(
                canvas,
                hx + back_sign * hr * 0.44,
                hy - hr * 0.02,
                hr * 0.24,
                hr * 0.36,
                (*h_mid, 220),
                1.35,
                0.76,
            )
            # Deterministic hard mask over the side back-head region.
            # Use direct pixel overpaint so skin tones cannot leak through via
            # alpha blending in this area (the known "bald spot" artifact).
            mask_cx = hx + back_sign * hr * 0.58
            mask_cy = hy + hr * 0.02
            mask_rx = hr * 0.52
            mask_ry = hr * 0.84
            x_min = max(0, int(np.floor(mask_cx - mask_rx - 1.0)))
            x_max = min(canvas.shape[1] - 1, int(np.ceil(mask_cx + mask_rx + 1.0)))
            y_min = max(0, int(np.floor(mask_cy - mask_ry - 1.0)))
            y_max = min(canvas.shape[0] - 1, int(np.ceil(mask_cy + mask_ry + 1.0)))
            for py in range(y_min, y_max + 1):
                for px in range(x_min, x_max + 1):
                    if canvas[py, px, 3] == 0:
                        continue
                    dx = (px + 0.5 - mask_cx) / max(0.001, mask_rx)
                    dy = (py + 0.5 - mask_cy) / max(0.001, mask_ry)
                    if dx * dx + dy * dy > 1.0:
                        continue
                    tone = h_shadow if dx >= 0.0 else h_mid
                    canvas[py, px, 0] = tone[0]
                    canvas[py, px, 1] = tone[1]
                    canvas[py, px, 2] = tone[2]
                    canvas[py, px, 3] = max(canvas[py, px, 3], 232)
            # Safety pass: force the back half of the side head ellipse to hair.
            # This makes side tall-hair silhouettes deterministic and prevents
            # any remaining skin-toned "bald spot" pixels.
            head_rx = max(0.001, hr * 0.95)
            head_ry = max(0.001, hr * 1.08)
            for py in range(y_min, y_max + 1):
                for px in range(x_min, x_max + 1):
                    if canvas[py, px, 3] == 0:
                        continue
                    head_dx = (px + 0.5 - hx) / head_rx
                    head_dy = (py + 0.5 - (hy + hr * 0.03)) / head_ry
                    if head_dx * head_dx + head_dy * head_dy > 1.5:
                        continue
                    if px + 0.5 < hx + back_sign * hr * 0.24:
                        continue
                    tone = h_shadow if px + 0.5 >= hx + back_sign * hr * 0.56 else h_mid
                    canvas[py, px, 0] = tone[0]
                    canvas[py, px, 1] = tone[1]
                    canvas[py, px, 2] = tone[2]
                    canvas[py, px, 3] = max(canvas[py, px, 3], 232)
        stamp_ellipse(
            canvas,
            hx + front_sign * hr * 0.56,
            hy - hr * 0.06,
            hr * 0.2,
            hr * 0.3,
            (*s_mid, 210),
            1.3,
            0.68,
        )
        stamp_fuzzy_circle(
            canvas,
            hx + front_sign * hr * 0.72,
            hy - hr * 0.03,
            hr * 0.12,
            (*s_highlight, 195),
            1.25,
            0.66,
        )
        # Stronger 2-pixel-like nose cue for clearer L-std/R-std direction read.
        nose_x = hx + front_sign * hr * 0.9
        nose_y = hy + hr * 0.01
        stamp_ellipse(
            canvas,
            nose_x,
            nose_y,
            hr * 0.08,
            hr * 0.06,
            (*s_shadow, 200),
            1.0,
            0.62,
        )
        stamp_fuzzy_circle(
            canvas,
            nose_x + front_sign * 0.18,
            nose_y + 0.02,
            hr * 0.06,
            (*s_shadow, 192),
            1.0,
            0.6,
        )
        if appearance.hair_style_idx != 4:
            # Tiny ear cue on the back side.
            stamp_fuzzy_circle(
                canvas,
                hx + back_sign * hr * 0.42,
                hy + hr * 0.02,
                hr * 0.1,
                (*s_shadow, 175),
                1.25,
                0.66,
            )


def _draw_character(
    canvas: np.ndarray,
    appearance: CharacterAppearance,
    pose: Pose,
) -> None:
    """Draw one character using pre-rolled appearance and pose."""
    params = appearance.body_params
    facing = pose.facing
    cx = params.canvas_size / 2.0

    hr = params.head_radius
    head_cy = hr + params.head_top_pad + pose.body_dy
    torso_cy = head_cy + hr + params.neck_gap + params.torso_ry * params.torso_cy_factor

    belly_cy = torso_cy + params.belly_cy_offset
    torso_bottom = torso_cy + params.torso_ry
    if params.belly_ry > 0:
        torso_bottom = belly_cy + params.belly_ry

    # 1) Legs (unless robe covers them).
    p_shadow, p_mid, _p_hi = appearance.pants_pal
    leg_bottom = params.canvas_size - 1
    leg_top = leg_bottom - 2
    is_thin_legs = params.leg_w1 <= 1.05 and params.arm_thickness == 1
    if params.leg_from_torso:
        leg_top = int(torso_bottom - 0.3)
    else:
        # Anchor leg start to torso silhouette to avoid "floating feet" gaps on
        # thin/slender builds while preserving a minimum visible leg length.
        min_leg_len = 4 if is_thin_legs else 3
        anchored_top = int(torso_bottom + 0.4)
        leg_top = min(anchored_top, leg_bottom - min_leg_len)

    if not appearance.covers_legs:
        if facing in {Facing.EAST, Facing.WEST}:
            leg_positions = [
                (cx + pose.left_leg_dx, leg_bottom + pose.left_leg_dy),
                (cx + pose.right_leg_dx, leg_bottom + pose.right_leg_dy),
            ]
        else:
            leg_positions = [
                (
                    cx - params.leg_spacing + pose.left_leg_dx,
                    leg_bottom + pose.left_leg_dy,
                ),
                (
                    cx + params.leg_spacing + pose.right_leg_dx,
                    leg_bottom + pose.right_leg_dy,
                ),
            ]

        for lx, l_bot in leg_positions:
            l_bot_clamped = min(float(params.canvas_size - 1), l_bot)
            draw_tapered_trunk(
                canvas,
                lx,
                int(l_bot_clamped),
                leg_top,
                params.leg_w1 + 0.2,
                params.leg_w2 + 0.2,
                (*p_shadow, 235),
            )
            draw_tapered_trunk(
                canvas,
                lx,
                int(l_bot_clamped),
                leg_top,
                params.leg_w1,
                params.leg_w2,
                (*p_mid, 230),
            )

        if facing in {Facing.NORTH, Facing.SOUTH} and is_thin_legs:
            # Add a tiny upper-leg mass so thin builds do not read as disconnected
            # "sticks" under the torso at gameplay zoom levels.
            thigh_y = min(float(params.canvas_size - 1), leg_top + 1.1)
            stamp_ellipse(
                canvas,
                cx,
                thigh_y,
                max(0.6, params.leg_spacing * 0.62),
                0.42,
                (*p_shadow, 220),
                1.25,
                0.72,
            )
            for side in (-1, 1):
                stamp_fuzzy_circle(
                    canvas,
                    cx + side * max(0.95, params.leg_spacing * 0.92),
                    thigh_y + 0.12,
                    max(0.28, params.leg_w1 * 0.32),
                    (*p_shadow, 205),
                    1.2,
                    0.68,
                )

        if facing in {Facing.EAST, Facing.WEST}:
            # Side profile gets a visible toe protrusion toward facing direction.
            front_sign = -1.0
            toe_base_x = (
                cx + pose.left_leg_dx + front_sign * max(0.55, params.leg_w1 * 0.52)
            )
            toe_tip_x = toe_base_x + front_sign * max(0.7, params.leg_w1 * 0.68)
            toe_y = min(
                float(params.canvas_size - 1.0),
                leg_bottom + pose.left_leg_dy - 0.25,
            )
            draw_thick_line(
                canvas,
                toe_base_x,
                toe_y,
                toe_tip_x,
                toe_y + 0.01,
                (*p_shadow, 235),
                1,
            )
            stamp_fuzzy_circle(
                canvas,
                toe_tip_x + front_sign * 0.08,
                toe_y,
                max(0.28, params.leg_w1 * 0.22),
                (*p_mid, 220),
                1.1,
                0.6,
            )

    # 2) Belly and torso.
    if params.belly_ry > 0 and not appearance.covers_legs:
        _draw_belly_mass(canvas, cx, belly_cy, params, appearance.cloth_pal, facing)

    torso_rng = np.random.default_rng(appearance.hair_seed + 101)
    appearance.clothing_fn(
        canvas,
        cx,
        torso_cy,
        params,
        appearance.skin_pal,
        appearance.cloth_pal,
        torso_rng,
        facing,
    )

    # Belly builds can show a front-like chest/belly divot from the back.
    # Fill the seam in NORTH-facing view so the body reads as a back mass.
    if (
        facing == Facing.NORTH
        and params.belly_ry > 0
        and not appearance.covers_legs
        and appearance.clothing_fn is not _draw_bare_torso
    ):
        _shadow, c_mid, _highlight = appearance.cloth_pal
        seam_y = torso_cy + params.torso_ry * 0.95
        seam_rx = max(params.torso_rx * 0.95, params.belly_rx * 0.68)
        seam_ry = max(0.85, params.belly_ry * 0.38)
        stamp_ellipse(
            canvas,
            cx,
            seam_y,
            seam_rx,
            seam_ry,
            (*c_mid, 218),
            1.7,
            0.8,
        )

    # Robes hide legs, but a tiny foot peek preserves grounded human read.
    if appearance.covers_legs:
        foot_y = min(float(params.canvas_size - 1), torso_bottom + 0.7)
        boot_rx = max(0.6, params.leg_w1 * 0.55)
        if facing in {Facing.EAST, Facing.WEST}:
            stamp_ellipse(
                canvas,
                cx + 0.2,
                foot_y,
                boot_rx * 0.95,
                0.72,
                (*p_shadow, 235),
                1.6,
                0.85,
            )
        else:
            for side in (-1, 1):
                stamp_ellipse(
                    canvas,
                    cx + side * max(0.85, params.leg_spacing * 0.68),
                    foot_y,
                    boot_rx,
                    0.72,
                    (*p_shadow, 235),
                    1.6,
                    0.85,
                )

    # 3) Arms.
    _s_shadow, s_mid, _s_hi = appearance.skin_pal
    _c_shadow, _c_mid, _c_hi = appearance.cloth_pal
    is_broad = params.shoulder_width > params.torso_rx
    is_belly = params.belly_ry > 0
    is_thin = (
        params.arm_thickness == 1 and params.hand_radius > 0 and params.neck_rx > 0
    )
    is_child = params.neck_rx == 0 and params.hand_radius == 0

    if is_child or is_belly:
        shoulder_factor = 0.18
    elif is_thin:
        shoulder_factor = 0.26
    else:
        shoulder_factor = 0.28
    shoulder_y = torso_cy - params.torso_ry * shoulder_factor

    if params.arm_end_from_belly and params.belly_ry > 0:
        arm_end_y = belly_cy + params.belly_ry * 0.22
    elif is_broad:
        arm_end_y = torso_cy + params.torso_ry * 0.62
    elif is_thin:
        arm_end_y = torso_cy + params.torso_ry * 0.68
    elif is_child:
        arm_end_y = torso_cy + params.torso_ry * 0.64
    else:
        arm_end_y = torso_cy + params.torso_ry * 0.66

    # Armor reads better when hands sit slightly lower on the torso silhouette.
    if appearance.clothing_fn is _draw_armor_torso:
        arm_end_y += 1.8

    arm_alpha = 220 if params.arm_thickness == 1 else 225
    hand_drop = 0.3 if is_thin else 0.5
    hand_alpha = 210 if is_thin else 215

    if appearance.clothing_fn is _draw_bare_torso:
        arm_shadow_rgb = (
            (_s_shadow[0] + s_mid[0]) // 2,
            (_s_shadow[1] + s_mid[1]) // 2,
            (_s_shadow[2] + s_mid[2]) // 2,
        )
        arm_mid_rgb = s_mid
    else:
        arm_shadow_rgb = _c_shadow
        arm_mid_rgb = _c_mid

    if facing in {Facing.EAST, Facing.WEST}:
        # Side profile uses a clear foreground arm plus a faint background arm.
        front_sign = -1.0
        back_sign = 1.0
        front_arm_x = cx + front_sign * params.torso_rx * 0.62
        front_hand_x = front_arm_x + front_sign * 0.18
        far_arm_x = cx + back_sign * params.torso_rx * 0.34
        hand_radius = min(0.9, params.hand_radius) if params.hand_radius > 0 else 0.0

        draw_thick_line(
            canvas,
            far_arm_x,
            shoulder_y + 0.05,
            far_arm_x + back_sign * 0.1,
            arm_end_y + pose.right_arm_dy + 0.05,
            (*arm_shadow_rgb, max(145, arm_alpha - 55)),
            max(1, params.arm_thickness - 1),
        )

        stamp_fuzzy_circle(
            canvas,
            front_arm_x,
            shoulder_y,
            max(0.45, params.arm_thickness * 0.38),
            (*arm_shadow_rgb, 185),
            1.45,
            0.72,
        )

        draw_thick_line(
            canvas,
            front_arm_x + front_sign * 0.08,
            shoulder_y,
            front_hand_x + front_sign * 0.08,
            arm_end_y + pose.left_arm_dy,
            (*arm_shadow_rgb, max(165, arm_alpha - 30)),
            params.arm_thickness,
        )
        draw_thick_line(
            canvas,
            front_arm_x,
            shoulder_y,
            front_hand_x,
            arm_end_y + pose.left_arm_dy,
            (*arm_mid_rgb, arm_alpha),
            params.arm_thickness,
        )
        if hand_radius > 0:
            stamp_fuzzy_circle(
                canvas,
                front_hand_x,
                arm_end_y + pose.left_arm_dy + hand_drop,
                hand_radius,
                (*s_mid, hand_alpha),
                2.0,
                0.85,
            )
    else:
        # Keep arms attached at the shoulder edge and bias to mostly vertical fall.
        arm_swing_scale = 0.5 if appearance.clothing_fn is _draw_armor_torso else 1.0
        shoulder_base = params.shoulder_width if is_broad else params.torso_rx
        if is_broad:
            shoulder_anchor = (
                params.torso_rx * 0.82
                + (params.shoulder_width - params.torso_rx) * 0.12
            )
            hand_inset = 0.24
        elif is_belly:
            shoulder_anchor = shoulder_base * 0.85
            hand_inset = 0.2
        elif is_child:
            shoulder_anchor = shoulder_base * 0.8
            hand_inset = 0.12
        elif is_thin:
            shoulder_anchor = shoulder_base * 0.82
            hand_inset = 0.14
        else:
            shoulder_anchor = shoulder_base * 0.84
            hand_inset = 0.18
        for side, arm_dy in [(-1, pose.left_arm_dy), (1, pose.right_arm_dy)]:
            arm_x = cx + side * shoulder_anchor
            hand_x = arm_x - side * hand_inset
            hand_y = arm_end_y + arm_dy * arm_swing_scale

            stamp_fuzzy_circle(
                canvas,
                arm_x,
                shoulder_y,
                max(0.45, params.arm_thickness * 0.38),
                (*arm_shadow_rgb, 175),
                1.45,
                0.72,
            )

            # Two-pass arm stroke gives a readable limb edge without looking detached.
            draw_thick_line(
                canvas,
                arm_x + side * 0.08,
                shoulder_y + 0.05,
                hand_x + side * 0.08,
                hand_y + 0.05,
                (*arm_shadow_rgb, max(165, arm_alpha - 30)),
                params.arm_thickness,
            )
            draw_thick_line(
                canvas,
                arm_x,
                shoulder_y,
                hand_x,
                hand_y,
                (*arm_mid_rgb, arm_alpha),
                params.arm_thickness,
            )
            if params.hand_radius > 0:
                stamp_fuzzy_circle(
                    canvas,
                    hand_x,
                    hand_y + hand_drop,
                    params.hand_radius,
                    (*s_mid, hand_alpha),
                    2.0,
                    0.85,
                )

    # 4) Neck (hidden from back).
    if facing != Facing.NORTH and params.neck_rx > 0 and params.neck_ry > 0:
        if is_belly or is_broad:
            neck_y = head_cy + hr + 0.1
        elif is_thin:
            neck_y = head_cy + hr + 0.15
        else:
            neck_y = head_cy + hr + params.neck_gap * 0.5
        stamp_ellipse(
            canvas,
            cx,
            neck_y,
            params.neck_rx,
            params.neck_ry,
            (*s_mid, 225),
            2.0,
            0.85,
        )

    # 5) Head and hair.
    _draw_head_and_hair(canvas, appearance, cx, head_cy, hr, facing)

    if facing in {Facing.EAST, Facing.WEST}:
        # Final nose overlay: snap to near-integer pixel centers so it remains
        # visible across walk frames with subpixel body bob.
        front_sign = -1.0
        nose_x = float(round(cx + front_sign * hr * 0.9))
        nose_y = float(round(head_cy - hr * 0.02))
        stamp_fuzzy_circle(
            canvas,
            nose_x,
            nose_y,
            max(0.24, hr * 0.08),
            (*_s_shadow, 230),
            1.0,
            0.9,
        )
        stamp_fuzzy_circle(
            canvas,
            nose_x + front_sign * 0.35,
            nose_y + 0.02,
            max(0.2, hr * 0.06),
            (*_s_shadow, 220),
            1.0,
            0.85,
        )


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------


def roll_character_appearance(seed: int, size: int = 20) -> CharacterAppearance:
    """Roll one deterministic appearance from a seed."""
    return CharacterAppearance.from_seed(seed, size)


def _render_pose(appearance: CharacterAppearance, pose: Pose) -> np.ndarray:
    """Render one pose from a rolled appearance."""
    canvas_size = appearance.body_params.canvas_size
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    _draw_character(canvas, appearance, pose)
    # Side profile drawing is authored left-facing. Materialize EAST-facing
    # sprites by mirroring here so runtime only needs UV selection.
    if pose.facing is Facing.EAST:
        canvas = np.ascontiguousarray(canvas[:, ::-1, :])
    darken_rim(canvas, (35, 35, 25))
    return canvas


def draw_character_pose(appearance: CharacterAppearance, pose: Pose) -> np.ndarray:
    """Render one pose from pre-rolled appearance data."""
    return _render_pose(appearance, pose)


def generate_character(
    seed: int,
    size: int = 20,
    *,
    pose: Pose = POSE_STAND,
    appearance: CharacterAppearance | None = None,
) -> np.ndarray:
    """Generate a single character sprite for one pose."""
    resolved_appearance = appearance or roll_character_appearance(seed, size)
    return _render_pose(resolved_appearance, pose)


def generate_character_pose_set(seed: int, size: int = 20) -> list[np.ndarray]:
    """Generate front/back/left/right standing poses for one seed."""
    appearance = roll_character_appearance(seed, size)
    return [_render_pose(appearance, pose) for pose in POSES]


def generate_character_sprite(seed: int, size: int = 20) -> np.ndarray:
    """Backward-compatible wrapper returning front-standing pose."""
    return generate_character_pose_set(seed, size)[0]


def _generate_caricature_for_build(seed: int, size: int, build_idx: int) -> np.ndarray:
    """Generate the standing pose for a specific body build template."""
    appearance = CharacterAppearance.from_seed(seed, size, forced_build_idx=build_idx)
    return _render_pose(appearance, POSE_STAND)


def generate_caricature_standard(seed: int, size: int = 20) -> np.ndarray:
    """Standard chibi body template."""
    return _generate_caricature_for_build(seed, size, 0)


def generate_caricature_belly(seed: int, size: int = 20) -> np.ndarray:
    """Belly-dominant body template."""
    return _generate_caricature_for_build(seed, size, 1)


def generate_caricature_broad(seed: int, size: int = 20) -> np.ndarray:
    """Broad/shoulder-dominant body template."""
    return _generate_caricature_for_build(seed, size, 2)


def generate_caricature_thin(seed: int, size: int = 20) -> np.ndarray:
    """Thin/narrow body template."""
    return _generate_caricature_for_build(seed, size, 3)


def generate_caricature_child(seed: int, size: int = 20) -> np.ndarray:
    """Child body template."""
    return _generate_caricature_for_build(seed, size, 4)


CARICATURE_TEMPLATES: list[tuple[str, Callable[[int, int], np.ndarray]]] = [
    ("standard", generate_caricature_standard),
    ("belly", generate_caricature_belly),
    ("broad", generate_caricature_broad),
    ("thin", generate_caricature_thin),
    ("child", generate_caricature_child),
]


__all__ = [
    "CARICATURE_TEMPLATES",
    "CHARACTER_DIRECTIONAL_POSE_COUNT",
    "CLOTHING_DEFS",
    "CLOTHING_PALETTES",
    "CLOTHING_STYLE_NAMES",
    "HAIR_PALETTES",
    "HAIR_STYLE_NAMES",
    "HUMANOID_GLYPHS",
    "PANTS_PALETTES",
    "POSES",
    "POSE_BACK_STAND",
    "POSE_FRONT_STAND",
    "POSE_LEFT_STAND",
    "POSE_RIGHT_STAND",
    "POSE_STAND",
    "SKIN_PALETTES",
    "BodyParams",
    "CharacterAppearance",
    "Facing",
    "Pose",
    "character_sprite_seed",
    "draw_character_pose",
    "generate_caricature_belly",
    "generate_caricature_broad",
    "generate_caricature_child",
    "generate_caricature_standard",
    "generate_caricature_thin",
    "generate_character",
    "generate_character_pose_set",
    "generate_character_sprite",
    "roll_character_appearance",
]
