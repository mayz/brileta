"""Procedural humanoid character sprite generation with directional output.

This module provides deterministic sprite generation for humanoid actors. A single
seed rolls one character appearance, then that appearance is rendered in four
directional standing poses (front/back/left/right).
"""

from __future__ import annotations

import colorsys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from brileta import colors
from brileta.sprites.primitives import Palette3
from brileta.types import Facing, MapDecorationSeed, SpatialSeed
from brileta.util import rng as brileta_rng

if TYPE_CHECKING:
    from .clothing import ClothingDrawFn

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
# Palette hue-shifting
# ---------------------------------------------------------------------------
#
# Hand-drawn pixel art rarely runs a single hue straight up a lightness ramp.
# It bends the ramp: shadow tones lean cool (toward blue/purple) and gain a
# little saturation, highlight tones lean warm (toward yellow/orange). We bend
# hue at constant HLS lightness so the value spacing (and the contrast
# guardrails downstream) survive; only the hue and saturation move.

_SHADOW_HUE: float = 0.63  # blue/purple, in colorsys 0..1 hue units
_HIGHLIGHT_HUE: float = 0.09  # warm orange
_SHADOW_HUE_PULL: float = 0.12  # fraction of the way toward the cool hue
_HIGHLIGHT_HUE_PULL: float = 0.10  # fraction of the way toward the warm hue
_SHADOW_SAT_BOOST: float = 0.06  # small saturation rise in shadows


def _bend_hue(
    color: colors.Color,
    target_hue: float,
    hue_pull: float,
    sat_delta: float,
) -> colors.Color:
    """Rotate ``color``'s hue toward ``target_hue`` at constant lightness.

    ``hue_pull`` is the fraction of the shortest-path hue distance to travel;
    ``sat_delta`` nudges saturation. Lightness (HLS ``l``) is untouched so the
    ramp's value spacing is preserved.
    """
    r, g, b = (channel / 255.0 for channel in color)
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    # Rotate along the shortest path around the hue wheel.
    delta_h = ((target_hue - h + 0.5) % 1.0) - 0.5
    h = (h + delta_h * hue_pull) % 1.0
    s = min(1.0, max(0.0, s + sat_delta))
    r, g, b = colorsys.hls_to_rgb(h, lightness, s)
    return (round(r * 255), round(g * 255), round(b * 255))


def _hue_shift_ramp(ramp: Palette3) -> Palette3:
    """Bend a (shadow, mid, highlight) ramp cool-in-shadow, warm-in-highlight."""
    shadow, mid, highlight = ramp
    return (
        _bend_hue(shadow, _SHADOW_HUE, _SHADOW_HUE_PULL, _SHADOW_SAT_BOOST),
        mid,
        _bend_hue(highlight, _HIGHLIGHT_HUE, _HIGHLIGHT_HUE_PULL, 0.0),
    )


_RAW_SKIN_PALETTES: list[Palette3] = [
    ((150, 105, 75), (200, 160, 125), (230, 200, 170)),
    ((140, 95, 60), (190, 140, 100), (225, 185, 150)),
    ((115, 70, 40), (165, 110, 65), (200, 150, 105)),
    ((80, 50, 30), (125, 75, 45), (165, 115, 80)),
    ((55, 35, 20), (95, 55, 30), (135, 90, 55)),
]

_RAW_HAIR_PALETTES: list[Palette3] = [
    ((15, 12, 8), (35, 25, 18), (55, 42, 30)),
    ((50, 30, 15), (85, 55, 25), (125, 80, 40)),
    ((95, 45, 12), (155, 80, 18), (195, 125, 45)),
    ((135, 105, 45), (195, 170, 75), (225, 205, 125)),
    ((90, 90, 95), (145, 145, 150), (190, 190, 195)),
    ((170, 160, 150), (210, 200, 190), (235, 225, 218)),
]

_RAW_CLOTHING_PALETTES: list[Palette3] = [
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

_RAW_PANTS_PALETTES: list[Palette3] = [
    ((25, 25, 30), (50, 50, 60), (75, 75, 85)),
    ((40, 30, 20), (75, 55, 35), (110, 85, 55)),
    ((20, 25, 45), (35, 45, 80), (55, 70, 115)),
    ((18, 18, 18), (35, 35, 35), (55, 55, 55)),
    ((50, 45, 35), (90, 80, 60), (130, 120, 95)),
]

# Hue-shifted ramps used everywhere. The transform is deterministic (no RNG),
# so sprites stay byte-identical for a given seed.
SKIN_PALETTES: list[Palette3] = [_hue_shift_ramp(p) for p in _RAW_SKIN_PALETTES]
HAIR_PALETTES: list[Palette3] = [_hue_shift_ramp(p) for p in _RAW_HAIR_PALETTES]
CLOTHING_PALETTES: list[Palette3] = [_hue_shift_ramp(p) for p in _RAW_CLOTHING_PALETTES]
PANTS_PALETTES: list[Palette3] = [_hue_shift_ramp(p) for p in _RAW_PANTS_PALETTES]


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
# Walk cycle poses
# ---------------------------------------------------------------------------
#
# A 2-frame walk per facing (walk-A / walk-B) plus the standing pose. The two
# walk frames swap which leg leads and counter-swing the arms, so alternating
# them while moving reads as a step. Offsets are deliberately small (~1px): at
# 20x20 a bigger stride pops the silhouette. body_dy is left at 0 so the two
# walk frames never jump vertically against each other (no bob popping); the
# stride and arm swing carry the motion.
#
# Front/back legs stride vertically (one foot plants low, the other lifts);
# side legs stride horizontally (front foot forward, back foot back). EAST walk
# frames reuse the WEST-oriented offsets and rely on the renderer's canvas
# mirror, exactly like the standing poses.

_LEG_STRIDE_V = 1.6  # front/back: lifted foot rises (planted foot clamps to floor)
_LEG_STRIDE_H = 2.1  # side horizontal foot swing
_ARM_SWING = 1.3  # counter-swing of the hands (must be visible at 1x)

POSE_FRONT_WALK_A = Pose(
    "front_walk_a",
    facing=Facing.SOUTH,
    left_leg_dy=_LEG_STRIDE_V,
    right_leg_dy=-_LEG_STRIDE_V,
    left_arm_dy=-_ARM_SWING,
    right_arm_dy=_ARM_SWING,
)
POSE_FRONT_WALK_B = Pose(
    "front_walk_b",
    facing=Facing.SOUTH,
    left_leg_dy=-_LEG_STRIDE_V,
    right_leg_dy=_LEG_STRIDE_V,
    left_arm_dy=_ARM_SWING,
    right_arm_dy=-_ARM_SWING,
)
POSE_BACK_WALK_A = Pose(
    "back_walk_a",
    facing=Facing.NORTH,
    left_leg_dy=_LEG_STRIDE_V,
    right_leg_dy=-_LEG_STRIDE_V,
    left_arm_dy=-_ARM_SWING,
    right_arm_dy=_ARM_SWING,
)
POSE_BACK_WALK_B = Pose(
    "back_walk_b",
    facing=Facing.NORTH,
    left_leg_dy=-_LEG_STRIDE_V,
    right_leg_dy=_LEG_STRIDE_V,
    left_arm_dy=_ARM_SWING,
    right_arm_dy=-_ARM_SWING,
)
POSE_LEFT_WALK_A = Pose(
    "left_walk_a",
    facing=Facing.WEST,
    left_leg_dx=-_LEG_STRIDE_H,
    right_leg_dx=_LEG_STRIDE_H,
    left_arm_dy=_ARM_SWING,
    right_arm_dy=-_ARM_SWING,
)
POSE_LEFT_WALK_B = Pose(
    "left_walk_b",
    facing=Facing.WEST,
    left_leg_dx=_LEG_STRIDE_H,
    right_leg_dx=-_LEG_STRIDE_H,
    left_arm_dy=-_ARM_SWING,
    right_arm_dy=_ARM_SWING,
)
POSE_RIGHT_WALK_A = Pose(
    "right_walk_a",
    facing=Facing.EAST,
    left_leg_dx=-_LEG_STRIDE_H,
    right_leg_dx=_LEG_STRIDE_H,
    left_arm_dy=_ARM_SWING,
    right_arm_dy=-_ARM_SWING,
)
POSE_RIGHT_WALK_B = Pose(
    "right_walk_b",
    facing=Facing.EAST,
    left_leg_dx=_LEG_STRIDE_H,
    right_leg_dx=-_LEG_STRIDE_H,
    left_arm_dy=-_ARM_SWING,
    right_arm_dy=_ARM_SWING,
)

# Frames per facing, in selection order: stand, walk-A, walk-B. The frame index
# is 0 when idle and 1 + walk_frame (1 or 2) while moving.
CHARACTER_FRAMES_PER_FACING: int = 3

# The full render/atlas set: (facings x frames) poses, grouped by facing in the
# same facing order as CHARACTER_DIRECTIONAL_POSE_INDEX (S, N, W, E), each facing
# followed by its two walk frames. Front-stand stays at index 0 so the initial
# resting sprite is unchanged.
CHARACTER_POSES: tuple[Pose, ...] = (
    POSE_FRONT_STAND,
    POSE_FRONT_WALK_A,
    POSE_FRONT_WALK_B,
    POSE_BACK_STAND,
    POSE_BACK_WALK_A,
    POSE_BACK_WALK_B,
    POSE_LEFT_STAND,
    POSE_LEFT_WALK_A,
    POSE_LEFT_WALK_B,
    POSE_RIGHT_STAND,
    POSE_RIGHT_WALK_A,
    POSE_RIGHT_WALK_B,
)

CHARACTER_POSE_COUNT: int = len(CHARACTER_POSES)


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
    arm_shoulder_factor: float
    arm_end_torso_factor: float
    arm_end_belly_factor: float
    arm_shoulder_anchor_torso_factor: float
    arm_shoulder_anchor_shoulder_width_factor: float
    arm_hand_inset: float
    arm_hand_drop: float
    arm_hand_alpha: int
    neck_y_offset: float
    arm_end_from_belly: bool = False
    leg_from_torso: bool = False


BodyParamRollFn = Callable[[np.random.Generator, int], BodyParams]


@dataclass(frozen=True)
class CharacterPresentationProfile:
    """Weighted visual presentation knobs for humanoid character sprites.

    These weights bias generated appearance without making hard promises about
    gender or body shape. A profile changes only visual selection probabilities.
    """

    build_weights: tuple[float, ...]
    hair_weights: tuple[float, ...]
    clothing_weights: tuple[float, ...]


NEUTRAL_PRESENTATION = CharacterPresentationProfile(
    build_weights=(1.0, 1.0, 1.0, 1.0, 0.35),
    hair_weights=(0.7, 1.6, 1.4, 1.0, 0.8),
    clothing_weights=(0.45, 1.4, 1.4, 1.2, 0.55, 0.55),
)
MASC_PRESENTATION = CharacterPresentationProfile(
    build_weights=(1.3, 1.1, 1.4, 0.75, 0.25),
    hair_weights=(1.0, 2.4, 0.9, 0.35, 1.1),
    clothing_weights=(0.55, 1.5, 1.5, 1.2, 0.8, 0.3),
)
FEM_PRESENTATION = CharacterPresentationProfile(
    build_weights=(1.25, 0.65, 0.45, 1.25, 0.25),
    hair_weights=(0.25, 0.75, 1.7, 2.4, 0.45),
    clothing_weights=(0.15, 1.4, 1.4, 1.2, 0.25, 1.1),
)


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
        presentation_profile: CharacterPresentationProfile | None = None,
    ) -> CharacterAppearance:
        """Roll appearance data deterministically from ``seed``.

        Build selection intentionally uses a dedicated RNG stream to preserve the
        historical behavior of ``generate_character_sprite(seed)``.
        """
        # Keep imports local to avoid module import cycles while preserving
        # deterministic roll behavior.
        from .clothing import CLOTHING_DEFS
        from .hair import HAIR_STYLES

        chooser_rng = np.random.default_rng(seed)
        if forced_build_idx is None:
            if presentation_profile is None:
                build_idx = int(chooser_rng.integers(len(_BODY_PARAM_BUILD_ROLLERS)))
            else:
                build_idx = _weighted_index(
                    chooser_rng,
                    presentation_profile.build_weights,
                    len(_BODY_PARAM_BUILD_ROLLERS),
                )
        else:
            build_idx = forced_build_idx

        build_name, roll_fn = _BODY_PARAM_BUILD_ROLLERS[build_idx]

        rng = np.random.default_rng(seed)
        skin_pal, hair_pal, cloth_pal, pants_pal = _roll_palettes(rng)
        body_params = roll_fn(rng, size)
        if presentation_profile is None:
            hair_style_idx = int(rng.integers(len(HAIR_STYLES)))
        else:
            hair_style_idx = _weighted_index(
                rng, presentation_profile.hair_weights, len(HAIR_STYLES)
            )

        if presentation_profile is None:
            clothing_style_idx = int(rng.integers(len(CLOTHING_DEFS)))
        else:
            clothing_style_idx = _weighted_index(
                rng, presentation_profile.clothing_weights, len(CLOTHING_DEFS)
            )
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
# Build parameter rolling
# ---------------------------------------------------------------------------


def _weighted_index(
    rng: np.random.Generator,
    weights: tuple[float, ...],
    expected_len: int,
) -> int:
    """Return a deterministic weighted index, validating profile shape."""
    if len(weights) != expected_len:
        msg = f"Expected {expected_len} weights, got {len(weights)}"
        raise ValueError(msg)

    weight_array = np.array(weights, dtype=np.float64)
    if bool(np.any(weight_array < 0.0)):
        raise ValueError("Weights must be non-negative")

    total = float(weight_array.sum())
    if total <= 0.0:
        raise ValueError("At least one weight must be positive")

    probabilities = weight_array / total
    return int(rng.choice(expected_len, p=probabilities))


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
    neck_gap = 0.3 + torso_ry * 0.07
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
        neck_gap=neck_gap,
        hand_radius=0.8,
        arm_shoulder_factor=0.28,
        arm_end_torso_factor=0.66,
        arm_end_belly_factor=0.22,
        arm_shoulder_anchor_torso_factor=0.84,
        arm_shoulder_anchor_shoulder_width_factor=0.0,
        arm_hand_inset=0.18,
        arm_hand_drop=0.5,
        arm_hand_alpha=215,
        neck_y_offset=neck_gap * 0.5,
    )


def _roll_belly_body_params(rng: np.random.Generator, size: int) -> BodyParams:
    chest_ry = 1.1 + float(rng.uniform(0, 0.3))
    # Rounder, wider belly so this build reads clearly heavier than standard.
    belly_ry = 3.8 + float(rng.uniform(0, 1.0))
    return BodyParams(
        canvas_size=size,
        head_radius=3.0 + float(rng.uniform(0, 0.4)),
        head_top_pad=1.5,
        torso_rx=3.2 + float(rng.uniform(0, 0.4)),
        torso_ry=chest_ry,
        torso_cy_factor=0.4,
        belly_rx=4.2 + float(rng.uniform(0, 0.8)),
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
        arm_shoulder_factor=0.18,
        arm_end_torso_factor=0.66,
        arm_end_belly_factor=0.22,
        arm_shoulder_anchor_torso_factor=0.85,
        arm_shoulder_anchor_shoulder_width_factor=0.0,
        arm_hand_inset=0.2,
        arm_hand_drop=0.5,
        arm_hand_alpha=215,
        neck_y_offset=0.1,
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
        arm_shoulder_factor=0.28,
        arm_end_torso_factor=0.62,
        arm_end_belly_factor=0.22,
        arm_shoulder_anchor_torso_factor=0.7,
        arm_shoulder_anchor_shoulder_width_factor=0.12,
        arm_hand_inset=0.24,
        arm_hand_drop=0.5,
        arm_hand_alpha=215,
        neck_y_offset=0.1,
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
        arm_shoulder_factor=0.26,
        arm_end_torso_factor=0.68,
        arm_end_belly_factor=0.22,
        arm_shoulder_anchor_torso_factor=0.82,
        arm_shoulder_anchor_shoulder_width_factor=0.0,
        arm_hand_inset=0.14,
        arm_hand_drop=0.3,
        arm_hand_alpha=210,
        neck_y_offset=0.15,
    )


def _roll_child_body_params(rng: np.random.Generator, size: int) -> BodyParams:
    return BodyParams(
        canvas_size=max(8, size - 5),
        # Bigger head-to-body ratio: larger head over a small narrow torso is
        # the strongest cue that reads "child" at 1x.
        head_radius=3.9 + float(rng.uniform(0, 0.3)),
        head_top_pad=1.2,
        torso_rx=2.0,
        torso_ry=1.25,
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
        arm_shoulder_factor=0.18,
        arm_end_torso_factor=0.64,
        arm_end_belly_factor=0.22,
        arm_shoulder_anchor_torso_factor=0.8,
        arm_shoulder_anchor_shoulder_width_factor=0.0,
        arm_hand_inset=0.12,
        arm_hand_drop=0.3,
        arm_hand_alpha=210,
        neck_y_offset=0.15,
        leg_from_torso=True,
    )


_BODY_PARAM_BUILD_ROLLERS: tuple[tuple[str, BodyParamRollFn], ...] = (
    ("standard", _roll_standard_body_params),
    ("belly", _roll_belly_body_params),
    ("broad", _roll_broad_body_params),
    ("thin", _roll_thin_body_params),
    ("child", _roll_child_body_params),
)
