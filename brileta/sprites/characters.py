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
    CanvasStamper,
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
# Data-driven stamp specs
# ---------------------------------------------------------------------------

HairDrawFn = Callable[
    [np.ndarray, float, float, float, Palette3, np.random.Generator, Facing],
    None,
]


@dataclass(frozen=True)
class LinearExpr:
    """Linear expression against a named draw value."""

    base_key: str | None = None
    scale_key: str | None = None
    factor: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class EllipseStampSpec:
    """One ellipse stamp resolved from draw context values."""

    cx: LinearExpr
    cy: LinearExpr
    rx: LinearExpr
    ry: LinearExpr
    tone_idx: int
    alpha: int
    falloff: float
    hardness: float


@dataclass(frozen=True)
class CircleStampSpec:
    """One circle stamp resolved from draw context values."""

    cx: LinearExpr
    cy: LinearExpr
    radius: LinearExpr
    tone_idx: int
    alpha: int
    falloff: float
    hardness: float


ToneStampSpec = EllipseStampSpec | CircleStampSpec


@dataclass(frozen=True)
class ShapeContext:
    """Shared scalar values for resolving stamp expressions."""

    values: dict[str, float]


def _expr(
    base_key: str | None = None,
    scale_key: str | None = None,
    factor: float = 1.0,
    offset: float = 0.0,
) -> LinearExpr:
    """Convenience constructor for ``LinearExpr``."""
    return LinearExpr(
        base_key=base_key, scale_key=scale_key, factor=factor, offset=offset
    )


def _resolve_expr(expr: LinearExpr, context: ShapeContext) -> float:
    """Resolve a linear expression to a concrete scalar."""
    value = expr.offset
    if expr.base_key is not None:
        if expr.scale_key is None:
            value += context.values[expr.base_key] * expr.factor
        else:
            value += context.values[expr.base_key]
    if expr.scale_key is not None:
        value += context.values[expr.scale_key] * expr.factor
    return value


def _ellipses_are_disjoint(ellipses: list[tuple[float, float, float, float]]) -> bool:
    """Return True when ellipses are safely non-overlapping.

    The +4.0 margin is intentionally conservative to preserve byte-identical
    output compared to sequential stamping on tiny sprites.
    """
    for i, (cx_a, cy_a, rx_a, ry_a) in enumerate(ellipses):
        for cx_b, cy_b, rx_b, ry_b in ellipses[i + 1 :]:
            if abs(cx_a - cx_b) <= (rx_a + rx_b + 4.0) and abs(cy_a - cy_b) <= (
                ry_a + ry_b + 4.0
            ):
                return False
    return True


def _circles_are_disjoint(circles: list[tuple[float, float, float]]) -> bool:
    """Return True when circles are safely non-overlapping."""
    for i, (cx_a, cy_a, r_a) in enumerate(circles):
        for cx_b, cy_b, r_b in circles[i + 1 :]:
            dx = cx_a - cx_b
            dy = cy_a - cy_b
            max_r = r_a + r_b + 2.0
            if dx * dx + dy * dy <= max_r * max_r:
                return False
    return True


def _stamp_tone_specs(
    canvas: np.ndarray,
    palette: Palette3,
    specs: tuple[ToneStampSpec, ...],
    context: ShapeContext,
) -> None:
    """Render data-driven specs in-order with safe opportunistic batching."""
    stamper = CanvasStamper(canvas)

    run_kind: str | None = None
    run_rgba: colors.ColorRGBA | None = None
    run_falloff = 0.0
    run_hardness = 0.0
    ellipse_run: list[tuple[float, float, float, float]] = []
    circle_run: list[tuple[float, float, float]] = []

    def flush() -> None:
        nonlocal run_kind
        if run_kind == "ellipse" and run_rgba is not None:
            if len(ellipse_run) > 1 and _ellipses_are_disjoint(ellipse_run):
                stamper.batch_stamp_ellipses(
                    ellipse_run,
                    run_rgba,
                    falloff=run_falloff,
                    hardness=run_hardness,
                )
            else:
                for cx, cy, rx, ry in ellipse_run:
                    stamper.stamp_ellipse(
                        cx,
                        cy,
                        rx,
                        ry,
                        run_rgba,
                        falloff=run_falloff,
                        hardness=run_hardness,
                    )
        if run_kind == "circle" and run_rgba is not None:
            for cx, cy, radius in circle_run:
                # Keep circles on the historical fuzzy-circle path for
                # byte-identical output with pre-refactor sprites.
                stamp_fuzzy_circle(
                    canvas,
                    cx,
                    cy,
                    radius,
                    run_rgba,
                    falloff=run_falloff,
                    hardness=run_hardness,
                )
        run_kind = None
        ellipse_run.clear()
        circle_run.clear()

    for spec in specs:
        if isinstance(spec, EllipseStampSpec):
            kind = "ellipse"
            rgba: colors.ColorRGBA = (*palette[spec.tone_idx], spec.alpha)
            falloff = spec.falloff
            hardness = spec.hardness
            shape = (
                _resolve_expr(spec.cx, context),
                _resolve_expr(spec.cy, context),
                _resolve_expr(spec.rx, context),
                _resolve_expr(spec.ry, context),
            )
        else:
            kind = "circle"
            rgba = (*palette[spec.tone_idx], spec.alpha)
            falloff = spec.falloff
            hardness = spec.hardness
            shape = (
                _resolve_expr(spec.cx, context),
                _resolve_expr(spec.cy, context),
                _resolve_expr(spec.radius, context),
            )

        if (
            run_kind != kind
            or run_rgba != rgba
            or run_falloff != falloff
            or run_hardness != hardness
        ):
            flush()
            run_kind = kind
            run_rgba = rgba
            run_falloff = falloff
            run_hardness = hardness

        if kind == "ellipse":
            ellipse_run.append(shape)  # type: ignore[arg-type]
        else:
            circle_run.append(shape)  # type: ignore[arg-type]

    flush()


def _head_ellipse(
    cx_factor: float,
    cy_factor: float,
    rx_factor: float,
    ry_factor: float,
    *,
    tone_idx: int,
    alpha: int,
    falloff: float,
    hardness: float,
    cx_base: str = "hx",
    cy_base: str = "hy",
    cx_scale: str = "hr",
    cy_scale: str = "hr",
    cy_offset: float = 0.0,
) -> EllipseStampSpec:
    """Build a head-relative ellipse spec using factor inputs."""
    return EllipseStampSpec(
        cx=_expr(cx_base, cx_scale, cx_factor),
        cy=_expr(cy_base, cy_scale, cy_factor, offset=cy_offset),
        rx=_expr(None, "hr", rx_factor),
        ry=_expr(None, "hr", ry_factor),
        tone_idx=tone_idx,
        alpha=alpha,
        falloff=falloff,
        hardness=hardness,
    )


_HAIR_SHORT_NORTH: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0, -0.15, 1.15, 0.95, tone_idx=1, alpha=235, falloff=1.8, hardness=0.85
    ),
    _head_ellipse(
        -0.0, -0.3, 0.8, 0.6, tone_idx=2, alpha=200, falloff=1.5, hardness=0.75
    ),
)
_HAIR_SHORT_SIDE: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        0.04,
        -0.12,
        0.9,
        0.62,
        tone_idx=1,
        alpha=235,
        falloff=1.9,
        hardness=0.86,
        cy_base="cap_cy",
    ),
    _head_ellipse(
        0.52, -0.02, 0.25, 0.34, tone_idx=1, alpha=220, falloff=1.45, hardness=0.76
    ),
    _head_ellipse(
        -0.22,
        -0.34,
        0.36,
        0.24,
        tone_idx=2,
        alpha=198,
        falloff=1.35,
        hardness=0.72,
        cy_base="cap_cy",
    ),
)
_HAIR_SHORT_SOUTH: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0,
        -0.12,
        1.08,
        0.54,
        tone_idx=1,
        alpha=235,
        falloff=1.9,
        hardness=0.86,
        cy_base="cap_cy",
    ),
    EllipseStampSpec(
        cx=_expr("hx"),
        cy=_expr("cap_cy", None, offset=-0.7),
        rx=_expr(None, "hr", 0.62),
        ry=_expr(None, "hr", 0.36),
        tone_idx=2,
        alpha=200,
        falloff=1.5,
        hardness=0.75,
    ),
)

_HAIR_MEDIUM_NORTH: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0, -0.1, 1.25, 1.1, tone_idx=0, alpha=235, falloff=1.6, hardness=0.82
    ),
    _head_ellipse(
        -0.0, -0.2, 1.15, 1.0, tone_idx=1, alpha=225, falloff=1.5, hardness=0.8
    ),
    _head_ellipse(
        -0.85, 0.3, 0.45, 0.7, tone_idx=0, alpha=220, falloff=1.5, hardness=0.78
    ),
    _head_ellipse(
        0.85, 0.3, 0.45, 0.7, tone_idx=0, alpha=220, falloff=1.5, hardness=0.78
    ),
    _head_ellipse(
        -0.0, -0.4, 0.7, 0.5, tone_idx=2, alpha=195, falloff=1.3, hardness=0.7
    ),
)
_HAIR_MEDIUM_SIDE: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        0.06, -0.48, 1.12, 0.7, tone_idx=0, alpha=235, falloff=1.6, hardness=0.82
    ),
    _head_ellipse(
        0.05, -0.56, 1.0, 0.6, tone_idx=1, alpha=225, falloff=1.5, hardness=0.8
    ),
    _head_ellipse(
        0.78, 0.3, 0.4, 0.76, tone_idx=0, alpha=220, falloff=1.5, hardness=0.78
    ),
    _head_ellipse(
        -0.54, 0.02, 0.12, 0.24, tone_idx=1, alpha=214, falloff=1.35, hardness=0.74
    ),
    _head_ellipse(
        -0.3, -0.76, 0.48, 0.32, tone_idx=2, alpha=195, falloff=1.3, hardness=0.7
    ),
)
_HAIR_MEDIUM_SOUTH: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0, -0.46, 1.18, 0.74, tone_idx=0, alpha=235, falloff=1.6, hardness=0.82
    ),
    _head_ellipse(
        -0.0, -0.56, 1.06, 0.64, tone_idx=1, alpha=225, falloff=1.5, hardness=0.8
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", -0.85),
        cy=_expr("hy", "hr", 0.12),
        rx=_expr(None, "hr_side_scale", 0.4),
        ry=_expr(None, "hr_side_scale", 0.55),
        tone_idx=0,
        alpha=220,
        falloff=1.5,
        hardness=0.78,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", 0.85),
        cy=_expr("hy", "hr", 0.12),
        rx=_expr(None, "hr_side_scale", 0.4),
        ry=_expr(None, "hr_side_scale", 0.55),
        tone_idx=0,
        alpha=220,
        falloff=1.5,
        hardness=0.78,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "highlight_jitter", 1.0),
        cy=_expr("hy", "hr", -0.75),
        rx=_expr(None, "hr", 0.58),
        ry=_expr(None, "hr", 0.38),
        tone_idx=2,
        alpha=195,
        falloff=1.3,
        hardness=0.7,
    ),
)

_HAIR_LONG_NORTH: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0, 0.0, 1.3, 1.1, tone_idx=0, alpha=235, falloff=1.6, hardness=0.82
    ),
    _head_ellipse(
        -0.0, -0.1, 1.2, 1.0, tone_idx=1, alpha=225, falloff=1.5, hardness=0.8
    ),
    EllipseStampSpec(
        cx=_expr("hx"),
        cy=_expr("hy", "drape_main", 0.3),
        rx=_expr(None, "hr", 0.9),
        ry=_expr(None, "drape_main", 0.45),
        tone_idx=0,
        alpha=220,
        falloff=1.7,
        hardness=0.78,
    ),
    EllipseStampSpec(
        cx=_expr("hx"),
        cy=_expr("hy", "drape_main", 0.25),
        rx=_expr(None, "hr", 0.75),
        ry=_expr(None, "drape_main", 0.38),
        tone_idx=1,
        alpha=215,
        falloff=1.5,
        hardness=0.75,
    ),
    _head_ellipse(
        -0.0, -0.3, 0.7, 0.6, tone_idx=2, alpha=190, falloff=1.3, hardness=0.68
    ),
)
_HAIR_LONG_SIDE: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        0.1, -0.38, 1.15, 0.8, tone_idx=0, alpha=235, falloff=1.6, hardness=0.82
    ),
    _head_ellipse(
        0.08, -0.46, 1.02, 0.7, tone_idx=1, alpha=225, falloff=1.5, hardness=0.8
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", 0.62),
        cy=_expr("hy", "back_drape", 0.2),
        rx=_expr(None, "hr", 0.32),
        ry=_expr(None, "back_drape", 0.3),
        tone_idx=0,
        alpha=215,
        falloff=1.8,
        hardness=0.78,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", 0.58),
        cy=_expr("hy", "back_drape", 0.15),
        rx=_expr(None, "hr", 0.24),
        ry=_expr(None, "back_drape", 0.24),
        tone_idx=1,
        alpha=210,
        falloff=1.6,
        hardness=0.75,
    ),
    _head_ellipse(
        -0.52, 0.12, 0.1, 0.28, tone_idx=1, alpha=208, falloff=1.35, hardness=0.72
    ),
    _head_ellipse(
        -0.26, -0.7, 0.56, 0.4, tone_idx=2, alpha=190, falloff=1.3, hardness=0.68
    ),
)
_HAIR_LONG_SOUTH: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0, -0.48, 1.14, 0.72, tone_idx=0, alpha=235, falloff=1.6, hardness=0.82
    ),
    _head_ellipse(
        -0.0, -0.58, 1.02, 0.62, tone_idx=1, alpha=225, falloff=1.5, hardness=0.8
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", -0.78),
        cy=_expr("hy", "drape_left", 0.28),
        rx=_expr(None, "hr_side_scale", 0.3),
        ry=_expr(None, "drape_left_scaled", 0.22),
        tone_idx=0,
        alpha=215,
        falloff=1.8,
        hardness=0.78,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", -0.78),
        cy=_expr("hy", "drape_left", 0.24),
        rx=_expr(None, "hr_side_scale", 0.21),
        ry=_expr(None, "drape_left_scaled", 0.18),
        tone_idx=1,
        alpha=210,
        falloff=1.6,
        hardness=0.75,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", 0.78),
        cy=_expr("hy", "drape_right", 0.28),
        rx=_expr(None, "hr_side_scale", 0.3),
        ry=_expr(None, "drape_right_scaled", 0.22),
        tone_idx=0,
        alpha=215,
        falloff=1.8,
        hardness=0.78,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", 0.78),
        cy=_expr("hy", "drape_right", 0.24),
        rx=_expr(None, "hr_side_scale", 0.21),
        ry=_expr(None, "drape_right_scaled", 0.18),
        tone_idx=1,
        alpha=210,
        falloff=1.6,
        hardness=0.75,
    ),
    _head_ellipse(
        -0.0, -0.74, 0.66, 0.42, tone_idx=2, alpha=190, falloff=1.3, hardness=0.68
    ),
)
_HAIR_LONG_FALLBACK: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0, -0.35, 1.18, 0.82, tone_idx=0, alpha=235, falloff=1.6, hardness=0.82
    ),
    _head_ellipse(
        -0.0, -0.45, 1.08, 0.72, tone_idx=1, alpha=225, falloff=1.5, hardness=0.8
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", -0.7),
        cy=_expr("hy", "drape_left", 0.22),
        rx=_expr(None, "hr_side_scale", 0.36),
        ry=_expr(None, "drape_left_scaled", 0.28),
        tone_idx=0,
        alpha=215,
        falloff=1.8,
        hardness=0.78,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", -0.7),
        cy=_expr("hy", "drape_left", 0.18),
        rx=_expr(None, "hr_side_scale", 0.24),
        ry=_expr(None, "drape_left_scaled", 0.22),
        tone_idx=1,
        alpha=210,
        falloff=1.6,
        hardness=0.75,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", 0.7),
        cy=_expr("hy", "drape_right", 0.22),
        rx=_expr(None, "hr_side_scale", 0.36),
        ry=_expr(None, "drape_right_scaled", 0.28),
        tone_idx=0,
        alpha=215,
        falloff=1.8,
        hardness=0.78,
    ),
    EllipseStampSpec(
        cx=_expr("hx", "hr", 0.7),
        cy=_expr("hy", "drape_right", 0.18),
        rx=_expr(None, "hr_side_scale", 0.24),
        ry=_expr(None, "drape_right_scaled", 0.22),
        tone_idx=1,
        alpha=210,
        falloff=1.6,
        hardness=0.75,
    ),
    _head_ellipse(
        -0.0, -0.68, 0.68, 0.44, tone_idx=2, alpha=190, falloff=1.3, hardness=0.68
    ),
)

_HAIR_TALL_NORTH: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0, -0.1, 1.1, 1.0, tone_idx=0, alpha=230, falloff=1.7, hardness=0.82
    ),
    _head_ellipse(
        -0.0, -1.0, 0.65, 1.1, tone_idx=1, alpha=225, falloff=1.6, hardness=0.8
    ),
    _head_ellipse(
        -0.0, -1.2, 0.45, 0.8, tone_idx=2, alpha=200, falloff=1.3, hardness=0.7
    ),
)
_HAIR_TALL_SIDE: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        0.04, -1.12, 0.66, 1.12, tone_idx=0, alpha=230, falloff=1.8, hardness=0.82
    ),
    _head_ellipse(
        0.02, -1.24, 0.52, 0.94, tone_idx=1, alpha=220, falloff=1.5, hardness=0.78
    ),
    _head_ellipse(
        0.22, -0.24, 0.34, 0.48, tone_idx=0, alpha=220, falloff=1.45, hardness=0.74
    ),
    _head_ellipse(
        0.18, -0.28, 0.25, 0.36, tone_idx=1, alpha=210, falloff=1.35, hardness=0.7
    ),
    _head_ellipse(
        0.58, -0.02, 0.3, 0.44, tone_idx=0, alpha=214, falloff=1.4, hardness=0.74
    ),
    _head_ellipse(
        -0.22, -0.34, 0.2, 0.26, tone_idx=2, alpha=198, falloff=1.3, hardness=0.7
    ),
)
_HAIR_TALL_SOUTH: tuple[ToneStampSpec, ...] = (
    _head_ellipse(
        -0.0, -1.2, 0.7, 1.2, tone_idx=0, alpha=230, falloff=1.8, hardness=0.82
    ),
    _head_ellipse(
        -0.0, -1.3, 0.55, 1.0, tone_idx=1, alpha=220, falloff=1.5, hardness=0.78
    ),
    _head_ellipse(
        -0.0, -1.5, 0.35, 0.7, tone_idx=2, alpha=200, falloff=1.3, hardness=0.7
    ),
)


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
    """Short crop hairstyle rendered from declarative specs."""
    cap_cy = hy - hr * 0.45
    context = ShapeContext({"hx": hx, "hy": hy, "hr": hr, "cap_cy": cap_cy})
    if facing == Facing.NORTH:
        specs = _HAIR_SHORT_NORTH
    elif facing in {Facing.EAST, Facing.WEST}:
        specs = _HAIR_SHORT_SIDE
    else:
        specs = _HAIR_SHORT_SOUTH
    _stamp_tone_specs(canvas, pal, specs, context)


def _hair_medium(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Medium hair with side drape rendered from declarative specs."""
    context_values: dict[str, float] = {"hx": hx, "hy": hy, "hr": hr}
    if facing == Facing.NORTH:
        specs = _HAIR_MEDIUM_NORTH
    elif facing in {Facing.EAST, Facing.WEST}:
        specs = _HAIR_MEDIUM_SIDE
    else:
        side_scale = 0.72 if _luma(pal[1]) > 155.0 else 1.0
        context_values["hr_side_scale"] = hr * side_scale
        context_values["highlight_jitter"] = float(rng.uniform(-0.4, 0.4))
        specs = _HAIR_MEDIUM_SOUTH
    _stamp_tone_specs(canvas, pal, specs, ShapeContext(context_values))


def _hair_long(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Long hair rendered from declarative specs."""
    context_values: dict[str, float] = {"hx": hx, "hy": hy, "hr": hr}
    if facing == Facing.NORTH:
        context_values["drape_main"] = hr * float(rng.uniform(2.2, 3.0))
        specs = _HAIR_LONG_NORTH
    elif facing in {Facing.EAST, Facing.WEST}:
        context_values["back_drape"] = hr * float(rng.uniform(2.1, 3.0))
        specs = _HAIR_LONG_SIDE
    elif facing == Facing.SOUTH:
        side_scale = 0.74 if _luma(pal[1]) > 155.0 else 1.0
        context_values["hr_side_scale"] = hr * side_scale
        context_values["drape_left"] = hr * float(rng.uniform(2.0, 3.0))
        context_values["drape_right"] = hr * float(rng.uniform(2.0, 3.0))
        context_values["drape_left_scaled"] = context_values["drape_left"] * side_scale
        context_values["drape_right_scaled"] = (
            context_values["drape_right"] * side_scale
        )
        specs = _HAIR_LONG_SOUTH
    else:
        side_scale = 0.74 if _luma(pal[1]) > 155.0 else 1.0
        context_values["hr_side_scale"] = hr * side_scale
        context_values["drape_left"] = hr * float(rng.uniform(2.0, 3.0))
        context_values["drape_right"] = hr * float(rng.uniform(2.0, 3.0))
        context_values["drape_left_scaled"] = context_values["drape_left"] * side_scale
        context_values["drape_right_scaled"] = (
            context_values["drape_right"] * side_scale
        )
        specs = _HAIR_LONG_FALLBACK
    _stamp_tone_specs(canvas, pal, specs, ShapeContext(context_values))


def _hair_tall(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    _rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Tall upward hair rendered from declarative specs."""
    if facing == Facing.NORTH:
        specs = _HAIR_TALL_NORTH
    elif facing in {Facing.EAST, Facing.WEST}:
        specs = _HAIR_TALL_SIDE
    else:
        specs = _HAIR_TALL_SOUTH
    _stamp_tone_specs(canvas, pal, specs, ShapeContext({"hx": hx, "hy": hy, "hr": hr}))


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
    """Draw one torso mass with three-tone shading from data specs."""
    if facing == Facing.NORTH:
        highlight_alpha = 185
        highlight_rx = rx * 0.55
        highlight_ry = ry * 0.4
    else:
        highlight_alpha = 210
        highlight_rx = rx * 0.7
        highlight_ry = ry * 0.5

    specs: tuple[ToneStampSpec, ...] = (
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("cy", None, offset=0.3),
            rx=_expr("rx", None, offset=0.3),
            ry=_expr("ry", None, offset=0.3),
            tone_idx=0,
            alpha=240,
            falloff=2.0,
            hardness=0.88,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("cy"),
            rx=_expr("rx"),
            ry=_expr("ry"),
            tone_idx=1,
            alpha=235,
            falloff=2.0,
            hardness=0.88,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("cy", "ry", -0.25),
            rx=_expr("highlight_rx"),
            ry=_expr("highlight_ry"),
            tone_idx=2,
            alpha=highlight_alpha,
            falloff=1.8,
            hardness=0.75,
        ),
    )
    _stamp_tone_specs(
        canvas,
        palette,
        specs,
        ShapeContext(
            {
                "cx": cx,
                "cy": cy,
                "rx": rx,
                "ry": ry,
                "highlight_rx": highlight_rx,
                "highlight_ry": highlight_ry,
            }
        ),
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
    hi_alpha = 175 if facing == Facing.NORTH else 190
    hi_rx = params.belly_rx * (0.4 if facing == Facing.NORTH else 0.5)
    hi_ry = params.belly_ry * (0.3 if facing == Facing.NORTH else 0.4)

    specs: tuple[ToneStampSpec, ...] = (
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("belly_cy", None, offset=0.3),
            rx=_expr("belly_rx", None, offset=0.3),
            ry=_expr("belly_ry", None, offset=0.3),
            tone_idx=0,
            alpha=240,
            falloff=2.0,
            hardness=0.85,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("belly_cy"),
            rx=_expr("belly_rx"),
            ry=_expr("belly_ry"),
            tone_idx=1,
            alpha=235,
            falloff=2.0,
            hardness=0.85,
        ),
        EllipseStampSpec(
            cx=_expr("cx", "belly_rx", -0.1),
            cy=_expr("belly_cy", "belly_ry", -0.1),
            rx=_expr("hi_rx"),
            ry=_expr("hi_ry"),
            tone_idx=2,
            alpha=hi_alpha,
            falloff=1.6,
            hardness=0.7,
        ),
    )
    _stamp_tone_specs(
        canvas,
        cloth_pal,
        specs,
        ShapeContext(
            {
                "cx": cx,
                "belly_cy": belly_cy,
                "belly_rx": params.belly_rx,
                "belly_ry": params.belly_ry,
                "hi_rx": hi_rx,
                "hi_ry": hi_ry,
            }
        ),
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
    shoulder_half_w = params.shoulder_width
    torso_ry = params.torso_ry
    hi_alpha = 185 if facing == Facing.NORTH else 200
    specs: tuple[ToneStampSpec, ...] = (
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy", "torso_ry", -0.2),
            rx=_expr("shoulder_half_w"),
            ry=_expr("torso_ry", None, offset=0.0, factor=0.8),
            tone_idx=0,
            alpha=240,
            falloff=2.2,
            hardness=0.9,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy", "torso_ry", -0.3),
            rx=_expr("shoulder_half_w", None, offset=-0.3),
            ry=_expr("torso_ry", None, factor=0.7),
            tone_idx=1,
            alpha=235,
            falloff=2.2,
            hardness=0.9,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy", "torso_ry", 0.3),
            rx=_expr("shoulder_half_w", None, offset=-1.5),
            ry=_expr("torso_ry", None, factor=0.6),
            tone_idx=1,
            alpha=230,
            falloff=2.0,
            hardness=0.85,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy", "torso_ry", -0.35),
            rx=_expr("shoulder_half_w", None, factor=0.45),
            ry=_expr("torso_ry", None, factor=0.4),
            tone_idx=2,
            alpha=hi_alpha,
            falloff=1.6,
            hardness=0.72,
        ),
    )
    _stamp_tone_specs(
        canvas,
        cloth_pal,
        specs,
        ShapeContext(
            {
                "cx": cx,
                "torso_cy": torso_cy,
                "torso_ry": torso_ry,
                "shoulder_half_w": shoulder_half_w,
            }
        ),
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
    if facing in {Facing.EAST, Facing.WEST}:
        # Side profile should be narrower than front/back plate width.
        rx = max(1.8, params.torso_rx * 0.78)
    else:
        rx = max(params.torso_rx + 0.4, params.shoulder_width * 0.9)
    ry = max(1.4, params.torso_ry - 0.2)

    specs: list[ToneStampSpec] = [
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy", None, offset=0.3),
            rx=_expr("rx", None, offset=0.3),
            ry=_expr("ry", None, offset=0.3),
            tone_idx=0,
            alpha=245,
            falloff=2.2,
            hardness=0.92,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy"),
            rx=_expr("rx"),
            ry=_expr("ry"),
            tone_idx=1,
            alpha=240,
            falloff=2.2,
            hardness=0.92,
        ),
    ]

    if facing == Facing.NORTH:
        specs.append(
            EllipseStampSpec(
                cx=_expr("cx"),
                cy=_expr("torso_cy", "ry", -0.25),
                rx=_expr("rx", None, factor=0.4),
                ry=_expr("ry", None, factor=0.3),
                tone_idx=2,
                alpha=190,
                falloff=1.5,
                hardness=0.7,
            )
        )
    elif facing == Facing.SOUTH:
        specs.append(
            EllipseStampSpec(
                cx=_expr("cx", "rx", -0.15),
                cy=_expr("torso_cy", "ry", -0.3),
                rx=_expr("rx", None, factor=0.4),
                ry=_expr("ry", None, factor=0.35),
                tone_idx=2,
                alpha=215,
                falloff=1.5,
                hardness=0.7,
            )
        )
    else:
        # Side-view plate highlight and shoulder sleeve.
        specs.extend(
            (
                EllipseStampSpec(
                    cx=_expr("cx", "rx", -0.08),
                    cy=_expr("torso_cy", "ry", -0.26),
                    rx=_expr("rx", None, factor=0.36),
                    ry=_expr("ry", None, factor=0.34),
                    tone_idx=2,
                    alpha=208,
                    falloff=1.45,
                    hardness=0.68,
                ),
                CircleStampSpec(
                    cx=_expr("cx", "rx", -1.0, offset=-0.25),
                    cy=_expr("torso_cy", "ry", -0.42),
                    radius=_expr(None, None, offset=1.15),
                    tone_idx=1,
                    alpha=226,
                    falloff=1.8,
                    hardness=0.82,
                ),
                EllipseStampSpec(
                    cx=_expr("cx", "rx", -0.72),
                    cy=_expr("torso_cy", "ry", 0.06),
                    rx=_expr(None, None, offset=0.6),
                    ry=_expr(None, None, offset=0.82),
                    tone_idx=0,
                    alpha=220,
                    falloff=1.4,
                    hardness=0.72,
                ),
                EllipseStampSpec(
                    cx=_expr("cx", "rx", -0.62),
                    cy=_expr("torso_cy", "ry", 0.34),
                    rx=_expr(None, None, offset=0.52),
                    ry=_expr(None, None, offset=0.3),
                    tone_idx=2,
                    alpha=195,
                    falloff=1.15,
                    hardness=0.62,
                ),
                CircleStampSpec(
                    cx=_expr("cx", None, offset=0.22),
                    cy=_expr("torso_cy", "ry", -0.38),
                    radius=_expr(None, None, offset=0.65),
                    tone_idx=1,
                    alpha=205,
                    falloff=1.5,
                    hardness=0.72,
                ),
            )
        )

    if facing in {Facing.NORTH, Facing.SOUTH}:
        specs.extend(
            (
                CircleStampSpec(
                    cx=_expr("cx", "rx", -1.0, offset=-0.5),
                    cy=_expr("torso_cy", "ry", -0.5),
                    radius=_expr(None, None, offset=1.5),
                    tone_idx=1,
                    alpha=230,
                    falloff=2.0,
                    hardness=0.88,
                ),
                CircleStampSpec(
                    cx=_expr("cx", "rx", 1.0, offset=0.5),
                    cy=_expr("torso_cy", "ry", -0.5),
                    radius=_expr(None, None, offset=1.5),
                    tone_idx=1,
                    alpha=230,
                    falloff=2.0,
                    hardness=0.88,
                ),
            )
        )

    _stamp_tone_specs(
        canvas,
        cloth_pal,
        tuple(specs),
        ShapeContext({"cx": cx, "torso_cy": torso_cy, "rx": rx, "ry": ry}),
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
    ry = params.torso_ry + 2.0

    hi_alpha = 185 if facing == Facing.NORTH else 210
    hi_rx = params.torso_rx * (0.5 if facing == Facing.NORTH else 0.7)
    hi_ry = ry * (0.3 if facing == Facing.NORTH else 0.4)
    hi_cy_factor = -0.1 if facing == Facing.NORTH else -0.15

    specs: tuple[ToneStampSpec, ...] = (
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy", None, offset=1.0),
            rx=_expr("torso_rx", None, offset=0.8),
            ry=_expr("ry", None, offset=0.3),
            tone_idx=0,
            alpha=240,
            falloff=2.0,
            hardness=0.85,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy", None, offset=0.8),
            rx=_expr("torso_rx", None, offset=0.5),
            ry=_expr("ry"),
            tone_idx=1,
            alpha=235,
            falloff=2.0,
            hardness=0.85,
        ),
        EllipseStampSpec(
            cx=_expr("cx"),
            cy=_expr("torso_cy", "ry", hi_cy_factor),
            rx=_expr("hi_rx"),
            ry=_expr("hi_ry"),
            tone_idx=2,
            alpha=hi_alpha,
            falloff=1.6,
            hardness=0.7,
        ),
    )
    _stamp_tone_specs(
        canvas,
        cloth_pal,
        specs,
        ShapeContext(
            {
                "cx": cx,
                "torso_cy": torso_cy,
                "torso_rx": params.torso_rx,
                "ry": ry,
                "hi_rx": hi_rx,
                "hi_ry": hi_ry,
            }
        ),
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


@dataclass(frozen=True)
class CharacterDrawContext:
    """Shared geometry and appearance values for layer rendering."""

    canvas: np.ndarray
    appearance: CharacterAppearance
    pose: Pose
    params: BodyParams
    facing: Facing
    cx: float
    hr: float
    head_cy: float
    torso_cy: float
    belly_cy: float
    torso_bottom: float
    leg_bottom: int
    leg_top: int
    is_thin_legs: bool


CharacterLayerFn = Callable[[CharacterDrawContext], None]
CharacterLayerStep = tuple[str, CharacterLayerFn]


def _build_character_draw_context(
    canvas: np.ndarray,
    appearance: CharacterAppearance,
    pose: Pose,
) -> CharacterDrawContext:
    """Resolve geometric anchors once for the layer pipeline."""
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

    return CharacterDrawContext(
        canvas=canvas,
        appearance=appearance,
        pose=pose,
        params=params,
        facing=facing,
        cx=cx,
        hr=hr,
        head_cy=head_cy,
        torso_cy=torso_cy,
        belly_cy=belly_cy,
        torso_bottom=torso_bottom,
        leg_bottom=leg_bottom,
        leg_top=leg_top,
        is_thin_legs=is_thin_legs,
    )


def _layer_legs(context: CharacterDrawContext) -> None:
    """Draw leg masses and toe/readability cues."""
    canvas = context.canvas
    appearance = context.appearance
    pose = context.pose
    params = context.params
    facing = context.facing
    cx = context.cx
    leg_bottom = context.leg_bottom
    leg_top = context.leg_top
    p_shadow, p_mid, _p_hi = appearance.pants_pal

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

        if facing in {Facing.NORTH, Facing.SOUTH} and context.is_thin_legs:
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


def _layer_belly(context: CharacterDrawContext) -> None:
    """Draw belly/body mass that sits behind torso clothing."""
    if context.params.belly_ry > 0 and not context.appearance.covers_legs:
        _draw_belly_mass(
            context.canvas,
            context.cx,
            context.belly_cy,
            context.params,
            context.appearance.cloth_pal,
            context.facing,
        )


def _layer_torso(context: CharacterDrawContext) -> None:
    """Draw torso clothing and robe/boot coverage."""
    canvas = context.canvas
    appearance = context.appearance
    params = context.params
    facing = context.facing
    cx = context.cx
    torso_cy = context.torso_cy
    torso_bottom = context.torso_bottom
    p_shadow, _p_mid, _p_hi = appearance.pants_pal

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


@dataclass(frozen=True)
class ArmLayerState:
    """Shared arm geometry/colors reused by back/front arm layers."""

    shoulder_y: float
    arm_end_y: float
    arm_alpha: int
    hand_drop: float
    hand_alpha: int
    skin_mid: colors.Color
    arm_shadow_rgb: colors.Color
    arm_mid_rgb: colors.Color
    is_broad: bool
    is_belly: bool
    is_thin: bool
    is_child: bool


def _build_arm_layer_state(context: CharacterDrawContext) -> ArmLayerState:
    """Resolve arm shading and anchor values for arm layer passes."""
    appearance = context.appearance
    params = context.params
    torso_cy = context.torso_cy
    belly_cy = context.belly_cy

    s_shadow, s_mid, _s_hi = appearance.skin_pal
    c_shadow, c_mid, _c_hi = appearance.cloth_pal
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
            (s_shadow[0] + s_mid[0]) // 2,
            (s_shadow[1] + s_mid[1]) // 2,
            (s_shadow[2] + s_mid[2]) // 2,
        )
        arm_mid_rgb = s_mid
    else:
        arm_shadow_rgb = c_shadow
        arm_mid_rgb = c_mid

    return ArmLayerState(
        shoulder_y=shoulder_y,
        arm_end_y=arm_end_y,
        arm_alpha=arm_alpha,
        hand_drop=hand_drop,
        hand_alpha=hand_alpha,
        skin_mid=s_mid,
        arm_shadow_rgb=arm_shadow_rgb,
        arm_mid_rgb=arm_mid_rgb,
        is_broad=is_broad,
        is_belly=is_belly,
        is_thin=is_thin,
        is_child=is_child,
    )


def _draw_frontback_arm(
    context: CharacterDrawContext,
    arm_state: ArmLayerState,
    side: int,
    arm_dy: float,
) -> None:
    """Draw one arm for NORTH/SOUTH facings, preserving prior per-side order."""
    appearance = context.appearance
    params = context.params
    arm_swing_scale = 0.5 if appearance.clothing_fn is _draw_armor_torso else 1.0
    shoulder_base = params.shoulder_width if arm_state.is_broad else params.torso_rx
    if arm_state.is_broad:
        shoulder_anchor = (
            params.torso_rx * 0.82 + (params.shoulder_width - params.torso_rx) * 0.12
        )
        hand_inset = 0.24
    elif arm_state.is_belly:
        shoulder_anchor = shoulder_base * 0.85
        hand_inset = 0.2
    elif arm_state.is_child:
        shoulder_anchor = shoulder_base * 0.8
        hand_inset = 0.12
    elif arm_state.is_thin:
        shoulder_anchor = shoulder_base * 0.82
        hand_inset = 0.14
    else:
        shoulder_anchor = shoulder_base * 0.84
        hand_inset = 0.18

    arm_x = context.cx + side * shoulder_anchor
    hand_x = arm_x - side * hand_inset
    hand_y = arm_state.arm_end_y + arm_dy * arm_swing_scale

    stamp_fuzzy_circle(
        context.canvas,
        arm_x,
        arm_state.shoulder_y,
        max(0.45, params.arm_thickness * 0.38),
        (*arm_state.arm_shadow_rgb, 175),
        1.45,
        0.72,
    )

    # Two-pass arm stroke gives a readable limb edge without looking detached.
    draw_thick_line(
        context.canvas,
        arm_x + side * 0.08,
        arm_state.shoulder_y + 0.05,
        hand_x + side * 0.08,
        hand_y + 0.05,
        (*arm_state.arm_shadow_rgb, max(165, arm_state.arm_alpha - 30)),
        params.arm_thickness,
    )
    draw_thick_line(
        context.canvas,
        arm_x,
        arm_state.shoulder_y,
        hand_x,
        hand_y,
        (*arm_state.arm_mid_rgb, arm_state.arm_alpha),
        params.arm_thickness,
    )
    if params.hand_radius > 0:
        stamp_fuzzy_circle(
            context.canvas,
            hand_x,
            hand_y + arm_state.hand_drop,
            params.hand_radius,
            (*arm_state.skin_mid, arm_state.hand_alpha),
            2.0,
            0.85,
        )


def _layer_back_arm(context: CharacterDrawContext) -> None:
    """Draw the farther arm layer for the current facing."""
    arm_state = _build_arm_layer_state(context)
    if context.facing in {Facing.EAST, Facing.WEST}:
        # Side profile uses a clear foreground arm plus a faint background arm.
        back_sign = 1.0
        far_arm_x = context.cx + back_sign * context.params.torso_rx * 0.34
        draw_thick_line(
            context.canvas,
            far_arm_x,
            arm_state.shoulder_y + 0.05,
            far_arm_x + back_sign * 0.1,
            arm_state.arm_end_y + context.pose.right_arm_dy + 0.05,
            (*arm_state.arm_shadow_rgb, max(145, arm_state.arm_alpha - 55)),
            max(1, context.params.arm_thickness - 1),
        )
        return
    _draw_frontback_arm(context, arm_state, side=-1, arm_dy=context.pose.left_arm_dy)


def _layer_front_arm(context: CharacterDrawContext) -> None:
    """Draw the nearer arm layer for the current facing."""
    arm_state = _build_arm_layer_state(context)
    if context.facing in {Facing.EAST, Facing.WEST}:
        front_sign = -1.0
        front_arm_x = context.cx + front_sign * context.params.torso_rx * 0.62
        front_hand_x = front_arm_x + front_sign * 0.18
        hand_radius = (
            min(0.9, context.params.hand_radius)
            if context.params.hand_radius > 0
            else 0.0
        )

        stamp_fuzzy_circle(
            context.canvas,
            front_arm_x,
            arm_state.shoulder_y,
            max(0.45, context.params.arm_thickness * 0.38),
            (*arm_state.arm_shadow_rgb, 185),
            1.45,
            0.72,
        )
        draw_thick_line(
            context.canvas,
            front_arm_x + front_sign * 0.08,
            arm_state.shoulder_y,
            front_hand_x + front_sign * 0.08,
            arm_state.arm_end_y + context.pose.left_arm_dy,
            (*arm_state.arm_shadow_rgb, max(165, arm_state.arm_alpha - 30)),
            context.params.arm_thickness,
        )
        draw_thick_line(
            context.canvas,
            front_arm_x,
            arm_state.shoulder_y,
            front_hand_x,
            arm_state.arm_end_y + context.pose.left_arm_dy,
            (*arm_state.arm_mid_rgb, arm_state.arm_alpha),
            context.params.arm_thickness,
        )
        if hand_radius > 0:
            stamp_fuzzy_circle(
                context.canvas,
                front_hand_x,
                arm_state.arm_end_y + context.pose.left_arm_dy + arm_state.hand_drop,
                hand_radius,
                (*arm_state.skin_mid, arm_state.hand_alpha),
                2.0,
                0.85,
            )
        return
    _draw_frontback_arm(context, arm_state, side=1, arm_dy=context.pose.right_arm_dy)


def _layer_neck(context: CharacterDrawContext) -> None:
    """Draw neck for visible facings."""
    params = context.params
    appearance = context.appearance
    if context.facing != Facing.NORTH and params.neck_rx > 0 and params.neck_ry > 0:
        is_broad = params.shoulder_width > params.torso_rx
        is_belly = params.belly_ry > 0
        is_thin = (
            params.arm_thickness == 1 and params.hand_radius > 0 and params.neck_rx > 0
        )
        if is_belly or is_broad:
            neck_y = context.head_cy + context.hr + 0.1
        elif is_thin:
            neck_y = context.head_cy + context.hr + 0.15
        else:
            neck_y = context.head_cy + context.hr + params.neck_gap * 0.5
        stamp_ellipse(
            context.canvas,
            context.cx,
            neck_y,
            params.neck_rx,
            params.neck_ry,
            (*appearance.skin_pal[1], 225),
            2.0,
            0.85,
        )


def _apply_tall_side_hair_mask(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    h_shadow: colors.Color,
    h_mid: colors.Color,
) -> None:
    """Hard-mask side tall-hair back-head region to prevent bald-spot leaks."""
    back_sign = 1.0
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


def _layer_head(context: CharacterDrawContext) -> None:
    """Draw base head mass before hair."""
    appearance = context.appearance
    facing = context.facing
    hx = context.cx
    hy = context.head_cy
    hr = context.hr
    values = {"hx": hx, "hy": hy, "hr": hr}

    if facing == Facing.NORTH:
        specs: tuple[ToneStampSpec, ...] = (
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy", None, offset=0.2),
                radius=_expr("hr", None, offset=0.2),
                tone_idx=0,
                alpha=240,
                falloff=2.0,
                hardness=0.88,
            ),
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy"),
                radius=_expr("hr"),
                tone_idx=1,
                alpha=235,
                falloff=2.0,
                hardness=0.88,
            ),
        )
    elif facing in {Facing.EAST, Facing.WEST}:
        specs_list: list[ToneStampSpec] = [
            EllipseStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy", None, offset=0.2),
                rx=_expr("hr", None, factor=0.85),
                ry=_expr("hr", None, offset=0.2),
                tone_idx=0,
                alpha=240,
                falloff=2.0,
                hardness=0.88,
            ),
            EllipseStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy"),
                rx=_expr("hr", None, factor=0.8),
                ry=_expr("hr"),
                tone_idx=1,
                alpha=235,
                falloff=2.0,
                hardness=0.88,
            ),
            EllipseStampSpec(
                cx=_expr("hx", "hr", -0.14),
                cy=_expr("hy", "hr", -0.2),
                rx=_expr("hr", None, factor=0.45),
                ry=_expr("hr", None, factor=0.55),
                tone_idx=2,
                alpha=210,
                falloff=1.6,
                hardness=0.75,
            ),
            EllipseStampSpec(
                cx=_expr("hx", "hr", -0.26),
                cy=_expr("hy", "hr", -0.12),
                rx=_expr("hr", None, factor=0.34),
                ry=_expr("hr", None, factor=0.48),
                tone_idx=2,
                alpha=205,
                falloff=1.5,
                hardness=0.72,
            ),
        ]
        if appearance.hair_style_idx != 4:
            specs_list.append(
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.28),
                    cy=_expr("hy", "hr", 0.03),
                    rx=_expr("hr", None, factor=0.3),
                    ry=_expr("hr", None, factor=0.46),
                    tone_idx=0,
                    alpha=195,
                    falloff=1.5,
                    hardness=0.72,
                )
            )
        specs_list.append(
            CircleStampSpec(
                cx=_expr("hx", "hr", -0.63),
                cy=_expr("hy", "hr", -0.02),
                radius=_expr("hr", None, factor=0.12),
                tone_idx=2,
                alpha=190,
                falloff=1.4,
                hardness=0.7,
            )
        )
        specs = tuple(specs_list)
    else:
        specs = (
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy", None, offset=0.2),
                radius=_expr("hr", None, offset=0.2),
                tone_idx=0,
                alpha=240,
                falloff=2.0,
                hardness=0.88,
            ),
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy"),
                radius=_expr("hr"),
                tone_idx=1,
                alpha=235,
                falloff=2.0,
                hardness=0.88,
            ),
            CircleStampSpec(
                cx=_expr("hx"),
                cy=_expr("hy", "hr", -0.22),
                radius=_expr("hr", None, factor=0.6),
                tone_idx=2,
                alpha=210,
                falloff=1.6,
                hardness=0.75,
            ),
        )

    _stamp_tone_specs(context.canvas, appearance.skin_pal, specs, ShapeContext(values))


def _layer_hair(context: CharacterDrawContext) -> None:
    """Draw hairstyle and post-hair readability overlays."""
    canvas = context.canvas
    appearance = context.appearance
    facing = context.facing
    hx = context.cx
    hy = context.head_cy
    hr = context.hr
    h_shadow, h_mid, _h_highlight = appearance.hair_pal

    hair_fn = HAIR_STYLES[appearance.hair_style_idx]
    hair_rng = np.random.default_rng(appearance.hair_seed)
    hair_fn(canvas, hx, hy, hr, appearance.hair_pal, hair_rng, facing)

    if facing == Facing.SOUTH:
        is_long_front_hair = appearance.hair_style_idx == 3
        is_medium_front_hair = appearance.hair_style_idx == 2
        values = {"hx": hx, "hy": hy, "hr": hr}
        if is_long_front_hair or is_medium_front_hair:
            specs_list: list[ToneStampSpec] = [
                EllipseStampSpec(
                    cx=_expr("hx"),
                    cy=_expr("hy", "hr", -(0.26 if is_long_front_hair else 0.22)),
                    rx=_expr("hr", None, factor=(0.24 if is_long_front_hair else 0.2)),
                    ry=_expr("hr", None, factor=(0.13 if is_long_front_hair else 0.11)),
                    tone_idx=1,
                    alpha=210,
                    falloff=1.35,
                    hardness=0.7,
                )
            ]
            if is_long_front_hair:
                specs_list.append(
                    EllipseStampSpec(
                        cx=_expr("hx"),
                        cy=_expr("hy", "hr", -0.36),
                        rx=_expr("hr", None, factor=0.23),
                        ry=_expr("hr", None, factor=0.12),
                        tone_idx=1,
                        alpha=205,
                        falloff=1.25,
                        hardness=0.68,
                    )
                )
            _stamp_tone_specs(
                canvas,
                appearance.hair_pal,
                tuple(specs_list),
                ShapeContext(values),
            )
        else:
            face_specs: tuple[ToneStampSpec, ...] = (
                EllipseStampSpec(
                    cx=_expr("hx"),
                    cy=_expr("hy", "hr", -0.05),
                    rx=_expr("hr", None, factor=0.44),
                    ry=_expr("hr", None, factor=0.36),
                    tone_idx=1,
                    alpha=205,
                    falloff=1.35,
                    hardness=0.72,
                ),
                EllipseStampSpec(
                    cx=_expr("hx"),
                    cy=_expr("hy", "hr", -0.22),
                    rx=_expr("hr", None, factor=0.28),
                    ry=_expr("hr", None, factor=0.2),
                    tone_idx=2,
                    alpha=190,
                    falloff=1.2,
                    hardness=0.65,
                ),
            )
            _stamp_tone_specs(
                canvas, appearance.skin_pal, face_specs, ShapeContext(values)
            )
    elif facing in {Facing.EAST, Facing.WEST}:
        values = {"hx": hx, "hy": hy, "hr": hr}
        if appearance.hair_style_idx == 4:
            tall_specs: tuple[ToneStampSpec, ...] = (
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.42),
                    cy=_expr("hy", "hr", -0.08),
                    rx=_expr("hr", None, factor=0.32),
                    ry=_expr("hr", None, factor=0.46),
                    tone_idx=0,
                    alpha=224,
                    falloff=1.4,
                    hardness=0.76,
                ),
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.36),
                    cy=_expr("hy", "hr", -0.12),
                    rx=_expr("hr", None, factor=0.24),
                    ry=_expr("hr", None, factor=0.34),
                    tone_idx=1,
                    alpha=214,
                    falloff=1.3,
                    hardness=0.72,
                ),
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.52),
                    cy=_expr("hy", "hr", 0.02),
                    rx=_expr("hr", None, factor=0.34),
                    ry=_expr("hr", None, factor=0.5),
                    tone_idx=0,
                    alpha=232,
                    falloff=1.5,
                    hardness=0.82,
                ),
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.44),
                    cy=_expr("hy", "hr", -0.02),
                    rx=_expr("hr", None, factor=0.24),
                    ry=_expr("hr", None, factor=0.36),
                    tone_idx=1,
                    alpha=220,
                    falloff=1.35,
                    hardness=0.76,
                ),
            )
            _stamp_tone_specs(
                canvas, appearance.hair_pal, tall_specs, ShapeContext(values)
            )
            _apply_tall_side_hair_mask(canvas, hx, hy, hr, h_shadow, h_mid)

        face_specs_list: list[ToneStampSpec] = [
            EllipseStampSpec(
                cx=_expr("hx", "hr", -0.56),
                cy=_expr("hy", "hr", -0.06),
                rx=_expr("hr", None, factor=0.2),
                ry=_expr("hr", None, factor=0.3),
                tone_idx=1,
                alpha=210,
                falloff=1.3,
                hardness=0.68,
            ),
            CircleStampSpec(
                cx=_expr("hx", "hr", -0.72),
                cy=_expr("hy", "hr", -0.03),
                radius=_expr("hr", None, factor=0.12),
                tone_idx=2,
                alpha=195,
                falloff=1.25,
                hardness=0.66,
            ),
            EllipseStampSpec(
                cx=_expr("hx", "hr", -0.9),
                cy=_expr("hy", "hr", 0.01),
                rx=_expr("hr", None, factor=0.08),
                ry=_expr("hr", None, factor=0.06),
                tone_idx=0,
                alpha=200,
                falloff=1.0,
                hardness=0.62,
            ),
            CircleStampSpec(
                cx=_expr("hx", "hr", -0.9, offset=-0.18),
                cy=_expr("hy", "hr", 0.01, offset=0.02),
                radius=_expr("hr", None, factor=0.06),
                tone_idx=0,
                alpha=192,
                falloff=1.0,
                hardness=0.6,
            ),
        ]
        if appearance.hair_style_idx != 4:
            face_specs_list.append(
                CircleStampSpec(
                    cx=_expr("hx", "hr", 0.42),
                    cy=_expr("hy", "hr", 0.02),
                    radius=_expr("hr", None, factor=0.1),
                    tone_idx=0,
                    alpha=175,
                    falloff=1.25,
                    hardness=0.66,
                )
            )
        _stamp_tone_specs(
            canvas,
            appearance.skin_pal,
            tuple(face_specs_list),
            ShapeContext(values),
        )


def _layer_face_final(context: CharacterDrawContext) -> None:
    """Draw final side-profile nose overlay."""
    if context.facing in {Facing.EAST, Facing.WEST}:
        front_sign = -1.0
        nose_x = float(round(context.cx + front_sign * context.hr * 0.9))
        nose_y = float(round(context.head_cy - context.hr * 0.02))
        s_shadow = context.appearance.skin_pal[0]
        stamp_fuzzy_circle(
            context.canvas,
            nose_x,
            nose_y,
            max(0.24, context.hr * 0.08),
            (*s_shadow, 230),
            1.0,
            0.9,
        )
        stamp_fuzzy_circle(
            context.canvas,
            nose_x + front_sign * 0.35,
            nose_y + 0.02,
            max(0.2, context.hr * 0.06),
            (*s_shadow, 220),
            1.0,
            0.85,
        )


_CHARACTER_LAYER_PIPELINE_SOUTH: tuple[CharacterLayerStep, ...] = (
    # SOUTH-facing keeps torso mass below both arm passes.
    ("legs", _layer_legs),
    ("belly", _layer_belly),
    ("torso", _layer_torso),
    ("back_arm", _layer_back_arm),
    ("front_arm", _layer_front_arm),
    ("neck", _layer_neck),
    ("head", _layer_head),
    ("hair", _layer_hair),
    ("face", _layer_face_final),
)

_CHARACTER_LAYER_PIPELINE_NORTH: tuple[CharacterLayerStep, ...] = (
    # NORTH-facing shares the same z-stack but has different per-layer internals.
    ("legs", _layer_legs),
    ("belly", _layer_belly),
    ("torso", _layer_torso),
    ("back_arm", _layer_back_arm),
    ("front_arm", _layer_front_arm),
    ("neck", _layer_neck),
    ("head", _layer_head),
    ("hair", _layer_hair),
    ("face", _layer_face_final),
)

_CHARACTER_LAYER_PIPELINE_SIDE: tuple[CharacterLayerStep, ...] = (
    # Side profile has explicit arm depth: back arm then front arm.
    ("legs", _layer_legs),
    ("belly", _layer_belly),
    ("torso", _layer_torso),
    ("back_arm", _layer_back_arm),
    ("front_arm", _layer_front_arm),
    ("neck", _layer_neck),
    ("head", _layer_head),
    ("hair", _layer_hair),
    ("face", _layer_face_final),
)

_CHARACTER_LAYER_PIPELINES_BY_FACING: dict[Facing, tuple[CharacterLayerStep, ...]] = {
    Facing.SOUTH: _CHARACTER_LAYER_PIPELINE_SOUTH,
    Facing.NORTH: _CHARACTER_LAYER_PIPELINE_NORTH,
    Facing.WEST: _CHARACTER_LAYER_PIPELINE_SIDE,
    Facing.EAST: _CHARACTER_LAYER_PIPELINE_SIDE,
}


def _draw_character(
    canvas: np.ndarray,
    appearance: CharacterAppearance,
    pose: Pose,
) -> None:
    """Draw one character via an explicit ordered layer pipeline."""
    context = _build_character_draw_context(canvas, appearance, pose)
    layer_pipeline = _CHARACTER_LAYER_PIPELINES_BY_FACING[context.facing]
    for _layer_name, layer_fn in layer_pipeline:
        layer_fn(context)


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
