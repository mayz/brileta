"""Clothing and torso layer rendering."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from brileta.sprites.primitives import stamp_ellipse, stamp_fuzzy_circle
from brileta.types import Facing

from .appearance import BodyParams, Palette3

if TYPE_CHECKING:
    from .renderer import CharacterDrawContext


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

ToneEllipseSpec = tuple[float, float, float, float, int, int, float, float]
ToneCircleSpec = tuple[float, float, float, int, int, float, float]


def _stamp_tone_ellipses(
    canvas: np.ndarray, palette: Palette3, specs: tuple[ToneEllipseSpec, ...]
) -> None:
    """Stamp a sequence of palette-indexed ellipse specs in-order."""
    for cx, cy, rx, ry, tone_idx, alpha, falloff, hardness in specs:
        stamp_ellipse(
            canvas,
            cx,
            cy,
            rx,
            ry,
            (*palette[tone_idx], alpha),
            falloff,
            hardness,
        )


def _stamp_tone_circles(
    canvas: np.ndarray, palette: Palette3, specs: tuple[ToneCircleSpec, ...]
) -> None:
    """Stamp a sequence of palette-indexed fuzzy-circle specs in-order."""
    for cx, cy, radius, tone_idx, alpha, falloff, hardness in specs:
        stamp_fuzzy_circle(
            canvas,
            cx,
            cy,
            radius,
            (*palette[tone_idx], alpha),
            falloff,
            hardness,
        )


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
    if facing == Facing.NORTH:
        highlight_alpha = 185
        highlight_rx = rx * 0.55
        highlight_ry = ry * 0.4
    else:
        highlight_alpha = 210
        highlight_rx = rx * 0.7
        highlight_ry = ry * 0.5

    _stamp_tone_ellipses(
        canvas,
        palette,
        (
            (cx, cy + 0.3, rx + 0.3, ry + 0.3, 0, 240, 2.0, 0.88),
            (cx, cy, rx, ry, 1, 235, 2.0, 0.88),
            (
                cx,
                cy + ry * -0.25,
                highlight_rx,
                highlight_ry,
                2,
                highlight_alpha,
                1.8,
                0.75,
            ),
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

    _stamp_tone_ellipses(
        canvas,
        cloth_pal,
        (
            (
                cx,
                belly_cy + 0.3,
                params.belly_rx + 0.3,
                params.belly_ry + 0.3,
                0,
                240,
                2.0,
                0.85,
            ),
            (cx, belly_cy, params.belly_rx, params.belly_ry, 1, 235, 2.0, 0.85),
            (
                cx + params.belly_rx * -0.1,
                belly_cy + params.belly_ry * -0.1,
                hi_rx,
                hi_ry,
                2,
                hi_alpha,
                1.6,
                0.7,
            ),
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
    _stamp_tone_ellipses(
        canvas,
        cloth_pal,
        (
            (
                cx,
                torso_cy + torso_ry * -0.2,
                shoulder_half_w,
                torso_ry * 0.8,
                0,
                240,
                2.2,
                0.9,
            ),
            (
                cx,
                torso_cy + torso_ry * -0.3,
                shoulder_half_w - 0.3,
                torso_ry * 0.7,
                1,
                235,
                2.2,
                0.9,
            ),
            (
                cx,
                torso_cy + torso_ry * 0.3,
                shoulder_half_w - 1.5,
                torso_ry * 0.6,
                1,
                230,
                2.0,
                0.85,
            ),
            (
                cx,
                torso_cy + torso_ry * -0.35,
                shoulder_half_w * 0.45,
                torso_ry * 0.4,
                2,
                hi_alpha,
                1.6,
                0.72,
            ),
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

    _stamp_tone_ellipses(
        canvas,
        cloth_pal,
        (
            (cx, torso_cy + 0.3, rx + 0.3, ry + 0.3, 0, 245, 2.2, 0.92),
            (cx, torso_cy, rx, ry, 1, 240, 2.2, 0.92),
        ),
    )

    if facing == Facing.NORTH:
        _stamp_tone_ellipses(
            canvas,
            cloth_pal,
            ((cx, torso_cy + ry * -0.25, rx * 0.4, ry * 0.3, 2, 190, 1.5, 0.7),),
        )
    elif facing == Facing.SOUTH:
        _stamp_tone_ellipses(
            canvas,
            cloth_pal,
            (
                (
                    cx + rx * -0.15,
                    torso_cy + ry * -0.3,
                    rx * 0.4,
                    ry * 0.35,
                    2,
                    215,
                    1.5,
                    0.7,
                ),
            ),
        )
    else:
        # Side-view plate highlight and shoulder sleeve.
        _stamp_tone_ellipses(
            canvas,
            cloth_pal,
            (
                (
                    cx + rx * -0.08,
                    torso_cy + ry * -0.26,
                    rx * 0.36,
                    ry * 0.34,
                    2,
                    208,
                    1.45,
                    0.68,
                ),
                (cx + rx * -0.72, torso_cy + ry * 0.06, 0.6, 0.82, 0, 220, 1.4, 0.72),
                (
                    cx + rx * -0.62,
                    torso_cy + ry * 0.34,
                    0.52,
                    0.3,
                    2,
                    195,
                    1.15,
                    0.62,
                ),
            ),
        )
        _stamp_tone_circles(
            canvas,
            cloth_pal,
            (
                (cx + rx * -1.0 - 0.25, torso_cy + ry * -0.42, 1.15, 1, 226, 1.8, 0.82),
                (cx + 0.22, torso_cy + ry * -0.38, 0.65, 1, 205, 1.5, 0.72),
            ),
        )

    if facing in {Facing.NORTH, Facing.SOUTH}:
        _stamp_tone_circles(
            canvas,
            cloth_pal,
            (
                (cx + rx * -1.0 - 0.5, torso_cy + ry * -0.5, 1.5, 1, 230, 2.0, 0.88),
                (cx + rx * 1.0 + 0.5, torso_cy + ry * -0.5, 1.5, 1, 230, 2.0, 0.88),
            ),
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

    _stamp_tone_ellipses(
        canvas,
        cloth_pal,
        (
            (cx, torso_cy + 1.0, params.torso_rx + 0.8, ry + 0.3, 0, 240, 2.0, 0.85),
            (cx, torso_cy + 0.8, params.torso_rx + 0.5, ry, 1, 235, 2.0, 0.85),
            (
                cx,
                torso_cy + ry * hi_cy_factor,
                hi_rx,
                hi_ry,
                2,
                hi_alpha,
                1.6,
                0.7,
            ),
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
