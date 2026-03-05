"""Clothing and torso layer rendering."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from brileta.sprites.primitives import PaletteBrush, stamp_ellipse
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

    b = PaletteBrush(canvas, palette)
    b.ellipse(cx, cy + 0.3, rx + 0.3, ry + 0.3, tone=0, alpha=240)
    b.ellipse(cx, cy, rx, ry)
    b.ellipse(
        cx,
        cy + ry * -0.25,
        highlight_rx,
        highlight_ry,
        tone=2,
        alpha=highlight_alpha,
        falloff=1.8,
        hardness=0.75,
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

    b = PaletteBrush(canvas, cloth_pal, falloff=2.0, hardness=0.85)
    b.ellipse(
        cx,
        belly_cy + 0.3,
        params.belly_rx + 0.3,
        params.belly_ry + 0.3,
        tone=0,
        alpha=240,
    )
    b.ellipse(cx, belly_cy, params.belly_rx, params.belly_ry)
    b.ellipse(
        cx + params.belly_rx * -0.1,
        belly_cy + params.belly_ry * -0.1,
        hi_rx,
        hi_ry,
        tone=2,
        alpha=hi_alpha,
        falloff=1.6,
        hardness=0.7,
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
    sw = params.shoulder_width
    ry = params.torso_ry
    hi_alpha = 185 if facing == Facing.NORTH else 200

    b = PaletteBrush(canvas, cloth_pal, falloff=2.2, hardness=0.9)
    b.ellipse(cx, torso_cy + ry * -0.2, sw, ry * 0.8, tone=0, alpha=240)
    b.ellipse(cx, torso_cy + ry * -0.3, sw - 0.3, ry * 0.7)
    b.ellipse(
        cx,
        torso_cy + ry * 0.3,
        sw - 1.5,
        ry * 0.6,
        alpha=230,
        falloff=2.0,
        hardness=0.85,
    )
    b.ellipse(
        cx,
        torso_cy + ry * -0.35,
        sw * 0.45,
        ry * 0.4,
        tone=2,
        alpha=hi_alpha,
        falloff=1.6,
        hardness=0.72,
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
        canvas, cx, torso_cy, params.torso_rx, params.torso_ry, skin_pal, facing
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
        canvas, cx, torso_cy, params.torso_rx, params.torso_ry, cloth_pal, facing
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
        rx = max(1.8, params.torso_rx * 0.78)
    else:
        rx = max(params.torso_rx + 0.4, params.shoulder_width * 0.9)
    ry = max(1.4, params.torso_ry - 0.2)

    b = PaletteBrush(canvas, cloth_pal, falloff=2.2, hardness=0.92)
    b.ellipse(cx, torso_cy + 0.3, rx + 0.3, ry + 0.3, tone=0, alpha=245)
    b.ellipse(cx, torso_cy, rx, ry, alpha=240)

    if facing == Facing.NORTH:
        b.ellipse(
            cx,
            torso_cy + ry * -0.25,
            rx * 0.4,
            ry * 0.3,
            tone=2,
            alpha=190,
            falloff=1.5,
            hardness=0.7,
        )
    elif facing == Facing.SOUTH:
        b.ellipse(
            cx + rx * -0.15,
            torso_cy + ry * -0.3,
            rx * 0.4,
            ry * 0.35,
            tone=2,
            alpha=215,
            falloff=1.5,
            hardness=0.7,
        )
    else:
        # Side-view plate highlight and shoulder sleeve.
        b.ellipse(
            cx + rx * -0.08,
            torso_cy + ry * -0.26,
            rx * 0.36,
            ry * 0.34,
            tone=2,
            alpha=208,
            falloff=1.45,
            hardness=0.68,
        )
        b.ellipse(
            cx + rx * -0.72,
            torso_cy + ry * 0.06,
            0.6,
            0.82,
            tone=0,
            alpha=220,
            falloff=1.4,
            hardness=0.72,
        )
        b.ellipse(
            cx + rx * -0.62,
            torso_cy + ry * 0.34,
            0.52,
            0.3,
            tone=2,
            alpha=195,
            falloff=1.15,
            hardness=0.62,
        )
        b.circle(
            cx + rx * -1.0 - 0.25,
            torso_cy + ry * -0.42,
            1.15,
            alpha=226,
            falloff=1.8,
            hardness=0.82,
        )
        b.circle(
            cx + 0.22,
            torso_cy + ry * -0.38,
            0.65,
            alpha=205,
            falloff=1.5,
            hardness=0.72,
        )

    if facing in {Facing.NORTH, Facing.SOUTH}:
        b.circle(
            cx + rx * -1.0 - 0.5,
            torso_cy + ry * -0.5,
            1.5,
            alpha=230,
            falloff=2.0,
            hardness=0.88,
        )
        b.circle(
            cx + rx * 1.0 + 0.5,
            torso_cy + ry * -0.5,
            1.5,
            alpha=230,
            falloff=2.0,
            hardness=0.88,
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

    b = PaletteBrush(canvas, cloth_pal, falloff=2.0, hardness=0.85)
    b.ellipse(cx, torso_cy + 1.0, params.torso_rx + 0.8, ry + 0.3, tone=0, alpha=240)
    b.ellipse(cx, torso_cy + 0.8, params.torso_rx + 0.5, ry)
    b.ellipse(
        cx,
        torso_cy + ry * hi_cy_factor,
        hi_rx,
        hi_ry,
        tone=2,
        alpha=hi_alpha,
        falloff=1.6,
        hardness=0.7,
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
        stamp_ellipse(canvas, cx, seam_y, seam_rx, seam_ry, (*c_mid, 218), 1.7, 0.8)

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
