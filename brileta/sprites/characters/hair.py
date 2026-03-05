"""Hair style rendering and hair layer composition."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from brileta.sprites.primitives import HeadBrush, Palette3
from brileta.types import Facing

from .appearance import _luma

if TYPE_CHECKING:
    from .renderer import CharacterDrawContext


HairDrawFn = Callable[
    [np.ndarray, float, float, float, Palette3, np.random.Generator, Facing],
    None,
]

# Named indices into HAIR_STYLES for readability checks elsewhere.
HAIR_IDX_BALD = 0
HAIR_IDX_SHORT = 1
HAIR_IDX_MEDIUM = 2
HAIR_IDX_LONG = 3
HAIR_IDX_TALL = 4


# ---------------------------------------------------------------------------
# Short hair
# ---------------------------------------------------------------------------


def _draw_hair_short_north(b: HeadBrush) -> None:
    b.rel_ellipse(-0.0, -0.15, 1.15, 0.95, alpha=235, falloff=1.8, hardness=0.85)
    b.rel_ellipse(-0.0, -0.3, 0.8, 0.6, tone=2, alpha=200, falloff=1.5, hardness=0.75)


def _draw_hair_short_side(b: HeadBrush) -> None:
    cap_cy = b.hy - b.hr * 0.45
    b.ellipse(
        b.hx + b.hr * 0.04,
        cap_cy + b.hr * -0.12,
        b.hr * 0.9,
        b.hr * 0.62,
        alpha=235,
        falloff=1.9,
        hardness=0.86,
    )
    b.rel_ellipse(0.52, -0.02, 0.25, 0.34, alpha=220, falloff=1.45, hardness=0.76)
    b.ellipse(
        b.hx + b.hr * -0.22,
        cap_cy + b.hr * -0.34,
        b.hr * 0.36,
        b.hr * 0.24,
        tone=2,
        alpha=198,
        falloff=1.35,
        hardness=0.72,
    )


def _draw_hair_short_south(b: HeadBrush) -> None:
    cap_cy = b.hy - b.hr * 0.45
    b.ellipse(
        b.hx,
        cap_cy + b.hr * -0.12,
        b.hr * 1.08,
        b.hr * 0.54,
        alpha=235,
        falloff=1.9,
        hardness=0.86,
    )
    b.ellipse(
        b.hx,
        cap_cy - 0.7,
        b.hr * 0.62,
        b.hr * 0.36,
        tone=2,
        alpha=200,
        falloff=1.5,
        hardness=0.75,
    )


# ---------------------------------------------------------------------------
# Medium hair
# ---------------------------------------------------------------------------


def _draw_hair_medium_north(b: HeadBrush) -> None:
    b.rel_ellipse(-0.0, -0.1, 1.25, 1.1, tone=0, alpha=235, falloff=1.6, hardness=0.82)
    b.rel_ellipse(-0.0, -0.2, 1.15, 1.0, alpha=225, falloff=1.5, hardness=0.8)
    b.rel_ellipse(-0.85, 0.3, 0.45, 0.7, tone=0, alpha=220, falloff=1.5, hardness=0.78)
    b.rel_ellipse(0.85, 0.3, 0.45, 0.7, tone=0, alpha=220, falloff=1.5, hardness=0.78)
    b.rel_ellipse(-0.0, -0.4, 0.7, 0.5, tone=2, alpha=195, falloff=1.3, hardness=0.7)


def _draw_hair_medium_side(b: HeadBrush) -> None:
    b.rel_ellipse(0.06, -0.48, 1.12, 0.7, tone=0, alpha=235, falloff=1.6, hardness=0.82)
    b.rel_ellipse(0.05, -0.56, 1.0, 0.6, alpha=225, falloff=1.5, hardness=0.8)
    b.rel_ellipse(0.78, 0.3, 0.4, 0.76, tone=0, alpha=220, falloff=1.5, hardness=0.78)
    b.rel_ellipse(-0.54, 0.02, 0.12, 0.24, alpha=214, falloff=1.35, hardness=0.74)
    b.rel_ellipse(-0.3, -0.76, 0.48, 0.32, tone=2, alpha=195, falloff=1.3, hardness=0.7)


def _draw_hair_medium_south(
    b: HeadBrush,
    hr_side_scale: float,
    highlight_jitter: float,
) -> None:
    b.rel_ellipse(
        -0.0, -0.46, 1.18, 0.74, tone=0, alpha=235, falloff=1.6, hardness=0.82
    )
    b.rel_ellipse(-0.0, -0.56, 1.06, 0.64, alpha=225, falloff=1.5, hardness=0.8)
    for side in (-1.0, 1.0):
        b.ellipse(
            b.hx + b.hr * 0.85 * side,
            b.hy + b.hr * 0.12,
            hr_side_scale * 0.4,
            hr_side_scale * 0.55,
            tone=0,
            alpha=220,
            falloff=1.5,
            hardness=0.78,
        )
    b.ellipse(
        b.hx + highlight_jitter,
        b.hy + b.hr * -0.75,
        b.hr * 0.58,
        b.hr * 0.38,
        tone=2,
        alpha=195,
        falloff=1.3,
        hardness=0.7,
    )


# ---------------------------------------------------------------------------
# Long hair
# ---------------------------------------------------------------------------


def _draw_hair_long_north(b: HeadBrush, drape_main: float) -> None:
    b.rel_ellipse(-0.0, 0.0, 1.3, 1.1, tone=0, alpha=235, falloff=1.6, hardness=0.82)
    b.rel_ellipse(-0.0, -0.1, 1.2, 1.0, alpha=225, falloff=1.5, hardness=0.8)
    b.ellipse(
        b.hx,
        b.hy + drape_main * 0.3,
        b.hr * 0.9,
        drape_main * 0.45,
        tone=0,
        alpha=220,
        falloff=1.7,
        hardness=0.78,
    )
    b.ellipse(
        b.hx,
        b.hy + drape_main * 0.25,
        b.hr * 0.75,
        drape_main * 0.38,
        alpha=215,
        falloff=1.5,
        hardness=0.75,
    )
    b.rel_ellipse(-0.0, -0.3, 0.7, 0.6, tone=2, alpha=190, falloff=1.3, hardness=0.68)


def _draw_hair_long_side(b: HeadBrush, back_drape: float) -> None:
    b.rel_ellipse(0.1, -0.38, 1.15, 0.8, tone=0, alpha=235, falloff=1.6, hardness=0.82)
    b.rel_ellipse(0.08, -0.46, 1.02, 0.7, alpha=225, falloff=1.5, hardness=0.8)
    b.ellipse(
        b.hx + b.hr * 0.62,
        b.hy + back_drape * 0.2,
        b.hr * 0.32,
        back_drape * 0.3,
        tone=0,
        alpha=215,
        falloff=1.8,
        hardness=0.78,
    )
    b.ellipse(
        b.hx + b.hr * 0.58,
        b.hy + back_drape * 0.15,
        b.hr * 0.24,
        back_drape * 0.24,
        alpha=210,
        falloff=1.6,
        hardness=0.75,
    )
    b.rel_ellipse(-0.52, 0.12, 0.1, 0.28, alpha=208, falloff=1.35, hardness=0.72)
    b.rel_ellipse(-0.26, -0.7, 0.56, 0.4, tone=2, alpha=190, falloff=1.3, hardness=0.68)


def _draw_hair_long_south(
    b: HeadBrush,
    hr_side_scale: float,
    drape_left: float,
    drape_right: float,
    drape_left_scaled: float,
    drape_right_scaled: float,
) -> None:
    b.rel_ellipse(
        -0.0, -0.48, 1.14, 0.72, tone=0, alpha=235, falloff=1.6, hardness=0.82
    )
    b.rel_ellipse(-0.0, -0.58, 1.02, 0.62, alpha=225, falloff=1.5, hardness=0.8)
    for side, drape, drape_scaled in (
        (-1.0, drape_left, drape_left_scaled),
        (1.0, drape_right, drape_right_scaled),
    ):
        # Shadow drape strand.
        b.ellipse(
            b.hx + b.hr * 0.78 * side,
            b.hy + drape * 0.28,
            hr_side_scale * 0.3,
            drape_scaled * 0.22,
            tone=0,
            alpha=215,
            falloff=1.8,
            hardness=0.78,
        )
        # Mid-tone drape strand.
        b.ellipse(
            b.hx + b.hr * 0.78 * side,
            b.hy + drape * 0.24,
            hr_side_scale * 0.21,
            drape_scaled * 0.18,
            alpha=210,
            falloff=1.6,
            hardness=0.75,
        )
    b.rel_ellipse(
        -0.0, -0.74, 0.66, 0.42, tone=2, alpha=190, falloff=1.3, hardness=0.68
    )


# ---------------------------------------------------------------------------
# Tall hair
# ---------------------------------------------------------------------------


def _draw_hair_tall_north(b: HeadBrush) -> None:
    b.rel_ellipse(-0.0, -0.1, 1.1, 1.0, tone=0, alpha=230, falloff=1.7, hardness=0.82)
    b.rel_ellipse(-0.0, -1.0, 0.65, 1.1, alpha=225, falloff=1.6, hardness=0.8)
    b.rel_ellipse(-0.0, -1.2, 0.45, 0.8, tone=2, alpha=200, falloff=1.3, hardness=0.7)


def _draw_hair_tall_side(b: HeadBrush) -> None:
    b.rel_ellipse(
        0.04, -1.12, 0.66, 1.12, tone=0, alpha=230, falloff=1.8, hardness=0.82
    )
    b.rel_ellipse(0.02, -1.24, 0.52, 0.94, alpha=220, falloff=1.5, hardness=0.78)
    b.rel_ellipse(
        0.46, -0.22, 0.52, 0.72, tone=0, alpha=224, falloff=1.5, hardness=0.78
    )
    b.rel_ellipse(0.4, -0.28, 0.38, 0.54, alpha=214, falloff=1.38, hardness=0.74)
    b.rel_ellipse(
        0.22, -0.24, 0.34, 0.48, tone=0, alpha=220, falloff=1.45, hardness=0.74
    )
    b.rel_ellipse(0.18, -0.28, 0.25, 0.36, alpha=210, falloff=1.35, hardness=0.7)
    b.rel_ellipse(
        0.64, -0.04, 0.34, 0.5, tone=0, alpha=216, falloff=1.42, hardness=0.75
    )
    b.rel_ellipse(-0.22, -0.34, 0.2, 0.26, tone=2, alpha=198, falloff=1.3, hardness=0.7)


def _draw_hair_tall_south(b: HeadBrush) -> None:
    b.rel_ellipse(-0.0, -1.2, 0.7, 1.2, tone=0, alpha=230, falloff=1.8, hardness=0.82)
    b.rel_ellipse(-0.0, -1.3, 0.55, 1.0, alpha=220, falloff=1.5, hardness=0.78)
    b.rel_ellipse(-0.0, -1.5, 0.35, 0.7, tone=2, alpha=200, falloff=1.3, hardness=0.7)


# ---------------------------------------------------------------------------
# Top-level hair style dispatch functions
# ---------------------------------------------------------------------------


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
    b = HeadBrush(canvas, pal, hx, hy, hr)
    if facing == Facing.NORTH:
        _draw_hair_short_north(b)
    elif facing in {Facing.EAST, Facing.WEST}:
        _draw_hair_short_side(b)
    else:
        _draw_hair_short_south(b)


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
    b = HeadBrush(canvas, pal, hx, hy, hr)
    if facing == Facing.NORTH:
        _draw_hair_medium_north(b)
    elif facing in {Facing.EAST, Facing.WEST}:
        _draw_hair_medium_side(b)
    else:
        side_scale = 0.72 if _luma(pal[1]) > 155.0 else 1.0
        _draw_hair_medium_south(b, hr * side_scale, float(rng.uniform(-0.4, 0.4)))


def _hair_long(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    rng: np.random.Generator,
    facing: Facing,
) -> None:
    """Long hair with back/side drape."""
    b = HeadBrush(canvas, pal, hx, hy, hr)
    if facing == Facing.NORTH:
        _draw_hair_long_north(b, hr * float(rng.uniform(2.2, 3.0)))
    elif facing in {Facing.EAST, Facing.WEST}:
        _draw_hair_long_side(b, hr * float(rng.uniform(2.1, 3.0)))
    else:
        side_scale = 0.74 if _luma(pal[1]) > 155.0 else 1.0
        drape_left = hr * float(rng.uniform(2.0, 3.0))
        drape_right = hr * float(rng.uniform(2.0, 3.0))
        _draw_hair_long_south(
            b,
            hr * side_scale,
            drape_left,
            drape_right,
            drape_left * side_scale,
            drape_right * side_scale,
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
    """Tall upward hair."""
    b = HeadBrush(canvas, pal, hx, hy, hr)
    if facing == Facing.NORTH:
        _draw_hair_tall_north(b)
    elif facing in {Facing.EAST, Facing.WEST}:
        _draw_hair_tall_side(b)
    else:
        _draw_hair_tall_south(b)


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


def _layer_hair(context: CharacterDrawContext) -> None:
    """Draw hairstyle and post-hair readability overlays."""
    canvas = context.canvas
    appearance = context.appearance
    facing = context.facing
    hx = context.cx
    hy = context.head_cy
    hr = context.hr

    hair_fn = HAIR_STYLES[appearance.hair_style_idx]
    hair_rng = np.random.default_rng(appearance.hair_seed)
    hair_fn(canvas, hx, hy, hr, appearance.hair_pal, hair_rng, facing)

    hb = HeadBrush(canvas, appearance.hair_pal, hx, hy, hr)
    sb = HeadBrush(canvas, appearance.skin_pal, hx, hy, hr)

    if facing == Facing.SOUTH:
        is_long_front_hair = appearance.hair_style_idx == HAIR_IDX_LONG
        is_medium_front_hair = appearance.hair_style_idx == HAIR_IDX_MEDIUM
        if is_long_front_hair or is_medium_front_hair:
            cy_f = -(0.26 if is_long_front_hair else 0.22)
            rx_f = 0.24 if is_long_front_hair else 0.2
            ry_f = 0.13 if is_long_front_hair else 0.11
            hb.rel_ellipse(0.0, cy_f, rx_f, ry_f, alpha=210, falloff=1.35, hardness=0.7)
            if is_long_front_hair:
                hb.rel_ellipse(
                    0.0, -0.36, 0.23, 0.12, alpha=205, falloff=1.25, hardness=0.68
                )
        else:
            sb.rel_ellipse(
                0.0, -0.05, 0.44, 0.36, alpha=205, falloff=1.35, hardness=0.72
            )
            sb.rel_ellipse(
                0.0, -0.22, 0.28, 0.2, tone=2, alpha=190, falloff=1.2, hardness=0.65
            )
    elif facing in {Facing.EAST, Facing.WEST}:
        if appearance.hair_style_idx == HAIR_IDX_TALL:
            hb.rel_ellipse(
                0.5, -0.18, 0.38, 0.54, tone=0, alpha=228, falloff=1.45, hardness=0.76
            )
            hb.rel_ellipse(
                0.44, -0.22, 0.28, 0.4, alpha=216, falloff=1.3, hardness=0.72
            )
            hb.rel_ellipse(
                0.64, -0.02, 0.34, 0.5, tone=0, alpha=230, falloff=1.5, hardness=0.82
            )
            hb.rel_ellipse(
                0.56, -0.08, 0.25, 0.38, alpha=220, falloff=1.35, hardness=0.76
            )

        sb.rel_ellipse(-0.56, -0.06, 0.2, 0.3, alpha=210, falloff=1.3, hardness=0.68)
        sb.rel_ellipse(
            -0.9, 0.01, 0.08, 0.06, tone=0, alpha=200, falloff=1.0, hardness=0.62
        )
        sb.rel_circle(
            -0.72, -0.03, 0.12, tone=2, alpha=195, falloff=1.25, hardness=0.66
        )
        sb.circle(
            hx + hr * -0.9 - 0.18,
            hy + hr * 0.01 + 0.02,
            hr * 0.06,
            tone=0,
            alpha=192,
            falloff=1.0,
            hardness=0.6,
        )
        if appearance.hair_style_idx != HAIR_IDX_TALL:
            sb.rel_circle(
                0.42, 0.02, 0.1, tone=0, alpha=175, falloff=1.25, hardness=0.66
            )
