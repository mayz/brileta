"""Hair style specs and hair layer rendering."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from brileta import colors
from brileta.types import Facing

from .appearance import Palette3
from .specs import (
    CircleStampSpec,
    EllipseStampSpec,
    ShapeContext,
    ToneStampSpec,
    _expr,
    _head_ellipse,
    _stamp_tone_specs,
)

if TYPE_CHECKING:
    from .renderer import CharacterDrawContext


HairDrawFn = Callable[
    [np.ndarray, float, float, float, Palette3, np.random.Generator, Facing],
    None,
]


def _luma(color: colors.Color) -> float:
    """Return perceptual luminance for an RGB color."""
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


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
        0.46, -0.22, 0.52, 0.72, tone_idx=0, alpha=224, falloff=1.5, hardness=0.78
    ),
    _head_ellipse(
        0.4, -0.28, 0.38, 0.54, tone_idx=1, alpha=214, falloff=1.38, hardness=0.74
    ),
    _head_ellipse(
        0.22, -0.24, 0.34, 0.48, tone_idx=0, alpha=220, falloff=1.45, hardness=0.74
    ),
    _head_ellipse(
        0.18, -0.28, 0.25, 0.36, tone_idx=1, alpha=210, falloff=1.35, hardness=0.7
    ),
    _head_ellipse(
        0.64, -0.04, 0.34, 0.5, tone_idx=0, alpha=216, falloff=1.42, hardness=0.75
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
                    cx=_expr("hx", "hr", 0.5),
                    cy=_expr("hy", "hr", -0.18),
                    rx=_expr("hr", None, factor=0.38),
                    ry=_expr("hr", None, factor=0.54),
                    tone_idx=0,
                    alpha=228,
                    falloff=1.45,
                    hardness=0.76,
                ),
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.44),
                    cy=_expr("hy", "hr", -0.22),
                    rx=_expr("hr", None, factor=0.28),
                    ry=_expr("hr", None, factor=0.4),
                    tone_idx=1,
                    alpha=216,
                    falloff=1.3,
                    hardness=0.72,
                ),
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.64),
                    cy=_expr("hy", "hr", -0.02),
                    rx=_expr("hr", None, factor=0.34),
                    ry=_expr("hr", None, factor=0.5),
                    tone_idx=0,
                    alpha=230,
                    falloff=1.5,
                    hardness=0.82,
                ),
                EllipseStampSpec(
                    cx=_expr("hx", "hr", 0.56),
                    cy=_expr("hy", "hr", -0.08),
                    rx=_expr("hr", None, factor=0.25),
                    ry=_expr("hr", None, factor=0.38),
                    tone_idx=1,
                    alpha=220,
                    falloff=1.35,
                    hardness=0.76,
                ),
            )
            _stamp_tone_specs(
                canvas, appearance.hair_pal, tall_specs, ShapeContext(values)
            )

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
