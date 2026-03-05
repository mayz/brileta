"""Clothing specs and torso clothing layer rendering."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from brileta.sprites.primitives import stamp_ellipse
from brileta.types import Facing

from .appearance import BodyParams, Palette3
from .specs import (
    CircleStampSpec,
    EllipseStampSpec,
    ShapeContext,
    ToneStampSpec,
    _expr,
    _stamp_tone_specs,
)

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
