"""Hair style rendering and hair layer composition."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from brileta.sprites.primitives import stamp_ellipse, stamp_fuzzy_circle
from brileta.types import Facing

from .appearance import Palette3, _luma

if TYPE_CHECKING:
    from .renderer import CharacterDrawContext


HairDrawFn = Callable[
    [np.ndarray, float, float, float, Palette3, np.random.Generator, Facing],
    None,
]

HeadEllipseSpec = tuple[float, float, float, float, int, int, float, float]
HeadCircleSpec = tuple[float, float, float, int, int, float, float]

# Named indices into HAIR_STYLES for readability checks elsewhere.
HAIR_IDX_BALD = 0
HAIR_IDX_SHORT = 1
HAIR_IDX_MEDIUM = 2
HAIR_IDX_LONG = 3
HAIR_IDX_TALL = 4


def _tone_rgba(
    palette: Palette3, tone_idx: int, alpha: int
) -> tuple[int, int, int, int]:
    """Pack one palette tone with alpha for primitive stamp calls."""
    return (*palette[tone_idx], alpha)


def _stamp_head_ellipse(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    cx_factor: float,
    cy_factor: float,
    rx_factor: float,
    ry_factor: float,
    *,
    tone_idx: int,
    alpha: int,
    falloff: float,
    hardness: float,
    cx_base: float | None = None,
    cy_base: float | None = None,
    cx_scale: float | None = None,
    cy_scale: float | None = None,
    cy_offset: float = 0.0,
) -> None:
    """Stamp one head-relative ellipse from factor-based inputs."""
    resolved_cx_base = hx if cx_base is None else cx_base
    resolved_cy_base = hy if cy_base is None else cy_base
    resolved_cx_scale = hr if cx_scale is None else cx_scale
    resolved_cy_scale = hr if cy_scale is None else cy_scale
    stamp_ellipse(
        canvas,
        resolved_cx_base + resolved_cx_scale * cx_factor,
        resolved_cy_base + resolved_cy_scale * cy_factor + cy_offset,
        hr * rx_factor,
        hr * ry_factor,
        _tone_rgba(pal, tone_idx, alpha),
        falloff,
        hardness,
    )


def _stamp_head_ellipse_specs(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    specs: tuple[HeadEllipseSpec, ...],
    *,
    cx_base: float | None = None,
    cy_base: float | None = None,
    cx_scale: float | None = None,
    cy_scale: float | None = None,
    cy_offset: float = 0.0,
) -> None:
    """Stamp a sequence of head-relative ellipse specs in-order."""
    for (
        cx_factor,
        cy_factor,
        rx_factor,
        ry_factor,
        tone_idx,
        alpha,
        falloff,
        hardness,
    ) in specs:
        _stamp_head_ellipse(
            canvas,
            hx,
            hy,
            hr,
            pal,
            cx_factor,
            cy_factor,
            rx_factor,
            ry_factor,
            tone_idx=tone_idx,
            alpha=alpha,
            falloff=falloff,
            hardness=hardness,
            cx_base=cx_base,
            cy_base=cy_base,
            cx_scale=cx_scale,
            cy_scale=cy_scale,
            cy_offset=cy_offset,
        )


def _stamp_head_circle(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    cx_factor: float,
    cy_factor: float,
    radius_factor: float,
    *,
    tone_idx: int,
    alpha: int,
    falloff: float,
    hardness: float,
    cx_offset: float = 0.0,
    cy_offset: float = 0.0,
) -> None:
    """Stamp one head-relative fuzzy circle."""
    stamp_fuzzy_circle(
        canvas,
        hx + hr * cx_factor + cx_offset,
        hy + hr * cy_factor + cy_offset,
        hr * radius_factor,
        _tone_rgba(pal, tone_idx, alpha),
        falloff,
        hardness,
    )


def _stamp_head_circle_specs(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    specs: tuple[HeadCircleSpec, ...],
) -> None:
    """Stamp a sequence of head-relative circle specs in-order."""
    for (
        cx_factor,
        cy_factor,
        radius_factor,
        tone_idx,
        alpha,
        falloff,
        hardness,
    ) in specs:
        _stamp_head_circle(
            canvas,
            hx,
            hy,
            hr,
            pal,
            cx_factor,
            cy_factor,
            radius_factor,
            tone_idx=tone_idx,
            alpha=alpha,
            falloff=falloff,
            hardness=hardness,
        )


def _draw_hair_short_north(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (-0.0, -0.15, 1.15, 0.95, 1, 235, 1.8, 0.85),
            (-0.0, -0.3, 0.8, 0.6, 2, 200, 1.5, 0.75),
        ),
    )


def _draw_hair_short_side(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
) -> None:
    cap_cy = hy - hr * 0.45
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        ((0.04, -0.12, 0.9, 0.62, 1, 235, 1.9, 0.86),),
        cy_base=cap_cy,
    )
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        ((0.52, -0.02, 0.25, 0.34, 1, 220, 1.45, 0.76),),
    )
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        ((-0.22, -0.34, 0.36, 0.24, 2, 198, 1.35, 0.72),),
        cy_base=cap_cy,
    )


def _draw_hair_short_south(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
) -> None:
    cap_cy = hy - hr * 0.45
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        ((-0.0, -0.12, 1.08, 0.54, 1, 235, 1.9, 0.86),),
        cy_base=cap_cy,
    )
    stamp_ellipse(
        canvas,
        hx,
        cap_cy - 0.7,
        hr * 0.62,
        hr * 0.36,
        _tone_rgba(pal, 2, 200),
        1.5,
        0.75,
    )


def _draw_hair_medium_north(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (-0.0, -0.1, 1.25, 1.1, 0, 235, 1.6, 0.82),
            (-0.0, -0.2, 1.15, 1.0, 1, 225, 1.5, 0.8),
            (-0.85, 0.3, 0.45, 0.7, 0, 220, 1.5, 0.78),
            (0.85, 0.3, 0.45, 0.7, 0, 220, 1.5, 0.78),
            (-0.0, -0.4, 0.7, 0.5, 2, 195, 1.3, 0.7),
        ),
    )


def _draw_hair_medium_side(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (0.06, -0.48, 1.12, 0.7, 0, 235, 1.6, 0.82),
            (0.05, -0.56, 1.0, 0.6, 1, 225, 1.5, 0.8),
            (0.78, 0.3, 0.4, 0.76, 0, 220, 1.5, 0.78),
            (-0.54, 0.02, 0.12, 0.24, 1, 214, 1.35, 0.74),
            (-0.3, -0.76, 0.48, 0.32, 2, 195, 1.3, 0.7),
        ),
    )


def _draw_hair_medium_south(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    hr_side_scale: float,
    highlight_jitter: float,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (-0.0, -0.46, 1.18, 0.74, 0, 235, 1.6, 0.82),
            (-0.0, -0.56, 1.06, 0.64, 1, 225, 1.5, 0.8),
        ),
    )
    for side in (-1.0, 1.0):
        stamp_ellipse(
            canvas,
            hx + hr * 0.85 * side,
            hy + hr * 0.12,
            hr_side_scale * 0.4,
            hr_side_scale * 0.55,
            _tone_rgba(pal, 0, 220),
            1.5,
            0.78,
        )
    stamp_ellipse(
        canvas,
        hx + highlight_jitter,
        hy + hr * -0.75,
        hr * 0.58,
        hr * 0.38,
        _tone_rgba(pal, 2, 195),
        1.3,
        0.7,
    )


def _draw_hair_long_north(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    drape_main: float,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (-0.0, 0.0, 1.3, 1.1, 0, 235, 1.6, 0.82),
            (-0.0, -0.1, 1.2, 1.0, 1, 225, 1.5, 0.8),
        ),
    )
    stamp_ellipse(
        canvas,
        hx,
        hy + drape_main * 0.3,
        hr * 0.9,
        drape_main * 0.45,
        _tone_rgba(pal, 0, 220),
        1.7,
        0.78,
    )
    stamp_ellipse(
        canvas,
        hx,
        hy + drape_main * 0.25,
        hr * 0.75,
        drape_main * 0.38,
        _tone_rgba(pal, 1, 215),
        1.5,
        0.75,
    )
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        ((-0.0, -0.3, 0.7, 0.6, 2, 190, 1.3, 0.68),),
    )


def _draw_hair_long_side(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    back_drape: float,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (0.1, -0.38, 1.15, 0.8, 0, 235, 1.6, 0.82),
            (0.08, -0.46, 1.02, 0.7, 1, 225, 1.5, 0.8),
        ),
    )
    stamp_ellipse(
        canvas,
        hx + hr * 0.62,
        hy + back_drape * 0.2,
        hr * 0.32,
        back_drape * 0.3,
        _tone_rgba(pal, 0, 215),
        1.8,
        0.78,
    )
    stamp_ellipse(
        canvas,
        hx + hr * 0.58,
        hy + back_drape * 0.15,
        hr * 0.24,
        back_drape * 0.24,
        _tone_rgba(pal, 1, 210),
        1.6,
        0.75,
    )
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (-0.52, 0.12, 0.1, 0.28, 1, 208, 1.35, 0.72),
            (-0.26, -0.7, 0.56, 0.4, 2, 190, 1.3, 0.68),
        ),
    )


def _draw_hair_long_south(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
    hr_side_scale: float,
    drape_left: float,
    drape_right: float,
    drape_left_scaled: float,
    drape_right_scaled: float,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (-0.0, -0.48, 1.14, 0.72, 0, 235, 1.6, 0.82),
            (-0.0, -0.58, 1.02, 0.62, 1, 225, 1.5, 0.8),
        ),
    )
    for (
        side,
        drape,
        drape_scaled,
        cy_factor,
        rx_factor,
        ry_factor,
        tone_idx,
        alpha,
        falloff,
        hardness,
    ) in (
        (-1.0, drape_left, drape_left_scaled, 0.28, 0.3, 0.22, 0, 215, 1.8, 0.78),
        (-1.0, drape_left, drape_left_scaled, 0.24, 0.21, 0.18, 1, 210, 1.6, 0.75),
        (1.0, drape_right, drape_right_scaled, 0.28, 0.3, 0.22, 0, 215, 1.8, 0.78),
        (1.0, drape_right, drape_right_scaled, 0.24, 0.21, 0.18, 1, 210, 1.6, 0.75),
    ):
        stamp_ellipse(
            canvas,
            hx + hr * 0.78 * side,
            hy + drape * cy_factor,
            hr_side_scale * rx_factor,
            drape_scaled * ry_factor,
            _tone_rgba(pal, tone_idx, alpha),
            falloff,
            hardness,
        )
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        ((-0.0, -0.74, 0.66, 0.42, 2, 190, 1.3, 0.68),),
    )


def _draw_hair_tall_north(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (-0.0, -0.1, 1.1, 1.0, 0, 230, 1.7, 0.82),
            (-0.0, -1.0, 0.65, 1.1, 1, 225, 1.6, 0.8),
            (-0.0, -1.2, 0.45, 0.8, 2, 200, 1.3, 0.7),
        ),
    )


def _draw_hair_tall_side(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (0.04, -1.12, 0.66, 1.12, 0, 230, 1.8, 0.82),
            (0.02, -1.24, 0.52, 0.94, 1, 220, 1.5, 0.78),
            (0.46, -0.22, 0.52, 0.72, 0, 224, 1.5, 0.78),
            (0.4, -0.28, 0.38, 0.54, 1, 214, 1.38, 0.74),
            (0.22, -0.24, 0.34, 0.48, 0, 220, 1.45, 0.74),
            (0.18, -0.28, 0.25, 0.36, 1, 210, 1.35, 0.7),
            (0.64, -0.04, 0.34, 0.5, 0, 216, 1.42, 0.75),
            (-0.22, -0.34, 0.2, 0.26, 2, 198, 1.3, 0.7),
        ),
    )


def _draw_hair_tall_south(
    canvas: np.ndarray,
    hx: float,
    hy: float,
    hr: float,
    pal: Palette3,
) -> None:
    _stamp_head_ellipse_specs(
        canvas,
        hx,
        hy,
        hr,
        pal,
        (
            (-0.0, -1.2, 0.7, 1.2, 0, 230, 1.8, 0.82),
            (-0.0, -1.3, 0.55, 1.0, 1, 220, 1.5, 0.78),
            (-0.0, -1.5, 0.35, 0.7, 2, 200, 1.3, 0.7),
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
    """Short crop hairstyle."""
    if facing == Facing.NORTH:
        _draw_hair_short_north(canvas, hx, hy, hr, pal)
    elif facing in {Facing.EAST, Facing.WEST}:
        _draw_hair_short_side(canvas, hx, hy, hr, pal)
    else:
        _draw_hair_short_south(canvas, hx, hy, hr, pal)


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
    if facing == Facing.NORTH:
        _draw_hair_medium_north(canvas, hx, hy, hr, pal)
    elif facing in {Facing.EAST, Facing.WEST}:
        _draw_hair_medium_side(canvas, hx, hy, hr, pal)
    else:
        side_scale = 0.72 if _luma(pal[1]) > 155.0 else 1.0
        _draw_hair_medium_south(
            canvas,
            hx,
            hy,
            hr,
            pal,
            hr * side_scale,
            float(rng.uniform(-0.4, 0.4)),
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
    """Long hair with back/side drape."""
    if facing == Facing.NORTH:
        _draw_hair_long_north(
            canvas, hx, hy, hr, pal, hr * float(rng.uniform(2.2, 3.0))
        )
    elif facing in {Facing.EAST, Facing.WEST}:
        _draw_hair_long_side(canvas, hx, hy, hr, pal, hr * float(rng.uniform(2.1, 3.0)))
    else:
        side_scale = 0.74 if _luma(pal[1]) > 155.0 else 1.0
        drape_left = hr * float(rng.uniform(2.0, 3.0))
        drape_right = hr * float(rng.uniform(2.0, 3.0))
        _draw_hair_long_south(
            canvas,
            hx,
            hy,
            hr,
            pal,
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
    if facing == Facing.NORTH:
        _draw_hair_tall_north(canvas, hx, hy, hr, pal)
    elif facing in {Facing.EAST, Facing.WEST}:
        _draw_hair_tall_side(canvas, hx, hy, hr, pal)
    else:
        _draw_hair_tall_south(canvas, hx, hy, hr, pal)


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

    if facing == Facing.SOUTH:
        is_long_front_hair = appearance.hair_style_idx == HAIR_IDX_LONG
        is_medium_front_hair = appearance.hair_style_idx == HAIR_IDX_MEDIUM
        if is_long_front_hair or is_medium_front_hair:
            _stamp_head_ellipse_specs(
                canvas,
                hx,
                hy,
                hr,
                appearance.hair_pal,
                (
                    (
                        0.0,
                        -(0.26 if is_long_front_hair else 0.22),
                        (0.24 if is_long_front_hair else 0.2),
                        (0.13 if is_long_front_hair else 0.11),
                        1,
                        210,
                        1.35,
                        0.7,
                    ),
                ),
            )
            if is_long_front_hair:
                _stamp_head_ellipse_specs(
                    canvas,
                    hx,
                    hy,
                    hr,
                    appearance.hair_pal,
                    ((0.0, -0.36, 0.23, 0.12, 1, 205, 1.25, 0.68),),
                )
        else:
            _stamp_head_ellipse_specs(
                canvas,
                hx,
                hy,
                hr,
                appearance.skin_pal,
                (
                    (0.0, -0.05, 0.44, 0.36, 1, 205, 1.35, 0.72),
                    (0.0, -0.22, 0.28, 0.2, 2, 190, 1.2, 0.65),
                ),
            )
    elif facing in {Facing.EAST, Facing.WEST}:
        if appearance.hair_style_idx == HAIR_IDX_TALL:
            _stamp_head_ellipse_specs(
                canvas,
                hx,
                hy,
                hr,
                appearance.hair_pal,
                (
                    (0.5, -0.18, 0.38, 0.54, 0, 228, 1.45, 0.76),
                    (0.44, -0.22, 0.28, 0.4, 1, 216, 1.3, 0.72),
                    (0.64, -0.02, 0.34, 0.5, 0, 230, 1.5, 0.82),
                    (0.56, -0.08, 0.25, 0.38, 1, 220, 1.35, 0.76),
                ),
            )

        _stamp_head_ellipse_specs(
            canvas,
            hx,
            hy,
            hr,
            appearance.skin_pal,
            (
                (-0.56, -0.06, 0.2, 0.3, 1, 210, 1.3, 0.68),
                (-0.9, 0.01, 0.08, 0.06, 0, 200, 1.0, 0.62),
            ),
        )
        _stamp_head_circle_specs(
            canvas,
            hx,
            hy,
            hr,
            appearance.skin_pal,
            ((-0.72, -0.03, 0.12, 2, 195, 1.25, 0.66),),
        )
        _stamp_head_circle(
            canvas,
            hx,
            hy,
            hr,
            appearance.skin_pal,
            -0.9,
            0.01,
            0.06,
            tone_idx=0,
            alpha=192,
            falloff=1.0,
            hardness=0.6,
            cx_offset=-0.18,
            cy_offset=0.02,
        )
        if appearance.hair_style_idx != HAIR_IDX_TALL:
            _stamp_head_circle_specs(
                canvas,
                hx,
                hy,
                hr,
                appearance.skin_pal,
                ((0.42, 0.02, 0.1, 0, 175, 1.25, 0.66),),
            )
