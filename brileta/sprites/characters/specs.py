"""Data-driven stamp spec primitives and render helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brileta import colors
from brileta.sprites.primitives import CanvasStamper, stamp_fuzzy_circle

from .appearance import Palette3


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
