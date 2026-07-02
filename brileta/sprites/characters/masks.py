"""ASCII pixel-mask parsing and head-anchored rasterization.

Track B of the sprite roadmap replaces geometric ellipse hair/head layers with
hand-authored pixel masks. A mask is a multi-line string literal where each
character is one pixel:

    '.'   transparent
    '1'   palette shadow tone   (palette[0])
    '2'   palette mid tone      (palette[1])
    '3'   palette highlight tone (palette[2])

Masks are authored once at a nominal head size and rasterized onto the canvas
anchored to the live head geometry (center + radius), so builds with different
head radii scale the same silhouette instead of needing separate art.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brileta.sprites.primitives import Palette3

# Digit -> palette tone index. '1' is the darkest ramp entry.
_CHAR_TO_TONE: dict[str, int] = {"1": 0, "2": 1, "3": 2}


@dataclass(frozen=True)
class PixelMask:
    """A parsed pixel mask plus its head-relative placement box.

    ``tones`` is an ``(rows, cols)`` int8 grid: -1 transparent, else 0/1/2 for
    the palette tone. The mask is placed so its grid maps onto the canvas box
    ``[hx + hr*x0_f, hx + hr*x1_f] x [hy + hr*y0_f, hy + hr*y1_f]`` (right/bottom
    edges exclusive in head-relative space), where ``(hx, hy)`` is the head
    center and ``hr`` the head radius.
    """

    tones: np.ndarray
    x0_f: float
    y0_f: float
    x1_f: float
    y1_f: float


def parse_mask(
    text: str,
    *,
    x0_f: float,
    y0_f: float,
    x1_f: float,
    y1_f: float,
) -> PixelMask:
    """Parse a mask string literal into a :class:`PixelMask`.

    Leading/trailing blank lines are ignored. Rows are right-padded with
    transparent cells so a ragged literal still yields a rectangular grid.
    """
    lines = [line for line in text.splitlines() if line.strip() != ""]
    width = max(len(line) for line in lines)
    grid = np.full((len(lines), width), -1, dtype=np.int8)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            tone = _CHAR_TO_TONE.get(ch)
            if tone is not None:
                grid[r, c] = tone
    return PixelMask(tones=grid, x0_f=x0_f, y0_f=y0_f, x1_f=x1_f, y1_f=y1_f)


def render_mask(
    canvas: np.ndarray,
    mask: PixelMask,
    palette: Palette3,
    hx: float,
    hy: float,
    hr: float,
    *,
    mirror: bool = False,
) -> None:
    """Rasterize *mask* onto *canvas*, anchored to the head geometry.

    Each canvas pixel inside the placement box samples the nearest mask cell
    (nearest-neighbor, so the silhouette stays crisp and gap-free at any head
    radius). Opaque cells are written as fully opaque palette-toned pixels.
    ``mirror`` flips the mask horizontally for per-character variety on
    front/back facings.
    """
    rows, cols = mask.tones.shape
    left = hx + hr * mask.x0_f
    right = hx + hr * mask.x1_f
    top = hy + hr * mask.y0_f
    bottom = hy + hr * mask.y1_f

    h, w = canvas.shape[:2]
    px_start = max(0, int(np.floor(left)))
    px_end = min(w, int(np.ceil(right)))
    py_start = max(0, int(np.floor(top)))
    py_end = min(h, int(np.ceil(bottom)))
    span_x = right - left
    span_y = bottom - top
    if span_x <= 0 or span_y <= 0:
        return

    for py in range(py_start, py_end):
        # +0.5 samples the pixel center so rounding is symmetric.
        v = (py + 0.5 - top) / span_y
        row = int(v * rows)
        if row < 0 or row >= rows:
            continue
        for px in range(px_start, px_end):
            u = (px + 0.5 - left) / span_x
            col = int(u * cols)
            if col < 0 or col >= cols:
                continue
            tone = int(mask.tones[row, cols - 1 - col if mirror else col])
            if tone < 0:
                continue
            r, g, b = palette[tone]
            canvas[py, px, 0] = r
            canvas[py, px, 1] = g
            canvas[py, px, 2] = b
            canvas[py, px, 3] = 255
