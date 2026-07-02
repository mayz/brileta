"""Hand-authored hair pixel masks (Track B).

Each mask is a string literal rasterized by :mod:`masks` onto the head-anchored
box. Digits index the hair palette ramp (1=shadow, 2=mid, 3=highlight); '.' is
transparent. Light comes from the top-left, so highlight (3) clusters upper-left
and shadow (1) sits lower-right and along the silhouette rim. Shading moves in
2-4px clusters, never single-pixel noise.

Side masks are authored left-facing (nose points screen-left); the renderer
mirrors the whole canvas for the EAST facing. On profiles the hair mass shifts
toward the back of the skull (screen-right) so the face reads.

Placement boxes are in head-radius units, anchored at the head center
``(hx, hy)``: the grid maps onto ``[hx+hr*x0_f, hx+hr*x1_f] x
[hy+hr*y0_f, hy+hr*y1_f]``. ``y=-1`` is the crown, ``y=+1`` the chin.
"""

from __future__ import annotations

from .masks import PixelMask, parse_mask

# ---------------------------------------------------------------------------
# Short crop — a cap hugging the crown, hairline across the upper forehead.
# ---------------------------------------------------------------------------

_SHORT_SOUTH = """
..333222..
.33322221.
2332222221
2222222221
1222222111
11......11
"""

_SHORT_NORTH = """
..333222..
.33322221.
2332222221
2222222221
1222222211
1122222111
"""

_SHORT_SIDE = """
..3332221.
.33322221.
.32222221.
..2222221.
...222211.
....11121.
"""

# ---------------------------------------------------------------------------
# Medium — covers the forehead and frames the face down to the jaw.
# ---------------------------------------------------------------------------

_MEDIUM_SOUTH = """
..333222..
.33322221.
2333222221
2222222221
2222222221
2222222221
221....122
221....122
221....122
.11....11.
"""

_MEDIUM_NORTH = """
..333222..
.33322221.
2332222221
2222222221
2222222221
2222222221
2222222211
.12222211.
"""

_MEDIUM_SIDE = """
..3332221.
.33322221.
.322222221
.222222221
..22222221
...2222221
....122221
.....11221
"""

# ---------------------------------------------------------------------------
# Long — cap plus curtains draping past the neck onto the shoulders.
# ---------------------------------------------------------------------------

_LONG_SOUTH = """
..333222..
.33322221.
2333222221
2222222221
2222222221
221....122
221....122
221....122
221....122
221....122
.21....12.
.11....11.
"""

_LONG_NORTH = """
..333222..
.33322221.
2332222221
2222222221
2222222221
2222222221
2222222221
2222222221
2222222221
2222222221
.12222221.
"""

_LONG_SIDE = """
..3332221.
.33322221.
.322222221
.222222221
..22222221
...2222221
....2222221
.....222221
.....122221
......12221
......11221
"""

# ---------------------------------------------------------------------------
# Tall — upward volume rising well above the crown.
# ---------------------------------------------------------------------------

_TALL_SOUTH = """
..3322..
.33322..
.333222.
2332222.
2222221.
22222221
1222221.
11....11
"""

_TALL_NORTH = """
..3322..
.33322..
.333222.
2332222.
22222221
22222221
1222221.
"""

_TALL_SIDE = """
..33321..
.3332221.
333222221
233222221
.22222221
..2222221
..1222221
...112221
"""


def _mask(
    text: str, y0: float, y1: float, x0: float = -1.08, x1: float = 1.08
) -> PixelMask:
    return parse_mask(text, x0_f=x0, y0_f=y0, x1_f=x1, y1_f=y1)


SHORT_MASKS: dict[str, PixelMask] = {
    "south": _mask(_SHORT_SOUTH, -1.15, 0.2),
    "north": _mask(_SHORT_NORTH, -1.15, 0.45),
    "side": _mask(_SHORT_SIDE, -1.15, 0.2, x0=-1.0, x1=1.2),
}

MEDIUM_MASKS: dict[str, PixelMask] = {
    "south": _mask(_MEDIUM_SOUTH, -1.15, 1.0),
    "north": _mask(_MEDIUM_NORTH, -1.15, 0.9),
    "side": _mask(_MEDIUM_SIDE, -1.15, 1.0, x0=-1.0, x1=1.25),
}

LONG_MASKS: dict[str, PixelMask] = {
    "south": _mask(_LONG_SOUTH, -1.15, 2.1),
    "north": _mask(_LONG_NORTH, -1.15, 2.0),
    "side": _mask(_LONG_SIDE, -1.15, 2.0, x0=-1.0, x1=1.25),
}

TALL_MASKS: dict[str, PixelMask] = {
    "south": _mask(_TALL_SOUTH, -1.55, 0.2, x0=-0.95, x1=0.95),
    "north": _mask(_TALL_NORTH, -1.55, 0.35, x0=-0.95, x1=0.95),
    "side": _mask(_TALL_SIDE, -1.55, 0.25, x0=-0.95, x1=1.2),
}

# Keyed by hair_style_idx (see hair.HAIR_STYLES). Bald (0) has no mask.
MASK_SETS: dict[int, dict[str, PixelMask]] = {
    1: SHORT_MASKS,
    2: MEDIUM_MASKS,
    3: LONG_MASKS,
    4: TALL_MASKS,
}
