"""Hand-authored clothing collar/neckline pixel masks (Track B for clothing).

The hair playbook (see :mod:`hair_masks`) applied to the highest-salience
clothing edge: the neckline. At 20x20 a same-hue collar shadow is invisible
(attempt 1 died on the 1x veto), so the neckline is instead defined by what
the garment reveals or covers at the throat, which is high contrast at 1x:

  * robe  -> a skin V-neck, skin showing down the chest (skin vs dark robe)
  * armor -> a bright metal gorget covering the throat (metal vs skin)
  * shirt -> a plain crew band, the neutral middle

Digits index whatever palette ``clothing._layer_collar`` passes for the style
(skin for the robe front/side, a bright-metal cloth ramp for armor, a cloth
ramp for shirt): 1=shadow/dark, 2=mid, 3=highlight. '.' is transparent. Blocks
are kept 2px thick so they survive the alpha-hardening pass rather than thinning
to nothing.

Masks are anchored to the torso-top geometry and scaled by ``torso_rx``. Side
masks are authored left-facing (front of body points screen-left); the renderer
mirrors the whole canvas for EAST, same as hair.
"""

from __future__ import annotations

from .masks import PixelMask, parse_mask

# ---------------------------------------------------------------------------
# Shirt — a plain crew band: a thin garment collar seam at the throat.
# ---------------------------------------------------------------------------

_SHIRT_SOUTH = """
.11....11.
.12....21.
..122221..
"""

_SHIRT_NORTH = """
.11....11.
.12111121.
..122221..
"""

_SHIRT_SIDE = """
.11.....
.221....
..122...
"""

# ---------------------------------------------------------------------------
# Armor — a high metal gorget: a solid band covering the throat, bright rim.
# ---------------------------------------------------------------------------

_ARMOR_SOUTH = """
.33333333.
.32222223.
.32211223.
.13322331.
"""

_ARMOR_NORTH = """
.33333333.
.32222223.
.32222223.
.13333331.
"""

_ARMOR_SIDE = """
.3333333.
.3222223.
.3221133.
.1332233.
"""

# ---------------------------------------------------------------------------
# Robe — a skin V-neck: a solid wedge of skin opening down the chest.
# (Rendered in the SKIN palette for front/side; NORTH uses a cloth cowl band.)
# ---------------------------------------------------------------------------

_ROBE_SOUTH = """
.11....11.
.122..221.
.1222221..
..122221..
...1221...
....11....
"""

_ROBE_NORTH = """
.11111111.
.12222221.
..122221..
"""

_ROBE_SIDE = """
.111....
.1221...
.12221..
.12221..
..1221..
...111..
"""


def _collar(
    text: str,
    y0: float,
    y1: float,
    x0: float = -1.0,
    x1: float = 1.0,
) -> PixelMask:
    return parse_mask(text, x0_f=x0, y0_f=y0, x1_f=x1, y1_f=y1)


SHIRT_COLLAR: dict[str, PixelMask] = {
    "south": _collar(_SHIRT_SOUTH, -0.15, 0.6),
    "north": _collar(_SHIRT_NORTH, -0.2, 0.6),
    "side": _collar(_SHIRT_SIDE, -0.15, 0.6, x0=-0.95, x1=1.05),
}

ARMOR_COLLAR: dict[str, PixelMask] = {
    "south": _collar(_ARMOR_SOUTH, -0.55, 0.75),
    "north": _collar(_ARMOR_NORTH, -0.55, 0.7),
    "side": _collar(_ARMOR_SIDE, -0.55, 0.75, x0=-0.95, x1=1.05),
}

ROBE_COLLAR: dict[str, PixelMask] = {
    "south": _collar(_ROBE_SOUTH, -0.1, 1.15),
    "north": _collar(_ROBE_NORTH, -0.2, 0.55),
    "side": _collar(_ROBE_SIDE, -0.1, 1.15, x0=-0.95, x1=1.05),
}

# Keyed by clothing style name (see clothing.CLOTHING_STYLE_NAMES). "bare" has
# no collar.
COLLAR_MASKS: dict[str, dict[str, PixelMask]] = {
    "shirt": SHIRT_COLLAR,
    "armor": ARMOR_COLLAR,
    "robe": ROBE_COLLAR,
}
