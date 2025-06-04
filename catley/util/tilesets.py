from __future__ import annotations

import numpy as np
import tcod.tileset

from catley import colors


def derive_outlined_tileset(
    tileset: tcod.tileset.Tileset,
    *,
    color: colors.ColorRGBA = (*colors.MAGENTA, 255),
) -> tcod.tileset.Tileset:
    """Return a tileset of 1px magenta outlines based on ``tileset``.

    Each tile in the returned tileset will contain only an outline of the
    corresponding glyph from ``tileset``. The outline is 1 pixel wide and the
    rest of the tile will be fully transparent. This tileset can be rendered on
    top of the original to give characters a glowing effect or on its own for an
    outlined style.
    """
    outlined = tcod.tileset.Tileset(tileset.tile_width, tileset.tile_height)
    outline_rgba = np.asarray(color, dtype=np.uint8)

    for codepoint in tcod.tileset.CHARMAP_CP437:
        src = tileset.get_tile(codepoint)
        alpha = src[:, :, 3] > 0
        padded = np.pad(alpha, 1)
        neighbors = [
            padded[:-2, :-2],
            padded[:-2, 1:-1],
            padded[:-2, 2:],
            padded[1:-1, :-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, :-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        dilated = np.logical_or.reduce(neighbors)
        outline_mask = dilated & ~alpha
        tile = np.zeros_like(src)
        tile[outline_mask] = outline_rgba
        outlined.set_tile(codepoint, tile)

    return outlined
