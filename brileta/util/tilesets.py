from __future__ import annotations

import numpy as np

from brileta import colors

# CP437 to Unicode mapping, see https://en.wikipedia.org/wiki/Code_page_437
CHARMAP_CP437: tuple[int, ...] = (
    0,
    9786,
    9787,
    9829,
    9830,
    9827,
    9824,
    8226,
    9688,
    9675,
    9689,
    9794,
    9792,
    9834,
    9835,
    9788,
    9658,
    9668,
    8597,
    8252,
    182,
    167,
    9644,
    8616,
    8593,
    8595,
    8594,
    8592,
    8735,
    8596,
    9650,
    9660,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    8962,
    199,
    252,
    233,
    226,
    228,
    224,
    229,
    231,
    234,
    235,
    232,
    239,
    238,
    236,
    196,
    197,
    201,
    230,
    198,
    244,
    246,
    242,
    251,
    249,
    255,
    214,
    220,
    162,
    163,
    165,
    8359,
    402,
    225,
    237,
    243,
    250,
    241,
    209,
    170,
    186,
    191,
    8976,
    172,
    189,
    188,
    161,
    171,
    187,
    9617,
    9618,
    9619,
    9474,
    9508,
    9569,
    9570,
    9558,
    9557,
    9571,
    9553,
    9559,
    9565,
    9564,
    9563,
    9488,
    9492,
    9524,
    9516,
    9500,
    9472,
    9532,
    9566,
    9567,
    9562,
    9556,
    9577,
    9574,
    9568,
    9552,
    9580,
    9575,
    9576,
    9572,
    9573,
    9561,
    9560,
    9554,
    9555,
    9579,
    9578,
    9496,
    9484,
    9608,
    9604,
    9612,
    9616,
    9600,
    945,
    223,
    915,
    960,
    931,
    963,
    181,
    964,
    934,
    920,
    937,
    948,
    8734,
    966,
    949,
    8745,
    8801,
    177,
    8805,
    8804,
    8992,
    8993,
    247,
    8776,
    176,
    8729,
    183,
    8730,
    8319,
    178,
    9632,
    160,
)

# Create a reverse mapping from Unicode code points to CP437 positions
# This allows us to convert Unicode characters to their tileset indices
UNICODE_TO_CP437: dict[int, int] = {
    unicode_codepoint: cp437_index
    for cp437_index, unicode_codepoint in enumerate(CHARMAP_CP437)
}


def unicode_to_cp437(codepoint: int, fallback: int = ord("?")) -> int:
    """Convert a Unicode code point to its CP437 position.

    Args:
        codepoint: The Unicode code point to convert
        fallback: The character to use if the codepoint isn't in CP437

    Returns:
        The CP437 position (0-255) for the character
    """
    return UNICODE_TO_CP437.get(codepoint, fallback)


def derive_outlined_atlas(
    atlas_pixels: np.ndarray,
    tile_width: int,
    tile_height: int,
    columns: int,
    rows: int,
    color: colors.ColorRGBA = colors.COMBAT_OUTLINE,
) -> np.ndarray:
    """Create an outlined version of a tileset atlas as a numpy array.

    Takes a tileset atlas image (as RGBA numpy array) and returns a new atlas
    where each tile contains only a 1-pixel outline of the original glyph shape.
    This works with any graphics backend that uses numpy arrays.

    The algorithm for each tile:
    1. Extract the alpha channel to find where the glyph is
    2. Dilate the mask by 1 pixel in all 8 directions
    3. Subtract the original mask to get only the outline pixels
    4. Fill those pixels with the outline color

    Args:
        atlas_pixels: RGBA pixel array of the tileset atlas (height, width, 4)
        tile_width: Width of each tile in pixels
        tile_height: Height of each tile in pixels
        columns: Number of tile columns in the atlas
        rows: Number of tile rows in the atlas
        color: RGBA color for the outline (default: combat red)

    Returns:
        New RGBA pixel array with only the outlines, same dimensions as input
    """
    height, width = atlas_pixels.shape[:2]
    outlined = np.zeros_like(atlas_pixels)
    outline_rgba = np.asarray(color, dtype=np.uint8)

    # Process each tile in the atlas
    for row in range(rows):
        for col in range(columns):
            # Calculate tile bounds
            y1 = row * tile_height
            y2 = y1 + tile_height
            x1 = col * tile_width
            x2 = x1 + tile_width

            # Bounds check
            if y2 > height or x2 > width:
                continue

            # Extract the tile
            tile = atlas_pixels[y1:y2, x1:x2]

            # Get the alpha mask (where alpha > 0)
            alpha = tile[:, :, 3] > 0

            # Dilate the mask by 1 pixel in all 8 directions
            # Pad with zeros to handle edge cases
            padded = np.pad(alpha, 1, mode="constant", constant_values=False)

            # Get all 8 neighbors plus center
            neighbors = [
                padded[:-2, :-2],  # top-left
                padded[:-2, 1:-1],  # top
                padded[:-2, 2:],  # top-right
                padded[1:-1, :-2],  # left
                padded[1:-1, 1:-1],  # center
                padded[1:-1, 2:],  # right
                padded[2:, :-2],  # bottom-left
                padded[2:, 1:-1],  # bottom
                padded[2:, 2:],  # bottom-right
            ]

            # Dilated mask is True wherever any neighbor is True
            dilated = np.logical_or.reduce(neighbors)

            # Outline is the dilated area minus the original glyph
            outline_mask = dilated & ~alpha

            # Create the outlined tile
            outlined_tile = np.zeros_like(tile)
            outlined_tile[outline_mask] = outline_rgba

            # Write back to the output atlas
            outlined[y1:y2, x1:x2] = outlined_tile

    return outlined
