"""
Tile system for the game.

Instead of having separate arrays for each tile property (walkable, transparent,
character, colors), we use numpy structured arrays to pack everything together.

Benefits:
  - Performance: One array operation instead of updating 5+ separate arrays.
  - Memory: More cache-friendly, less memory fragmentation, no Python object overhead.
  - TCOD compatibility: Console.tiles_rgb expects exactly this data structure.
  - Maintainability: All tile properties defined in one place.
  - One tiles[x,y] lookup vs tiles[x,y].walkable, tiles[x,y].char, etc.
  - Bulk operations: tiles["walkable"][large_area] = True (thousands at once)
"""

import numpy as np

from catley import colors

# Defines the structure that TCOD's Console.tiles_rgb expects.
# It *must* match this exact format, or rendering will break.
# Console.tiles_rgb[x, y] expects exactly:
#   (character_code, [r,g,b], [r,g,b])
TileAppearance = np.dtype(
    [
        ("ch", np.int32),  # Character code (like ord('@') = 64)
        ("fg", "3B"),  # Foreground RGB: 3 unsigned bytes (0-255 each)
        ("bg", "3B"),  # Background RGB: 3 unsigned bytes (0-255 each)
    ]
)

# Tile data type definition. Includes both game logic and rendering properties.
#
# This is for *core properties* only. Only include properties that are:
# 1. Accessed frequently (every movement, every FOV calculation, every render).
# 2. Apply to most/all tiles in the game.
# 3. Fundamental to basic game mechanics.
#
# Adding fields here affects *all* tiles and makes the core data structure larger.
# Ask: "Is this checked as often as walkable/transparent?" If no, put it elsewhere.
#
# Don't add situational properties here (flammable, slippery, fragile, etc.)
# Those should go in (for example):
# - Behavior system (TileBehavior enum + separate handling)
# - Auxiliary property maps (only created when needed)
# - Component systems attached to specific tiles
Tile = np.dtype(
    [
        # Game logic properties.
        # "walkable" - Can actors move through this tile?
        ("walkable", bool),
        # "transparent" - Can actors see through this tile?
        ("transparent", bool),
        # Rendering properties.
        # "dark" - What player sees for explored but not currently visible tiles.
        ("dark", TileAppearance),
        # "light" - What player sees for currently visible tiles.
        ("light", TileAppearance),
    ]
)


# Tile type factory function.
def make_tile_type(
    *,  # Forces keyword arguments - prevents bugs from wrong parameter order
    walkable: bool,
    transparent: bool,
    dark: tuple[int, colors.Color, colors.Color],
    light: tuple[int, colors.Color, colors.Color] | None = None,
) -> np.ndarray:
    """
    Create a new tile TYPE definition (template) with the given properties.

    This creates a reusable tile type that can be assigned to many map positions.
    Think of it like defining a "class" of tile (wall, floor, door, etc.)

    - Makes tile type creation less error-prone
    - Provides defaults and validation
    - Hides the numpy array creation complexity
    - Makes code more readable: make_tile_type(walkable=True, ...) vs np.array(...)

    Args:
        walkable: Can actors walk through this type of tile?
        transparent: Can actors see through this type of tile?
        dark: (character, fg_color, bg_color) when tile not in FOV
        light: (character, fg_color, bg_color) when tile in FOV
               If None, uses same as dark

    Returns:
        A numpy array representing a tile type definition that can be
        assigned to map positions: map.tiles[x, y] = wall_tile_type
    """
    if light is None:
        light = dark

    return np.array((walkable, transparent, dark, light), dtype=Tile)


# Constants representing tile type definitions
#
# Later on, we can add more tile types like DOOR, STAIRS, WATER, etc.
FLOOR = make_tile_type(
    walkable=True,
    transparent=True,
    dark=(ord(" "), colors.WHITE, colors.DARK_GROUND),
    light=(ord(" "), colors.WHITE, colors.LIGHT_GROUND),
)

WALL = make_tile_type(
    walkable=False,
    transparent=False,
    dark=(ord(" "), colors.WHITE, colors.DARK_WALL),
    light=(ord(" "), colors.WHITE, colors.LIGHT_WALL),
)
