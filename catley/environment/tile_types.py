"""
Tile Type system for the game using the flyweight pattern.

This module defines:
- `TileTypeAppearance`: What a tile type looks like (character, fg/bg colors).
- `TileTypeData`: The intrinsic properties of a *type* of tile (walkable, transparent,
  and its dark/light appearances). These are the flyweight objects.
- An automated registration system for `TileTypeData` instances. Each registered
  tile type is assigned a unique integer ID (`TileTypeID`). `GameMap` will store a
  NumPy array of these `TileTypeID`s, significantly reducing memory usage compared
  to storing full `TileTypeData` for every tile cell.
- Helper functions to efficiently get maps of specific properties (e.g., a boolean
  map of all walkable tiles) from a `TileTypeID` map. These are crucial for
  systems like FOV calculation and rendering.
"""

import numpy as np

from catley import colors

# Defines the structure for the visual appearance of a tile type.
# Used within TileTypeData for 'dark' and 'light' states.
#
# This structure is designed to be backend-agnostic and compatible with
# GlyphBuffer. The RGB values are converted to RGBA as needed during
# rendering (typically adding full alpha=255).
TileTypeAppearance = np.dtype(
    [
        ("ch", np.int32),  # Character code (e.g., ord('@'))
        ("fg", "3B"),  # Foreground RGB: 3 unsigned bytes (0-255 each)
        ("bg", "3B"),  # Background RGB: 3 unsigned bytes (0-255 each)
    ]
)

# Defines the intrinsic data for a *type* of tile (flyweight).
# This includes its game logic properties (walkable, transparent)
# and its visual appearances under different lighting conditions.
TileTypeData = np.dtype(
    [
        ("walkable", bool),
        ("transparent", bool),  # FOV/line-of-sight (exploration & targeting)
        ("cover_bonus", np.int8),  # Defensive bonus from using this tile as cover
        ("casts_shadows", bool),  # Light occlusion (visual shadows)
        ("display_name", "U32"),  # Human-readable name (Unicode string, max 32 chars)
        # Appearance when explored but not in FOV
        ("dark", TileTypeAppearance),
        # Appearance when in FOV (before dynamic lighting)
        ("light", TileTypeAppearance),
    ]
)

# --- Automated Tile Type Registration System ---

# Internal list to store all registered TileTypeData instances.
# The index of a tile type in this list becomes its TileTypeID.
_registered_tile_type_data_list: list[np.ndarray] = []  # Stores TileTypeData instances

# Dictionary to map symbolic names (e.g., "WALL") to their assigned TileTypeID.
# Useful for debugging or potentially loading maps from definitions using names.
_tile_type_name_to_id_map: dict[str, int] = {}


def register_tile_type(name: str, tile_type_data_instance: np.ndarray) -> int:
    """
    Registers a new tile type, assigns it a unique ID, and stores it.
    It also dynamically creates a global constant for this ID in the format
    TILE_TYPE_ID_{NAME} (e.g., TILE_TYPE_ID_WALL).

    Args:
        name: A symbolic, unique string name for this tile type (e.g., "WALL", "FLOOR").
              Case-insensitive for registration checking and constant creation.
        tile_type_data_instance: The numpy array (structured with TileTypeData dtype)
                                 containing the properties of this tile type.

    Returns:
        The automatically assigned unique integer ID for this tile type.

    Raises:
        ValueError: If the name (case-insensitive) is already registered.
    """
    # Use uppercase for consistent checking and constant naming
    normalized_name = name.upper()
    if normalized_name in _tile_type_name_to_id_map:
        raise ValueError(
            f"Tile type name '{name}' (as '{normalized_name}') is already registered."
        )

    # The ID is simply the next available index in our list.
    tile_type_id = len(_registered_tile_type_data_list)
    _registered_tile_type_data_list.append(tile_type_data_instance)
    _tile_type_name_to_id_map[normalized_name] = tile_type_id

    # Dynamically create a global constant for this ID (e.g., TILE_TYPE_ID_WALL = 0).
    # This makes using IDs in code more readable and less prone to magic numbers.
    globals()[f"TILE_TYPE_ID_{normalized_name}"] = tile_type_id

    return tile_type_id


# --- End Automated Tile Type Registration System ---


def make_tile_type_data(
    *,  # Forces keyword arguments - prevents bugs from wrong parameter order
    walkable: bool,
    transparent: bool,
    display_name: str,
    cover_bonus: int = 0,
    casts_shadows: bool = False,
    dark: tuple[int, colors.Color, colors.Color],  # (char_code, fg_color, bg_color)
    light: tuple[int, colors.Color, colors.Color] | None = None,
) -> np.ndarray:  # Returns an instance of TileTypeData
    """
    Helper function to create a TileTypeData instance.

    Args:
        walkable: Can actors walk through this type of tile?
        transparent: Is this tile see-through for FOV/targeting?
        display_name: Human-readable name for UI display (e.g., "Wall", "Closed Door")
        cover_bonus: Defensive bonus granted when using this tile as cover.
        casts_shadows: Does this tile block light for shadow casting?
        dark:  A (char_code, fg_color, bg_color) tuple defining
               the 'dark' TileTypeAppearance.
        light: A (char_code, fg_color, bg_color) tuple defining
               the 'light' TileTypeAppearance. If None, use the same
               tuple as 'dark'.

    Returns:
        A numpy array structured with the TileTypeData dtype.
    """
    if light is None:
        light = dark
    # Create the TileTypeAppearance structs for dark and light
    dark_appearance = np.array((dark[0], dark[1], dark[2]), dtype=TileTypeAppearance)
    light_appearance = np.array(
        (light[0], light[1], light[2]), dtype=TileTypeAppearance
    )

    return np.array(
        (
            walkable,
            transparent,
            cover_bonus,
            casts_shadows,
            display_name,
            dark_appearance,
            light_appearance,
        ),
        dtype=TileTypeData,
    )


# --- Define and Register Core Tile Types ---
# The act of defining these automatically registers them and creates their ID constants.
# For example, after the WALL line, TILE_TYPE_ID_WALL will exist and be 0.

# It's good practice to register the most common/default tile type first (often WALL
# or VOID) if its ID (e.g., 0) is used as a default fill value in map creation.
_wall_data = make_tile_type_data(
    walkable=False,
    transparent=False,
    display_name="Wall",
    dark=(ord(" "), colors.DARK_GREY, colors.DARK_WALL),
    light=(ord(" "), colors.LIGHT_GREY, colors.LIGHT_WALL),
)
register_tile_type("WALL", _wall_data)

_floor_data = make_tile_type_data(
    walkable=True,
    transparent=True,
    display_name="Floor",
    dark=(ord(" "), colors.DARK_GREY, colors.DARK_GROUND),
    light=(ord(" "), colors.LIGHT_GREY, colors.LIGHT_GROUND),
)
register_tile_type("FLOOR", _floor_data)

_outdoor_floor_data = make_tile_type_data(
    walkable=True,
    transparent=True,
    display_name="Ground",
    dark=(ord(" "), colors.DARK_GREY, colors.OUTDOOR_DARK_GROUND),
    light=(ord(" "), colors.LIGHT_GREY, colors.OUTDOOR_LIGHT_GROUND),
)
register_tile_type("OUTDOOR_FLOOR", _outdoor_floor_data)

_outdoor_wall_data = make_tile_type_data(
    walkable=False,
    transparent=False,
    display_name="Rock Wall",
    casts_shadows=True,
    dark=(ord("#"), colors.LIGHT_GREY, colors.OUTDOOR_DARK_WALL),
    light=(ord("#"), colors.LIGHT_GREY, colors.OUTDOOR_LIGHT_WALL),
)
register_tile_type("OUTDOOR_WALL", _outdoor_wall_data)

_door_closed_data = make_tile_type_data(
    walkable=False,
    transparent=False,
    display_name="Closed Door",
    dark=(ord("+"), colors.ORANGE, colors.DARK_WALL),
    light=(ord("+"), colors.LIGHT_ORANGE, colors.LIGHT_WALL),
)
register_tile_type("DOOR_CLOSED", _door_closed_data)

_door_open_data = make_tile_type_data(
    walkable=True,
    transparent=True,
    display_name="Open Door",
    dark=(ord("'"), colors.ORANGE, colors.DARK_GROUND),
    light=(ord("'"), colors.LIGHT_ORANGE, colors.LIGHT_GROUND),
)
register_tile_type("DOOR_OPEN", _door_open_data)

_boulder_data = make_tile_type_data(
    walkable=False,
    transparent=True,
    display_name="Boulder",
    cover_bonus=2,
    casts_shadows=True,
    dark=(ord("#"), colors.DARK_GREY, colors.DARK_GROUND),
    light=(ord("#"), colors.LIGHT_GREY, colors.LIGHT_GROUND),
)
register_tile_type("BOULDER", _boulder_data)


# --- Pre-calculated Property Arrays for Efficient Lookups ---
# These arrays are built *after* all tile types have been registered.
# They allow for fast, vectorized conversion from a map of TileTypeIDs
# to a map of a specific property (e.g., walkability).
# This is safe because Python executes modules top-to-bottom, so by this point,
# _registered_tile_type_data_list is fully populated by the register_tile_type calls.

_tile_type_properties_walkable = np.array(
    [t["walkable"] for t in _registered_tile_type_data_list], dtype=bool
)
_tile_type_properties_transparent = np.array(
    [t["transparent"] for t in _registered_tile_type_data_list], dtype=bool
)
_tile_type_properties_dark_appearance = np.array(
    [t["dark"] for t in _registered_tile_type_data_list], dtype=TileTypeAppearance
)
_tile_type_properties_light_appearance = np.array(
    [t["light"] for t in _registered_tile_type_data_list], dtype=TileTypeAppearance
)
_tile_type_properties_display_name = np.array(
    [t["display_name"] for t in _registered_tile_type_data_list], dtype="U32"
)

# --- Public Helper Functions for Accessing Tile Properties ---


def get_walkable_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """
    Converts a map of TileTypeIDs into a boolean map of walkability.
    True means the tile at that position is walkable.
    """
    return _tile_type_properties_walkable[tile_type_ids_map]


def get_transparent_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """
    Converts a map of TileTypeIDs into a boolean map of transparency.
    True means the tile at that position is transparent (for FOV).
    """
    return _tile_type_properties_transparent[tile_type_ids_map]


def get_dark_appearance_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """
    Converts a map of TileTypeIDs into a map of 'dark' TileTypeAppearance structs.
    Used for rendering explored tiles not currently in FOV.
    """
    return _tile_type_properties_dark_appearance[tile_type_ids_map]


def get_light_appearance_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """
    Converts a map of TileTypeIDs into a map of 'light' TileTypeAppearance structs.
    Used as the base for rendering tiles currently in FOV, before dynamic lighting.
    """
    return _tile_type_properties_light_appearance[tile_type_ids_map]


def get_tile_type_data_by_id(tile_type_id: int) -> np.ndarray:  # Returns TileTypeData
    """
    Retrieves the full TileTypeData instance for a given TileTypeID.
    Useful if you need to access multiple properties of a specific tile type
    without constructing full property maps.
    """
    if 0 <= tile_type_id < len(_registered_tile_type_data_list):
        return _registered_tile_type_data_list[tile_type_id]
    # Provide a more informative error if an ID is out of bounds.
    raise IndexError(
        f"Invalid TileTypeID: {tile_type_id}. "
        f"Registered IDs are 0 to {len(_registered_tile_type_data_list) - 1}."
    )


def get_tile_type_name_by_id(tile_type_id: int) -> str:
    """
    Get the human-readable name of a tile type by its ID.

    Args:
        tile_type_id: The ID of the tile type

    Returns:
        The name of the tile type in a human-readable format (e.g., "Wall", "Floor")
    """
    if 0 <= tile_type_id < len(_tile_type_properties_display_name):
        return str(_tile_type_properties_display_name[tile_type_id])
    return f"Unknown Tile (ID: {tile_type_id})"
