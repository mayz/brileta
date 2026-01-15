"""
Tile Type system for the game using the flyweight pattern.

This module defines:
- `TileTypeAppearance`: What a tile type looks like (character, fg/bg colors).
- `TileTypeData`: The intrinsic properties of a *type* of tile (walkable, transparent,
  and its dark/light appearances). These are the flyweight objects.
- `TileTypeID`: An IntEnum of all tile type IDs. To add a new tile type, add it to
  this enum first, then register its data.
- Helper functions to efficiently get maps of specific properties (e.g., a boolean
  map of all walkable tiles) from a `TileTypeID` map. These are crucial for
  systems like FOV calculation and rendering.
"""

from enum import IntEnum, auto

import numpy as np

from catley import colors
from catley.game.enums import ImpactMaterial


class TileTypeID(IntEnum):
    """
    All tile type IDs. To add a new tile type:
    1. Add it to this enum
    2. Register its data with register_tile_type() below
    """

    WALL = 0
    FLOOR = auto()
    OUTDOOR_FLOOR = auto()
    OUTDOOR_WALL = auto()
    DOOR_CLOSED = auto()
    DOOR_OPEN = auto()
    BOULDER = auto()
    # Hazardous terrain types
    ACID_POOL = auto()
    HOT_COALS = auto()


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

# Animation parameters for tiles that have dynamic color/glyph effects.
# Colors oscillate via random walk, creating organic shimmer effects.
TileAnimationParams = np.dtype(
    [
        ("animates", np.bool_),  # Does this tile animate?
        ("fg_variation", "3B"),  # RGB variation range for foreground glyph color
        ("bg_variation", "3B"),  # RGB variation range for background color
        ("glyph_flicker_chance", np.float32),  # Probability of hiding glyph (0.0-1.0)
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
        ("material", np.int8),  # ImpactMaterial enum value for impact sounds
        ("hazard_damage", "U8"),  # Dice string for damage (e.g., "1d4"), empty = safe
        ("hazard_damage_type", "U16"),  # Damage type: "acid", "fire", "electric", etc.
        ("display_name", "U32"),  # Human-readable name (Unicode string, max 32 chars)
        # Appearance when explored but not in FOV
        ("dark", TileTypeAppearance),
        # Appearance when in FOV (before dynamic lighting)
        ("light", TileTypeAppearance),
        # Animation parameters (color oscillation, glyph flicker)
        ("animation", TileAnimationParams),
    ]
)

# --- Tile Type Registration System ---

# Dictionary to store registered TileTypeData instances, keyed by TileTypeID.
_tile_type_definitions: dict[TileTypeID, np.ndarray] = {}


def register_tile_type(
    tile_id: TileTypeID, tile_type_data_instance: np.ndarray
) -> None:
    """
    Register tile type data for a TileTypeID.

    Args:
        tile_id: The TileTypeID enum value for this tile type.
        tile_type_data_instance: The numpy array (structured with TileTypeData dtype)
                                 containing the properties of this tile type.

    Raises:
        ValueError: If the tile_id is already registered.
    """
    if tile_id in _tile_type_definitions:
        raise ValueError(f"Tile type {tile_id.name} is already registered.")
    _tile_type_definitions[tile_id] = tile_type_data_instance


# --- End Tile Type Registration System ---


def make_tile_type_data(
    *,  # Forces keyword arguments - prevents bugs from wrong parameter order
    walkable: bool,
    transparent: bool,
    display_name: str,
    material: ImpactMaterial = ImpactMaterial.STONE,
    cover_bonus: int = 0,
    casts_shadows: bool = False,
    hazard_damage: str = "",
    hazard_damage_type: str = "",
    dark: tuple[int, colors.Color, colors.Color],  # (char_code, fg_color, bg_color)
    light: tuple[int, colors.Color, colors.Color] | None = None,
    animation: tuple[tuple[int, int, int], tuple[int, int, int], float]
    | None = None,  # (fg_variation, bg_variation, glyph_flicker_chance)
) -> np.ndarray:  # Returns an instance of TileTypeData
    """
    Helper function to create a TileTypeData instance.

    Args:
        walkable: Can actors walk through this type of tile?
        transparent: Is this tile see-through for FOV/targeting?
        display_name: Human-readable name for UI display (e.g., "Wall", "Closed Door")
        material: Impact material for sound effects (default: STONE).
        cover_bonus: Defensive bonus granted when using this tile as cover.
        casts_shadows: Does this tile block light for shadow casting?
        hazard_damage: Dice string for damage (e.g., "1d4"). Empty string = safe.
        hazard_damage_type: Type of damage (e.g., "acid", "fire"). Empty string if safe.
        dark:  A (char_code, fg_color, bg_color) tuple defining
               the 'dark' TileTypeAppearance.
        light: A (char_code, fg_color, bg_color) tuple defining
               the 'light' TileTypeAppearance. If None, use the same
               tuple as 'dark'.
        animation: Optional animation parameters as a tuple of
                   (fg_variation, bg_variation, glyph_flicker_chance).
                   fg_variation and bg_variation are RGB tuples specifying
                   the range of color oscillation. glyph_flicker_chance is
                   the probability (0.0-1.0) of hiding the glyph per frame.
                   If None, the tile does not animate.

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

    # Create animation params (defaults to no animation)
    if animation is None:
        animation_params = np.array(
            (False, (0, 0, 0), (0, 0, 0), 0.0), dtype=TileAnimationParams
        )
    else:
        fg_var, bg_var, flicker = animation
        animation_params = np.array(
            (True, fg_var, bg_var, flicker), dtype=TileAnimationParams
        )

    return np.array(
        (
            walkable,
            transparent,
            cover_bonus,
            casts_shadows,
            material.value,
            hazard_damage,
            hazard_damage_type,
            display_name,
            dark_appearance,
            light_appearance,
            animation_params,
        ),
        dtype=TileTypeData,
    )


# --- Register Core Tile Types ---
# Each TileTypeID enum value must have its data registered here.
# WALL is registered first (ID 0) as it's the default fill value in map creation.

register_tile_type(
    TileTypeID.WALL,
    make_tile_type_data(
        walkable=False,
        transparent=False,
        display_name="Wall",
        material=ImpactMaterial.STONE,
        dark=(ord(" "), colors.DARK_GREY, colors.DARK_WALL),
        light=(ord(" "), colors.LIGHT_GREY, colors.LIGHT_WALL),
    ),
)

register_tile_type(
    TileTypeID.FLOOR,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Floor",
        material=ImpactMaterial.STONE,
        dark=(ord(" "), colors.DARK_GREY, colors.DARK_GROUND),
        light=(ord(" "), colors.LIGHT_GREY, colors.LIGHT_GROUND),
    ),
)

register_tile_type(
    TileTypeID.OUTDOOR_FLOOR,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Ground",
        material=ImpactMaterial.STONE,
        dark=(ord(" "), colors.DARK_GREY, colors.OUTDOOR_DARK_GROUND),
        light=(ord(" "), colors.LIGHT_GREY, colors.OUTDOOR_LIGHT_GROUND),
    ),
)

register_tile_type(
    TileTypeID.OUTDOOR_WALL,
    make_tile_type_data(
        walkable=False,
        transparent=False,
        display_name="Rock Wall",
        material=ImpactMaterial.STONE,
        casts_shadows=True,
        dark=(ord("#"), colors.LIGHT_GREY, colors.OUTDOOR_DARK_WALL),
        light=(ord("#"), colors.LIGHT_GREY, colors.OUTDOOR_LIGHT_WALL),
    ),
)

register_tile_type(
    TileTypeID.DOOR_CLOSED,
    make_tile_type_data(
        walkable=False,
        transparent=False,
        display_name="Closed Door",
        material=ImpactMaterial.WOOD,
        dark=(ord("+"), colors.ORANGE, colors.DARK_WALL),
        light=(ord("+"), colors.LIGHT_ORANGE, colors.LIGHT_WALL),
    ),
)

register_tile_type(
    TileTypeID.DOOR_OPEN,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Open Door",
        material=ImpactMaterial.WOOD,
        dark=(ord("'"), colors.ORANGE, colors.DARK_GROUND),
        light=(ord("'"), colors.LIGHT_ORANGE, colors.LIGHT_GROUND),
    ),
)

register_tile_type(
    TileTypeID.BOULDER,
    make_tile_type_data(
        walkable=False,
        transparent=True,
        display_name="Boulder",
        material=ImpactMaterial.STONE,
        cover_bonus=2,
        casts_shadows=True,
        dark=(ord("#"), colors.DARK_GREY, colors.DARK_GROUND),
        light=(ord("#"), colors.LIGHT_GREY, colors.LIGHT_GROUND),
    ),
)

# --- Hazardous Terrain Types ---
# These tiles deal damage to actors who end their turn standing on them.

register_tile_type(
    TileTypeID.ACID_POOL,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Acid Pool",
        material=ImpactMaterial.FLESH,  # Splashy sound on impact
        hazard_damage="1d4",
        hazard_damage_type="acid",
        dark=(ord("~"), (0, 40, 0), (0, 40, 0)),
        light=(ord("~"), (50, 140, 50), (50, 140, 50)),
        animation=((60, 100, 60), (60, 100, 60), 0.0),
    ),
)

register_tile_type(
    TileTypeID.HOT_COALS,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Hot Coals",
        material=ImpactMaterial.STONE,
        hazard_damage="1d6",
        hazard_damage_type="fire",
        dark=(ord("."), (40, 10, 0), (40, 10, 0)),
        light=(ord("."), (200, 100, 40), (200, 100, 40)),
        animation=((100, 100, 60), (100, 100, 60), 0.0),
    ),
)


# --- Pre-calculated Property Arrays for Efficient Lookups ---
# These arrays are built *after* all tile types have been registered.
# They allow for fast, vectorized conversion from a map of TileTypeIDs
# to a map of a specific property (e.g., walkability).
# The list is built in enum order so TileTypeID values can be used as indices directly.

# Build indexed list from enum order for numpy array indexing
_registered_tile_type_data_list = [_tile_type_definitions[tid] for tid in TileTypeID]

# Also maintain name-to-id map for backwards compatibility with string lookups
_tile_type_name_to_id_map = {tid.name: tid.value for tid in TileTypeID}

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
_tile_type_properties_hazard_damage = np.array(
    [t["hazard_damage"] for t in _registered_tile_type_data_list], dtype="U8"
)
_tile_type_properties_hazard_damage_type = np.array(
    [t["hazard_damage_type"] for t in _registered_tile_type_data_list], dtype="U16"
)
_tile_type_properties_animation = np.array(
    [t["animation"] for t in _registered_tile_type_data_list], dtype=TileAnimationParams
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


def get_tile_material(tile_type_id: int) -> ImpactMaterial:
    """
    Get the impact material for a tile type.

    Used by the audio system to play appropriate impact sounds when
    projectiles hit environmental surfaces.

    Args:
        tile_type_id: The ID of the tile type

    Returns:
        The ImpactMaterial enum value for this tile type.
        Defaults to STONE for unknown tile IDs.
    """
    if 0 <= tile_type_id < len(_registered_tile_type_data_list):
        material_value = int(_registered_tile_type_data_list[tile_type_id]["material"])
        return ImpactMaterial(material_value)
    return ImpactMaterial.STONE


def get_tile_hazard_info(tile_type_id: int) -> tuple[str, str]:
    """
    Get the hazard damage dice and type for a tile type.

    Used by the turn manager to apply environmental damage to actors
    who end their turn on hazardous terrain.

    Args:
        tile_type_id: The ID of the tile type

    Returns:
        A tuple of (damage_dice, damage_type). Returns ("", "") for safe tiles.
        damage_dice is a dice string like "1d4" or "2d6+1".
    """
    if 0 <= tile_type_id < len(_tile_type_properties_hazard_damage):
        damage_dice = str(_tile_type_properties_hazard_damage[tile_type_id])
        damage_type = str(_tile_type_properties_hazard_damage_type[tile_type_id])
        return (damage_dice, damage_type)
    return ("", "")


def get_animation_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """
    Converts a map of TileTypeIDs into a map of TileAnimationParams structs.

    Used by the rendering system to determine which tiles animate and
    how their colors should oscillate.

    Args:
        tile_type_ids_map: A 2D numpy array of TileTypeID values.

    Returns:
        A 2D numpy array of TileAnimationParams structs matching the input shape.
    """
    return _tile_type_properties_animation[tile_type_ids_map]
