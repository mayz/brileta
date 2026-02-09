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
from typing import NamedTuple

import numpy as np

from catley import colors
from catley.game.enums import ImpactMaterial
from catley.util.dice import Dice

# --- Hazard Cost Constants ---
# Used by pathfinding to make AI avoid hazardous tiles.
# Cost formula: HAZARD_BASE_COST + (avg_damage * HAZARD_DAMAGE_SCALING)
HAZARD_BASE_COST = 5
HAZARD_DAMAGE_SCALING = 2


class TileTypeID(IntEnum):
    """
    All tile type IDs. To add a new tile type:
    1. Add it to this enum
    2. Register its data with register_tile_type() below
    """

    WALL = 0
    FLOOR = auto()
    COBBLESTONE = auto()
    DOOR_CLOSED = auto()
    DOOR_OPEN = auto()
    BOULDER = auto()
    # Outdoor terrain variety
    GRASS = auto()
    DIRT_PATH = auto()
    GRAVEL = auto()
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

# RGB drift range for animated tile color oscillation.
# Each component specifies the maximum per-channel variation (R, G, B).
DriftRange = tuple[int, int, int]


class TileAnimationConfig(NamedTuple):
    """Configuration for tile color/glyph animation effects.

    Replaces the raw tuple parameter in make_tile_type_data() with named fields
    for clarity. Unpacks identically to the old (fg_drift, bg_drift, flicker) tuple.
    """

    fg_drift_range: DriftRange
    bg_drift_range: DriftRange
    glyph_flicker_chance: float


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

# Light emission parameters for tiles that glow (e.g., lava, acid pools, hot coals).
# These tiles contribute light to the lightmap, illuminating surrounding tiles.
TileEmissionParams = np.dtype(
    [
        ("emits_light", np.bool_),  # Does this tile emit light?
        ("light_color", "3B"),  # RGB emission color (0-255 each)
        ("light_radius", np.uint8),  # Light radius in tiles (typically 2-4)
        ("light_intensity", np.float32),  # Emission strength (0.0-1.0)
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
        (
            "shadow_height",
            np.uint8,
        ),  # Shadow height (0 = no shadow, 1+ = tiles of shadow)
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
        # Light emission parameters (glowing tiles like lava, acid, coals)
        ("emission", TileEmissionParams),
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
    shadow_height: int = 0,
    hazard_damage: str = "",
    hazard_damage_type: str = "",
    dark: tuple[int, colors.Color, colors.Color],  # (char_code, fg_color, bg_color)
    light: tuple[int, colors.Color, colors.Color] | None = None,
    animation: TileAnimationConfig | None = None,
    emission: tuple[colors.Color, int, float]
    | None = None,  # (light_color, light_radius, light_intensity)
) -> np.ndarray:  # Returns an instance of TileTypeData
    """
    Helper function to create a TileTypeData instance.

    Args:
        walkable: Can actors walk through this type of tile?
        transparent: Is this tile see-through for FOV/targeting?
        display_name: Human-readable name for UI display (e.g., "Wall", "Closed Door")
        material: Impact material for sound effects (default: STONE).
        cover_bonus: Defensive bonus granted when using this tile as cover.
        shadow_height: Height for shadow casting (0 = no shadow, higher = longer shadow).
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
        emission: Optional light emission parameters as a tuple of
                  (light_color, light_radius, light_intensity).
                  light_color is an RGB tuple for the emitted light color.
                  light_radius is the radius in tiles (typically 2-4).
                  light_intensity is the emission strength (0.0-1.0).
                  If None, the tile does not emit light.

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

    # Create emission params (defaults to no emission)
    if emission is None:
        emission_params = np.array((False, (0, 0, 0), 0, 0.0), dtype=TileEmissionParams)
    else:
        light_color, light_radius, light_intensity = emission
        emission_params = np.array(
            (True, light_color, light_radius, light_intensity), dtype=TileEmissionParams
        )

    return np.array(
        (
            walkable,
            transparent,
            cover_bonus,
            shadow_height,
            material.value,
            hazard_damage,
            hazard_damage_type,
            display_name,
            dark_appearance,
            light_appearance,
            animation_params,
            emission_params,
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
        shadow_height=4,
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
    TileTypeID.COBBLESTONE,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Cobblestone",
        material=ImpactMaterial.STONE,
        dark=(ord(" "), (54, 40, 27), colors.OUTDOOR_DARK_GROUND),  # FG for pool glyphs
        light=(ord(" "), (113, 93, 64), colors.OUTDOOR_LIGHT_GROUND),
    ),
)

register_tile_type(
    TileTypeID.DOOR_CLOSED,
    make_tile_type_data(
        walkable=False,
        transparent=False,
        display_name="Closed Door",
        material=ImpactMaterial.WOOD,
        shadow_height=4,  # Door is set in a wall - casts same shadow as building
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
        shadow_height=4,  # Doorframe and wall above still cast building shadow
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
        shadow_height=2,
        dark=(ord("#"), colors.DARK_GREY, colors.DARK_GROUND),
        light=(ord("#"), colors.LIGHT_GREY, colors.LIGHT_GROUND),
    ),
)

# --- Outdoor Terrain Variety ---
# These tiles provide visual variety for outdoor areas.
# All are walkable and transparent.

register_tile_type(
    TileTypeID.GRASS,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Grass",
        material=ImpactMaterial.FLESH,  # Soft impact sound
        dark=(
            ord(" "),
            (30, 50, 25),
            (30, 50, 25),
        ),  # FG ≈ BG; independent jitter creates variation
        light=(ord(" "), (70, 110, 50), (70, 110, 50)),
    ),
)

register_tile_type(
    TileTypeID.DIRT_PATH,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Dirt Path",
        material=ImpactMaterial.STONE,
        dark=(ord(" "), (40, 31, 22), (45, 35, 25)),  # FG for pool glyphs (subtle)
        light=(ord(" "), (93, 74, 49), (100, 80, 55)),
    ),
)

register_tile_type(
    TileTypeID.GRAVEL,
    make_tile_type_data(
        walkable=True,
        transparent=True,
        display_name="Gravel",
        material=ImpactMaterial.STONE,
        dark=(
            ord(" "),
            (50, 45, 40),
            (50, 45, 40),
        ),  # FG ≈ BG; independent jitter creates variation
        light=(ord(" "), (95, 90, 80), (95, 90, 80)),
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
        animation=TileAnimationConfig((60, 100, 60), (60, 100, 60), 0.0),
        emission=((80, 180, 80), 2, 0.5),  # Green glow, radius 2
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
        dark=(ord("."), (30, 8, 0), (30, 8, 0)),
        light=(ord("."), (150, 75, 30), (150, 75, 30)),
        animation=TileAnimationConfig((80, 60, 40), (80, 60, 40), 0.0),
        emission=((180, 100, 40), 2, 0.5),  # Orange/red glow, radius 2
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
_tile_type_properties_shadow_height = np.array(
    [t["shadow_height"] for t in _registered_tile_type_data_list], dtype=np.uint8
)
_tile_type_properties_emission = np.array(
    [t["emission"] for t in _registered_tile_type_data_list], dtype=TileEmissionParams
)


def _compute_hazard_cost(damage_dice: str) -> int:
    """Compute pathfinding cost from a hazard damage dice string.

    Returns 1 for safe tiles (empty dice string), higher values for hazards.
    """
    if not damage_dice:
        return 1

    avg_damage = Dice(damage_dice).average()
    return int(HAZARD_BASE_COST + avg_damage * HAZARD_DAMAGE_SCALING)


_tile_type_properties_hazard_cost = np.array(
    [
        _compute_hazard_cost(str(t["hazard_damage"]))
        for t in _registered_tile_type_data_list
    ],
    dtype=np.int16,
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


def get_hazard_cost_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """
    Converts a map of TileTypeIDs into a map of pathfinding hazard costs.

    Returns an int16 array where:
    - 1 = safe tile (normal movement cost)
    - >1 = hazardous tile (higher cost to discourage pathfinding through it)

    Used by the pathfinding system to make AI avoid hazardous terrain.
    """
    return _tile_type_properties_hazard_cost[tile_type_ids_map]


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


def get_hazard_cost(tile_type_id: int) -> int:
    """
    Get the pathfinding cost for a tile based on its hazard level.

    Used by the pathfinding system to make AI avoid hazardous tiles when
    possible. Safe tiles return 1 (normal cost), while hazardous tiles
    return higher values proportional to their danger level.

    Args:
        tile_type_id: The ID of the tile type.

    Returns:
        An integer cost value: 1 for safe tiles, higher for hazards.
    """
    if 0 <= tile_type_id < len(_tile_type_properties_hazard_cost):
        return int(_tile_type_properties_hazard_cost[tile_type_id])
    return 1


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


def get_shadow_height_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """
    Vectorized lookup for tile shadow heights.

    Converts a map of TileTypeIDs into a uint8 map where 0 means
    no shadow and higher values indicate taller shadow casters
    (longer shadows).

    Args:
        tile_type_ids_map: A 2D numpy array of TileTypeID values.

    Returns:
        A 2D uint8 numpy array matching the input shape.
    """
    return _tile_type_properties_shadow_height[tile_type_ids_map]


def get_emission_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """
    Converts a map of TileTypeIDs into a map of TileEmissionParams structs.

    Used by the lighting system to determine which tiles emit light and
    how they should contribute to the lightmap.

    Args:
        tile_type_ids_map: A 2D numpy array of TileTypeID values.

    Returns:
        A 2D numpy array of TileEmissionParams structs matching the input shape.
    """
    return _tile_type_properties_emission[tile_type_ids_map]


def get_sub_tile_jitter_map(tile_type_ids_map: np.ndarray) -> np.ndarray:
    """Converts a map of TileTypeIDs into a float32 map of sub-tile jitter amplitudes.

    Non-zero values tell the fragment shader to apply per-pixel brightness
    variation within each tile cell. Currently only COBBLESTONE uses this.

    Args:
        tile_type_ids_map: A 2D numpy array of TileTypeID values.

    Returns:
        A float32 numpy array matching the input shape.
    """
    return _tile_type_properties_sub_tile_jitter[tile_type_ids_map]


# --- Per-tile visual decoration (glyph pools + color jitter) ---
#
# Terrain tiles use glyph pools for visual variety: each tile position gets a
# deterministic glyph chosen from its terrain type's pool via spatial hashing.
# This is separate from TileTypeAppearance because variable-length lists can't
# be stored in fixed numpy dtypes. The pool overrides `ch` at render time,
# while `fg` from the registration is still used as the base glyph color.
#
# Each entry also specifies a brightness jitter amplitude. Jitter is applied
# equally to all RGB channels (brightness, not hue) for both fg and bg,
# keeping glyph contrast consistent while varying overall tile lightness.


class TerrainDecorationDef(NamedTuple):
    """Per-terrain-type visual decoration parameters.

    Controls which glyphs appear on terrain tiles and how their colors vary.

    Color variation works at two complementary scales:

    - **Tile-level** (bg_jitter / fg_jitter): each tile gets a slightly
      different brightness via spatial hash. Makes a wall look like it's built
      from individual stone blocks rather than a single painted surface.

    - **Sub-tile pixel-level** (sub_tile_jitter): the fragment shader applies
      per-pixel brightness noise within each tile cell. Adds fine grain or
      texture *within* a single block.

    Both shift brightness (all RGB channels equally) without changing hue.
    They can be used independently or together - bg_jitter for block-to-block
    variation, sub_tile_jitter for surface grain within each block.
    """

    glyphs: list[str]
    """Characters drawn on this terrain (chosen per-tile via spatial hash)."""

    bg_jitter: int = 0
    """Per-tile brightness jitter applied equally to fg and bg.
    Each tile's background shifts by a random offset in [-bg_jitter, +bg_jitter]
    (same offset to all RGB channels, so hue is preserved). Creates
    tile-to-tile color variation - e.g. each stone in a wall is a slightly
    different shade."""

    fg_jitter: int = 0
    """Independent per-tile brightness jitter applied only to fg.
    When > 0, uses separate hash bits so some glyphs end up lighter than bg
    and some darker (bidirectional contrast). When 0, fg is locked to the
    same offset as bg (unidirectional - preserves the base fg-bg gap)."""

    sub_tile_jitter: float = 0.0
    """Fragment shader noise amplitude for per-pixel brightness variation
    within each tile cell. Adds fine surface grain at a smaller scale than
    bg_jitter's tile-level variation."""


# Glyphs are chosen to be visually distinct between terrain types:
#   GRASS: tall organic marks (blades), bidirectional - some catch light, some in shadow
#   DIRT: minimal dots (packed earth), unidirectional darker
#   GRAVEL: chunky double marks (pebbles), bidirectional - varied pebble faces
#   COBBLESTONE: dense horizontal marks (fitted stone joints), unidirectional darker
#   WALL/FLOOR: no visible glyph, subtle bg_jitter for stone block variation
_TERRAIN_DECORATION_DEFS: dict[TileTypeID, TerrainDecorationDef] = {
    TileTypeID.WALL: TerrainDecorationDef(glyphs=[" "], sub_tile_jitter=0.012),
    TileTypeID.COBBLESTONE: TerrainDecorationDef(
        glyphs=[" "], bg_jitter=2, sub_tile_jitter=0.02
    ),
    TileTypeID.GRASS: TerrainDecorationDef(
        glyphs=['"', "'", ",", "`"], bg_jitter=4, fg_jitter=12
    ),
    TileTypeID.DIRT_PATH: TerrainDecorationDef(glyphs=[".", ","], bg_jitter=4),
    TileTypeID.GRAVEL: TerrainDecorationDef(
        glyphs=[":", ";"], bg_jitter=2, fg_jitter=8
    ),
}


class TerrainGlyphPool(NamedTuple):
    """Compiled (numpy) version of TerrainDecorationDef for vectorized render-time lookup."""

    glyph_codes: np.ndarray
    """Character codes as int32 numpy array for direct indexing."""

    bg_jitter: int
    fg_jitter: int
    sub_tile_jitter: float


TERRAIN_GLYPH_POOLS: dict[int, TerrainGlyphPool] = {
    tid: TerrainGlyphPool(
        np.array([ord(ch) for ch in dec.glyphs], dtype=np.int32),
        dec.bg_jitter,
        dec.fg_jitter,
        dec.sub_tile_jitter,
    )
    for tid, dec in _TERRAIN_DECORATION_DEFS.items()
}

# Sub-tile jitter amplitude per tile type, built from TerrainDecorationDef.
# Non-zero values cause the fragment shader to apply per-pixel brightness
# variation within each tile cell using a PCG hash.
_tile_type_properties_sub_tile_jitter = np.array(
    [
        _TERRAIN_DECORATION_DEFS[tid].sub_tile_jitter
        if tid in _TERRAIN_DECORATION_DEFS
        else 0.0
        for tid in TileTypeID
    ],
    dtype=np.float32,
)


def apply_terrain_decoration(
    chars: np.ndarray,
    fg_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    tile_ids: np.ndarray,
    world_x: np.ndarray,
    world_y: np.ndarray,
    decoration_seed: int,
) -> None:
    """Apply per-tile glyph selection and color jitter for terrain variety.

    Uses a deterministic spatial hash so the same world position always gets
    the same glyph and color offset, producing stable visuals without storing
    per-tile state.

    Args:
        chars: Output char codes, modified in-place. Shape: (N,).
        fg_rgb: Output fg colors, modified in-place. Shape: (N, 3).
        bg_rgb: Output bg colors, modified in-place. Shape: (N, 3).
        tile_ids: TileTypeID values for each tile. Shape: (N,).
        world_x: World x coordinates. Shape: (N,).
        world_y: World y coordinates. Shape: (N,).
        decoration_seed: Per-map seed for deterministic decoration.
    """
    # Spatial hash: deterministic per-tile value from world position and seed.
    # Uses large primes to minimize correlation between nearby tiles.
    h = (
        world_x.astype(np.int64) * 73856093
        ^ world_y.astype(np.int64) * 19349663
        ^ np.int64(decoration_seed)
    )

    for tid, glyph_pool in TERRAIN_GLYPH_POOLS.items():
        mask = tile_ids == tid
        if not np.any(mask):
            continue

        tile_hash = h[mask]

        # Select glyph from pool using hash (low bits).
        codes = glyph_pool.glyph_codes
        chars[mask] = codes[np.abs(tile_hash % len(codes)).astype(np.intp)]

        # BG brightness jitter: same offset to all RGB channels (shifts
        # lightness without shifting hue). Uses hash bits 8-15.
        bg_jitter_range = 2 * glyph_pool.bg_jitter + 1
        bg_brightness = (tile_hash >> 8) % bg_jitter_range - glyph_pool.bg_jitter
        bg_offset = np.column_stack([bg_brightness, bg_brightness, bg_brightness])

        bg_rgb[mask] = np.clip(
            bg_rgb[mask].astype(np.int16) + bg_offset, 0, 255
        ).astype(np.uint8)

        # FG jitter: when fg_jitter > 0, fg gets its own independent offset
        # from hash bits 16-23. This creates bidirectional contrast - some
        # glyphs lighter than bg, some darker. When 0, fg is locked to the
        # same offset as bg (preserving the base fg-bg gap).
        if glyph_pool.fg_jitter > 0:
            fg_jitter_range = 2 * glyph_pool.fg_jitter + 1
            fg_brightness = (tile_hash >> 16) % fg_jitter_range - glyph_pool.fg_jitter
            fg_offset = np.column_stack([fg_brightness, fg_brightness, fg_brightness])
        else:
            fg_offset = bg_offset

        fg_rgb[mask] = np.clip(
            fg_rgb[mask].astype(np.int16) + fg_offset, 0, 255
        ).astype(np.uint8)
