import tcod

from catley.environment.map import GameMap
from catley.game.items.item_core import Item
from catley.game.items.item_types import WeaponProperty


def get_range_category(distance: int, weapon: Item) -> str:
    """Determine range category based on distance and weapon"""
    ranged_attack = weapon.ranged_attack
    if ranged_attack is None:
        return "melee_only"

    if distance == 1:
        return "adjacent"
    if distance <= ranged_attack.optimal_range:
        return "close"
    if distance <= ranged_attack.max_range:
        return "far"
    return "out_of_range"


def get_range_modifier(weapon: Item, range_category: str) -> dict | None:
    """Get attack modifiers based on weapon and range"""
    ranged_attack = weapon.ranged_attack
    if ranged_attack is None:
        return None

    modifiers = {
        "adjacent": {"has_advantage": True},  # Point blank
        "close": {},  # Optimal range
        "far": {"has_disadvantage": True},  # Long shot
        "out_of_range": None,  # Impossible
    }

    # Special weapon properties can modify this
    if WeaponProperty.SCOPED in ranged_attack.properties:
        # Scoped weapons get advantage at far range instead of disadvantage
        modifiers["far"] = {"has_advantage": True}

    return modifiers.get(range_category)


def has_line_of_sight(
    game_map: GameMap, start_x: int, start_y: int, end_x: int, end_y: int
) -> bool:
    """Check if there's clear line of sight using TCOD"""
    # Get line points using optimized Bresenham
    line_points = get_line(start_x, start_y, end_x, end_y)

    # Check all points except start and end for transparency
    return all(game_map.transparent[x, y] for x, y in line_points[1:-1])


def get_line(
    start_x: int, start_y: int, end_x: int, end_y: int
) -> list[tuple[int, int]]:
    """Return Bresenham line points from (start_x, start_y) to (end_x, end_y)."""

    return [tuple(pt) for pt in tcod.los.bresenham((start_x, start_y), (end_x, end_y))]


def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate Chebyshev distance between two points."""

    # Chebyshev distance properly handles diagonal adjacency by returning 1
    # when moving one step diagonally.  Manhattan distance would return 2 in
    # that case which caused melee attacks to be skipped when standing
    # diagonally adjacent to a target.
    return max(abs(x2 - x1), abs(y2 - y1))
