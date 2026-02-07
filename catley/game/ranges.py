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

    # Scoped weapons are awkward at close range ("closer than distant")
    # but the scope compensates for distance at far range
    if WeaponProperty.SCOPED in ranged_attack.properties:
        modifiers["adjacent"] = {"has_disadvantage": True}
        modifiers["close"] = {"has_disadvantage": True}
        modifiers["far"] = {}  # Neutral instead of disadvantage

    return modifiers.get(range_category)


def has_line_of_sight(
    game_map: GameMap, start_x: int, start_y: int, end_x: int, end_y: int
) -> bool:
    """Check if there's clear line of sight between two points.

    Uses Bresenham's line algorithm to walk the grid between start and end,
    checking that all intermediate tiles are transparent.
    """
    line_points = get_line(start_x, start_y, end_x, end_y)

    # Check all points except start and end for transparency
    return all(game_map.transparent[x, y] for x, y in line_points[1:-1])


def get_line(
    start_x: int, start_y: int, end_x: int, end_y: int
) -> list[tuple[int, int]]:
    """Return Bresenham line points from (start_x, start_y) to (end_x, end_y).

    Walks the grid in unit steps along the major axis, accumulating error
    to decide when to step along the minor axis.
    """
    points: list[tuple[int, int]] = []
    dx = abs(end_x - start_x)
    dy = abs(end_y - start_y)
    sx = 1 if start_x < end_x else -1
    sy = 1 if start_y < end_y else -1
    err = dx - dy
    x, y = start_x, start_y

    while True:
        points.append((x, y))
        if x == end_x and y == end_y:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return points


def calculate_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate Chebyshev distance between two points."""

    # Chebyshev distance properly handles diagonal adjacency by returning 1
    # when moving one step diagonally.  Manhattan distance would return 2 in
    # that case which caused melee attacks to be skipped when standing
    # diagonally adjacent to a target.
    return max(abs(x2 - x1), abs(y2 - y1))
