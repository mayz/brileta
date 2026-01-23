from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import tcod.path

if TYPE_CHECKING:
    from catley.environment.map import GameMap
    from catley.game.actors import Actor
    from catley.util.spatial import SpatialIndex

from catley.types import WorldTilePos


def find_region_path(
    game_map: GameMap,
    start_region_id: int,
    end_region_id: int,
) -> list[int] | None:
    """
    Find a path through the region graph using BFS.

    This is the high-level component of hierarchical pathfinding (HPA*).
    It finds a sequence of regions to traverse to get from the start region
    to the end region.

    Args:
        game_map: The GameMap instance containing the regions graph.
        start_region_id: The ID of the starting region.
        end_region_id: The ID of the destination region.

    Returns:
        A list of region IDs representing the path from start to end,
        including both the start and end regions. Returns None if no
        path exists (regions are not connected).
    """
    if start_region_id == end_region_id:
        return [start_region_id]

    if start_region_id not in game_map.regions:
        return None
    if end_region_id not in game_map.regions:
        return None

    # BFS on the region graph
    queue: deque[int] = deque([start_region_id])
    came_from: dict[int, int] = {start_region_id: -1}

    while queue:
        current = queue.popleft()

        if current == end_region_id:
            # Reconstruct path
            path: list[int] = []
            node = current
            while node != -1:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        current_region = game_map.regions.get(current)
        if current_region is None:
            continue

        for neighbor_id in current_region.connections:
            if neighbor_id not in came_from:
                came_from[neighbor_id] = current
                queue.append(neighbor_id)

    return None  # No path found


def find_local_path(
    game_map: GameMap,
    actor_spatial_index: SpatialIndex[Actor],
    pathing_actor: Actor,
    start_pos: WorldTilePos,
    end_pos: WorldTilePos,
) -> list[WorldTilePos]:
    """
    Calculate a path from a start to an end position using A*.

    This is the low-level (local) component of the pathfinding system.
    For cross-region paths, use find_region_path() to get the high-level
    region sequence, then use find_local_path() to pathfind within or between
    adjacent regions.

    This pathfinder is aware of static unwalkable map tiles and
    environmental hazards. It generates a temporary cost map for each
    request. Blocking actors are intentionally ignored here - collisions
    with moving actors are handled at movement execution time via the
    collision detection system.

    Hazardous tiles (acid pools, hot coals, fire actors) are assigned higher
    costs so the pathfinder prefers to route around them when alternatives
    exist, but will still traverse them if no other path is available.

    Args:
        game_map: The GameMap instance, used for static terrain checks.
        actor_spatial_index: The spatial index, used for dynamic obstacle checks.
        pathing_actor: The actor requesting the path. This actor is ignored as an
            obstacle.
        start_pos: The (x, y) starting coordinate for the path.
        end_pos: The (x, y) target coordinate for the path.

    Returns:
        A list of (x, y) tuples representing the path from start to end.
        The list does not include the start point. Returns an empty list
        if no path is found.
    """
    from catley.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost_map

    # Build cost array: start with hazard costs for all tiles, then mask
    # non-walkable tiles to 0. This ensures hazards anywhere on the map
    # are considered, not just within the start-end bounding box.
    hazard_costs = get_hazard_cost_map(game_map.tiles)
    cost = np.where(game_map.walkable, hazard_costs, 0).astype(np.int16)

    # Bounding box for spatial index queries (actors near the path)
    x1 = max(0, min(start_pos[0], end_pos[0]))
    y1 = max(0, min(start_pos[1], end_pos[1]))
    x2 = min(game_map.width - 1, max(start_pos[0], end_pos[0]))
    y2 = min(game_map.height - 1, max(start_pos[1], end_pos[1]))

    nearby_actors = actor_spatial_index.get_in_bounds(x1, y1, x2, y2)

    for actor in nearby_actors:
        if (
            hasattr(actor, "damage_per_turn")
            and actor.damage_per_turn > 0
            and cost[actor.x, actor.y] > 0
        ):
            # Fire hazards (campfires, barrel fires, torches) are high-cost
            fire_cost = HAZARD_BASE_COST + actor.damage_per_turn
            cost[actor.x, actor.y] = max(cost[actor.x, actor.y], fire_cost)

    astar = tcod.path.AStar(cost=cost, diagonal=1)

    path: list[WorldTilePos] = astar.get_path(
        start_pos[0], start_pos[1], end_pos[0], end_pos[1]
    )

    return path
