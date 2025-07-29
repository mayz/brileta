from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tcod.path

if TYPE_CHECKING:
    from catley.environment.map import GameMap
    from catley.game.actors import Actor
    from catley.util.spatial import SpatialIndex

from catley.types import WorldTilePos


def find_path(
    game_map: GameMap,
    actor_spatial_index: SpatialIndex[Actor],
    pathing_actor: Actor,
    start_pos: WorldTilePos,
    end_pos: WorldTilePos,
) -> list[WorldTilePos]:
    """
    Calculates a path from a start to an end position using A*.

    ################################################################################
    TODO: This implementation has scaling limitations for large maps.
          A more robust hierarchical pathfinding (HPA*) system is planned.
          See the "Pathfinding Scaling Strategy - Catley" note for the full spec.
    ################################################################################

    This pathfinder is aware of both static, unwalkable map tiles and
    dynamic, blocking actors. It generates a temporary cost map for each
    request to ensure the path is valid for the current game state.

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

    cost = np.array(game_map.walkable, dtype=np.int8)

    x1 = max(0, min(start_pos[0], end_pos[0]))
    y1 = max(0, min(start_pos[1], end_pos[1]))
    x2 = min(game_map.width - 1, max(start_pos[0], end_pos[0]))
    y2 = min(game_map.height - 1, max(start_pos[1], end_pos[1]))

    nearby_actors = actor_spatial_index.get_in_bounds(x1, y1, x2, y2)

    for actor in nearby_actors:
        if actor.blocks_movement and actor is not pathing_actor:
            cost[actor.x, actor.y] = 0

    astar = tcod.path.AStar(cost=cost, diagonal=1)

    path: list[WorldTilePos] = astar.get_path(
        start_pos[0], start_pos[1], end_pos[0], end_pos[1]
    )

    return path
