from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from brileta.environment.map import GameMap
    from brileta.game.actors import Actor
    from brileta.util.spatial import SpatialIndex

from brileta.game.enums import StepBlock
from brileta.types import WorldTilePos

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld

# Native pathfinding is required at runtime.
try:
    from brileta.util._native import astar as _c_astar
except ImportError as exc:  # pragma: no cover - fails fast by design
    raise ImportError(
        "brileta.util._native is required. "
        "Build native extensions with `make` (or `uv pip install -e .`)."
    ) from exc


def probe_step(
    game_map: GameMap,
    game_world: GameWorld,
    x: int,
    y: int,
    *,
    exclude_actor: Actor | None = None,
    can_open_doors: bool = False,
) -> StepBlock | None:
    """Check whether a single tile is passable and return the reason if not.

    This is the authoritative definition of "what blocks a tile" for all
    per-tile movement checks (AI planning, movement execution, stunt
    destination validation, etc.).  Adding a new blocking condition here
    automatically propagates to every call site.

    The check order matches MoveExecutor so the probe and the executor
    always agree: bounds -> door -> walkable -> actor occupancy.

    Args:
        game_map: The GameMap for bounds, tile type, and walkability.
        game_world: The GameWorld for actor occupancy lookups.
        x: Tile X coordinate to test.
        y: Tile Y coordinate to test.
        exclude_actor: An actor to ignore for occupancy (typically the
            moving actor itself, who is allowed to "stand on" the tile).
        can_open_doors: If True, treat closed doors as passable. The
            actual door-opening is handled downstream by ActionRouter
            when the move intent bumps the door.

    Returns:
        ``None`` if the tile is passable, or a :class:`StepBlock` value
        explaining why the tile is blocked.
    """
    from brileta.environment.tile_types import TileTypeID
    from brileta.game.actors.container import Container

    # 1. Bounds
    if not (0 <= x < game_map.width and 0 <= y < game_map.height):
        return StepBlock.OUT_OF_BOUNDS

    # 2. Closed door (checked before walkable because doors are non-walkable
    #    tiles that require special handling - opening, not rerouting).
    #    Door-capable actors treat closed doors as passable; the actual
    #    opening happens via ActionRouter's bump-to-open fallback.
    is_closed_door = game_map.tiles[x, y] == TileTypeID.DOOR_CLOSED
    if is_closed_door and not can_open_doors:
        return StepBlock.CLOSED_DOOR

    # 3. Static walkability (walls, deep water, etc.).
    # Closed doors are non-walkable but already handled above for
    # door-capable actors, so skip them here.
    if not game_map.walkable[x, y] and not is_closed_door:
        return StepBlock.WALL

    # 4. Dynamic actor occupancy
    blocker = game_world.get_actor_at_location(x, y)
    if blocker is not None and blocker is not exclude_actor and blocker.blocks_movement:
        if isinstance(blocker, Container):
            return StepBlock.BLOCKED_BY_CONTAINER
        return StepBlock.BLOCKED_BY_ACTOR

    return None


def find_closest_adjacent_tile(
    target_x: int,
    target_y: int,
    reference_x: int,
    reference_y: int,
    game_map: GameMap,
    game_world: GameWorld,
    reference_actor: Actor | None = None,
) -> WorldTilePos | None:
    """Find the closest walkable, unoccupied tile adjacent to a target position.

    This is used when an actor needs to interact with something they can't
    stand on (e.g., a container, closed door). It finds the best cardinal-
    adjacent tile to approach from, prioritizing tiles closest to the
    reference position.

    Args:
        target_x: X coordinate of the target (e.g., container position).
        target_y: Y coordinate of the target.
        reference_x: X coordinate to measure distance from (e.g., player position).
        reference_y: Y coordinate to measure distance from.
        game_map: The GameMap for bounds and walkability checks.
        game_world: The GameWorld for actor occupancy checks.
        reference_actor: If provided, this actor is allowed to occupy an
            adjacent tile (used when the reference is an actor that might
            already be standing on a valid adjacent tile).

    Returns:
        The (x, y) position of the closest valid adjacent tile, or None if
        no valid adjacent tile exists.
    """
    # Cardinal adjacent positions, sorted by Manhattan distance from reference
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    adjacent = [(target_x + dx, target_y + dy) for dx, dy in offsets]
    adjacent.sort(key=lambda p: abs(p[0] - reference_x) + abs(p[1] - reference_y))

    for adj_x, adj_y in adjacent:
        if (
            probe_step(
                game_map, game_world, adj_x, adj_y, exclude_actor=reference_actor
            )
            is not None
        ):
            continue
        return (adj_x, adj_y)

    return None


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
    *,
    can_open_doors: bool = False,
) -> list[WorldTilePos]:
    """
    Calculate a path from a start to an end position using A*.

    This is the low-level (local) component of the pathfinding system.
    For cross-region paths, use find_region_path() to get the high-level
    region sequence, then use find_local_path() to pathfind within or between
    adjacent regions.

    This pathfinder is aware of static unwalkable map tiles, blocking
    actors (NPCs, furniture, etc.), and environmental hazards. It generates
    a temporary cost map for each request. All blocking actors are treated
    as impassable obstacles during path planning.

    Hazardous tiles (acid pools, hot coals, fire actors) are assigned higher
    costs so the pathfinder prefers to route around them when alternatives
    exist, but will still traverse them if no other path is available.

    When ``can_open_doors`` is True, closed doors are treated as passable
    with extra cost (DOOR_TRAVERSAL_COST) instead of being impassable.
    This lets door-capable NPCs plan routes through closed doors while
    still preferring open paths when they exist.

    Args:
        game_map: The GameMap instance, used for static terrain checks.
        actor_spatial_index: The spatial index, used for dynamic obstacle checks.
        pathing_actor: The actor requesting the path. This actor is ignored as an
            obstacle.
        start_pos: The (x, y) starting coordinate for the path.
        end_pos: The (x, y) target coordinate for the path.
        can_open_doors: If True, treat closed doors as high-cost passable
            tiles rather than walls. Defaults to False.

    Returns:
        A list of (x, y) tuples representing the path from start to end.
        The list does not include the start point. Returns an empty list
        if no path is found.
    """
    from brileta.environment.tile_types import (
        DOOR_TRAVERSAL_COST,
        HAZARD_BASE_COST,
        TileTypeID,
        get_hazard_cost_map,
    )

    # Build cost array: start with hazard costs for all tiles, then mask
    # non-walkable tiles to 0. This ensures hazards anywhere on the map
    # are considered, not just within the start-end bounding box.
    hazard_costs = get_hazard_cost_map(game_map.tiles)
    cost = np.where(game_map.walkable, hazard_costs, 0).astype(np.int16)

    # Door-capable actors can pathfind through closed doors at extra cost.
    # Doors are non-walkable (cost 0 after the mask above), so we overlay
    # them with DOOR_TRAVERSAL_COST to make them passable but expensive.
    if can_open_doors:
        door_mask = game_map.tiles == TileTypeID.DOOR_CLOSED
        cost[door_mask] = DOOR_TRAVERSAL_COST

    # Bounding box for spatial index queries (actors near the path)
    x1 = max(0, min(start_pos[0], end_pos[0]))
    y1 = max(0, min(start_pos[1], end_pos[1]))
    x2 = min(game_map.width - 1, max(start_pos[0], end_pos[0]))
    y2 = min(game_map.height - 1, max(start_pos[1], end_pos[1]))

    nearby_actors = actor_spatial_index.get_in_bounds(x1, y1, x2, y2)

    for actor in nearby_actors:
        # All blocking actors are treated as impassable obstacles.
        #
        # NOTE: Do NOT "optimize" this to ignore mobile actors. We tried that
        # (reasoning: "actors move, so pre-computed avoidance is unreliable").
        # It broke pathfinding because Character.get_next_action() validates
        # each step using find_local_path - if this function ignores blocking
        # actors, validation always passes, so rerouting never triggers when
        # an actor blocks the path. Treat all blockers as obstacles here; the
        # validation mechanism handles dynamic rerouting when actors move.
        if actor.blocks_movement and actor is not pathing_actor:
            cost[actor.x, actor.y] = 0

        damage_per_turn = getattr(actor, "damage_per_turn", 0)
        if damage_per_turn > 0 and cost[actor.x, actor.y] > 0:
            # Fire hazards (campfires, barrel fires, torches) are high-cost
            fire_cost = HAZARD_BASE_COST + damage_per_turn
            cost[actor.x, actor.y] = max(cost[actor.x, actor.y], fire_cost)

    path: list[WorldTilePos] = _astar(cost, start_pos, end_pos)
    return path


def _astar(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[WorldTilePos]:
    """A* pathfinding on a 2D cost grid with diagonal movement.

    Diagonal moves cost sqrt(2) times the destination cell weight, while
    cardinal moves cost 1x.  This prevents the zigzag/parabola paths that
    occur when diagonals are free.

    Args:
        cost: Array of shape (width, height) indexed as cost[x, y].
              0 means impassable, positive values are traversal weights.
        start: (x, y) start position.
        goal: (x, y) goal position.

    Returns:
        List of (x, y) positions from start to goal, excluding start.
        Empty list if no path exists or start == goal.
    """
    # The C extension needs a C-contiguous int16 array.
    c_cost = np.ascontiguousarray(cost, dtype=np.int16)
    return _c_astar(c_cost, start[0], start[1], goal[0], goal[1])
