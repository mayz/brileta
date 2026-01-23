"""Default action lookup for right-click execution.

This module provides the mapping from target types to default actions,
enabling quick right-click execution of the "obvious" action for any
target. The design is intentionally simple - a direct lookup table
rather than a scoring matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.discovery.types import TargetType

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.base import GameIntent
    from catley.game.actors import Actor
    from catley.types import WorldTilePos


# Default action IDs by target type.
# These map to the action IDs used by the discovery system.
DEFAULT_ACTION_IDS: dict[TargetType, str] = {
    TargetType.NPC: "talk",  # Pathfind to NPC, then talk
    TargetType.CONTAINER: "search",  # Pathfind to container, then search
    TargetType.DOOR_CLOSED: "open",  # Pathfind adjacent, then open
    TargetType.DOOR_OPEN: "close",  # Pathfind adjacent, then close
    TargetType.ITEM_PILE: "pickup",  # Pathfind to items, then pick up
    TargetType.FLOOR: "walk",  # Pathfind to tile
}


def classify_target(
    controller: Controller,
    target: Actor | WorldTilePos | None,
) -> TargetType | None:
    """Determine the TargetType for a given target.

    Args:
        controller: The game controller for accessing game state.
        target: An Actor, a tile position (x, y), or None.

    Returns:
        The TargetType for the target, or None if it cannot be classified.
    """
    from catley.environment.tile_types import TileTypeID
    from catley.game.actors import Character
    from catley.game.actors.container import Container

    if target is None:
        return None

    gw = controller.gw
    gm = gw.game_map

    # Handle Actor targets
    if isinstance(target, Character):
        # Any character (player excluded) is an NPC
        if target is not gw.player:
            return TargetType.NPC
        return None

    if isinstance(target, Container):
        return TargetType.CONTAINER

    # Handle tile position targets (tuple of x, y)
    if isinstance(target, tuple) and len(target) == 2:
        x, y = target

        # Check bounds
        if not (0 <= x < gm.width and 0 <= y < gm.height):
            return None

        # Check for items at the tile
        items = gw.get_pickable_items_at_location(x, y)
        if items:
            return TargetType.ITEM_PILE

        # Check for actors at the tile
        actor_at_tile = gw.get_actor_at_location(x, y)
        if actor_at_tile is not None:
            if isinstance(actor_at_tile, Character) and actor_at_tile is not gw.player:
                return TargetType.NPC
            if isinstance(actor_at_tile, Container):
                return TargetType.CONTAINER

        # Check tile type
        tile_id = gm.tiles[x, y]
        if tile_id == TileTypeID.DOOR_CLOSED:
            return TargetType.DOOR_CLOSED
        if tile_id == TileTypeID.DOOR_OPEN:
            return TargetType.DOOR_OPEN

        # Check if it's a walkable floor tile
        if gm.walkable[x, y]:
            return TargetType.FLOOR

    return None


def get_default_action_id(target_type: TargetType) -> str | None:
    """Get the default action ID for a target type.

    Args:
        target_type: The classified target type.

    Returns:
        The action ID string, or None if no default exists.
    """
    return DEFAULT_ACTION_IDS.get(target_type)


def _pathfind_to_adjacent_and_execute(
    controller: Controller,
    target_x: int,
    target_y: int,
    intent: GameIntent,
) -> bool:
    """Pathfind to a tile adjacent to (target_x, target_y) and execute intent.

    Finds the first adjacent walkable tile (not a wall, not blocked by another
    actor) that has a valid path, then starts pathfinding with the intent to
    execute upon arrival.

    Args:
        controller: The game controller.
        target_x: X coordinate of the target tile.
        target_y: Y coordinate of the target tile.
        intent: The GameIntent to execute when the player arrives.

    Returns:
        True if pathfinding was started, False if no valid adjacent tile found.
    """
    from catley.environment.tile_types import TileTypeID
    from catley.util.pathfinding import find_local_path

    gw = controller.gw
    gm = gw.game_map
    player = gw.player

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        adj_x, adj_y = target_x + dx, target_y + dy
        if not (0 <= adj_x < gm.width and 0 <= adj_y < gm.height):
            continue
        if gm.tiles[adj_x, adj_y] == TileTypeID.WALL:
            continue

        blocker = gw.get_actor_at_location(adj_x, adj_y)
        if blocker is not None and blocker is not player:
            continue

        path = find_local_path(
            gm,
            gw.actor_spatial_index,
            player,
            (player.x, player.y),
            (adj_x, adj_y),
        )
        if path:
            return controller.start_actor_pathfinding(
                player, (adj_x, adj_y), final_intent=intent
            )

    return False


def execute_default_action(
    controller: Controller,
    target: Actor | WorldTilePos,
) -> bool:
    """Execute the default action for a target.

    This is the main entry point for right-click default action execution.
    It classifies the target, determines the default action, and executes it
    (including pathfinding if the target is not adjacent).

    In combat mode, NPCs are attacked rather than talked to.

    Args:
        controller: The game controller.
        target: The target to act on (Actor or tile position).

    Returns:
        True if an action was initiated, False otherwise.
    """
    from catley.game.actions.combat import AttackIntent
    from catley.game.actions.environment import (
        CloseDoorIntent,
        OpenDoorIntent,
        SearchContainerIntent,
    )
    from catley.game.actions.misc import PickupItemsAtLocationIntent
    from catley.game.actions.social import TalkIntent
    from catley.game.actors import Character
    from catley.game.actors.container import Container

    target_type = classify_target(controller, target)
    if target_type is None:
        return False

    player = controller.gw.player
    gw = controller.gw
    gm = gw.game_map

    # Get target position
    if isinstance(target, (Character, Container)):
        target_x, target_y = target.x, target.y
    elif isinstance(target, tuple):
        target_x, target_y = target
    else:
        return False

    # Calculate distance to target
    distance = max(abs(target_x - player.x), abs(target_y - player.y))

    # Handle each target type
    match target_type:
        case TargetType.NPC:
            if not isinstance(target, Character):
                return False

            # In combat mode, attack the NPC instead of talking
            if controller.is_combat_mode():
                intent = AttackIntent(controller, player, target)
                controller.queue_action(intent)
                return True

            # Not in combat - talk to NPC (pathfind to them if not adjacent)
            if distance == 1:
                # Adjacent - talk immediately
                intent = TalkIntent(controller, player, target)
                controller.queue_action(intent)
                return True
            # Not adjacent - pathfind then talk
            final_intent = TalkIntent(controller, player, target)
            return controller.start_actor_pathfinding(
                player, (target_x, target_y), final_intent=final_intent
            )

        case TargetType.CONTAINER:
            # Search container - pathfind to it if not adjacent
            container = target if isinstance(target, Container) else None
            if container is None:
                # Try to get container at tile position
                actors_at_tile = gw.actor_spatial_index.get_at_point(target_x, target_y)
                for actor in actors_at_tile:
                    if isinstance(actor, Container):
                        container = actor
                        break
            if container is None:
                return False

            if distance == 1:
                # Adjacent - search immediately
                search_intent = SearchContainerIntent(controller, player, container)
                controller.queue_action(search_intent)
                return True
            # Not adjacent - pathfind to adjacent tile then search
            final_intent = SearchContainerIntent(controller, player, container)
            # Find adjacent walkable tile
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                adj_x, adj_y = container.x + dx, container.y + dy
                if (
                    0 <= adj_x < gm.width
                    and 0 <= adj_y < gm.height
                    and gm.walkable[adj_x, adj_y]
                ):
                    blocker = gw.get_actor_at_location(adj_x, adj_y)
                    if blocker is None or blocker is player:
                        return controller.start_actor_pathfinding(
                            player, (adj_x, adj_y), final_intent=final_intent
                        )
            return False

        case TargetType.DOOR_CLOSED:
            # Open door - pathfind to adjacent tile if not adjacent
            open_intent = OpenDoorIntent(controller, player, target_x, target_y)
            if distance == 1:
                controller.queue_action(open_intent)
                return True
            return _pathfind_to_adjacent_and_execute(
                controller, target_x, target_y, open_intent
            )

        case TargetType.DOOR_OPEN:
            # Close door - pathfind to adjacent tile if not adjacent
            close_intent = CloseDoorIntent(controller, player, target_x, target_y)
            if distance == 1:
                controller.queue_action(close_intent)
                return True
            return _pathfind_to_adjacent_and_execute(
                controller, target_x, target_y, close_intent
            )

        case TargetType.ITEM_PILE:
            # Pick up items - pathfind to the tile if not at it
            if distance == 0:
                # Standing on items - pick up immediately
                pickup_intent = PickupItemsAtLocationIntent(controller, player)
                controller.queue_action(pickup_intent)
                return True
            # Not at items - pathfind then pick up
            final_intent = PickupItemsAtLocationIntent(controller, player)
            return controller.start_actor_pathfinding(
                player, (target_x, target_y), final_intent=final_intent
            )

        case TargetType.FLOOR:
            # Walk to tile
            if distance == 0:
                return False  # Already there
            return controller.start_actor_pathfinding(player, (target_x, target_y))

    return False
