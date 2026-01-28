"""Default action lookup for right-click execution.

This module provides the mapping from target types to default actions,
enabling quick right-click execution of the "obvious" action for any
target. The design is intentionally simple - a direct lookup table
rather than a scoring matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from catley.game.actions.discovery.types import TargetType
from catley.game.actors import Actor, ItemPile
from catley.types import WorldTilePos

if TYPE_CHECKING:
    from catley.controller import Controller


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

    if isinstance(target, ItemPile):
        return TargetType.ITEM_PILE

    # Handle tile position targets (tuple of x, y)
    if isinstance(target, tuple) and len(target) == 2:
        x, y = cast(WorldTilePos, target)

        # Check bounds
        if not (0 <= x < gm.width and 0 <= y < gm.height):
            return None

        # Check for items at the tile
        items = gw.get_pickable_items_at_location(x, y)
        if items:
            return TargetType.ITEM_PILE

        # Check for ItemPile with countables (may have no items)
        # Need to check all actors at tile since get_actor_at_location prioritizes
        # blocking actors (player) over non-blocking (ItemPile)
        actors_at_tile = gw.actor_spatial_index.get_at_point(x, y)
        for actor in actors_at_tile:
            if isinstance(actor, ItemPile):
                return TargetType.ITEM_PILE

        # Check for other actors at the tile
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
    from catley.game.action_plan import WalkToPlan
    from catley.game.actions.combat import AttackIntent
    from catley.game.actions.environment import (
        CloseDoorIntent,
        CloseDoorPlan,
        OpenDoorIntent,
        OpenDoorPlan,
        SearchContainerPlan,
    )
    from catley.game.actions.misc import PickupItemsAtLocationIntent, PickupItemsPlan
    from catley.game.actions.social import TalkPlan
    from catley.game.actors import Character
    from catley.game.actors.container import Container

    target_type = classify_target(controller, target)
    if target_type is None:
        return False

    player = controller.gw.player
    gw = controller.gw

    # Get target position
    if isinstance(target, (Character, Container, ItemPile)):
        target_x, target_y = target.x, target.y
    elif isinstance(target, tuple):
        target_x, target_y = cast(WorldTilePos, target)
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

            # Not in combat - talk to NPC (uses ActionPlan for approach)
            controller.start_plan(
                player,
                TalkPlan,
                target_actor=target,
                target_position=(target.x, target.y),
            )
            return True

        case TargetType.CONTAINER:
            # Search container - uses ActionPlan for approach
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

            controller.start_plan(
                player,
                SearchContainerPlan,
                target_actor=container,
                target_position=(container.x, container.y),
            )
            return True

        case TargetType.DOOR_CLOSED:
            # Open door - pathfind to adjacent tile if not adjacent
            if distance == 1:
                open_intent = OpenDoorIntent(controller, player, target_x, target_y)
                controller.queue_action(open_intent)
                return True
            return controller.start_plan(
                player, OpenDoorPlan, target_position=(target_x, target_y)
            )

        case TargetType.DOOR_OPEN:
            # Close door - pathfind to adjacent tile if not adjacent
            if distance == 1:
                close_intent = CloseDoorIntent(controller, player, target_x, target_y)
                controller.queue_action(close_intent)
                return True
            return controller.start_plan(
                player, CloseDoorPlan, target_position=(target_x, target_y)
            )

        case TargetType.ITEM_PILE:
            # Pick up items - pathfind to the tile if not at it
            if distance == 0:
                # Standing on items - pick up immediately
                pickup_intent = PickupItemsAtLocationIntent(controller, player)
                controller.queue_action(pickup_intent)
                return True
            # Not at items - use ActionPlan to pathfind then pick up
            return controller.start_plan(
                player, PickupItemsPlan, target_position=(target_x, target_y)
            )

        case TargetType.FLOOR:
            # Walk to tile
            if distance == 0:
                return False  # Already there
            return controller.start_plan(
                player, WalkToPlan, target_position=(target_x, target_y)
            )

    return False
