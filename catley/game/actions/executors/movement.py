from __future__ import annotations

from typing import TYPE_CHECKING

from catley.environment.tile_types import TileTypeID
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor
from catley.game.actors.container import Container

if TYPE_CHECKING:
    from catley.game.actions.movement import MoveIntent


class MoveExecutor(ActionExecutor):
    """Executes movement intents by reporting movement results and collisions."""

    def execute(self, intent: MoveIntent) -> GameActionResult | None:  # type: ignore[override]
        game_map = intent.controller.gw.game_map

        # Check map boundaries first
        if not (
            0 <= intent.newx < game_map.width and 0 <= intent.newy < game_map.height
        ):
            return GameActionResult(succeeded=False, block_reason="out_of_bounds")

        # Check for doors
        tile_id = game_map.tiles[intent.newx, intent.newy]
        if tile_id == TileTypeID.DOOR_CLOSED:
            return GameActionResult(
                succeeded=False,
                blocked_by=(intent.newx, intent.newy),
                block_reason="door",
            )

        # Check if tile is walkable (walls, etc.)
        if not game_map.walkable[intent.newx, intent.newy]:
            return GameActionResult(succeeded=False, block_reason="wall")

        # Check for blocking actors
        blocking_actor = intent.controller.gw.get_actor_at_location(
            intent.newx, intent.newy
        )
        if blocking_actor and blocking_actor.blocks_movement:
            # Distinguish containers from other blocking actors
            if isinstance(blocking_actor, Container):
                return GameActionResult(
                    succeeded=False, blocked_by=blocking_actor, block_reason="container"
                )
            return GameActionResult(
                succeeded=False, blocked_by=blocking_actor, block_reason="actor"
            )

        # Success! Move the actor (this automatically creates animation)
        intent.actor.move(intent.dx, intent.dy, intent.controller)

        # Plan advancement is handled by TurnManager (player) and
        # process_all_npc_reactions (NPC) after action execution.

        # Movement uses 0ms presentation by default - the existing
        # AUTOPILOT_MOVE_INTERVAL and animation system handle pacing.
        # ApproachStep can pass custom timing if dramatic pacing is needed.
        presentation_ms = (
            intent.presentation_ms if intent.presentation_ms is not None else 0
        )

        return GameActionResult(
            succeeded=True, should_update_fov=True, presentation_ms=presentation_ms
        )
