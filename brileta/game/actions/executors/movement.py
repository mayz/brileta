from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import config
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.enums import StepBlock
from brileta.util.pathfinding import probe_step

if TYPE_CHECKING:
    from brileta.game.actions.movement import MoveIntent


class MoveExecutor(ActionExecutor):
    """Executes movement intents by reporting movement results and collisions."""

    def execute(self, intent: MoveIntent) -> GameActionResult | None:  # type: ignore[override]
        game_map = intent.controller.gw.game_map
        game_world = intent.controller.gw

        block = probe_step(game_map, game_world, intent.newx, intent.newy)
        if block is not None:
            return self._blocked_result(block, intent)

        # Use intent's duration, or fall back to autopilot default.
        duration_ms = (
            intent.duration_ms
            if intent.duration_ms is not None
            else config.AUTOPILOT_MOVE_DURATION_MS
        )

        # Success! Move the actor with the specified animation duration.
        intent.actor.move(
            intent.dx, intent.dy, intent.controller, duration=duration_ms / 1000.0
        )

        # Plan advancement is handled by TurnManager (player) and
        # process_all_npc_reactions (NPC) after action execution.

        # Return duration_ms as duration_ms so TurnManager waits
        # for the animation to complete before processing the next action.
        return GameActionResult(
            succeeded=True, should_update_fov=True, duration_ms=duration_ms
        )

    @staticmethod
    def _blocked_result(block: StepBlock, intent: MoveIntent) -> GameActionResult:
        """Convert a StepBlock into the appropriate failed GameActionResult."""
        match block:
            case StepBlock.OUT_OF_BOUNDS:
                return GameActionResult(succeeded=False, block_reason="out_of_bounds")
            case StepBlock.CLOSED_DOOR:
                return GameActionResult(
                    succeeded=False,
                    blocked_by=(intent.newx, intent.newy),
                    block_reason="door",
                )
            case StepBlock.WALL:
                return GameActionResult(succeeded=False, block_reason="wall")
            case StepBlock.BLOCKED_BY_CONTAINER:
                blocker = intent.controller.gw.get_actor_at_location(
                    intent.newx, intent.newy
                )
                return GameActionResult(
                    succeeded=False, blocked_by=blocker, block_reason="container"
                )
            case StepBlock.BLOCKED_BY_ACTOR:
                blocker = intent.controller.gw.get_actor_at_location(
                    intent.newx, intent.newy
                )
                return GameActionResult(
                    succeeded=False, blocked_by=blocker, block_reason="actor"
                )
            case _:  # Defensive: treat unknown blocks as impassable
                return GameActionResult(succeeded=False, block_reason="blocked")
