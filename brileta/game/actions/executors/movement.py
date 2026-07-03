from __future__ import annotations

from brileta import config
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.actions.movement import MoveIntent
from brileta.game.enums import ActionBlockReason, StepBlock
from brileta.util.pathfinding import probe_step


class MoveExecutor(ActionExecutor[MoveIntent]):
    """Executes movement intents by reporting movement results and collisions."""

    def execute(self, intent: MoveIntent) -> GameActionResult | None:
        game_map = intent.controller.gw.game_map
        game_world = intent.controller.gw

        block = probe_step(game_map, game_world, intent.newx, intent.newy)
        if block is not None:
            return self._blocked_result(block, intent)

        # Use intent's duration, or derive one from how the actor is paced.
        ease_power = config.DEFAULT_MOVE_EASE_POWER
        if intent.duration_ms is not None:
            duration_ms = intent.duration_ms
            ease_power = intent.ease_power
        elif (
            intent.actor is not game_world.player
            and intent.actor.energy is not None
            and not intent.controller.is_combat_mode()
        ):
            # Explore-mode NPC: size the glide to fill the gap until its next
            # ambient step, so consecutive steps chain into a continuous
            # linear stroll instead of a 100ms zip and a freeze. Undershoot
            # slightly (0.9) so a glide always finishes before the next
            # step's animation takes over the actor's render position.
            step_interval_s = intent.actor.energy.ambient_step_interval_s()
            duration_ms = int(step_interval_s * 0.9 * 1000)
            ease_power = 1.0
        else:
            duration_ms = config.AUTOPILOT_MOVE_DURATION_MS

        # Success! Move the actor with the specified animation duration.
        intent.actor.move(
            intent.dx,
            intent.dy,
            intent.controller,
            duration=duration_ms / 1000.0,
            ease_power=ease_power,
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
                return GameActionResult(
                    succeeded=False,
                    block_reason=ActionBlockReason.STEP_BLOCKED,
                    step_block=StepBlock.OUT_OF_BOUNDS,
                )
            case StepBlock.CLOSED_DOOR:
                return GameActionResult(
                    succeeded=False,
                    blocked_by=(intent.newx, intent.newy),
                    block_reason=ActionBlockReason.STEP_BLOCKED,
                    step_block=StepBlock.CLOSED_DOOR,
                )
            case StepBlock.WALL:
                return GameActionResult(
                    succeeded=False,
                    block_reason=ActionBlockReason.STEP_BLOCKED,
                    step_block=StepBlock.WALL,
                )
            case StepBlock.BLOCKED_BY_CONTAINER:
                blocker = intent.controller.gw.get_actor_at_location(
                    intent.newx, intent.newy
                )
                return GameActionResult(
                    succeeded=False,
                    blocked_by=blocker,
                    block_reason=ActionBlockReason.STEP_BLOCKED,
                    step_block=StepBlock.BLOCKED_BY_CONTAINER,
                )
            case StepBlock.BLOCKED_BY_ACTOR:
                blocker = intent.controller.gw.get_actor_at_location(
                    intent.newx, intent.newy
                )
                return GameActionResult(
                    succeeded=False,
                    blocked_by=blocker,
                    block_reason=ActionBlockReason.STEP_BLOCKED,
                    step_block=StepBlock.BLOCKED_BY_ACTOR,
                )
        raise ValueError(f"Unhandled StepBlock in MoveExecutor: {block!r}")
