from __future__ import annotations

from brileta import config
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.actions.movement import MoveIntent
from brileta.game.enums import ActionBlockReason, StepBlock
from brileta.util.pathfinding import probe_step

# Fraction of a speed-paced walker's step interval that one tile glide lasts.
# At 1.0 the glide fills exactly one interval, so on average a glide ends just
# as the next step's glide begins and steps chain into one continuous walk. The
# animation system tolerates timing jitter either way: if the next glide starts
# early it chains from the current render position (Actor.move) and this one
# supersedes itself silently (MoveAnimation). Sizing it below 1.0 would leave
# the actor parked on each tile for the remainder of the interval - a visible
# per-tile stutter. Nudge above 1.0 (e.g. 1.05) to guarantee overlap if jitter
# ever opens a seam.
_SPEED_PACED_GLIDE_FRACTION = 1.0


class MoveExecutor(ActionExecutor[MoveIntent]):
    """Executes movement intents by reporting movement results and collisions."""

    def execute(self, intent: MoveIntent) -> GameActionResult | None:
        game_map = intent.controller.gw.game_map
        game_world = intent.controller.gw

        block = probe_step(game_map, game_world, intent.newx, intent.newy)
        if block is not None:
            return self._blocked_result(block, intent)

        # How long a walk step glides is a pure function of the walker's SPEED:
        # the glide fills the time until that walker's next step. Same speed ->
        # same gap -> same glide -> same look, for everyone. WHY the actor is
        # walking (wander, routine, patrol) is irrelevant and must never appear
        # in this decision.
        #
        # "Time until the next step" comes from whichever clock paces the walker:
        #   - speed clock:  an actor stepping on its own energy/speed cadence
        #     (all self-directed NPC walking). Sized here from its speed.
        #   - key repeat:   the human player, paced by held-key timing. Supplies
        #     its own duration_ms sized to that cadence (see MoveIntentGenerator).
        #   - turn order:   combat, paced by discrete turns. Supplies its own
        #     duration_ms.
        # The formula is one rule; only the clock that feeds it differs.
        ease_power = config.DEFAULT_MOVE_EASE_POWER
        paced_by_speed_clock = (
            intent.actor.energy is not None
            and intent.actor is not game_world.player
            and not intent.controller.is_combat_mode()
        )
        if paced_by_speed_clock:
            # Glide fills the gap until this actor's next step at its current
            # speed, so consecutive steps chain into one continuous walk with no
            # per-tile stop. A duration stamped by a pathfinding plan (which
            # copies the player/combat clock) is the wrong clock here, so it is
            # ignored.
            step_interval_s = intent.actor.energy.ambient_step_interval_s()
            duration_ms = int(step_interval_s * _SPEED_PACED_GLIDE_FRACTION * 1000)
            ease_power = 1.0
        elif intent.duration_ms is not None:
            duration_ms = intent.duration_ms
            ease_power = intent.ease_power
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
