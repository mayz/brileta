"""Simple stateless utility actions for NPC AI.

These actions produce a single intent per tick and do not create goals.
Goal-backed behaviors (Wander, Flee, Patrol) live in the behaviors/ subpackage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from brileta.game import ranges
from brileta.game.action_plan import WalkToPlan

from .utility import (
    Consideration,
    Precondition,
    ResponseCurve,
    ResponseCurveType,
    UtilityAction,
    UtilityContext,
    has_escape_route,
    is_hostile,
    is_not_hostile,
    is_threat_present,
)

if TYPE_CHECKING:
    from brileta.game.actions.base import GameIntent


class AttackAction(UtilityAction):
    """Attack the target when adjacent, or pathfind toward them.

    Preconditions: threat must be present and NPC must be hostile.
    Scores higher with more health, higher threat, and lower disposition.
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [is_threat_present, is_hostile]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration("health_percent", ResponseCurve(ResponseCurveType.LINEAR)),
        Consideration("threat_level", ResponseCurve(ResponseCurveType.LINEAR)),
        # Within the hostile band, more hostile values score higher.
        Consideration("disposition", ResponseCurve(ResponseCurveType.INVERSE)),
    ]

    def __init__(
        self,
        base_score: float = 1.0,
        preconditions: list[Precondition] | None = None,
        considerations: list[Consideration] | None = None,
    ) -> None:
        super().__init__(
            action_id="attack",
            base_score=base_score,
            considerations=(
                considerations if considerations is not None else self.CONSIDERATIONS
            ),
            preconditions=(
                preconditions if preconditions is not None else self.PRECONDITIONS
            ),
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        from brileta.game.actions.combat import AttackIntent

        controller = context.controller
        actor = context.actor
        target = context.target
        if target is None:
            return None

        if context.can_attack:
            controller.stop_plan(actor)
            return AttackIntent(controller, actor, target)

        plan = actor.active_plan
        if plan is not None and plan.context.target_position is not None:
            gx, gy = plan.context.target_position
            if (
                ranges.calculate_distance(target.x, target.y, gx, gy) == 1
                and controller.gw.game_map.walkable[gx, gy]
            ):
                return None

        if context.best_attack_destination and controller.start_plan(
            actor, WalkToPlan, target_position=context.best_attack_destination
        ):
            return None

        return None


class AvoidAction(UtilityAction):
    """Move one step away from the target. Used by unfriendly NPCs.

    Preconditions: threat present, escape route exists, NPC not hostile.
    Avoidance peaks when the target is nearby (BELL curve on threat).
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [
        is_threat_present,
        has_escape_route,
        is_not_hostile,
    ]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration("disposition", ResponseCurve(ResponseCurveType.INVERSE)),
        # Avoid peaks when the target is nearby, not at max aggro range.
        Consideration(
            "threat_level",
            ResponseCurve(
                ResponseCurveType.BELL,
                peak=0.7,
                width=0.5,
            ),
        ),
    ]

    def __init__(self, base_score: float = 0.7) -> None:
        super().__init__(
            action_id="avoid",
            base_score=base_score,
            considerations=self.CONSIDERATIONS,
            preconditions=self.PRECONDITIONS,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        """Move away from the target using the best flee step."""
        from brileta.game.actions.movement import MoveIntent

        if context.best_flee_step is None:
            return None

        dx, dy = context.best_flee_step
        context.controller.stop_plan(context.actor)
        return MoveIntent(context.controller, context.actor, dx, dy)


class WatchAction(UtilityAction):
    """Stay put, facing the target. Used by wary/unfriendly NPCs.

    Preconditions: threat present and NPC not hostile.
    Scores higher with lower disposition (more unfriendly) and higher threat.
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [is_threat_present, is_not_hostile]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration("disposition", ResponseCurve(ResponseCurveType.INVERSE)),
        Consideration("threat_level", ResponseCurve(ResponseCurveType.LINEAR)),
    ]

    def __init__(self, base_score: float = 0.35) -> None:
        super().__init__(
            action_id="watch",
            base_score=base_score,
            considerations=self.CONSIDERATIONS,
            preconditions=self.PRECONDITIONS,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Do nothing - stay in place, watching.
        return None


class IdleAction(UtilityAction):
    """Do nothing this turn. Always a baseline fallback.

    No preconditions - idle is always available. Favors cautious/neutral
    dispositions and fades out for friendlier NPCs that should keep moving.
    """

    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration("threat_level", ResponseCurve(ResponseCurveType.INVERSE)),
        # Idle favors cautious/neutral dispositions and fades out
        # for friendlier NPCs that should keep moving.
        Consideration(
            "disposition",
            ResponseCurve(
                ResponseCurveType.BELL,
                peak=0.45,
                width=0.35,
            ),
        ),
    ]

    def __init__(self, base_score: float = 0.1) -> None:
        super().__init__(
            action_id="idle",
            base_score=base_score,
            considerations=self.CONSIDERATIONS,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        return None
