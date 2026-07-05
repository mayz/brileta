"""Surrender: SurrenderAction + SurrenderGoal (NUBS 7).

Surrender is the "can't flee" combat resolution. When a sapient NPC is cornered
(no escape route), hurt, and frightened, and isn't a fanatic, SurrenderAction
wins scoring and the NPC yields instead of fighting to the death. It adopts a
SurrenderGoal that makes it cower in place (a white-flag bubble, no attacks) and
waits for the player to respond via the conversation UI:

  - Accept: the NPC is pacified (large disposition swing) and disengages.
  - Refuse: the NPC drops back into a desperate fight.
  - Ignore: the NPC keeps cowering until the player decides or the threat leaves.

Because everything competes in the same scoring system, no special "am I losing"
check is needed - surrender simply outscores Attack once health is low, fear is
high, and there is nowhere to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from brileta.game.actors.ai.goals import Goal, GoalState
from brileta.game.actors.ai.utility import (
    Consideration,
    Precondition,
    ResponseCurve,
    ResponseCurveType,
    UtilityAction,
    UtilityContext,
    is_any_threat_perceived,
)

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors.core import NPC, Actor

# How far disposition swings toward the player when a surrender is accepted. From
# a typical raider's -75 this lands solidly non-hostile, so the pacified NPC
# stops attacking and drifts off under normal (now peaceable) AI.
_ACCEPT_DISPOSITION_SWING = 90


def has_no_escape_route(context: UtilityContext) -> bool:
    """Precondition: the NPC is cornered - no valid flee step this tick."""
    return not context.has_escape_route


def surrender_not_refused(context: UtilityContext) -> bool:
    """Precondition: the current threat hasn't already refused our surrender.

    Nothing in the scoring inputs changes when a surrender is refused - the NPC
    is still cornered, hurt, and afraid - so without this gate SurrenderAction
    would out-score Attack and re-adopt a fresh SurrenderGoal on the next tick,
    and "fights on" would never happen. Once the target the NPC would yield to
    has refused, surrender toward them is off the table and Attack takes over.
    """
    target = context.target
    return target is None or not context.actor.ai.was_surrender_refused_by(target)


class SurrenderAction(UtilityAction):
    """Yield to an overwhelming, inescapable threat.

    Preconditions: a threat is perceived and there is no escape route (Flee's
    own precondition has failed, so Flee scores 0 and Surrender is the live
    alternative to Attack). Scores on low health, fear (the strongest threat
    signal), and Agreeableness - agreeable NPCs give up sooner, while a low
    Agreeableness "fanatic" resists and fights on.
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [
        is_any_threat_perceived,
        has_no_escape_route,
        surrender_not_refused,
    ]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        # Low health drives surrender hard - a quadratic inverse ramp so a
        # lightly-wounded NPC keeps fighting but a badly-hurt one folds.
        Consideration(
            "health_percent",
            ResponseCurve(ResponseCurveType.INVERSE, exponent=2.0),
            weight=2.0,
        ),
        # Fear: the stronger of outgoing/incoming threat. A cornered NPC facing
        # a heavy threat is far likelier to yield.
        Consideration("max_threat", ResponseCurve(ResponseCurveType.LINEAR)),
        # Agreeableness centers on 1.0: an agreeable NPC (>0.5) surrenders more
        # readily, a disagreeable "fanatic" (<0.5) resists. Gain kept high so a
        # true fanatic's score is pushed well below Attack.
        Consideration(
            "agreeableness", ResponseCurve(ResponseCurveType.CENTERED, gain=1.2)
        ),
    ]

    def __init__(self, base_score: float = 1.4) -> None:
        super().__init__(
            action_id="surrender",
            base_score=base_score,
            considerations=self.CONSIDERATIONS,
            preconditions=self.PRECONDITIONS,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation goes through get_intent_with_goal in AIComponent.
        return None

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Adopt a SurrenderGoal and cower this tick (no intent)."""
        goal = SurrenderGoal(preconditions=[is_any_threat_perceived])
        actor.current_goal = goal
        goal.tick()
        return goal.get_next_action(actor, context.controller)


class SurrenderGoal(Goal):
    """Cower with a white flag until the player responds or the threat clears.

    The goal produces no intent (the NPC stands still, hands up). Its high,
    fixed progress makes it sticky so a surrendered NPC does not flip back to
    attacking on a marginal score wobble. It ends when the player accepts or
    refuses via conversation, or when no threat remains (the NPC recovers).
    """

    def __init__(self, *, preconditions: list[Precondition] | None = None) -> None:
        super().__init__(goal_id="surrender", preconditions=preconditions)
        self._actor_ref: NPC | None = None

    def get_base_score(self) -> float:
        return 1.4

    def get_considerations(self) -> list[Consideration]:
        """Continue on the same fear/health signals that triggered surrender."""
        return list(SurrenderAction.CONSIDERATIONS)

    @property
    def progress(self) -> float:
        # Fixed high progress: a yielding NPC should stay yielded, not oscillate.
        return 0.9

    def accept(self, npc: NPC, player: Actor) -> None:
        """Player accepted: pacify the NPC and end the surrender."""
        self._actor_ref = npc
        npc.ai.modify_disposition(player, _ACCEPT_DISPOSITION_SWING)
        npc.indicator = None
        self.state = GoalState.COMPLETED

    def refuse(self, npc: NPC, refuser: Actor) -> None:
        """Player refused: the NPC fights on (disposition stays hostile).

        Records the refusal against ``refuser`` so SurrenderAction is suppressed
        toward them next tick - otherwise, still cornered and hurt, the NPC would
        immediately re-adopt Surrender and never actually fight on.
        """
        self._actor_ref = npc
        npc.ai.record_surrender_refused(refuser)
        npc.indicator = None
        self.state = GoalState.FAILED

    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """No terminal check needed here.

        Recovery is handled by scoring, not this hook: the goal carries the
        ``is_any_threat_perceived`` precondition, so once the NPC no longer
        perceives a threat, ContinueGoal(surrender) scores 0, another action
        wins, and the brain abandons the goal - the NPC stands back up on its
        own. Accept/refuse handle the player-driven terminal transitions, and
        death clears the goal in NPC.take_damage.
        """
        self._actor_ref = npc

    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Cower: do nothing this tick."""
        self._actor_ref = npc
        return None
