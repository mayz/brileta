"""Social help-seeking: RequestHelpAction + RequestHelpGoal (NUBS 6).

RequestHelpAction is the utility-system entry point. When an NPC has an urgent
unmet need and a capable helper is nearby (the player, or a friendly NPC), it
wins scoring and AIComponent creates a RequestHelpGoal. The goal walks the NPC
up to the helper, signals (a bark plus a "!" bubble), and waits for interaction.

If the helper never engages, the goal records a failed attempt against that
specific helper and gives up. A per-target attempt count decays future scoring
toward that helper, so an ignored NPC stops pestering the same person while
staying free to ask someone else.

TradeAction is a Phase 7 stub: registered on the social tag but scored 0 so it
never wins until the conversation UI exists.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, ClassVar

from brileta.game import ranges
from brileta.game.action_plan import ActionPlan, ApproachStep
from brileta.game.actors.ai.goals import Goal, GoalState
from brileta.game.actors.ai.utility import (
    Consideration,
    Precondition,
    ResponseCurve,
    ResponseCurveType,
    UtilityAction,
    UtilityContext,
    is_no_threat_perceived,
    is_not_hostile,
)
from brileta.game.actors.indicators import IndicatorKind
from brileta.types import ActorId

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors.core import NPC, Actor

# Stop this far (Chebyshev tiles) from the helper before signaling, so the NPC
# comes up to conversational range without crowding onto the helper's tile.
_APPROACH_STOP_DISTANCE = 2
# Give up (fail) if the helper wanders beyond this while we approach/wait - they
# are clearly not engaging and chasing them forever looks broken. This is the
# *primary* give-up signal: an NPC only counts a failed attempt when the helper
# actually leaves, not while they are standing right there.
_GIVE_UP_DISTANCE = 8
# Backstop only: how many turns to keep waiting when the helper stays nearby but
# never engages, before giving up once and returning to normal life. A turn is
# ~1 real second at explore pace, so this is a long, patient wait (~30s) - the
# NPC must NOT bail while the player is right next to it (the old 6-turn timer
# made NPCs give up and wander off before the player could even click Talk).
_WAIT_PATIENCE_TURNS = 30

# Approach plan targeting the helper *actor* (not a fixed tile) so it re-paths as
# they move and stops _APPROACH_STOP_DISTANCE short. Crucially stop_distance > 0:
# the plan system only falls back to an adjacent tile (instead of cancelling)
# when the target tile is occupied - which the helper's own tile always is - if
# stop_distance is positive. A stop_distance-0 WalkToPlan onto the helper's tile
# fails to path and cancels every tick, so the NPC never actually moves.
_APPROACH_PLAN = ActionPlan(
    name="RequestHelpApproach",
    steps=[ApproachStep(stop_distance=_APPROACH_STOP_DISTANCE)],
)


def has_unmet_need(context: UtilityContext) -> bool:
    """Precondition: the NPC carries at least one need with non-zero urgency."""
    return context.max_need_urgency > 0.0


def has_help_candidate(context: UtilityContext) -> bool:
    """Precondition: a helper the NPC could approach is currently perceived."""
    return context.help_target is not None


class _RequestState(Enum):
    """Internal phases of a help request (kept off GoalState per its docstring)."""

    APPROACHING = auto()  # Walking up to the helper.
    SIGNALING = auto()  # Raise the bubble + bark, then move to WAITING.
    WAITING = auto()  # Stand and wait for the helper to engage.


class RequestHelpAction(UtilityAction):
    """Approach a nearby helper and ask for help with an urgent need.

    Preconditions: an unmet need, a perceived helper, no threat *perceived at
    all* (outgoing or incoming), and not hostile. Gating on incoming threat too
    means a hostile creature approaching cancels the request outright - the NPC
    reacts to the danger instead of calmly walking off to ask a favor, rather
    than relying on flee/watch to always outscore it (they don't at full health
    against a moderate incoming threat). Scores on need urgency, per-target
    failed-attempt decay, and Extraversion (gregarious NPCs ask sooner). Base
    score is high enough that an *urgent* need reliably overrides a routine or
    wander even for the least sociable resident.
    """

    PRECONDITIONS: ClassVar[list[Precondition]] = [
        has_unmet_need,
        has_help_candidate,
        is_no_threat_perceived,
        is_not_hostile,
    ]
    CONSIDERATIONS: ClassVar[list[Consideration]] = [
        Consideration("max_need_urgency", ResponseCurve(ResponseCurveType.LINEAR)),
        # help_attempt_decay is already a 0-1 multiplier (0.5 ** attempts), so a
        # LINEAR curve passes it straight through: x1.0 fresh, halving each miss.
        Consideration("help_attempt_decay", ResponseCurve(ResponseCurveType.LINEAR)),
        # Extraversion: gregarious NPCs ask sooner, reserved ones hold back.
        # Centered on 1.0 with a gentle gain (range ~0.8-1.2) so it nudges the
        # threshold without ever suppressing an urgent need below routine - an
        # introvert with a broken cart still comes to ask, just less eagerly.
        Consideration(
            "extraversion", ResponseCurve(ResponseCurveType.CENTERED, gain=0.4)
        ),
    ]

    def __init__(self, base_score: float = 0.9) -> None:
        super().__init__(
            action_id="request_help",
            base_score=base_score,
            considerations=self.CONSIDERATIONS,
            preconditions=self.PRECONDITIONS,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation is handled via get_intent_with_goal in AIComponent.
        return None

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Create a RequestHelpGoal targeting the chosen helper, return first intent."""
        helper = context.help_target
        if helper is None:
            return None
        goal = RequestHelpGoal(
            help_target_id=helper.actor_id,
            preconditions=self.preconditions,
        )
        actor.current_goal = goal
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)
        if goal.is_complete:
            actor.current_goal = None
        return intent


class RequestHelpGoal(Goal):
    """Approach a helper, signal, and wait - failing (per-target) if ignored.

    State machine APPROACHING -> SIGNALING -> WAITING, tracked internally. On
    completion via player interaction the goal is marked COMPLETED with no
    failed attempt; on patience timeout or the helper walking off it records a
    failed attempt against that helper and fails.
    """

    def __init__(
        self,
        help_target_id: ActorId,
        *,
        preconditions: list[Precondition] | None = None,
    ) -> None:
        super().__init__(goal_id="request_help", preconditions=preconditions)
        self._help_target_id = help_target_id
        self._phase = _RequestState.APPROACHING
        self._wait_remaining = _WAIT_PATIENCE_TURNS
        # Cached acting NPC so _on_state_changed can clear its indicator even
        # when the brain abandons the goal outside get_next_action.
        self._actor_ref: NPC | None = None

    def get_base_score(self) -> float:
        return 0.9

    def get_considerations(self) -> list[Consideration]:
        """Continue on the same signals that started the request."""
        return list(RequestHelpAction.CONSIDERATIONS)

    @property
    def progress(self) -> float:
        """Rises across phases so a mid-approach NPC resists distraction."""
        match self._phase:
            case _RequestState.APPROACHING:
                return 0.3
            case _RequestState.SIGNALING:
                return 0.6
            case _RequestState.WAITING:
                return 0.8

    def fulfill(self, npc: NPC) -> None:
        """Player engaged: complete the request with no failed attempt."""
        self._actor_ref = npc  # So the terminal hook can clear the bubble.
        self.state = GoalState.COMPLETED

    def decline(self, npc: NPC, helper: Actor) -> None:
        """Player said no: record a failed attempt so the NPC stops re-asking.

        A spoken "no" counts at least as much as a timeout (NUBS 7), so the
        per-target attempt tracking that already gates timed-out requests also
        gates declined ones - a declined helper's future RequestHelp score
        decays exactly as an ignored one's does.
        """
        self._actor_ref = npc
        npc.ai.record_failed_help_attempt(helper)
        self.state = GoalState.FAILED

    def _on_state_changed(
        self, previous_state: GoalState, new_state: GoalState
    ) -> None:
        """Clear the request bubble whenever the goal reaches a terminal state."""
        _ = previous_state
        terminal = (
            GoalState.COMPLETED,
            GoalState.FAILED,
            GoalState.ABANDONED,
        )
        if new_state in terminal and self._actor_ref is not None:
            self._actor_ref.indicator = None

    def evaluate_completion(self, npc: NPC, controller: Controller) -> None:
        """Fail if the helper is gone; otherwise defer to the state machine."""
        self._actor_ref = npc
        helper = controller.gw.get_actor_by_id(self._help_target_id)
        if helper is None or (helper.health and not helper.health.is_alive()):
            # Helper vanished - abandon quietly, no attempt recorded (they did
            # not ignore us so much as disappear).
            self.state = GoalState.ABANDONED

    def get_next_action(self, npc: NPC, controller: Controller) -> GameIntent | None:
        """Advance the approach -> signal -> wait machine for this tick."""
        self._actor_ref = npc
        helper = controller.gw.get_actor_by_id(self._help_target_id)
        if helper is None:
            self.state = GoalState.ABANDONED
            return None

        distance = ranges.calculate_distance(npc.x, npc.y, helper.x, helper.y)

        match self._phase:
            case _RequestState.APPROACHING:
                return self._approach(npc, controller, helper, distance)
            case _RequestState.SIGNALING:
                return self._signal(npc)
            case _RequestState.WAITING:
                return self._wait(npc, helper, distance)

    def _approach(
        self,
        npc: NPC,
        controller: Controller,
        helper: Actor,
        distance: int,
    ) -> GameIntent | None:
        """Walk toward the helper; switch to signaling once close enough."""
        if distance <= _APPROACH_STOP_DISTANCE:
            controller.stop_plan(npc)
            self._phase = _RequestState.SIGNALING
            return self._signal(npc)

        # Start an approach plan toward the helper actor (not a fixed tile) so
        # the plan system re-paths as they move and stops _APPROACH_STOP_DISTANCE
        # short. The brain cancels any stale plan from an interrupted goal before
        # we get here, so a missing plan reliably means "no approach in progress":
        # start one. While a plan runs the TurnManager drives it and this code
        # isn't reached; when it finishes/cancels we re-path to the helper's new
        # position next tick.
        if npc.active_plan is None:
            controller.start_plan(npc, _APPROACH_PLAN, target_actor=helper)
        return None

    def _signal(self, npc: NPC) -> GameIntent | None:
        """Raise the request bubble, bark once, then start waiting."""
        # Local import: barks -> ai package would cycle at module load time.
        from brileta.game.actors.barks import emit_bark

        npc.indicator = IndicatorKind.REQUEST
        emit_bark(npc, "Hey - over here!")
        self._phase = _RequestState.WAITING
        self._wait_remaining = _WAIT_PATIENCE_TURNS
        return None

    def _wait(self, npc: NPC, helper: Actor, distance: int) -> GameIntent | None:
        """Stand and wait for the helper to engage.

        The NPC only gives up (recording a failed attempt) when the helper
        actually leaves - moves beyond _GIVE_UP_DISTANCE. While the helper stays
        near, it keeps waiting with the bubble up, bailing only after the long
        _WAIT_PATIENCE_TURNS backstop so a parked-but-idle player eventually
        frees the NPC. It must NOT bail on a short timer while the helper is
        standing right there, or it flips between waiting and wandering off.
        """
        # Keep the bubble up - the AIComponent resets indicators each tick, so a
        # continuing WAITING goal must re-raise it.
        npc.indicator = IndicatorKind.REQUEST

        if distance > _GIVE_UP_DISTANCE:
            npc.ai.record_failed_help_attempt(helper)
            self.state = GoalState.FAILED
            return None

        if self._wait_remaining <= 0:
            npc.ai.record_failed_help_attempt(helper)
            self.state = GoalState.FAILED
            return None

        self._wait_remaining -= 1
        return None


class TradeAction(UtilityAction):
    """Phase 7 stub: barter/trade with a friendly NPC.

    Defined and registered on the social tag so the wiring exists, but scored 0
    (base_score 0.0) so it can never win. Phase 7's conversation UI replaces
    this with real trade mechanics.
    """

    def __init__(self, base_score: float = 0.0) -> None:
        super().__init__(action_id="trade", base_score=base_score)

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        return None
