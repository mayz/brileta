"""Tests for NUBS 6 social help-seeking: RequestHelp action, goal, and decay.

Validates:
- RequestHelp wins over idle/wander with an urgent need + perceived helper,
  and scores 0 with no need, no helper, or under threat.
- Per-target attempt decay multiplier at 0/1/2/3 attempts, and that failures
  toward one helper don't suppress asking another.
- Goal state machine APPROACHING -> SIGNALING -> WAITING, indicator handling,
  patience-timeout failure (records an attempt), and fulfillment (no attempt).
- Tag composition: residents get request_help, creatures do not.
"""

from __future__ import annotations

import pytest

from brileta import colors
from brileta.game.actors import NPC, Character
from brileta.game.actors.ai.behaviors.request_help import (
    _APPROACH_STOP_DISTANCE,
    RequestHelpAction,
    RequestHelpGoal,
    _RequestState,
)
from brileta.game.actors.ai.behaviors.wander import WanderAction
from brileta.game.actors.ai.goals import GoalState
from brileta.game.actors.ai.utility import UtilityContext
from brileta.game.actors.indicators import IndicatorKind
from brileta.game.actors.npc_core import compose_actions
from brileta.game.actors.npc_types import (
    BRIGAND_TYPE,
    DOG_TYPE,
    GIANT_SCORPION_TYPE,
    RESIDENT_TYPE,
    TROG_TYPE,
)
from tests.helpers import make_ai_world


def _social_context(
    *,
    max_need_urgency: float = 0.9,
    help_target_is_player: bool = True,
    threat_level: float = 0.0,
    incoming_threat: float = 0.0,
    disposition: float = 0.5,
    failed_help_attempts: int = 0,
) -> tuple[UtilityContext, object]:
    """Build a UtilityContext wired for social scoring plus the acting NPC."""
    controller, player, npc = make_ai_world(npc_x=3, npc_y=0, disposition=0)
    context = UtilityContext(
        controller=controller,
        actor=npc,
        target=player if threat_level > 0.0 else None,
        distance_to_target=3,
        health_percent=1.0,
        threat_level=threat_level,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=None,
        disposition=disposition,
        incoming_threat=incoming_threat,
        max_need_urgency=max_need_urgency,
        help_target=player if help_target_is_player else None,
        failed_help_attempts=failed_help_attempts,
    )
    return context, npc


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_request_help_beats_wander_with_urgent_need_and_helper() -> None:
    context, _ = _social_context(max_need_urgency=0.9)
    request = RequestHelpAction()
    wander = WanderAction()
    assert request.score(context) > wander.score(context)


@pytest.mark.parametrize(
    ("kwargs",),
    [
        ({"max_need_urgency": 0.0},),  # No need.
        ({"help_target_is_player": False},),  # No perceived helper.
        ({"threat_level": 0.8},),  # Under outgoing threat.
        ({"incoming_threat": 0.5},),  # Hostile approaching (even at full health).
    ],
)
def test_request_help_scores_zero_when_ineligible(kwargs: dict) -> None:
    context, _ = _social_context(**kwargs)
    assert RequestHelpAction().score(context) == 0.0


def test_urgent_need_beats_routine_for_least_sociable_npc() -> None:
    """An urgent need must override a routine even for an introverted,
    disciplined resident - the worst case where Extraversion damps the ask and
    Conscientiousness maximizes the routine's persistence bonus. Regression for
    residents that ignored needs and stayed on their routine."""
    from brileta.game.actors.ai.behaviors.routine import RoutineGoal
    from brileta.game.actors.ai.goals import ContinueGoalAction

    context, _ = _social_context(max_need_urgency=0.9)
    # Least sociable, most disciplined resident: min extraversion damps the ask,
    # max conscientiousness maximizes routine persistence.
    context.extraversion = 0.0
    context.conscientiousness = 1.0

    request_score = RequestHelpAction().score(context)
    routine_continue = ContinueGoalAction(RoutineGoal()).score(context)
    assert request_score > routine_continue


# ---------------------------------------------------------------------------
# Attempt decay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attempts", "expected"),
    [(0, 1.0), (1, 0.5), (2, 0.25), (3, 0.125)],
)
def test_attempt_decay_multiplier(attempts: int, expected: float) -> None:
    context, _ = _social_context(failed_help_attempts=attempts)
    assert context.get_input("help_attempt_decay") == pytest.approx(expected)


def test_attempts_are_per_target() -> None:
    """Failures toward one helper must not count against another."""
    controller, player, npc = make_ai_world(npc_x=3, npc_y=0, disposition=0)
    other = NPC(1, 1, "r", (255, 255, 255), "Neighbor", game_world=controller.gw)

    npc.ai.record_failed_help_attempt(player)
    npc.ai.record_failed_help_attempt(player)

    assert npc.ai.failed_help_attempts_toward(player) == 2
    assert npc.ai.failed_help_attempts_toward(other) == 0


def test_select_help_target_requires_capable_non_hostile_helper() -> None:
    """Runtime helper selection must reject anything that could never help.

    Only the player or a help-capable (RequestHelp-capable), mutually
    non-hostile NPC qualifies. Regression: a needy resident used to treat any
    perceived non-hostile actor as a helper, so it would walk up to a dog or a
    not-yet-provoked monster and wait on it forever, since its own disposition
    toward an un-met creature defaults to neutral, not hostile.
    """
    from typing import cast

    from brileta.game.actors.ai.perception import PerceivedActor
    from brileta.game.game_world import GameWorld
    from tests.helpers import DummyController, DummyGameWorld

    gw = DummyGameWorld(width=40, height=40)
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    resident = RESIDENT_TYPE.create(5, 5, "Resident", game_world=cast(GameWorld, gw))
    gw.add_actor(resident)
    dog = DOG_TYPE.create(6, 5, "Dog", game_world=cast(GameWorld, gw))
    gw.add_actor(dog)
    neighbor = RESIDENT_TYPE.create(7, 5, "Neighbor", game_world=cast(GameWorld, gw))
    gw.add_actor(neighbor)
    controller = DummyController(gw)

    def perceived(*actors: object) -> list[PerceivedActor]:
        return [
            PerceivedActor(
                actor=cast(Character, a),
                distance=abs(cast(Character, a).x - resident.x),
                perception_strength=0.8,
            )
            for a in actors
        ]

    # A dog can't help -> no target (so RequestHelp won't even fire).
    target, _ = resident.ai._select_help_target(controller, resident, perceived(dog))
    assert target is None

    # Another resident is a valid, non-hostile helper.
    target, _ = resident.ai._select_help_target(
        controller, resident, perceived(neighbor)
    )
    assert target is neighbor

    # A neighbor hostile toward the resident is excluded.
    neighbor.ai.set_hostile(resident)
    target, _ = resident.ai._select_help_target(
        controller, resident, perceived(neighbor)
    )
    assert target is None

    # The player always qualifies and is preferred over a non-helper.
    target, _ = resident.ai._select_help_target(
        controller, resident, perceived(dog, player)
    )
    assert target is player


# ---------------------------------------------------------------------------
# Goal state machine
# ---------------------------------------------------------------------------


def test_goal_approaches_when_far() -> None:
    controller, player, npc = make_ai_world(npc_x=6, npc_y=0, disposition=0)
    goal = RequestHelpGoal(help_target_id=player.actor_id)

    goal.get_next_action(npc, controller)

    assert goal._phase is _RequestState.APPROACHING
    assert npc.indicator is None
    assert npc.active_plan is not None


def test_approach_plan_actually_moves_toward_helper() -> None:
    """Regression: the approach must path toward the player and move.

    A stop_distance-0 plan aimed at the player's own (occupied) tile fails to
    path and the turn manager cancels it every tick, so the NPC never moves and
    only reacts once the player walks into signal range. The approach must use a
    positive stop distance so the plan falls back to an adjacent tile and closes
    the gap itself.
    """
    from brileta.game import ranges
    from brileta.game.actions.movement import MoveIntent

    controller, player, npc = make_ai_world(npc_x=8, npc_y=0, disposition=0)
    goal = RequestHelpGoal(help_target_id=player.actor_id)
    npc.current_goal = goal

    # First tick starts the approach plan (no direct intent yet).
    goal.get_next_action(npc, controller)
    assert npc.active_plan is not None

    # The turn manager's plan executor must yield a move that closes distance,
    # not None (which is what a cancelled/blocked plan returns).
    intent = controller.turn_manager._get_intent_from_plan(npc)
    assert isinstance(intent, MoveIntent)
    new_distance = ranges.calculate_distance(
        npc.x + intent.dx, npc.y + intent.dy, player.x, player.y
    )
    assert new_distance < 8


def test_brain_cancels_stale_plan_when_switching_goals() -> None:
    """Root-cause regression: switching from one goal to another must cancel the
    outgoing goal's movement plan.

    ``current_goal`` and ``active_plan`` are separate actor state, and the
    TurnManager drives the plan independently. When a resident drops its routine
    (whose plan walks it to its workplace) to ask for help, the brain must cancel
    that plan - otherwise the new RequestHelp goal rides the routine's plan and
    the NPC walks away from the player. This exercises AIComponent.get_action,
    not the goal in isolation, so it guards the fix at its real site.
    """
    from types import SimpleNamespace
    from typing import cast

    from brileta.game.actors.ai.behaviors.routine import RoutineGoal
    from brileta.game.actors.needs import Need, NeedType
    from brileta.game.game_world import GameWorld
    from tests.helpers import DummyController, DummyGameWorld

    gw = DummyGameWorld(width=60, height=60)
    player = Character(
        24, 20, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    gw.clock = SimpleNamespace(time_of_day=0.5)  # Workday: routine heads to anchor.
    npc = RESIDENT_TYPE.create(20, 20, "Resident", game_world=cast(GameWorld, gw))
    # Workplace far to the west, away from the player - the routine plan walks
    # the NPC in the wrong direction.
    npc.home_pos = npc.anchor_pos = (5, 20)
    gw.add_actor(npc)
    controller = DummyController(gw)

    # Tick once: the resident adopts its routine and starts a walk plan west.
    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, RoutineGoal)
    routine_plan = npc.active_plan
    assert routine_plan is not None

    # A need appears; RequestHelp outscores the routine and takes over.
    npc.needs.append(Need(type=NeedType.REPAIR, urgency=0.9))
    npc.ai.get_action(controller, npc)

    assert isinstance(npc.current_goal, RequestHelpGoal)
    # The stale routine plan must be gone; any plan now must target the player.
    assert npc.active_plan is not routine_plan
    if npc.active_plan is not None:
        assert npc.active_plan.context.target_actor is player


def test_goal_switch_cancels_plan_even_if_new_action_ignores_it() -> None:
    """Pin the invariant in isolation: the abandon-for-a-different-action path in
    get_action must itself cancel the outgoing goal's plan, regardless of whether
    the winning action touches it.

    A do-nothing stub action wins over the active goal's continuation; the stub
    never touches active_plan, so the plan can only be cleared by the central
    stop_plan on goal switch. Removing that line makes this fail even though the
    end-to-end RequestHelp behavior (which starts its own plan) would still pass.
    """
    from brileta.game.action_plan import WalkToPlan
    from brileta.game.actors.ai.behaviors.wander import WanderGoal
    from brileta.game.actors.ai.utility import UtilityAction

    controller, _player, npc = make_ai_world(npc_x=3, npc_y=0, disposition=0)

    class _StubAction(UtilityAction):
        """Always wins, and deliberately leaves active_plan untouched."""

        def __init__(self) -> None:
            super().__init__(action_id="stub", base_score=5.0)

        def get_intent(self, context: UtilityContext) -> None:
            return None

    npc.ai.brain.actions = [_StubAction()]

    # An active goal carrying an orphanable movement plan.
    npc.current_goal = WanderGoal()
    controller.start_plan(npc, WalkToPlan, target_position=(npc.x + 5, npc.y))
    assert npc.active_plan is not None

    npc.ai.get_action(controller, npc)

    # The stub (a different action) won, so the brain must have dropped the goal
    # AND cancelled its plan - the stub itself never would.
    assert npc.current_goal is None
    assert npc.active_plan is None


def test_goal_signals_then_waits_when_close() -> None:
    # Distance == stop distance -> transition straight to signaling/waiting.
    controller, player, npc = make_ai_world(
        npc_x=_APPROACH_STOP_DISTANCE, npc_y=0, disposition=0
    )
    goal = RequestHelpGoal(help_target_id=player.actor_id)

    goal.get_next_action(npc, controller)

    assert goal._phase is _RequestState.WAITING
    assert npc.indicator is IndicatorKind.REQUEST


def test_wait_holds_while_helper_stays_close() -> None:
    """Regression: the NPC must not give up while the helper is standing right
    next to it. The old short timer made it bail and re-ask in a loop before the
    player could interact; now it keeps waiting until the helper leaves or the
    long backstop expires."""
    controller, player, npc = make_ai_world(
        npc_x=_APPROACH_STOP_DISTANCE, npc_y=0, disposition=0
    )
    goal = RequestHelpGoal(help_target_id=player.actor_id)
    goal._phase = _RequestState.WAITING
    goal._actor_ref = npc

    # Wait far longer than the old 6-turn budget; the helper never leaves.
    for _ in range(10):
        goal.get_next_action(npc, controller)
        assert goal.state is GoalState.ACTIVE
        assert npc.indicator is IndicatorKind.REQUEST
    assert npc.ai.failed_help_attempts_toward(player) == 0


def test_patience_timeout_records_failed_attempt_and_fails() -> None:
    controller, player, npc = make_ai_world(
        npc_x=_APPROACH_STOP_DISTANCE, npc_y=0, disposition=0
    )
    goal = RequestHelpGoal(help_target_id=player.actor_id)
    goal._phase = _RequestState.WAITING
    goal._wait_remaining = 0
    goal._actor_ref = npc

    goal.get_next_action(npc, controller)

    assert goal.state is GoalState.FAILED
    assert npc.indicator is None
    assert npc.ai.failed_help_attempts_toward(player) == 1


def test_fulfill_completes_without_failed_attempt() -> None:
    _, player, npc = make_ai_world(
        npc_x=_APPROACH_STOP_DISTANCE, npc_y=0, disposition=0
    )
    goal = RequestHelpGoal(help_target_id=player.actor_id)
    npc.indicator = IndicatorKind.REQUEST

    goal.fulfill(npc)

    assert goal.state is GoalState.COMPLETED
    assert npc.indicator is None
    assert npc.ai.failed_help_attempts_toward(player) == 0


def test_wait_fails_if_helper_walks_far_away() -> None:
    controller, player, npc = make_ai_world(npc_x=0, npc_y=0, disposition=0)
    goal = RequestHelpGoal(help_target_id=player.actor_id)
    goal._phase = _RequestState.WAITING
    goal._actor_ref = npc
    # Move the helper well beyond the give-up distance.
    npc.teleport(50, 50)

    goal.get_next_action(npc, controller)

    assert goal.state is GoalState.FAILED
    assert npc.ai.failed_help_attempts_toward(player) == 1


# ---------------------------------------------------------------------------
# Tag composition
# ---------------------------------------------------------------------------


def test_residents_get_request_help_creatures_do_not() -> None:
    resident_ids = {a.action_id for a in compose_actions(RESIDENT_TYPE.tags)}
    assert "request_help" in resident_ids
    assert "trade" in resident_ids

    for creature in (TROG_TYPE, BRIGAND_TYPE, GIANT_SCORPION_TYPE, DOG_TYPE):
        ids = {a.action_id for a in compose_actions(creature.tags)}
        assert "request_help" not in ids
