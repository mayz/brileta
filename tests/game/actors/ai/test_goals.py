"""Tests for the Goal lifecycle and utility scoring.

Validates:
- ContinueGoalAction persistence bonus scales with goal.progress.
- Goals are abandoned when a higher-scoring action wins.
- Completed/failed goals are excluded from continue scoring.
- NPC picks a new action on the same tick after goal completion.
- NPC death clears current goal.
- UtilityBrain includes ContinueGoalAction in scoring.
"""

import pytest

from brileta.game.actions.movement import MoveIntent
from brileta.game.actors.ai.actions import IdleAction
from brileta.game.actors.ai.behaviors.flee import (
    _FLEE_SAFE_DISTANCE,
    FleeAction,
    FleeGoal,
)
from brileta.game.actors.ai.behaviors.patrol import PatrolGoal
from brileta.game.actors.ai.goals import (
    PERSISTENCE_MINIMUM,
    ContinueGoalAction,
    GoalState,
)
from brileta.game.actors.ai.utility import UtilityBrain, UtilityContext
from brileta.types import ActorId
from tests.helpers import make_ai_world

# ---------------------------------------------------------------------------
# ContinueGoalAction / Persistence Bonus Tests
# ---------------------------------------------------------------------------


def test_continue_goal_persistence_bonus_scales_with_progress() -> None:
    """ContinueGoalAction persistence bonus increases with goal.progress."""
    controller, player, npc = make_ai_world(npc_x=3, npc_y=0)

    # Create a FleeGoal with known state
    goal = FleeGoal(threat_actor_id=player.actor_id)
    goal._start_distance = 1
    goal._current_distance = 1

    context = UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=3,
        health_percent=0.2,
        threat_level=0.8,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=(1, 0),
    )

    # Score at low progress (just started fleeing)
    goal._current_distance = 2  # Small progress
    action_low = ContinueGoalAction(goal)
    score_low = action_low.score(context)

    # Score at high progress (nearly safe)
    goal._current_distance = _FLEE_SAFE_DISTANCE - 1  # Almost there
    action_high = ContinueGoalAction(goal)
    score_high = action_high.score(context)

    # Higher progress should give higher persistence bonus
    assert score_high > score_low, (
        f"Higher progress should score higher: {score_high} vs {score_low}"
    )


def test_continue_goal_beats_raw_action_at_zero_progress() -> None:
    """ContinueGoalAction has a persistence bonus even at progress 0.0.

    Without PERSISTENCE_MINIMUM, a goal at progress 0.0 would tie with the
    raw action and get recreated every tick (progress never advances). The
    minimum bonus ensures ContinueGoal always edges ahead.

    We verify this by computing the raw consideration score (without bonus)
    and checking ContinueGoalAction adds exactly PERSISTENCE_MINIMUM on top.
    """
    controller, player, npc = make_ai_world(npc_x=3, npc_y=0)

    goal = FleeGoal(threat_actor_id=player.actor_id)
    assert goal.progress == 0.0

    context = UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=3,
        health_percent=0.2,
        threat_level=0.8,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=(1, 0),
    )

    # Compute the raw consideration score the goal would get without bonus.
    raw_score = goal.get_base_score()
    for c in goal.get_considerations():
        raw_score *= c.evaluate(context)

    continue_score = ContinueGoalAction(goal).score(context)

    # ContinueGoal should exceed the raw score by exactly PERSISTENCE_MINIMUM
    assert continue_score > raw_score
    assert continue_score - raw_score == pytest.approx(PERSISTENCE_MINIMUM)


def test_flee_goal_continuation_inherits_action_disposition_scoring() -> None:
    """Continuation should preserve originating flee action considerations.

    Regression coverage: when a flee goal is created from a combatant
    FleeAction, ContinueGoalAction should include disposition INVERSE in
    its score, not drop that factor on tick 2+.
    """
    controller, player, npc = make_ai_world(npc_x=3, npc_y=0)
    flee_action = FleeAction()
    goal = FleeGoal(
        threat_actor_id=player.actor_id,
        preconditions=flee_action.preconditions,
        considerations=flee_action.considerations,
        base_score=flee_action.base_score,
    )
    assert any(c.input_key == "disposition" for c in goal.get_considerations())

    hostile_context = UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=3,
        health_percent=0.2,
        threat_level=0.8,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=(1, 0),
        disposition=0.125,  # normalized -75
    )
    neutral_context = UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=3,
        health_percent=0.2,
        threat_level=0.8,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=(1, 0),
        disposition=0.5,  # normalized 0
    )

    hostile_score = ContinueGoalAction(goal).score(hostile_context)
    neutral_score = ContinueGoalAction(goal).score(neutral_context)
    assert hostile_score > neutral_score


# ---------------------------------------------------------------------------
# Goal Abandonment Tests
# ---------------------------------------------------------------------------


def test_goal_abandoned_when_higher_action_wins() -> None:
    """An active goal is abandoned when a higher-scoring action wins.

    Scenario: NPC is fleeing, then gets healed to full health. Attack should
    now score higher than continue-fleeing, causing the flee goal to be abandoned.
    """
    controller, _player, npc = make_ai_world(npc_x=2, npc_y=0, npc_hp_damage=4)

    # Create flee goal
    action = npc.ai.get_action(controller, npc)
    assert npc.current_goal is not None
    assert isinstance(npc.current_goal, FleeGoal)

    # Simulate the first flee move
    if isinstance(action, MoveIntent):
        npc.move(action.dx, action.dy)

    # Heal the NPC to full - attack should now score higher
    npc.heal()
    assert npc.health.hp == npc.health.max_hp

    # Move NPC adjacent to player for attack opportunity
    npc.teleport(1, 0)

    # Next evaluation: attack should win, flee goal should be abandoned
    npc.ai.get_action(controller, npc)

    # The NPC should have abandoned the flee goal
    assert npc.current_goal is None


# ---------------------------------------------------------------------------
# Goal Completion Transition Tests
# ---------------------------------------------------------------------------


def test_npc_picks_new_action_same_tick_after_goal_completes() -> None:
    """When a goal completes at the top of AIComponent.get_action, the NPC
    proceeds to normal scoring and picks a new action on the same tick
    rather than doing nothing.
    """
    controller, _player, npc = make_ai_world(npc_x=2, npc_y=0, npc_hp_damage=4)

    # Create a flee goal
    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, FleeGoal)
    old_flee_goal = npc.current_goal

    # Teleport NPC to safe distance so the flee goal completes
    npc.teleport(_FLEE_SAFE_DISTANCE + 1, 0)

    # The NPC should complete the flee goal AND proceed to normal scoring
    # on this same tick, not return with nothing decided.
    result = npc.ai.get_action(controller, npc)

    # The old flee goal must be done
    assert old_flee_goal.state == GoalState.COMPLETED

    # The NPC should have completed the old goal and finished scoring the tick.
    # With no threat it may pick up a new goal (e.g., WanderGoal).
    assert result is None or isinstance(result, MoveIntent)
    assert not isinstance(npc.current_goal, FleeGoal)


def test_completed_goal_excluded_from_continue_scoring() -> None:
    """A goal in a terminal state should not produce a ContinueGoalAction.

    UtilityBrain.select_action guards on `not current_goal.is_complete`.
    Verify a FAILED goal is correctly excluded from scoring.
    """
    controller, player, npc = make_ai_world(npc_x=5, npc_y=0)

    brain = UtilityBrain([IdleAction(base_score=0.2)])

    context = UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=5,
        health_percent=1.0,
        threat_level=0.0,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=None,
    )

    # Create a FAILED goal
    goal = FleeGoal(threat_actor_id=player.actor_id)
    goal.state = GoalState.FAILED
    assert goal.is_complete

    # Scoring with a completed goal should NOT include ContinueGoalAction
    _action, scored = brain.select_action(context, current_goal=goal)
    scored_names = [s.display_name for s in scored]
    assert "ContinueGoal(Flee)" not in scored_names
    assert len(scored) == 1  # Only Idle


# ---------------------------------------------------------------------------
# NPC Death Clears Goal
# ---------------------------------------------------------------------------


def test_npc_death_clears_current_goal() -> None:
    """Killing an NPC with an active goal should clear current_goal."""
    _controller, _player, npc = make_ai_world(npc_x=3, npc_y=0)
    npc.current_goal = FleeGoal(threat_actor_id=ActorId(123))

    npc.take_damage(999)
    assert not npc.health.is_alive()
    assert npc.current_goal is None


# ---------------------------------------------------------------------------
# UtilityBrain.select_action with Goals
# ---------------------------------------------------------------------------


def test_utility_brain_includes_continue_goal_action() -> None:
    """UtilityBrain.select_action should score ContinueGoalAction alongside atomics."""
    controller, player, npc = make_ai_world(npc_x=15, npc_y=15)

    # Create a simple brain with just Idle
    brain = UtilityBrain([IdleAction(base_score=0.2)])

    context = UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=15,
        health_percent=1.0,
        threat_level=0.0,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=None,
    )

    # Without a goal: only Idle should appear in scored list
    _action, scored = brain.select_action(context, current_goal=None)
    assert len(scored) == 1
    assert scored[0].display_name == "Idle"

    # With a PatrolGoal: ContinueGoalAction should also appear
    goal = PatrolGoal([(10, 10), (20, 20)])
    action_with_goal, scored_with_goal = brain.select_action(context, current_goal=goal)
    scored_names = [s.display_name for s in scored_with_goal]
    assert "Idle" in scored_names
    assert "ContinueGoal(Patrol)" in scored_names

    # ContinueGoal(Patrol) should win over Idle (0.25 + persistence > 0.2)
    assert action_with_goal is not None
    assert isinstance(action_with_goal, ContinueGoalAction)
