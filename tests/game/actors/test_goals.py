"""Tests for the NUBS Goal system.

Validates:
- FleeGoal persists across multiple turns until safe distance reached.
- FleeGoal completes when threat dies mid-flee.
- ContinueGoalAction persistence bonus scales with goal.progress.
- PatrolGoal cycles between waypoints (unit tests only; PatrolAction is not
  currently wired into utility scoring).
- WanderAction creates WanderGoals that move via heading + step budgets.
- Goals are abandoned when a higher-scoring action wins.
- ai.force_hostile makes all NPCs use hostile behavior.
- Combat awareness: NPCs react to attacks from outside aggro range.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import pytest

from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC, Character
from brileta.game.actors.ai import disposition_label
from brileta.game.actors.goals import (
    _FLEE_SAFE_DISTANCE,
    ContinueGoalAction,
    FleeGoal,
    GoalState,
    PatrolGoal,
    WanderGoal,
)
from brileta.game.actors.utility import (
    Consideration,
    ResponseCurve,
    ResponseCurveType,
    UtilityContext,
)
from brileta.game.game_world import GameWorld
from brileta.game.turn_manager import TurnManager
from brileta.types import ActorId
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100


def make_world(
    npc_x: int = 3,
    npc_y: int = 0,
    npc_hp_damage: int = 0,
    disposition: int = -75,
    map_size: int = 80,
) -> tuple[DummyController, Character, NPC]:
    """Create a test world with player at origin and NPC at given position."""
    gw = DummyGameWorld(width=map_size, height=map_size)
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        npc_x,
        npc_y,
        "g",
        colors.RED,
        "Enemy",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    if disposition != 0:
        npc.ai.modify_disposition(player, disposition)
    controller = DummyController(gw)

    if npc_hp_damage > 0:
        npc.take_damage(npc_hp_damage)

    return controller, player, npc


# ---------------------------------------------------------------------------
# Disposition Label Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-100, "Hostile"),
        (-52, "Hostile"),
        (-51, "Hostile"),
        (-50, "Unfriendly"),
        (-21, "Unfriendly"),
        (-20, "Wary"),
        (0, "Approachable"),
        (20, "Approachable"),
        (21, "Friendly"),
        (100, "Ally"),
    ],
)
def test_disposition_label_boundaries(value: int, expected: str) -> None:
    """Disposition labels should match inclusive boundary expectations."""
    assert disposition_label(value) == expected


# ---------------------------------------------------------------------------
# FleeGoal Tests
# ---------------------------------------------------------------------------


def test_flee_goal_persists_across_multiple_turns() -> None:
    """A fleeing NPC continues fleeing across multiple turns until safe."""
    controller, _player, npc = make_world(npc_x=2, npc_y=0, npc_hp_damage=4)

    # NPC is adjacent to player at distance 2 and nearly dead.
    # First call should create a FleeGoal.
    action = npc.ai.get_action(controller, npc)

    # Flee action should produce a MoveIntent away from player
    assert isinstance(action, MoveIntent), f"Expected MoveIntent, got {type(action)}"
    assert action.dx > 0 or action.dy != 0  # Moving away from player at (0,0)

    # NPC should now have an active FleeGoal
    assert npc.current_goal is not None
    assert isinstance(npc.current_goal, FleeGoal)
    assert npc.current_goal.state == GoalState.ACTIVE

    # Simulate the move
    npc.move(action.dx, action.dy)

    # Second turn: the goal should persist (ContinueGoalAction should win)
    action2 = npc.ai.get_action(controller, npc)
    assert isinstance(action2, MoveIntent), (
        f"Expected MoveIntent on turn 2, got {type(action2)}"
    )

    # Goal should still be active
    assert npc.current_goal is not None
    assert npc.current_goal.state == GoalState.ACTIVE


def test_flee_goal_completes_at_safe_distance() -> None:
    """Fleeing NPC completes goal when reaching safe distance."""
    controller, _player, npc = make_world(npc_x=2, npc_y=0, npc_hp_damage=4)

    # Get initial flee action to create the goal
    npc.ai.get_action(controller, npc)
    assert npc.current_goal is not None

    flee_goal = npc.current_goal

    # Teleport NPC to safe distance
    npc.teleport(_FLEE_SAFE_DISTANCE + 1, 0)

    # Next AI evaluation should detect safe distance and complete the flee goal.
    # The NPC may then pick a new action (e.g., patrol) on the same tick.
    npc.ai.get_action(controller, npc)

    # Flee goal should be completed and no longer the active goal
    assert flee_goal.state == GoalState.COMPLETED
    assert npc.current_goal is not flee_goal


def test_flee_goal_abandoned_when_threat_dies() -> None:
    """A fleeing NPC whose threat dies mid-flee abandons the goal."""
    controller, player, npc = make_world(npc_x=2, npc_y=0, npc_hp_damage=4)

    # Create the flee goal
    npc.ai.get_action(controller, npc)
    assert npc.current_goal is not None
    assert isinstance(npc.current_goal, FleeGoal)

    # Simulate the first move
    npc.move(1, 0)

    # Kill the threat (player)
    player.take_damage(999)
    assert not player.health.is_alive()

    # Next evaluation: goal should complete because threat is dead
    npc.ai.get_action(controller, npc)
    assert npc.current_goal is None


def test_npc_attacks_when_no_escape_route() -> None:
    """With no escape route, FleeAction's score collapses and attack wins.

    NPC at (1,0) next to player at (0,0) with all escape tiles blocked.
    has_escape_route is False, so FleeAction's STEP consideration zeroes
    out. Attack wins and no FleeGoal is created.
    """
    controller, _player, npc = make_world(npc_x=1, npc_y=0, npc_hp_damage=4)

    # Block all tiles away from the player
    gm = controller.gw.game_map
    gm.walkable[2, 0] = False
    gm.walkable[2, 1] = False
    gm.walkable[1, 1] = False

    # NPC at (1,0), player at (0,0). The only open tiles are toward
    # the player, so has_escape_route is False and flee scores 0.
    action = npc.ai.get_action(controller, npc)

    # Attack should win - no FleeGoal created
    assert npc.current_goal is None
    assert action is not None


# ---------------------------------------------------------------------------
# ContinueGoalAction / Persistence Bonus Tests
# ---------------------------------------------------------------------------


def test_continue_goal_persistence_bonus_scales_with_progress() -> None:
    """ContinueGoalAction persistence bonus increases with goal.progress."""
    controller, player, npc = make_world(npc_x=3, npc_y=0)

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
    """ContinueGoalAction wins ties with the raw action even at progress 0.0.

    Without PERSISTENCE_MINIMUM, a goal at progress 0.0 would tie with the
    raw FleeAction. UnifiedAI would then abandon the goal and recreate it
    every tick, so progress would never advance. The minimum bonus ensures
    ContinueGoal always edges out the raw action it mirrors.
    """
    import pytest

    from brileta.game.actors.ai import FleeAction
    from brileta.game.actors.goals import PERSISTENCE_MINIMUM

    controller, player, npc = make_world(npc_x=3, npc_y=0)

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

    # FleeAction with the same considerations and base_score as FleeGoal
    flee_action = FleeAction(
        base_score=goal.get_base_score(),
        considerations=goal.get_considerations(),
        preconditions=[],
    )

    flee_score = flee_action.score(context)
    continue_score = ContinueGoalAction(goal).score(context)

    # ContinueGoal should beat the raw action by exactly PERSISTENCE_MINIMUM
    assert continue_score > flee_score
    assert continue_score - flee_score == pytest.approx(PERSISTENCE_MINIMUM)


# ---------------------------------------------------------------------------
# PatrolGoal Tests
# ---------------------------------------------------------------------------


def test_patrol_goal_advances_waypoints() -> None:
    """PatrolGoal advances to next waypoint when current one is reached."""
    controller, _player, npc = make_world(npc_x=5, npc_y=5)
    waypoints = [(5, 5), (10, 5), (10, 10)]

    goal = PatrolGoal(waypoints)
    npc.current_goal = goal

    # NPC is already at first waypoint (5,5)
    # get_next_action should advance to next waypoint and start a plan
    goal.get_next_action(npc, controller)

    # After being at waypoint 0, it should advance to waypoint 1
    assert goal._current_waypoint_index == 1

    # NPC should have an active plan targeting waypoint 1
    assert npc.active_plan is not None
    assert npc.active_plan.context.target_position == (10, 5)


def test_patrol_goal_cycles_back_to_start() -> None:
    """PatrolGoal wraps around to first waypoint after completing the route."""
    controller, _player, npc = make_world(npc_x=5, npc_y=5)
    waypoints = [(5, 5), (10, 5)]

    goal = PatrolGoal(waypoints)

    # Simulate reaching waypoint 0 (start)
    npc.teleport(5, 5)
    goal.get_next_action(npc, controller)
    assert goal._current_waypoint_index == 1

    # Simulate reaching waypoint 1
    npc.teleport(10, 5)
    npc.active_plan = None  # Clear plan so it creates a new one
    goal.get_next_action(npc, controller)

    # Should wrap back to waypoint 0
    assert goal._current_waypoint_index == 0
    assert goal._laps_completed == 1


def test_patrol_goal_progress_tracks_route() -> None:
    """PatrolGoal progress reflects position in the current lap.

    Without an NPC reference, progress falls back to waypoint-index only.
    """
    waypoints = [(0, 0), (5, 0), (5, 5), (0, 5)]
    goal = PatrolGoal(waypoints)

    assert goal.progress == 0.0  # At start, no NPC yet

    goal._current_waypoint_index = 2
    assert goal.progress == 0.5  # Halfway through (index-only)

    goal._current_waypoint_index = 3
    assert goal.progress == 0.75  # Three-quarters through (index-only)


def test_patrol_goal_progress_includes_spatial_distance() -> None:
    """PatrolGoal progress includes how close the NPC is to the current waypoint.

    With an NPC reference and initial waypoint distance tracked, progress
    smoothly advances between waypoints rather than jumping in discrete steps.
    """
    controller, _player, npc = make_world(npc_x=5, npc_y=5)
    waypoints = [(5, 5), (10, 5), (10, 10)]

    goal = PatrolGoal(waypoints)
    npc.current_goal = goal

    # NPC at (5,5) which is waypoint 0 - get_next_action advances to waypoint 1
    goal.get_next_action(npc, controller)
    assert goal._current_waypoint_index == 1

    # NPC is 5 tiles from waypoint 1 (10,5). Progress should be 1/3 (waypoint 1)
    # plus partial approach: distance=5, initial=5, so leg_progress=0.0
    assert goal.progress > 0.3  # At least past waypoint 0
    assert goal.progress < 0.67  # Not yet at waypoint 1

    # Move NPC halfway to waypoint 1
    npc.teleport(8, 5)  # 2 tiles from (10,5), started at 5 tiles away
    progress_mid = goal.progress
    assert progress_mid > goal._current_waypoint_index / len(waypoints)

    # Move NPC closer
    npc.teleport(9, 5)  # 1 tile from (10,5)
    progress_close = goal.progress
    assert progress_close > progress_mid  # Progress increased


def test_patrol_goal_requires_minimum_waypoints() -> None:
    """PatrolGoal raises ValueError with fewer than 2 waypoints."""
    import pytest

    with pytest.raises(ValueError, match="at least 2 waypoints"):
        PatrolGoal([(0, 0)])


# ---------------------------------------------------------------------------
# Goal Abandonment Tests
# ---------------------------------------------------------------------------


def test_goal_abandoned_when_higher_action_wins() -> None:
    """An active goal is abandoned when a higher-scoring action wins.

    Scenario: NPC is fleeing, then gets healed to full health. Attack should
    now score higher than continue-fleeing, causing the flee goal to be abandoned.
    """
    controller, _player, npc = make_world(npc_x=2, npc_y=0, npc_hp_damage=4)

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
    """When a goal completes at the top of UnifiedAI.get_action, the NPC
    proceeds to normal scoring and picks a new action on the same tick
    rather than doing nothing.
    """
    controller, _player, npc = make_world(npc_x=2, npc_y=0, npc_hp_damage=4)

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
    # With no threat and hostile disposition, IdleAction is the only option.
    assert result is None or isinstance(result, MoveIntent)
    assert npc.current_goal is None


def test_completed_goal_excluded_from_continue_scoring() -> None:
    """A goal in a terminal state should not produce a ContinueGoalAction.

    UtilityBrain.select_action guards on `not current_goal.is_complete`.
    Verify a FAILED goal is correctly excluded from scoring.
    """
    from brileta.game.actors.ai import IdleAction
    from brileta.game.actors.utility import UtilityBrain

    controller, player, npc = make_world(npc_x=5, npc_y=0)

    brain = UtilityBrain(
        [
            IdleAction(
                base_score=0.2,
                considerations=[
                    Consideration(
                        "threat_level",
                        ResponseCurve(ResponseCurveType.INVERSE),
                    ),
                ],
            ),
        ]
    )

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


def test_flee_goal_returns_none_when_cornered_midgoal() -> None:
    """When an NPC is cornered mid-flee, the flee goal should fail or be
    abandoned, and the NPC should gracefully fall back to another action.
    """
    controller, _player, npc = make_world(npc_x=2, npc_y=0, npc_hp_damage=4)

    # Create a flee goal
    action = npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, FleeGoal)
    original_flee_goal = npc.current_goal
    if isinstance(action, MoveIntent):
        npc.move(action.dx, action.dy)

    # Now wall off all escape routes so the goal can't flee further
    gm = controller.gw.game_map
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tx, ty = npc.x + dx, npc.y + dy
            if 0 <= tx < gm.width and 0 <= ty < gm.height:
                gm.walkable[tx, ty] = False

    # Next tick: flee has no escape route. The NPC should not crash.
    # The flee goal should no longer be ACTIVE (either failed, abandoned, or
    # replaced), and the NPC either picks a different action or skips the turn.
    npc.ai.get_action(controller, npc)

    # The flee goal must not remain active when fully cornered
    assert original_flee_goal.state != GoalState.ACTIVE, (
        "Cornered flee goal should not remain ACTIVE"
    )
    # If the NPC still has the flee goal, it must be in a terminal state
    if npc.current_goal is original_flee_goal:
        assert original_flee_goal.is_complete


# ---------------------------------------------------------------------------
# FleeGoal Progress Tests
# ---------------------------------------------------------------------------


def test_flee_goal_progress_starts_at_zero() -> None:
    """FleeGoal progress is 0 before first evaluation."""
    goal = FleeGoal(threat_actor_id=ActorId(12345))
    assert goal.progress == 0.0


def test_flee_goal_progress_already_at_safe_distance() -> None:
    """FleeGoal progress is 1.0 if start distance is already at safe distance."""
    goal = FleeGoal(threat_actor_id=ActorId(12345))
    goal._start_distance = _FLEE_SAFE_DISTANCE
    goal._current_distance = _FLEE_SAFE_DISTANCE
    assert goal.progress == 1.0

    # Also works if start distance exceeds safe distance
    goal._start_distance = _FLEE_SAFE_DISTANCE + 5
    goal._current_distance = _FLEE_SAFE_DISTANCE + 5
    assert goal.progress == 1.0


def test_flee_goal_progress_increases_with_distance() -> None:
    """FleeGoal progress increases as NPC moves away from threat."""
    goal = FleeGoal(threat_actor_id=ActorId(12345))
    goal._start_distance = 2
    goal._current_distance = 2
    assert goal.progress == 0.0

    goal._current_distance = 5
    expected = (5 - 2) / (_FLEE_SAFE_DISTANCE - 2)
    assert abs(goal.progress - expected) < 0.01

    goal._current_distance = _FLEE_SAFE_DISTANCE
    assert goal.progress == 1.0


# ---------------------------------------------------------------------------
# WanderAction / WanderGoal Tests
# ---------------------------------------------------------------------------


def test_wander_action_creates_wander_goal_when_safe() -> None:
    """Approachable NPC far from threat should create and start a WanderGoal."""
    controller, _player, npc = make_world(
        npc_x=15,
        npc_y=15,
        disposition=40,  # Friendly
    )

    assert npc.current_goal is None

    action = npc.ai.get_action(controller, npc)

    assert isinstance(npc.current_goal, WanderGoal)
    assert npc.current_goal.state == GoalState.ACTIVE
    assert isinstance(action, MoveIntent)


def test_wander_action_does_not_duplicate_existing_wander_goal() -> None:
    """When wander is active, ContinueGoalAction should keep the same goal."""
    controller, _player, npc = make_world(
        npc_x=15,
        npc_y=15,
        disposition=40,  # Friendly
    )

    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, WanderGoal)
    original_goal = npc.current_goal

    npc.ai.get_action(controller, npc)
    assert npc.current_goal is original_goal


def test_wander_goal_sidestep_when_forward_blocked() -> None:
    """WanderGoal should sidestep/turn when its current forward heading is blocked."""
    controller, _player, npc = make_world(
        npc_x=5,
        npc_y=5,
        disposition=40,  # Friendly
    )
    blocker = NPC(
        6,
        5,
        "b",
        colors.RED,
        "Blocker",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(blocker)

    goal = WanderGoal(
        minimum_segment_steps=5,
        maximum_segment_steps=5,
        max_stuck_turns=2,
        heading_jitter_chance=0.0,
    )
    # Force an east-facing segment to make forward direction deterministic.
    goal._heading = (1, 0)
    goal._segment_length = 5
    goal._steps_remaining = 5
    npc.current_goal = goal

    action = goal.get_next_action(npc, controller)
    assert isinstance(action, MoveIntent)
    assert (action.dx, action.dy) in {(0, -1), (0, 1), (1, -1), (1, 1)}


def test_wander_goal_restarts_segment_when_budget_exhausted() -> None:
    """WanderGoal should pick a fresh segment after spending its step budget."""
    controller, _player, npc = make_world(
        npc_x=5,
        npc_y=5,
        disposition=40,  # Friendly
    )

    goal = WanderGoal(
        minimum_segment_steps=1,
        maximum_segment_steps=1,
        max_stuck_turns=2,
        heading_jitter_chance=0.0,
    )
    npc.current_goal = goal

    first = goal.get_next_action(npc, controller)
    assert isinstance(first, MoveIntent)
    assert goal._steps_remaining == 0

    # Simulate successful movement so stuck tracking resets normally.
    npc.move(first.dx, first.dy)

    second = goal.get_next_action(npc, controller)
    assert isinstance(second, MoveIntent)
    assert goal._segment_length == 1
    assert goal._steps_remaining == 0


def test_wander_goal_restarts_heading_after_being_stuck() -> None:
    """WanderGoal should reset heading when blocked for max_stuck_turns."""
    controller, _player, npc = make_world(
        npc_x=5,
        npc_y=5,
        disposition=40,  # Friendly
    )
    game_map = controller.gw.game_map

    goal = WanderGoal(
        minimum_segment_steps=3,
        maximum_segment_steps=3,
        max_stuck_turns=1,
        heading_jitter_chance=0.0,
    )
    goal._heading = (1, 0)
    goal._segment_length = 3
    goal._steps_remaining = 3
    npc.current_goal = goal

    # Temporarily block all adjacent movement.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            game_map.walkable[npc.x + dx, npc.y + dy] = False

    blocked = goal.get_next_action(npc, controller)
    assert blocked is None
    assert goal._heading is None
    assert goal._steps_remaining == 0

    # Re-open one direction; next tick should pick a new segment and move.
    game_map.walkable[npc.x + 1, npc.y] = True
    resumed = goal.get_next_action(npc, controller)
    assert isinstance(resumed, MoveIntent)


# ---------------------------------------------------------------------------
# ai.force_hostile Tests
# ---------------------------------------------------------------------------


@contextmanager
def _override_live_variable(name: str, value: object):
    """Temporarily override (or register) a live variable to return *value*.

    Restores the original getter on exit so that test failures don't leak
    state into other tests.
    """
    from brileta.util.live_vars import live_variable_registry

    var = live_variable_registry.get_variable(name)
    if var is None:
        live_variable_registry.register(
            name,
            getter=lambda: value,
            setter=lambda v: None,
            description=f"Test override for {name}",
        )
        original_getter = None
    else:
        original_getter = var.getter
        var.getter = lambda: value

    try:
        yield
    finally:
        var = live_variable_registry.get_variable(name)
        if var is not None:
            if original_getter is not None:
                var.getter = original_getter
            else:
                var.getter = lambda: False


def test_force_hostile_makes_wary_npc_use_hostile_behavior() -> None:
    """With ai.force_hostile on, a Wary NPC should score hostile actions."""
    with _override_live_variable("ai.force_hostile", True):
        gw = DummyGameWorld()
        player = Character(
            0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        npc = NPC(
            3,
            0,
            "g",
            colors.RED,
            "Guard",
            game_world=cast(GameWorld, gw),
        )
        gw.player = player
        gw.add_actor(player)
        gw.add_actor(npc)
        npc.ai.modify_disposition(player, -10)  # Wary
        controller = DummyController(gw)

        # With the player within aggro range, force_hostile should make
        # hostile combat behavior eligible (attack/chase).
        result = npc.ai.get_action(controller, npc)
        assert result is not None or npc.active_plan is not None


# ---------------------------------------------------------------------------
# NPC Death Clears Goal
# ---------------------------------------------------------------------------


def test_npc_death_clears_current_goal() -> None:
    """Killing an NPC with an active goal should clear current_goal."""
    _controller, _player, npc = make_world(npc_x=3, npc_y=0)
    npc.current_goal = FleeGoal(threat_actor_id=ActorId(123))

    npc.take_damage(999)
    assert not npc.health.is_alive()
    assert npc.current_goal is None


# ---------------------------------------------------------------------------
# PatrolGoal Plan Reuse
# ---------------------------------------------------------------------------


def test_patrol_goal_reuses_existing_plan_for_same_waypoint() -> None:
    """PatrolGoal should not create a new plan if one already targets the current waypoint."""
    controller, _player, npc = make_world(npc_x=5, npc_y=5)
    waypoints = [(10, 5), (10, 10)]

    goal = PatrolGoal(waypoints)
    npc.current_goal = goal

    # First call creates a plan to waypoint 0
    goal.get_next_action(npc, controller)
    assert npc.active_plan is not None
    first_plan = npc.active_plan
    assert first_plan.context.target_position == (10, 5)

    # Second call with same waypoint should reuse the existing plan
    goal.get_next_action(npc, controller)
    assert npc.active_plan is first_plan


# ---------------------------------------------------------------------------
# FleeGoal Escape Direction Tests
# ---------------------------------------------------------------------------


def test_flee_goal_moves_away_from_threat() -> None:
    """FleeGoal should pick the tile that maximizes distance from the threat."""
    # Player (threat) at (0,0), NPC at (3,0). Escape should move to (4,0) or
    # diagonally away - any direction that increases Chebyshev distance.
    controller, player, npc = make_world(npc_x=3, npc_y=0)

    goal = FleeGoal(threat_actor_id=player.actor_id)
    goal.evaluate_completion(npc, controller)

    intent = goal.get_next_action(npc, controller)
    assert isinstance(intent, MoveIntent)
    # dx must be positive (moving away from player at x=0)
    assert intent.dx > 0 or (intent.dx == 0 and intent.dy != 0)


def test_flee_goal_avoids_blocked_tiles() -> None:
    """FleeGoal should skip tiles blocked by other actors."""
    controller, player, npc = make_world(npc_x=3, npc_y=0)
    gw = controller.gw

    # Block the direct escape tile at (4,0) with a wall-like actor
    blocker = NPC(
        4,
        0,
        "B",
        colors.WHITE,
        "Blocker",
        game_world=cast(GameWorld, gw),
    )
    gw.add_actor(blocker)

    goal = FleeGoal(threat_actor_id=player.actor_id)
    goal.evaluate_completion(npc, controller)

    intent = goal.get_next_action(npc, controller)
    assert isinstance(intent, MoveIntent)
    # Should not try to move into (4,0) where the blocker is
    assert not (intent.dx == 1 and intent.dy == 0)
    # But should still flee (diagonal or other direction)
    new_x = npc.x + intent.dx
    new_y = npc.y + intent.dy
    from brileta.game import ranges

    assert ranges.calculate_distance(new_x, new_y, player.x, player.y) > 3


def test_flee_goal_fails_when_fully_cornered() -> None:
    """FleeGoal.get_next_action should set FAILED when no escape tile exists."""
    # NPC at (1,0), player at (0,0). Block all tiles that increase distance.
    controller, player, npc = make_world(npc_x=1, npc_y=0)
    gm = controller.gw.game_map

    # Wall off everything to the right and diagonal
    gm.walkable[2, 0] = False
    gm.walkable[2, 1] = False
    gm.walkable[1, 1] = False

    goal = FleeGoal(threat_actor_id=player.actor_id)
    goal.evaluate_completion(npc, controller)

    intent = goal.get_next_action(npc, controller)
    assert intent is None
    assert goal.state == GoalState.FAILED


# ---------------------------------------------------------------------------
# UtilityBrain.select_action with Goals
# ---------------------------------------------------------------------------


def test_utility_brain_includes_continue_goal_action() -> None:
    """UtilityBrain.select_action should score ContinueGoalAction alongside atomics."""
    from brileta.game.actors.ai import IdleAction
    from brileta.game.actors.utility import UtilityBrain

    controller, player, npc = make_world(npc_x=15, npc_y=15)

    # Create a simple brain with just Idle
    brain = UtilityBrain(
        [
            IdleAction(
                base_score=0.2,
                considerations=[
                    Consideration(
                        "threat_level",
                        ResponseCurve(ResponseCurveType.INVERSE),
                    ),
                ],
            ),
        ]
    )

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


# ---------------------------------------------------------------------------
# Combat Awareness Tests
# ---------------------------------------------------------------------------


def test_combat_awareness_selects_attacker_outside_aggro_range() -> None:
    """An NPC shot from outside aggro range should target the attacker."""
    controller, player, npc = make_world(
        npc_x=20, npc_y=0, npc_hp_damage=0, disposition=0
    )

    # Player is at (0,0), NPC at (20,0) - well outside aggro_radius (10).
    # Without combat awareness, the NPC would just see the player as a
    # neutral fallback target with threat_level 0.
    context_before = npc.ai._build_context(controller, npc)
    assert context_before.threat_level == 0.0

    # Simulate being attacked: set hostile and notify.
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    # Now the NPC should perceive the attacker as a threat.
    context_after = npc.ai._build_context(controller, npc)
    assert context_after.threat_level > 0.0
    assert context_after.target is player


def test_combat_awareness_threat_decays_with_distance() -> None:
    """Awareness threat should be higher when closer, zero at flee distance."""
    controller, player, npc = make_world(
        npc_x=15, npc_y=0, npc_hp_damage=0, disposition=0
    )
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    # At distance 15 (outside aggro 10, inside flee distance 50)
    context_15 = npc.ai._build_context(controller, npc)
    threat_at_15 = context_15.threat_level

    # Teleport further away
    npc.teleport(40, 0)
    context_40 = npc.ai._build_context(controller, npc)
    threat_at_40 = context_40.threat_level

    assert threat_at_15 > threat_at_40 > 0.0

    # At or beyond _FLEE_SAFE_DISTANCE, threat should be 0
    npc.teleport(_FLEE_SAFE_DISTANCE, 0)
    context_safe = npc.ai._build_context(controller, npc)
    assert context_safe.threat_level == 0.0


def test_combat_awareness_npc_flees_from_ranged_attacker() -> None:
    """NPC attacked from range should flee, not keep wandering."""
    controller, player, npc = make_world(
        npc_x=15, npc_y=0, npc_hp_damage=4, disposition=0
    )

    # Before being attacked: NPC wanders (no threat)
    npc.ai.get_action(controller, npc)
    # NPC should have a WanderGoal (neutral, no threat)
    assert npc.current_goal is None or isinstance(npc.current_goal, WanderGoal)

    # Clear goal state for clean test
    if npc.current_goal is not None:
        npc.current_goal.abandon()
        npc.current_goal = None
    npc.active_plan = None

    # Simulate ranged attack
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    # NPC should now flee
    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, FleeGoal)


def test_combat_awareness_clears_when_attacker_dies() -> None:
    """Awareness should clear when the attacker is no longer alive."""
    controller, player, npc = make_world(
        npc_x=20, npc_y=0, npc_hp_damage=0, disposition=0
    )
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    assert npc.ai._last_attacker_id is not None

    # Kill the attacker
    player.health._hp = 0

    # Target selection should clear the awareness
    npc.ai._select_target_actor(controller, npc)
    assert npc.ai._last_attacker_id is None
