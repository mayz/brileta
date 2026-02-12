"""Tests for PatrolGoal behavior.

Validates:
- PatrolGoal advances waypoints and cycles back to start.
- Progress tracking (index-only and spatial).
- Plan reuse for the same waypoint.
- Minimum waypoint requirement.
"""

import pytest

from brileta.game.actors.ai.behaviors.patrol import PatrolGoal
from tests.helpers import make_ai_world


def test_patrol_goal_advances_waypoints() -> None:
    """PatrolGoal advances to next waypoint when current one is reached."""
    controller, _player, npc = make_ai_world(npc_x=5, npc_y=5)
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
    controller, _player, npc = make_ai_world(npc_x=5, npc_y=5)
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
    controller, _player, npc = make_ai_world(npc_x=5, npc_y=5)
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
    with pytest.raises(ValueError, match="at least 2 waypoints"):
        PatrolGoal([(0, 0)])


def test_patrol_goal_reuses_existing_plan_for_same_waypoint() -> None:
    """PatrolGoal should not create a new plan if one already targets the current waypoint."""
    controller, _player, npc = make_ai_world(npc_x=5, npc_y=5)
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
