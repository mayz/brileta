"""Tests for WanderAction / WanderGoal behavior.

Validates:
- WanderAction creates WanderGoals that stroll with pauses and heading drift.
- Sidestep behavior when forward is blocked.
- Linger/pause mechanics.
- Heading changes and stuck recovery.
- Speed reduction and restoration on goal state transitions.
"""

from typing import cast
from unittest.mock import patch

from brileta import colors
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC
from brileta.game.actors.ai.behaviors.wander import WanderGoal
from brileta.game.actors.ai.goals import GoalState
from brileta.game.game_world import GameWorld
from brileta.types import DIRECTIONS
from tests.helpers import make_ai_world


def test_wander_action_creates_wander_goal_when_safe() -> None:
    """Approachable NPC far from threat should create and start a WanderGoal."""
    import brileta.game.actors.ai.behaviors.wander as wander_module

    controller, _player, npc = make_ai_world(
        npc_x=15,
        npc_y=15,
        disposition=40,  # Friendly
    )

    assert npc.current_goal is None

    with patch.object(wander_module._rng, "random", return_value=1.0):
        action = npc.ai.get_action(controller, npc)

    assert isinstance(npc.current_goal, WanderGoal)
    assert npc.current_goal.state == GoalState.ACTIVE
    assert isinstance(action, MoveIntent)


def test_wander_action_does_not_duplicate_existing_wander_goal() -> None:
    """When wander is active, ContinueGoalAction should keep the same goal."""
    import brileta.game.actors.ai.behaviors.wander as wander_module

    controller, _player, npc = make_ai_world(
        npc_x=15,
        npc_y=15,
        disposition=40,  # Friendly
    )

    with patch.object(wander_module._rng, "random", return_value=1.0):
        npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, WanderGoal)
    original_goal = npc.current_goal

    with patch.object(wander_module._rng, "random", return_value=1.0):
        npc.ai.get_action(controller, npc)
    assert npc.current_goal is original_goal


def test_wander_goal_sidestep_when_forward_blocked() -> None:
    """WanderGoal should sidestep/turn when its current forward heading is blocked."""
    controller, _player, npc = make_ai_world(
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
        pause_chance=0.0,
        minimum_linger_turns=1,
        maximum_linger_turns=1,
        new_heading_chance=0.0,
        max_stuck_turns=2,
        heading_jitter_chance=0.0,
    )
    # Force an east-facing heading to make forward direction deterministic.
    goal._heading = (1, 0)
    npc.current_goal = goal

    action = goal.get_next_action(npc, controller)
    assert isinstance(action, MoveIntent)
    assert (action.dx, action.dy) in {(0, -1), (0, 1), (1, -1), (1, 1)}


def test_wander_goal_can_linger_then_resume() -> None:
    """WanderGoal pauses for linger turns, then resumes movement."""
    controller, _player, npc = make_ai_world(
        npc_x=5,
        npc_y=5,
        disposition=40,  # Friendly
    )

    goal = WanderGoal(
        pause_chance=1.0,
        minimum_linger_turns=2,
        maximum_linger_turns=2,
        new_heading_chance=0.0,
        max_stuck_turns=2,
        heading_jitter_chance=0.0,
    )
    goal._heading = (1, 0)
    npc.current_goal = goal

    # First tick: pause is triggered and linger is scheduled.
    first = goal.get_next_action(npc, controller)
    assert first is None
    assert goal._linger_remaining == 1

    # Disable new pauses so we can verify movement resumes after linger.
    goal._pause_chance = 0.0
    second = goal.get_next_action(npc, controller)
    assert second is None
    assert goal._linger_remaining == 0

    third = goal.get_next_action(npc, controller)
    assert isinstance(third, MoveIntent)


def test_wander_goal_does_not_retrigger_pause_while_lingering() -> None:
    """When already lingering, pause RNG should not be consulted again."""
    import brileta.game.actors.ai.behaviors.wander as wander_module

    controller, _player, npc = make_ai_world(
        npc_x=5,
        npc_y=5,
        disposition=40,  # Friendly
    )

    goal = WanderGoal(
        pause_chance=1.0,
        minimum_linger_turns=1,
        maximum_linger_turns=1,
        new_heading_chance=0.0,
        max_stuck_turns=2,
        heading_jitter_chance=0.0,
    )
    goal._heading = (1, 0)
    goal._linger_remaining = 2

    with patch.object(wander_module._rng, "random") as random_mock:
        action = goal.get_next_action(npc, controller)

    assert action is None
    assert goal._linger_remaining == 1
    random_mock.assert_not_called()


def test_wander_goal_rolls_new_heading_based_on_probability() -> None:
    """Heading-change probability should trigger a full heading repick."""
    controller, _player, npc = make_ai_world(
        npc_x=5,
        npc_y=5,
        disposition=40,  # Friendly
    )

    goal = WanderGoal(
        pause_chance=0.0,
        minimum_linger_turns=1,
        maximum_linger_turns=1,
        new_heading_chance=1.0,
        max_stuck_turns=2,
        heading_jitter_chance=0.0,
    )
    goal._heading = (1, 0)

    with patch.object(goal, "_pick_segment_heading", return_value=(-1, 0)) as picker:
        action = goal.get_next_action(npc, controller)

    assert isinstance(action, MoveIntent)
    assert (action.dx, action.dy) == (-1, 0)
    picker.assert_called_once()


def test_wander_goal_restarts_heading_after_being_stuck() -> None:
    """WanderGoal should pick a fresh heading when stuck for max_stuck_turns."""
    controller, _player, npc = make_ai_world(
        npc_x=5,
        npc_y=5,
        disposition=40,  # Friendly
    )
    game_map = controller.gw.game_map

    goal = WanderGoal(
        pause_chance=0.0,
        minimum_linger_turns=1,
        maximum_linger_turns=1,
        new_heading_chance=0.0,
        max_stuck_turns=1,
        heading_jitter_chance=0.0,
    )
    goal._heading = (1, 0)
    npc.current_goal = goal

    # Temporarily block all adjacent movement.
    for dx, dy in DIRECTIONS:
        game_map.walkable[npc.x + dx, npc.y + dy] = False

    blocked = goal.get_next_action(npc, controller)
    assert blocked is None
    assert goal._stuck_turns == 1

    # Re-open one direction; next tick should repick heading and move.
    game_map.walkable[npc.x + 1, npc.y] = True
    with patch.object(goal, "_pick_segment_heading", return_value=(1, 0)) as picker:
        resumed = goal.get_next_action(npc, controller)

    assert isinstance(resumed, MoveIntent)
    assert (resumed.dx, resumed.dy) == (1, 0)
    picker.assert_called_once()


def test_wander_action_reduces_speed_on_goal_activation() -> None:
    """WanderAction should apply a reduced stroll speed when goal starts."""
    import brileta.game.actors.ai.behaviors.wander as wander_module

    controller, _player, npc = make_ai_world(
        npc_x=15,
        npc_y=15,
        disposition=40,  # Friendly
    )
    original_speed = npc.energy.speed

    with (
        patch.object(wander_module._rng, "uniform", return_value=0.60),
        patch.object(wander_module._rng, "random", return_value=1.0),
    ):
        npc.ai.get_action(controller, npc)

    assert isinstance(npc.current_goal, WanderGoal)
    assert npc.energy.speed == max(1, round(original_speed * 0.60))


def test_wander_goal_restores_speed_on_abandon() -> None:
    """WanderGoal abandonment should restore the actor's original speed."""
    _controller, _player, npc = make_ai_world(
        npc_x=15,
        npc_y=15,
        disposition=40,  # Friendly
    )
    original_speed = npc.energy.speed
    goal = WanderGoal(
        speed_min_multiplier=0.60,
        speed_max_multiplier=0.60,
    )

    goal.apply_wander_speed(npc)
    assert npc.energy.speed == max(1, round(original_speed * 0.60))

    goal.abandon()

    assert goal.state == GoalState.ABANDONED
    assert npc.energy.speed == original_speed


def test_wander_goal_restores_speed_on_completion() -> None:
    """Terminal wander states should restore the actor's original speed."""
    _controller, _player, npc = make_ai_world(
        npc_x=15,
        npc_y=15,
        disposition=40,  # Friendly
    )
    original_speed = npc.energy.speed
    goal = WanderGoal(
        speed_min_multiplier=0.60,
        speed_max_multiplier=0.60,
    )

    goal.apply_wander_speed(npc)
    assert npc.energy.speed == max(1, round(original_speed * 0.60))

    goal.state = GoalState.COMPLETED

    assert npc.energy.speed == original_speed


def test_wander_goal_restores_speed_on_failed() -> None:
    """FAILED wander state should restore the actor's original speed."""
    _controller, _player, npc = make_ai_world(
        npc_x=15,
        npc_y=15,
        disposition=40,  # Friendly
    )
    original_speed = npc.energy.speed
    goal = WanderGoal(
        speed_min_multiplier=0.60,
        speed_max_multiplier=0.60,
    )

    goal.apply_wander_speed(npc)
    assert npc.energy.speed == max(1, round(original_speed * 0.60))

    goal.state = GoalState.FAILED

    assert npc.energy.speed == original_speed
