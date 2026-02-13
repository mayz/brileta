"""Tests for FleeGoal behavior.

Validates:
- FleeGoal persists across multiple turns until safe distance reached.
- FleeGoal completes when threat dies mid-flee.
- FleeGoal progress tracking.
- Escape direction selection and blocked-tile avoidance.
- Cornered NPC fallback behavior.
"""

from typing import cast

from brileta import colors
from brileta.game import ranges
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC
from brileta.game.actors.ai.behaviors.flee import _FLEE_SAFE_DISTANCE, FleeGoal
from brileta.game.actors.ai.goals import GoalState
from brileta.game.game_world import GameWorld
from brileta.types import DIRECTIONS, ActorId
from tests.helpers import make_ai_world

# ---------------------------------------------------------------------------
# FleeGoal Persistence Tests
# ---------------------------------------------------------------------------


def test_flee_goal_persists_across_multiple_turns() -> None:
    """A fleeing NPC continues fleeing across multiple turns until safe."""
    controller, _player, npc = make_ai_world(npc_x=2, npc_y=0, npc_hp_damage=4)

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
    controller, _player, npc = make_ai_world(npc_x=2, npc_y=0, npc_hp_damage=4)

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
    controller, player, npc = make_ai_world(npc_x=2, npc_y=0, npc_hp_damage=4)

    # Create the flee goal
    npc.ai.get_action(controller, npc)
    assert npc.current_goal is not None
    assert isinstance(npc.current_goal, FleeGoal)

    # Simulate the first move
    npc.move(1, 0)

    # Kill the threat (player)
    player.take_damage(999)
    assert not player.health.is_alive()

    # Next evaluation: goal should complete because threat is dead.
    # The NPC may immediately pick up a new goal (e.g., WanderGoal) on
    # the same tick, so just verify the flee goal is gone.
    npc.ai.get_action(controller, npc)
    assert not isinstance(npc.current_goal, FleeGoal)


def test_npc_attacks_when_no_escape_route() -> None:
    """With no escape route, FleeAction's score collapses and attack wins.

    NPC at (1,0) next to player at (0,0) with all escape tiles blocked.
    has_escape_route is False, so FleeAction's STEP consideration zeroes
    out. Attack wins and no FleeGoal is created.
    """
    controller, _player, npc = make_ai_world(npc_x=1, npc_y=0, npc_hp_damage=4)

    # Block all tiles that increase or maintain distance from the player.
    # This includes lateral tiles (same distance) to truly corner the NPC.
    gm = controller.gw.game_map
    gm.walkable[2, 0] = False
    gm.walkable[2, 1] = False
    gm.walkable[1, 1] = False
    gm.walkable[0, 1] = False  # lateral (same distance as current)

    # NPC at (1,0), player at (0,0). The only open tiles are toward
    # the player, so has_escape_route is False and flee scores 0.
    action = npc.ai.get_action(controller, npc)

    # Attack should win - no FleeGoal created
    assert npc.current_goal is None
    assert action is not None


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
# FleeGoal Escape Direction Tests
# ---------------------------------------------------------------------------


def test_flee_goal_moves_away_from_threat() -> None:
    """FleeGoal should pick the tile that maximizes distance from the threat."""
    # Player (threat) at (0,0), NPC at (3,0). Escape should move to (4,0) or
    # diagonally away - any direction that increases Chebyshev distance.
    controller, player, npc = make_ai_world(npc_x=3, npc_y=0)

    goal = FleeGoal(threat_actor_id=player.actor_id)
    goal.evaluate_completion(npc, controller)

    intent = goal.get_next_action(npc, controller)
    assert isinstance(intent, MoveIntent)
    # dx must be positive (moving away from player at x=0)
    assert intent.dx > 0 or (intent.dx == 0 and intent.dy != 0)


def test_flee_goal_avoids_blocked_tiles() -> None:
    """FleeGoal should skip tiles blocked by other actors."""
    controller, player, npc = make_ai_world(npc_x=3, npc_y=0)
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
    assert ranges.calculate_distance(new_x, new_y, player.x, player.y) > 3


def test_flee_goal_fails_when_fully_cornered() -> None:
    """FleeGoal.get_next_action should set FAILED when no escape tile exists."""
    # NPC at (1,0), player at (0,0). Block all non-approaching tiles.
    controller, player, npc = make_ai_world(npc_x=1, npc_y=0)
    gm = controller.gw.game_map

    # Wall off everything that increases or maintains distance.
    gm.walkable[2, 0] = False
    gm.walkable[2, 1] = False
    gm.walkable[1, 1] = False
    gm.walkable[0, 1] = False  # lateral (same distance as current)

    goal = FleeGoal(threat_actor_id=player.actor_id)
    goal.evaluate_completion(npc, controller)

    intent = goal.get_next_action(npc, controller)
    assert intent is None
    assert goal.state == GoalState.FAILED


def test_flee_goal_returns_none_when_cornered_midgoal() -> None:
    """When an NPC is cornered mid-flee, the flee goal should fail or be
    abandoned, and the NPC should gracefully fall back to another action.
    """
    controller, _player, npc = make_ai_world(npc_x=2, npc_y=0, npc_hp_damage=4)

    # Create a flee goal
    action = npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, FleeGoal)
    original_flee_goal = npc.current_goal
    if isinstance(action, MoveIntent):
        npc.move(action.dx, action.dy)

    # Now wall off all escape routes so the goal can't flee further
    gm = controller.gw.game_map
    for dx, dy in DIRECTIONS:
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
