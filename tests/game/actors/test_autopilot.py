from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.environment.tile_types import TileTypeID
from catley.game.action_plan import ActivePlan, ApproachStep, WalkToPlan
from catley.game.actions.base import GameIntent
from catley.game.actions.environment import OpenDoorPlan
from catley.game.actions.executors.movement import MoveExecutor
from catley.game.actions.movement import MoveIntent
from catley.game.actors import PC, Character
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100


def make_world() -> tuple[DummyController, Character]:
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw)
    return controller, player


class DummyFinalIntent(GameIntent):
    pass


def test_start_plan_success() -> None:
    """Test that start_plan creates an active plan."""
    controller, player = make_world()
    success = controller.start_plan(player, WalkToPlan, target_position=(1, 0))
    assert success
    plan = player.active_plan
    assert isinstance(plan, ActivePlan)
    assert plan.context.target_position == (1, 0)


def test_action_plan_approach_and_complete() -> None:
    """Test that ActionPlan approaches target and completes."""
    controller, player = make_world()
    controller.start_plan(player, WalkToPlan, target_position=(1, 0))

    # TurnManager should generate a MoveIntent from the plan
    intent = controller.turn_manager._get_intent_from_plan(player)
    assert isinstance(intent, MoveIntent)
    assert (intent.dx, intent.dy) == (1, 0)

    # Execute the move
    result = MoveExecutor().execute(intent)
    assert result is not None and result.succeeded

    # Simulate the TurnManager handling the approach result
    controller.turn_manager._on_approach_result(player, result)

    # Plan should complete since we arrived at destination
    assert player.active_plan is None


def test_open_door_plan_approach_then_open() -> None:
    """Test that door plan approaches then opens."""
    controller, player = make_world()
    gm = controller.gw.game_map

    # Place a closed door at (3, 0)
    gm.tiles[3, 0] = TileTypeID.DOOR_CLOSED
    gm.invalidate_property_caches()

    # Start an open door plan
    controller.start_plan(player, OpenDoorPlan, target_position=(3, 0))
    plan = player.active_plan
    assert plan is not None
    assert plan.context.target_position == (3, 0)

    # First step should be ApproachStep
    step = plan.get_current_step()
    assert isinstance(step, ApproachStep)
    assert step.stop_distance == 1  # Stop adjacent to door

    # Move toward door
    intent = controller.turn_manager._get_intent_from_plan(player)
    assert isinstance(intent, MoveIntent)


def test_plan_path_recalculation_when_blocked() -> None:
    """Test that plans recalculate path when blocked."""
    controller, player = make_world()
    gm = controller.gw.game_map

    # Start a walk plan
    controller.start_plan(player, WalkToPlan, target_position=(2, 0))

    # Get initial intent
    intent = controller.turn_manager._get_intent_from_plan(player)
    assert isinstance(intent, MoveIntent)

    # Block the path after getting the first move
    gm.walkable[1, 0] = False

    # Clear cached path to force recalculation
    plan = player.active_plan
    assert plan is not None
    plan.cached_path = None

    # Should find an alternate path or fail gracefully
    intent = controller.turn_manager._get_intent_from_plan(player)
    # Either finds alternate or returns None (and cancels plan)
    if intent is not None:
        assert isinstance(intent, MoveIntent)
        # Should not try to move to blocked tile
        assert not (intent.dx == 1 and intent.dy == 0)


def test_plan_single_step_path() -> None:
    """Test that single-step paths work correctly."""
    controller, player = make_world()

    # Create plan to adjacent tile
    controller.start_plan(player, WalkToPlan, target_position=(1, 0))

    intent = controller.turn_manager._get_intent_from_plan(player)
    assert isinstance(intent, MoveIntent)
    assert (intent.dx, intent.dy) == (1, 0)


def test_plan_unreachable_target() -> None:
    """Test that unreachable targets result in plan cancellation."""
    controller, player = make_world()
    gm = controller.gw.game_map

    # Surround the target position with walls to make it unreachable
    target = (5, 5)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            tx, ty = target[0] + dx, target[1] + dy
            if 0 <= tx < gm.width and 0 <= ty < gm.height:
                gm.tiles[tx, ty] = TileTypeID.WALL
    gm.invalidate_property_caches()

    # Create plan to unreachable target
    controller.start_plan(player, WalkToPlan, target_position=target)

    # Should return None (no path possible) and cancel plan
    intent = controller.turn_manager._get_intent_from_plan(player)
    assert intent is None
    assert player.active_plan is None


def test_stop_plan_cancels_plan() -> None:
    """Test that stop_plan cancels the active plan."""
    controller, player = make_world()

    controller.start_plan(player, WalkToPlan, target_position=(5, 5))
    assert player.active_plan is not None

    controller.stop_plan(player)
    assert player.active_plan is None


def test_manual_input_cancels_plan() -> None:
    """Test that manual input cancels the active plan.

    Uses PC (not Character) since the manual action cancellation logic
    is in PC.get_next_action().
    """
    gw = DummyGameWorld()
    player = PC(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw)

    # Start a plan
    controller.start_plan(player, WalkToPlan, target_position=(5, 5))
    assert player.active_plan is not None

    # Queue a manual action
    manual_intent = MoveIntent(controller, player, 0, 1)
    controller.queue_action(manual_intent)

    # Get next action should cancel plan and return manual action
    action = player.get_next_action(controller)
    assert action is manual_intent
    assert player.active_plan is None
