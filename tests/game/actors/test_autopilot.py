from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.game.actions.base import GameIntent
from catley.game.actions.environment import OpenDoorIntent
from catley.game.actions.executors.movement import MoveExecutor
from catley.game.actions.movement import MoveIntent
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.pathfinding_goal import PathfindingGoal
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


def test_start_actor_pathfinding_success() -> None:
    controller, player = make_world()
    success = controller.start_actor_pathfinding(player, (1, 0))
    assert success
    goal = player.pathfinding_goal
    assert isinstance(goal, PathfindingGoal)
    assert goal.target_pos == (1, 0)
    assert goal._cached_path and goal._cached_path[-1] == (1, 0)


def test_autopilot_moves_and_triggers_final_intent() -> None:
    controller, player = make_world()
    final_intent = DummyFinalIntent(controller, player)
    controller.start_actor_pathfinding(player, (1, 0), final_intent=final_intent)

    action = cast(MoveIntent, player.get_next_action(controller))
    assert isinstance(action, MoveIntent)
    assert (action.dx, action.dy) == (1, 0)

    MoveExecutor().execute(action)

    assert player.pathfinding_goal is None
    queued = controller.turn_manager.dequeue_player_action()
    assert queued is final_intent


def test_start_actor_pathfinding_handles_door() -> None:
    controller, player = make_world()
    gm = controller.gw.game_map
    gm.tiles[3, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    gm.invalidate_property_caches()

    success = controller.start_actor_pathfinding(
        player,
        (3, 0),
        final_intent=OpenDoorIntent(controller, player, 3, 0),
    )
    assert success
    assert player.pathfinding_goal is not None
    assert player.pathfinding_goal.target_pos == (2, 0)
