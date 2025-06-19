from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.action_router import ActionRouter
from catley.game.actions.movement import MoveIntent
from catley.game.actors import NPC, Character
from catley.game.enums import Disposition
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyControllerAutopilot(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(cast(Controller, self))
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100

    def update_fov(self) -> None:
        pass


def _make_autopilot_world() -> tuple[DummyControllerAutopilot, Character]:
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    controller = DummyControllerAutopilot(gw)
    return controller, player


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(cast(Controller, self))
        self.action_cost = 100
        self.update_fov_called = False
        self.message_log = type("Log", (), {"add_message": lambda *a, **kw: None})()

    def start_actor_pathfinding(self, *args, **kwargs):
        return Controller.start_actor_pathfinding(
            cast(Controller, self), *args, **kwargs
        )

    def stop_actor_pathfinding(self, *args, **kwargs):
        return Controller.stop_actor_pathfinding(
            cast(Controller, self), *args, **kwargs
        )

    def update_fov(self) -> None:
        self.update_fov_called = True


def _make_world() -> tuple[DummyController, Character]:
    gw = DummyGameWorld()
    player = Character(
        0,
        0,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw=gw)
    return controller, player


def make_world() -> tuple[DummyController, Character, NPC]:
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        3,
        0,
        "g",
        colors.RED,
        "Enemy",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw=gw)
    return controller, player, npc


def test_turn_manager_updates_fov_using_action_result() -> None:
    controller, player = _make_world()
    router = ActionRouter(cast(Controller, controller))

    intent = MoveIntent(cast(Controller, controller), player, 1, 0)
    router.execute_intent(intent)
    assert controller.update_fov_called

    controller.update_fov_called = False
    # Move into an out-of-bounds tile to ensure failure without follow-ups
    intent = MoveIntent(cast(Controller, controller), player, -2, 0)
    router.execute_intent(intent)
    assert not controller.update_fov_called


def test_is_player_turn_available_with_goal() -> None:
    controller, player = _make_autopilot_world()
    tm = controller.turn_manager
    assert not tm.is_player_turn_available()
    controller.start_actor_pathfinding(player, (1, 0))
    assert tm.is_player_turn_available()


def test_process_unified_round_handles_autopilot() -> None:
    controller, player = _make_autopilot_world()
    tm = controller.turn_manager
    controller.start_actor_pathfinding(player, (1, 0))
    tm.process_unified_round()
    assert (player.x, player.y) == (1, 0)
    assert player.pathfinding_goal is None
    assert not tm.is_player_turn_available()


def test_npc_autopilot_waits_without_player_turn() -> None:
    controller, player, npc = make_world()
    tm = controller.turn_manager
    # Hostile NPC sets a goal toward the player
    npc.ai.get_action(cast(Controller, controller), npc)
    assert npc.pathfinding_goal is not None
    assert not tm.is_player_turn_available()

    tm.process_unified_round()
    # No movement should occur without player action
    assert (npc.x, npc.y) == (3, 0)
