from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actions.movement import MoveIntent
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None

    def __post_init__(self) -> None:
        self.update_fov_called = False
        self.message_log = type("Log", (), {"add_message": lambda *a, **kw: None})()

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


def test_turn_manager_updates_fov_using_action_result() -> None:
    controller, player = _make_world()
    tm = TurnManager(cast(Controller, controller))

    intent = MoveIntent(cast(Controller, controller), player, 1, 0)
    tm._execute_action(intent)
    assert controller.update_fov_called

    controller.update_fov_called = False
    # Move into same tile (no actual move) by using dx=0
    intent = MoveIntent(cast(Controller, controller), player, 0, 0)
    tm._execute_action(intent)
    assert not controller.update_fov_called
