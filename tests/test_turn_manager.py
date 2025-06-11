from dataclasses import dataclass
from typing import cast

from game.game_world import GameWorld
from game.turn_manager import TurnManager

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.environment.map import GameMap
from catley.game.actions.base import GameAction, GameActionResult
from catley.game.actors import Character


class DummyGameWorld:
    def __init__(self) -> None:
        self.game_map = GameMap(3, 3)
        self.game_map.tiles[:] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
        self.actors: list[Character] = []
        self.player: Character | None = None


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
    gw.actors.append(player)
    controller = DummyController(gw=gw)
    return controller, player


class DummyAction(GameAction):
    def __init__(self, controller: Controller, actor: Character, flag: bool) -> None:
        super().__init__(controller, actor)
        self.flag = flag

    def execute(self) -> GameActionResult | None:
        if self.flag:
            return GameActionResult(should_update_fov=True)
        return None


def test_turn_manager_updates_fov_using_action_result() -> None:
    controller, player = _make_world()
    tm = TurnManager(cast(Controller, controller))

    action = DummyAction(cast(Controller, controller), player, True)
    tm._execute_action(action)
    assert controller.update_fov_called

    controller.update_fov_called = False
    action = DummyAction(cast(Controller, controller), player, False)
    tm._execute_action(action)
    assert not controller.update_fov_called
