from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.game.action_router import ActionRouter
from catley.game.actions.base import GameActionResult
from catley.game.actions.movement import MoveIntent
from catley.game.actors import Character
from catley.game.game_world import GameWorld
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


def test_router_opens_door_on_bump() -> None:
    controller, player = _make_world()
    gm = controller.gw.game_map
    gm.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]

    router = ActionRouter(cast(Controller, controller))
    intent = MoveIntent(cast(Controller, controller), player, 1, 0)
    router.execute_intent(intent)

    assert gm.tiles[1, 0] == tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
    assert controller.update_fov_called


def test_router_attacks_actor_on_bump() -> None:
    controller, player = _make_world()
    enemy = Character(
        1,
        0,
        "E",
        colors.WHITE,
        "Enemy",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(enemy)

    with patch(
        "catley.game.actions.executors.combat.AttackExecutor.execute",
        return_value=GameActionResult(),
    ) as mock_execute:
        router = ActionRouter(cast(Controller, controller))
        intent = MoveIntent(cast(Controller, controller), player, 1, 0)
        router.execute_intent(intent)
        assert mock_execute.called
        assert (player.x, player.y) == (0, 0)
        assert not controller.update_fov_called
