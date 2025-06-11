from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.game.actions.environment import CloseDoorAction, OpenDoorAction
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None


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


def test_open_door_action_opens_door() -> None:
    controller, player = _make_world()
    gm = controller.gw.game_map
    gm.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    _ = gm.walkable[1, 0]
    action = OpenDoorAction(cast(Controller, controller), player, 1, 0)
    action.execute()
    assert gm.tiles[1, 0] == tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
    assert gm.walkable[1, 0]
    assert gm.transparent[1, 0]


def test_close_door_action_closes_door() -> None:
    controller, player = _make_world()
    gm = controller.gw.game_map
    gm.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
    _ = gm.walkable[1, 0]
    action = CloseDoorAction(cast(Controller, controller), player, 1, 0)
    action.execute()
    assert gm.tiles[1, 0] == tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    assert not gm.walkable[1, 0]
    assert not gm.transparent[1, 0]
