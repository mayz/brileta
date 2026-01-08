from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.environment.tile_types import TileTypeID
from catley.game.actions.environment import CloseDoorIntent, OpenDoorIntent
from catley.game.actions.executors.doors import (
    CloseDoorExecutor,
    OpenDoorExecutor,
)
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
    gm.tiles[1, 0] = TileTypeID.DOOR_CLOSED
    _ = gm.walkable[1, 0]
    intent = OpenDoorIntent(cast(Controller, controller), player, 1, 0)
    executor = OpenDoorExecutor()
    executor.execute(intent)
    assert gm.tiles[1, 0] == TileTypeID.DOOR_OPEN
    assert gm.walkable[1, 0]
    assert gm.transparent[1, 0]


def test_close_door_action_closes_door() -> None:
    controller, player = _make_world()
    gm = controller.gw.game_map
    gm.tiles[1, 0] = TileTypeID.DOOR_OPEN
    _ = gm.walkable[1, 0]
    intent = CloseDoorIntent(cast(Controller, controller), player, 1, 0)
    executor = CloseDoorExecutor()
    executor.execute(intent)
    assert gm.tiles[1, 0] == TileTypeID.DOOR_CLOSED
    assert not gm.walkable[1, 0]
    assert not gm.transparent[1, 0]
