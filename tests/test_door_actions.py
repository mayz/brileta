from dataclasses import dataclass
from typing import cast

from game.game_world import GameWorld

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.environment.map import GameMap
from catley.game.actions.environment import CloseDoorAction, OpenDoorAction
from catley.game.actors import Character
from catley.util.spatial import SpatialHashGrid


class DummyGameWorld:
    def __init__(self) -> None:
        self.game_map = GameMap(3, 3)
        self.game_map.tiles[:] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
        self.actors: list[Character] = []
        self.player: Character | None = None
        self.actor_spatial_index = SpatialHashGrid(cell_size=16)

    def add_actor(self, actor: Character) -> None:
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)

    def remove_actor(self, actor: Character) -> None:
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            pass


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
    gw.actors.append(player)
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
