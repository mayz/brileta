from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actions import MoveAction
from catley.game.actors import Character
from catley.world import tile_types
from catley.world.game_state import GameWorld
from catley.world.map import GameMap


class DummyGameWorld:
    def __init__(self) -> None:
        self.game_map = GameMap(5, 5)
        self.game_map.tiles[:] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]  # All walkable
        self.actors: list[Character] = []
        self.player: Character | None = None


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None


def make_world() -> tuple[DummyController, Character]:
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
    controller = DummyController(gw=gw, frame_manager=object())
    return controller, player


def test_move_action_blocks_at_map_edges() -> None:
    controller, player = make_world()

    action = MoveAction(cast(Controller, controller), player, dx=-1, dy=0)
    action.execute()
    assert (player.x, player.y) == (0, 0)

    player.x = controller.gw.game_map.width - 1
    action = MoveAction(cast(Controller, controller), player, dx=1, dy=0)
    action.execute()
    assert player.x == controller.gw.game_map.width - 1


def test_move_action_checks_walkable_tiles() -> None:
    controller, player = make_world()
    gw = controller.gw
    player.x = 2
    player.y = 2
    # Make tile to the right a wall
    gw.game_map.tiles[3, 2] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[arg-type,attr-defined]
    action = MoveAction(cast(Controller, controller), player, dx=1, dy=0)
    action.execute()
    assert (player.x, player.y) == (2, 2)

    # Move to empty floor
    action = MoveAction(cast(Controller, controller), player, dx=-1, dy=0)
    action.execute()
    assert (player.x, player.y) == (1, 2)
