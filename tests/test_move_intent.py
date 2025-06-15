from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.game.actions.executors.movement import MoveExecutor
from catley.game.actions.movement import MoveIntent
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


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
    gw.add_actor(player)
    controller = DummyController(gw=gw, frame_manager=object())
    return controller, player


def test_move_blocks_at_map_edges() -> None:
    controller, player = make_world()

    intent = MoveIntent(cast(Controller, controller), player, dx=-1, dy=0)
    result = MoveExecutor().execute(intent)
    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "out_of_bounds"
    assert (player.x, player.y) == (0, 0)

    player.x = controller.gw.game_map.width - 1
    intent = MoveIntent(cast(Controller, controller), player, dx=1, dy=0)
    result = MoveExecutor().execute(intent)
    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "out_of_bounds"
    assert player.x == controller.gw.game_map.width - 1


def test_move_checks_walkable_tiles() -> None:
    controller, player = make_world()
    gw = controller.gw
    player.x = 2
    player.y = 2
    # Make tile to the right a wall
    gw.game_map.tiles[3, 2] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
    intent = MoveIntent(cast(Controller, controller), player, dx=1, dy=0)
    result = MoveExecutor().execute(intent)
    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "wall"
    assert (player.x, player.y) == (2, 2)

    # Move to empty floor
    intent = MoveIntent(cast(Controller, controller), player, dx=-1, dy=0)
    result = MoveExecutor().execute(intent)
    assert result is not None
    assert result.succeeded
    assert (player.x, player.y) == (1, 2)


def test_move_checks_closed_door() -> None:
    controller, player = make_world()
    gw = controller.gw
    gw.game_map.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    intent = MoveIntent(cast(Controller, controller), player, dx=1, dy=0)
    result = MoveExecutor().execute(intent)
    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "door"
    assert (player.x, player.y) == (0, 0)


def test_move_updates_spatial_index() -> None:
    controller, player = make_world()
    gw = controller.gw

    assert gw.actor_spatial_index.get_at_point(0, 0) == [player]
    assert not gw.actor_spatial_index.get_at_point(1, 1)

    intent = MoveIntent(cast(Controller, controller), player, dx=1, dy=1)
    result = MoveExecutor().execute(intent)
    assert result is not None
    assert result.succeeded

    assert (player.x, player.y) == (1, 1)
    assert not gw.actor_spatial_index.get_at_point(0, 0)
    assert gw.actor_spatial_index.get_at_point(1, 1) == [player]
