from dataclasses import dataclass
from typing import cast

import pytest

from brileta import colors
from brileta.controller import Controller
from brileta.environment.tile_types import TileTypeID
from brileta.game.actions.executors.movement import MoveExecutor
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import Character
from brileta.game.enums import StepBlock
from brileta.game.game_world import GameWorld
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
    gw.game_map.tiles[3, 2] = TileTypeID.WALL
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
    gw.game_map.tiles[1, 0] = TileTypeID.DOOR_CLOSED
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


def test_queued_moves_use_updated_position() -> None:
    """Queued movement intents should resolve using the actor's latest position."""
    controller, player = make_world()
    gw = controller.gw

    # Setup: open north and east but place a wall to the northeast.
    player.x = 1
    player.y = 1
    gw.game_map.tiles[2, 0] = TileTypeID.WALL
    gw.game_map.invalidate_property_caches()

    move_up = MoveIntent(cast(Controller, controller), player, 0, -1)
    move_right = MoveIntent(cast(Controller, controller), player, 1, 0)

    result_up = MoveExecutor().execute(move_up)
    assert result_up is not None and result_up.succeeded
    assert (player.x, player.y) == (1, 0)

    result_right = MoveExecutor().execute(move_right)
    assert result_right is not None
    assert not result_right.succeeded
    assert result_right.block_reason == "wall"
    # Player should remain at (1, 0) because the diagonal tile is a wall.
    assert (player.x, player.y) == (1, 0)


# --- _blocked_result mapping tests ---


@pytest.mark.parametrize(
    ("step_block", "expected_reason"),
    [
        (StepBlock.OUT_OF_BOUNDS, "out_of_bounds"),
        (StepBlock.WALL, "wall"),
        (StepBlock.CLOSED_DOOR, "door"),
        (StepBlock.BLOCKED_BY_ACTOR, "actor"),
        (StepBlock.BLOCKED_BY_CONTAINER, "container"),
    ],
)
def test_blocked_result_maps_step_block_to_reason(
    step_block: StepBlock,
    expected_reason: str,
) -> None:
    """Each StepBlock variant produces the correct block_reason string."""
    controller, player = make_world()
    intent = MoveIntent(cast(Controller, controller), player, dx=1, dy=0)
    result = MoveExecutor._blocked_result(step_block, intent)
    assert not result.succeeded
    assert result.block_reason == expected_reason


def test_move_blocked_by_container() -> None:
    """Moving into a container produces block_reason='container'."""
    from brileta.game.actors.container import Container

    controller, player = make_world()
    container = Container(1, 0, name="Crate", game_world=cast(GameWorld, controller.gw))
    controller.gw.add_actor(container)

    intent = MoveIntent(cast(Controller, controller), player, dx=1, dy=0)
    result = MoveExecutor().execute(intent)
    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "container"
    assert result.blocked_by is container
