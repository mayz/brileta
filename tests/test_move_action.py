from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.game.actions.movement import MoveAction
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    FISTS_TYPE,
    PISTOL_TYPE,
    SLEDGEHAMMER_TYPE,
    SNIPER_RIFLE_TYPE,
)
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


def test_ram_weapon_prefers_melee_over_ranged() -> None:
    controller, player = make_world()
    pistol = PISTOL_TYPE.create()
    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.equip_to_slot(pistol, 0)
    player.inventory.equip_to_slot(knife, 1)

    action = MoveAction(cast(Controller, controller), player, dx=1, dy=0)
    weapon = action._select_ram_weapon()

    assert weapon is knife


def test_ram_weapon_defaults_to_fists_when_only_ranged() -> None:
    controller, player = make_world()
    pistol = PISTOL_TYPE.create()
    player.inventory.equip_to_slot(pistol, 0)

    action = MoveAction(cast(Controller, controller), player, dx=1, dy=0)
    weapon = action._select_ram_weapon()

    assert weapon.item_type is FISTS_TYPE


def test_ram_weapon_ignores_ranged_preferred_weapons() -> None:
    controller, player = make_world()
    pistol = PISTOL_TYPE.create()
    sniper = SNIPER_RIFLE_TYPE.create()
    player.inventory.equip_to_slot(pistol, 0)
    player.inventory.equip_to_slot(sniper, 1)

    action = MoveAction(cast(Controller, controller), player, dx=1, dy=0)
    weapon = action._select_ram_weapon()

    assert weapon.item_type is FISTS_TYPE


def test_ram_weapon_prefers_active_weapon() -> None:
    controller, player = make_world()
    knife = COMBAT_KNIFE_TYPE.create()
    sledgehammer = SLEDGEHAMMER_TYPE.create()
    player.inventory.equip_to_slot(knife, 0)
    player.inventory.equip_to_slot(sledgehammer, 1)

    # Set sledgehammer as active weapon
    player.inventory.switch_to_weapon_slot(1)

    action = MoveAction(cast(Controller, controller), player, dx=1, dy=0)
    weapon = action._select_ram_weapon()

    # Should prefer the active sledgehammer over the knife
    assert weapon is sledgehammer


def test_move_action_opens_closed_door_when_bumped() -> None:
    controller, player = make_world()
    gw = controller.gw
    # Place a closed door to the right of the player
    gw.game_map.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]

    action = MoveAction(cast(Controller, controller), player, dx=1, dy=0)
    action.execute()

    # Player should have moved into the door tile and the door should now be open
    assert (player.x, player.y) == (1, 0)
    assert gw.game_map.tiles[1, 0] == tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]


def test_move_action_updates_spatial_index() -> None:
    """Verify that moving an actor updates its entry in the spatial index."""
    controller, player = make_world()
    gw = controller.gw

    assert gw.actor_spatial_index.get_at_point(0, 0) == [player]
    assert not gw.actor_spatial_index.get_at_point(1, 1)

    move_action = MoveAction(cast(Controller, controller), player, dx=1, dy=1)
    move_action.execute()

    assert (player.x, player.y) == (1, 1)
    assert not gw.actor_spatial_index.get_at_point(0, 0)
    assert gw.actor_spatial_index.get_at_point(1, 1) == [player]
