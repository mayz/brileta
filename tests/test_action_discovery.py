import types
from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actions.discovery import (
    ActionCategory,
    ActionContext,
    ActionDiscovery,
    ActionOption,
)
from catley.game.actions.environment import OpenDoorAction
from catley.game.actors import Character
from catley.game.enums import Disposition
from catley.game.items.capabilities import RangedAttack
from catley.game.items.item_types import COMBAT_KNIFE_TYPE, PISTOL_TYPE
from catley.util import dice
from catley.world import tile_types
from catley.world.game_state import GameWorld
from catley.world.map import GameMap


class DummyGameWorld:
    def __init__(self) -> None:
        self.actors: list[Character] = []
        self.player: Character | None = None
        self.selected_actor: Character | None = None
        self.items: dict[tuple[int, int], list] = {}
        self.game_map = GameMap(5, 5)

        # Default to all floor tiles for simplicity
        self.game_map.tiles[:] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]

    def get_pickable_items_at_location(self, x: int, y: int) -> list:
        return self.items.get((x, y), [])


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None
    message_log: object | None = None


def _make_context_world():
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    hostile = Character(
        5, 5, "H", colors.RED, "Hostile", game_world=cast(GameWorld, gw)
    )
    hostile.ai = types.SimpleNamespace(disposition=Disposition.HOSTILE)
    friend = Character(
        20, 20, "F", colors.WHITE, "Friend", game_world=cast(GameWorld, gw)
    )

    gw.actors.extend([player, hostile, friend])
    gw.player = player
    gw.selected_actor = hostile
    knife = COMBAT_KNIFE_TYPE.create()
    gw.items[(0, 0)] = [knife]

    controller = DummyController(gw=gw)
    return controller, player, hostile, friend, knife


def test_build_context_collects_nearby_items_and_hostiles() -> None:
    controller, player, hostile, friend, knife = _make_context_world()
    disc = ActionDiscovery()

    ctx = disc._build_context(cast(Controller, controller), player)

    assert hostile in ctx.nearby_actors
    assert friend not in ctx.nearby_actors
    assert ctx.in_combat
    assert ctx.items_on_ground == [knife]
    assert ctx.selected_actor == hostile


def _make_combat_world():
    gw = DummyGameWorld()
    player = Character(
        0,
        0,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
        strength=5,
        observation=5,
    )
    pistol = PISTOL_TYPE.create()
    ranged_attack = cast(RangedAttack, pistol.ranged_attack)
    ranged_attack.current_ammo = 0
    player.inventory.equip_to_slot(pistol, 0)

    melee_target = Character(
        1,
        0,
        "M",
        colors.WHITE,
        "Melee",
        game_world=cast(GameWorld, gw),
        agility=3,
    )
    ranged_target = Character(
        4,
        0,
        "R",
        colors.WHITE,
        "Ranged",
        game_world=cast(GameWorld, gw),
        agility=3,
    )

    gw.actors.extend([player, melee_target, ranged_target])
    gw.player = player

    controller = DummyController(gw=gw)
    return controller, player, melee_target, ranged_target, pistol


def test_get_combat_options_melee_ranged_and_reload() -> None:
    controller, player, melee_target, ranged_target, pistol = _make_combat_world()
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    opts = disc._get_combat_options(cast(Controller, controller), player, ctx)
    names = {o.name for o in opts}

    assert f"Melee {melee_target.name} with {pistol.name}" in names
    assert f"Shoot {ranged_target.name} with {pistol.name}" in names
    assert f"Reload {pistol.name}" in names

    melee_opt = next(o for o in opts if o.name.startswith("Melee"))
    expected_melee_prob = dice.calculate_check_roll_success_probability(
        player.stats.strength,
        melee_target.stats.agility + 10,
    )
    assert melee_opt.success_probability == expected_melee_prob
    ranged_opt = next(o for o in opts if o.name.startswith("Shoot"))
    expected_ranged_prob = dice.calculate_check_roll_success_probability(
        player.stats.observation,
        ranged_target.stats.agility + 10,
    )
    assert ranged_opt.success_probability == expected_ranged_prob


def test_get_combat_options_for_target_filters() -> None:
    controller, player, melee_target, ranged_target, pistol = _make_combat_world()
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    melee_only = disc._get_combat_options_for_target(
        cast(Controller, controller), player, melee_target, ctx
    )
    ranged_only = disc._get_combat_options_for_target(
        cast(Controller, controller), player, ranged_target, ctx
    )

    assert len(melee_only) == 1
    assert melee_only[0].name.startswith("Melee")
    assert len(ranged_only) == 1
    assert ranged_only[0].name.startswith("Ranged")


def test_sort_by_relevance_orders_actions() -> None:
    ctx = ActionContext(0, 0, [], [], True)
    opt1 = ActionOption(
        "a", "", ActionCategory.COMBAT, hotkey="a", success_probability=0.5
    )
    opt2 = ActionOption("b", "", ActionCategory.ITEMS)
    opt3 = ActionOption("c", "", ActionCategory.COMBAT, success_probability=0.8)

    disc = ActionDiscovery()
    ordered = disc._sort_by_relevance([opt2, opt3, opt1], ctx)

    assert ordered[0] == opt1
    assert ordered[1] == opt3
    assert ordered[2] == opt2


def test_environment_options_include_door_actions() -> None:
    gw = DummyGameWorld()
    gw.game_map.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    player = Character(0, 0, "@", colors.WHITE, "P", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.actors.append(player)
    controller = DummyController(gw=gw)
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)
    opts = disc._get_environment_options(cast(Controller, controller), player, ctx)
    names = {o.name for o in opts}
    assert "Open Door" in names
    action = next(o for o in opts if o.name == "Open Door").execute()
    assert isinstance(action, OpenDoorAction)
