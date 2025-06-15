import types
from dataclasses import dataclass
from typing import Any, cast

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.environment.map import GameMap
from catley.game import ranges
from catley.game.actions.base import GameAction
from catley.game.actions.discovery import (
    ActionCategory,
    ActionContext,
    ActionDiscovery,
    ActionOption,
    ActionRequirement,
    CombatIntentCache,
)
from catley.game.actions.environment import OpenDoorIntent
from catley.game.actors import Character, status_effects
from catley.game.enums import Disposition
from catley.game.game_world import GameWorld
from catley.game.items.capabilities import RangedAttack
from catley.game.items.item_types import COMBAT_KNIFE_TYPE, PISTOL_TYPE
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None
    message_log: object | None = None
    combat_intent_cache: CombatIntentCache | None = None

    def create_resolver(self, **kwargs: object) -> object:
        from catley.game.resolution.d20_system import D20System

        return D20System(**kwargs)  # type: ignore[call-arg]


def _make_context_world():
    gw = DummyGameWorld()
    gw.game_map = GameMap(30, 30)
    gw.game_map.gw = cast(GameWorld, gw)
    gw.game_map.tiles[:] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
    gw.game_map.visible[:] = True
    gw.items = {}
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    hostile = Character(
        5, 5, "H", colors.RED, "Hostile", game_world=cast(GameWorld, gw)
    )
    hostile.ai = cast(Any, types.SimpleNamespace(disposition=Disposition.HOSTILE))
    friend = Character(
        20, 20, "F", colors.WHITE, "Friend", game_world=cast(GameWorld, gw)
    )

    gw.add_actor(player)
    gw.add_actor(hostile)
    gw.add_actor(friend)
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
    ranged_attack.current_ammo = 1
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

    gw.add_actor(player)
    gw.add_actor(melee_target)
    gw.add_actor(ranged_target)
    gw.player = player

    controller = DummyController(gw=gw)
    return controller, player, melee_target, ranged_target, pistol


def test_get_combat_options_melee_ranged_and_reload() -> None:
    controller, player, melee_target, ranged_target, pistol = _make_combat_world()
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    opts = disc._get_combat_options(cast(Controller, controller), player, ctx)
    names = {o.name for o in opts}

    assert f"Melee attack with {pistol.name}" in names
    assert f"Ranged attack with {pistol.name}" in names
    assert f"Reload {pistol.name}" not in names

    inv_opts = disc._get_inventory_options(
        cast(Controller, controller),
        player,
        ctx,
    )
    inv_names = {o.name for o in inv_opts}
    assert f"Reload {pistol.name}" in inv_names

    melee_opt = next(o for o in opts if o.name.startswith("Melee attack"))
    expected_melee_prob = disc._calculate_combat_probability(
        cast(Controller, controller),
        player,
        melee_target,
        "strength",
    )
    assert melee_opt.success_probability == expected_melee_prob
    ranged_opt = next(o for o in opts if o.name.startswith("Ranged attack"))
    distance = ranges.calculate_distance(
        player.x, player.y, melee_target.x, melee_target.y
    )
    range_cat = ranges.get_range_category(distance, pistol)
    range_mods = ranges.get_range_modifier(pistol, range_cat)
    expected_ranged_prob = disc._calculate_combat_probability(
        cast(Controller, controller),
        player,
        melee_target,
        "observation",
        range_mods,
    )
    assert ranged_opt.success_probability == expected_ranged_prob


def test_get_combat_options_for_target_filters() -> None:
    controller, player, melee_target, ranged_target, pistol = _make_combat_world()
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    melee_opts = disc._get_combat_options_for_target(
        cast(Controller, controller), player, melee_target, ctx
    )
    ranged_only = disc._get_combat_options_for_target(
        cast(Controller, controller), player, ranged_target, ctx
    )

    assert len(melee_opts) == 2
    assert any(o.name.startswith("Pistol-Whip") for o in melee_opts)
    assert any(o.name.startswith("Shoot") for o in melee_opts)
    assert len(ranged_only) == 1
    assert ranged_only[0].name.startswith("Shoot")


def test_combat_options_ignore_dead_and_unseen() -> None:
    controller, player, melee_target, ranged_target, pistol = _make_combat_world()
    melee_target.health.hp = 0
    controller.gw.game_map.visible[ranged_target.x, ranged_target.y] = False
    controller.gw.game_map.tiles[2, 0] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]

    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)
    opts = disc._get_combat_options(cast(Controller, controller), player, ctx)
    names = {o.name for o in opts}
    assert names == {
        f"Melee attack with {pistol.name}",
        f"Ranged attack with {pistol.name}",
    }


def test_combat_option_probabilities_reflect_status_effects() -> None:
    controller, player, melee_target, ranged_target, pistol = _make_combat_world()
    player.status_effects.apply_status_effect(status_effects.OffBalanceEffect())
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    opts = disc._get_combat_options(cast(Controller, controller), player, ctx)
    melee_opt = next(o for o in opts if o.name.startswith("Melee attack"))
    expected_melee_prob = disc._calculate_combat_probability(
        cast(Controller, controller),
        player,
        melee_target,
        "strength",
    )
    assert melee_opt.success_probability == expected_melee_prob

    ranged_opt = next(o for o in opts if o.name.startswith("Ranged attack"))
    distance = ranges.calculate_distance(
        player.x, player.y, melee_target.x, melee_target.y
    )
    range_cat = ranges.get_range_category(distance, pistol)
    range_mods = ranges.get_range_modifier(pistol, range_cat)
    expected_ranged_prob = disc._calculate_combat_probability(
        cast(Controller, controller),
        player,
        melee_target,
        "observation",
        range_mods,
    )
    assert ranged_opt.success_probability == expected_ranged_prob


def test_sort_by_relevance_orders_actions() -> None:
    ctx = ActionContext(0, 0, [], [], True)
    opt1 = ActionOption(
        "id1",
        "a",
        "",
        ActionCategory.COMBAT,
        cast(type[GameAction], type(None)),
        requirements=[],
        static_params={},
        hotkey="a",
        success_probability=0.5,
    )
    opt2 = ActionOption(
        "id2",
        "b",
        "",
        ActionCategory.ITEMS,
        cast(type[GameAction], type(None)),
    )
    opt3 = ActionOption(
        "id3",
        "c",
        "",
        ActionCategory.COMBAT,
        cast(type[GameAction], type(None)),
        success_probability=0.8,
    )

    disc = ActionDiscovery()
    ordered = disc._sort_by_relevance([opt2, opt3, opt1], ctx)

    assert ordered[0] == opt1
    assert ordered[1] == opt3
    assert ordered[2] == opt2


def test_inventory_options_hide_weapon_switching_when_in_combat() -> None:
    controller, player, hostile, friend, _ = _make_context_world()
    pistol = PISTOL_TYPE.create()
    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.equip_to_slot(pistol, 0)
    player.inventory.equip_to_slot(knife, 1)
    disc = ActionDiscovery()

    ctx = disc._build_context(cast(Controller, controller), player)
    assert ctx.in_combat
    opts = disc._get_inventory_options(cast(Controller, controller), player, ctx)
    assert all(not o.name.startswith("Switch to") for o in opts)

    cast(Any, hostile.ai).disposition = Disposition.FRIENDLY  # Out of combat
    ctx = disc._build_context(cast(Controller, controller), player)
    assert not ctx.in_combat
    opts = disc._get_inventory_options(cast(Controller, controller), player, ctx)
    names = {o.name for o in opts}
    assert f"Switch to {knife.name}" in names


def test_target_specific_option_probabilities_reflect_status_effects() -> None:
    controller, player, melee_target, ranged_target, pistol = _make_combat_world()
    player.status_effects.apply_status_effect(status_effects.OffBalanceEffect())
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    melee_opts = disc._get_combat_options_for_target(
        cast(Controller, controller), player, melee_target, ctx
    )
    ranged_only = disc._get_combat_options_for_target(
        cast(Controller, controller), player, ranged_target, ctx
    )

    melee_opt = next(o for o in melee_opts if o.name.startswith("Pistol-Whip"))
    expected_melee_prob = disc._calculate_combat_probability(
        cast(Controller, controller),
        player,
        melee_target,
        "strength",
    )
    assert melee_opt.success_probability == expected_melee_prob

    ranged_opt = ranged_only[0]
    distance = ranges.calculate_distance(
        player.x, player.y, ranged_target.x, ranged_target.y
    )
    range_cat = ranges.get_range_category(distance, pistol)
    range_mods = ranges.get_range_modifier(pistol, range_cat)
    expected_ranged_prob = disc._calculate_combat_probability(
        cast(Controller, controller),
        player,
        ranged_target,
        "observation",
        range_mods,
    )
    assert ranged_opt.success_probability == expected_ranged_prob


def test_environment_options_include_door_actions() -> None:
    gw = DummyGameWorld()
    gw.game_map.tiles[1, 0] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    player = Character(0, 0, "@", colors.WHITE, "P", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw=gw)
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)
    opts = disc._get_environment_options(cast(Controller, controller), player, ctx)
    names = {o.name for o in opts}
    assert "Open Door" in names
    open_door_option = next(o for o in opts if o.name == "Open Door")
    assert open_door_option.action_class is OpenDoorIntent
    assert open_door_option.requirements == [ActionRequirement.TARGET_TILE]


def test_probability_descriptor_mapping() -> None:
    desc, color = ActionDiscovery.get_probability_descriptor(0.35)
    assert desc == "Unlikely"
    assert color == "orange"


def test_terminal_combat_actions_available_out_of_combat() -> None:
    """`_get_all_terminal_combat_actions` should work outside of combat."""
    controller, player, melee_target, ranged_target, _ = _make_combat_world()
    disc = ActionDiscovery()

    # No actor starts hostile so the player is technically out of combat.
    ctx = disc._build_context(cast(Controller, controller), player)
    assert not ctx.in_combat

    actions = disc._get_all_terminal_combat_actions(
        cast(Controller, controller), player
    )

    # Both targets should yield at least one attack option each.
    assert len(actions) >= 2
    assert all(callable(opt.execute) for opt in actions)
