from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from catley import colors
from catley.controller import Controller
from catley.environment.generators import GeneratedMapData
from catley.environment.map import GameMap
from catley.environment.tile_types import TileTypeID

if TYPE_CHECKING:
    from catley.environment.map import MapRegion
from catley.game import ranges
from catley.game.actions.base import GameIntent
from catley.game.actions.discovery import (
    ActionCategory,
    ActionContext,
    ActionDiscovery,
    ActionOption,
    ActionRequirement,
    CombatIntentCache,
)
from catley.game.actions.environment import OpenDoorIntent, SearchContainerIntent
from catley.game.actors import Character, status_effects
from catley.game.actors.container import create_bookcase
from catley.game.enums import Disposition
from catley.game.game_world import GameWorld
from catley.game.items.capabilities import RangedAttack
from catley.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    HUNTING_SHOTGUN_TYPE,
    PISTOL_TYPE,
)
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
    tiles = np.full((30, 30), TileTypeID.FLOOR, dtype=np.uint8, order="F")
    regions: dict[int, MapRegion] = {}
    map_data = GeneratedMapData(
        tiles=tiles,
        regions=regions,
        tile_to_region_id=np.full((30, 30), -1, dtype=np.int16, order="F"),
    )
    gw.game_map = GameMap(30, 30, map_data)
    gw.game_map.gw = cast(GameWorld, gw)
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
    controller, player, melee_target, _ranged_target, pistol = _make_combat_world()
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
    controller, player, melee_target, ranged_target, _pistol = _make_combat_world()
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
    controller.gw.game_map.tiles[2, 0] = TileTypeID.WALL

    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)
    opts = disc._get_combat_options(cast(Controller, controller), player, ctx)
    names = {o.name for o in opts}
    assert names == {
        f"Melee attack with {pistol.name}",
        f"Ranged attack with {pistol.name}",
    }


def test_combat_option_probabilities_reflect_status_effects() -> None:
    controller, player, melee_target, _ranged_target, pistol = _make_combat_world()
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
        cast(type[GameIntent], type(None)),
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
        cast(type[GameIntent], type(None)),
    )
    opt3 = ActionOption(
        "id3",
        "c",
        "",
        ActionCategory.COMBAT,
        cast(type[GameIntent], type(None)),
        success_probability=0.8,
    )

    disc = ActionDiscovery()
    ordered = disc._sort_by_relevance([opt2, opt3, opt1], ctx)

    assert ordered[0] == opt1
    assert ordered[1] == opt3
    assert ordered[2] == opt2


def test_inventory_options_hide_weapon_switching_when_in_combat() -> None:
    controller, player, hostile, _friend, _ = _make_context_world()
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
    gw.game_map.tiles[1, 0] = TileTypeID.DOOR_CLOSED
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
    # With a single door, it should not require tile selection
    assert open_door_option.requirements == []
    assert open_door_option.static_params == {"x": 1, "y": 0}


def test_environment_options_multiple_doors_require_selection() -> None:
    gw = DummyGameWorld()
    # Place two closed doors adjacent to the player
    gw.game_map.tiles[1, 0] = TileTypeID.DOOR_CLOSED
    gw.game_map.tiles[0, 1] = TileTypeID.DOOR_CLOSED
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
    # With multiple doors, it should require tile selection
    assert open_door_option.requirements == [ActionRequirement.TARGET_TILE]
    assert open_door_option.static_params == {}


def test_probability_descriptor_mapping() -> None:
    desc, color = ActionDiscovery.get_probability_descriptor(0.35)
    assert desc == "Unlikely"
    assert color == "orange"


def test_terminal_combat_actions_available_out_of_combat() -> None:
    """`_get_all_terminal_combat_actions` should work outside of combat."""
    controller, player, _melee_target, _ranged_target, _ = _make_combat_world()
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


def test_combat_options_exclude_self_targeting() -> None:
    """Combat options should not be available when targeting yourself."""
    controller, player, melee_target, _ranged_target, _pistol = _make_combat_world()
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    # Test that player cannot target themselves
    self_opts = disc._get_combat_options_for_target(
        cast(Controller, controller), player, player, ctx
    )

    # Should return empty list when targeting self
    assert len(self_opts) == 0

    # Verify that other targets still work normally
    other_opts = disc._get_combat_options_for_target(
        cast(Controller, controller), player, melee_target, ctx
    )
    assert len(other_opts) > 0


def test_tile_specific_door_actions() -> None:
    """Test that door actions are discovered for specific door tiles."""
    # Set up a game world with a door
    controller, player, _, _, _ = _make_combat_world()
    gw = controller.gw
    disc = ActionDiscovery()

    # Player is at (0, 0) by default
    assert player.x == 0
    assert player.y == 0

    # Build context for action discovery
    ctx = disc.context_builder.build_context(cast(Controller, controller), player)

    # Test 1: Adjacent door (player at 0,0, door at 1,0 - distance 1)
    gw.game_map.tiles[1, 0] = TileTypeID.DOOR_CLOSED

    adjacent_actions = disc.environment_discovery.discover_environment_actions_for_tile(
        cast(Controller, controller), player, ctx, 1, 0
    )

    # Should find direct door action for adjacent door (plus shoot action if armed)
    assert len(adjacent_actions) >= 1
    door_actions = [a for a in adjacent_actions if a.name == "Open Door"]
    assert len(door_actions) == 1
    adjacent_action = door_actions[0]
    assert "go to and" not in adjacent_action.name.lower()  # No movement required
    assert adjacent_action.action_class == OpenDoorIntent  # Direct action
    assert adjacent_action.execute is None  # No custom execute function

    # Test 2: Distant door (player at 0,0, door at 3,3 - distance 3)
    gw.game_map.tiles[3, 3] = TileTypeID.DOOR_CLOSED

    distant_actions = disc.environment_discovery.discover_environment_actions_for_tile(
        cast(Controller, controller), player, ctx, 3, 3
    )

    # Should find movement-required door action for distant door
    # (plus shoot action if player is armed)
    assert len(distant_actions) >= 1
    go_door_actions = [a for a in distant_actions if a.name == "Go to and Open Door"]
    assert len(go_door_actions) == 1
    distant_action = go_door_actions[0]
    assert "go to and" in distant_action.name.lower()  # Movement required
    assert distant_action.execute is not None  # Uses pathfinding execute function
    assert distant_action.action_class is None  # No direct action class

    # Test 3: Open door behavior
    gw.game_map.tiles[1, 0] = TileTypeID.DOOR_OPEN

    open_door_actions = (
        disc.environment_discovery.discover_environment_actions_for_tile(
            cast(Controller, controller), player, ctx, 1, 0
        )
    )

    # Should find close door action (plus shoot action if armed)
    assert len(open_door_actions) >= 1
    close_actions = [a for a in open_door_actions if "close" in a.name.lower()]
    assert len(close_actions) == 1
    close_action = close_actions[0]
    assert "close" in close_action.name.lower()

    # Test 4: Non-door tile (should return "Go here" action for walkable tiles)
    gw.game_map.tiles[1, 0] = TileTypeID.FLOOR
    gw.game_map.walkable[1, 0] = True  # Floor is walkable

    floor_actions = disc.environment_discovery.discover_environment_actions_for_tile(
        cast(Controller, controller), player, ctx, 1, 0
    )

    # Should find "Go here" action for a regular floor tile at distance > 0
    # (no shoot action since floor is walkable)
    assert len(floor_actions) == 1
    go_action = floor_actions[0]
    assert go_action.name == "Go here"
    assert go_action.description == "Walk to this location"
    assert go_action.execute is not None  # Uses pathfinding


def test_combat_actions_show_only_active_weapon() -> None:
    """Combat actions should only show the currently active weapon's attacks.

    The action panel shows actions for the active weapon only (not all equipped
    weapons) to reduce clutter and match Fallout 1 & 2's design. Players switch
    weapons explicitly to see different options.
    """
    # Set up player with two ranged weapons
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

    # Create and equip both weapons
    pistol = PISTOL_TYPE.create()
    shotgun = HUNTING_SHOTGUN_TYPE.create()

    # Give both weapons ammo
    pistol_ranged = cast(RangedAttack, pistol.ranged_attack)
    pistol_ranged.current_ammo = 6
    shotgun_ranged = cast(RangedAttack, shotgun.ranged_attack)
    shotgun_ranged.current_ammo = 2

    player.inventory.equip_to_slot(pistol, 0)
    player.inventory.equip_to_slot(shotgun, 1)

    # Create a target at ranged distance
    target = Character(
        4,
        0,
        "T",
        colors.RED,
        "Target",
        game_world=cast(GameWorld, gw),
        agility=3,
    )

    gw.add_actor(player)
    gw.add_actor(target)
    gw.player = player

    controller = DummyController(gw=gw)
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    # With pistol active (slot 0, the default), should only see pistol actions
    assert player.inventory.active_slot == 0
    opts = disc._get_combat_options_for_target(
        cast(Controller, controller), player, target, ctx
    )
    ranged_opts = [o for o in opts if "ranged" in o.id.lower()]

    # Should have exactly ONE ranged option (for active weapon only)
    assert len(ranged_opts) == 1, f"Expected 1 ranged option, got {len(ranged_opts)}"
    assert pistol.name in ranged_opts[0].id, (
        f"Expected pistol name '{pistol.name}' in action ID: {ranged_opts[0].id}"
    )

    # Switch to shotgun (slot 1) and verify actions change
    player.inventory.switch_to_slot(1)
    assert player.inventory.active_slot == 1

    ctx = disc._build_context(cast(Controller, controller), player)
    opts = disc._get_combat_options_for_target(
        cast(Controller, controller), player, target, ctx
    )
    ranged_opts = [o for o in opts if "ranged" in o.id.lower()]

    # Should now show shotgun actions instead
    assert len(ranged_opts) == 1, f"Expected 1 ranged option, got {len(ranged_opts)}"
    assert shotgun.name in ranged_opts[0].id, (
        f"Expected shotgun name '{shotgun.name}' in action ID: {ranged_opts[0].id}"
    )


def test_environment_options_include_container_search() -> None:
    """Test that search actions appear when player is adjacent to a container."""
    gw = DummyGameWorld()
    player = Character(0, 0, "@", colors.WHITE, "P", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)

    # Create a crate adjacent to the player
    crate = create_bookcase(x=1, y=0, game_world=cast(GameWorld, gw))
    gw.add_actor(crate)

    controller = DummyController(gw=gw)
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)
    opts = disc._get_environment_options(cast(Controller, controller), player, ctx)

    # Should find search action for adjacent container
    search_options = [o for o in opts if "Search" in o.name]
    assert len(search_options) == 1
    assert "Bookcase" in search_options[0].name
    assert search_options[0].action_class is SearchContainerIntent
    assert search_options[0].requirements == []
    assert search_options[0].static_params == {"target": crate}


def test_multiple_adjacent_containers_creates_multiple_options() -> None:
    """Test that multiple adjacent containers create separate search options."""
    gw = DummyGameWorld()
    player = Character(0, 0, "@", colors.WHITE, "P", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)

    # Create containers on two sides of the player
    bookcase1 = create_bookcase(x=1, y=0, game_world=cast(GameWorld, gw))
    bookcase2 = create_bookcase(x=0, y=1, game_world=cast(GameWorld, gw))
    gw.add_actor(bookcase1)
    gw.add_actor(bookcase2)

    controller = DummyController(gw=gw)
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)
    opts = disc._get_environment_options(cast(Controller, controller), player, ctx)

    search_options = [o for o in opts if "Search" in o.name]
    assert len(search_options) == 2
    # Both bookcases should have the same display name
    for opt in search_options:
        assert "Search Bookcase" in opt.name


def test_tile_specific_container_search_action() -> None:
    """Test that clicking on a container tile shows search action."""
    gw = DummyGameWorld()
    player = Character(0, 0, "@", colors.WHITE, "P", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)

    # Create a crate adjacent to the player
    crate = create_bookcase(x=1, y=0, game_world=cast(GameWorld, gw))
    gw.add_actor(crate)

    controller = DummyController(gw=gw)
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    opts = disc.environment_discovery.discover_environment_actions_for_tile(
        cast(Controller, controller), player, ctx, 1, 0
    )

    search_options = [o for o in opts if "Search" in o.name]
    assert len(search_options) == 1
    assert "Bookcase" in search_options[0].name
    assert search_options[0].action_class is SearchContainerIntent


def test_distant_container_offers_go_and_search() -> None:
    """Test that distant containers show 'Go to and Search' action."""
    gw = DummyGameWorld()
    player = Character(0, 0, "@", colors.WHITE, "P", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)

    # Create a crate far from the player
    crate = create_bookcase(x=5, y=5, game_world=cast(GameWorld, gw))
    gw.add_actor(crate)

    controller = DummyController(gw=gw)
    disc = ActionDiscovery()
    ctx = disc._build_context(cast(Controller, controller), player)

    opts = disc.environment_discovery.discover_environment_actions_for_tile(
        cast(Controller, controller), player, ctx, 5, 5
    )

    search_options = [o for o in opts if "Search" in o.name]
    assert len(search_options) == 1
    assert "Go to and Search" in search_options[0].name
    assert search_options[0].execute is not None  # Uses pathfinding
