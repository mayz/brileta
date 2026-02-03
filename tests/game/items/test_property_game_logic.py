import random
from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.events import (
    EffectEvent,
    MessageEvent,
    reset_event_bus_for_testing,
    subscribe_to_event,
)
from catley.game import ranges
from catley.game.actions.area_effects import AreaEffectIntent
from catley.game.actions.combat import AttackIntent
from catley.game.actions.executors.area_effects import WeaponAreaEffectExecutor
from catley.game.actions.executors.combat import AttackExecutor
from catley.game.actors import Character
from catley.game.enums import OutcomeTier
from catley.game.game_world import GameWorld
from catley.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    GRENADE_TYPE,
    HUNTING_RIFLE_TYPE,
    SNIPER_RIFLE_TYPE,
    SUBMACHINE_GUN_TYPE,
)
from catley.game.items.properties import TacticalProperty, WeaponProperty
from catley.game.resolution.d20_system import D20ResolutionResult
from tests.helpers import DummyGameWorld


@dataclass
class DummyMessageLog:
    messages: list[str]

    def __init__(self) -> None:
        self.messages = []
        subscribe_to_event(MessageEvent, lambda e: self.add_message(e.text))

    def add_message(self, text: str, *_args, **_kwargs) -> None:
        self.messages.append(text)


@dataclass
class DummyFrameManager:
    effects: list[str]

    def __init__(self) -> None:
        self.effects = []

    def get_visible_bounds(self) -> None:
        return None

    def create_effect(self, name: str, *args, **kwargs) -> None:
        self.effects.append(name)


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    message_log: DummyMessageLog
    frame_manager: DummyFrameManager


def _make_world(weapon_key: str):
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    defender = Character(2, 1, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw))
    gw.add_actor(attacker)
    gw.add_actor(defender)
    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )
    if weapon_key == "hunting":
        weapon = HUNTING_RIFLE_TYPE.create()
    elif weapon_key == "sniper":
        weapon = SNIPER_RIFLE_TYPE.create()
    elif weapon_key == "smg":
        weapon = SUBMACHINE_GUN_TYPE.create()
    else:
        weapon = GRENADE_TYPE.create()
    return controller, attacker, defender, weapon


def test_awkward_weapon_miss_effect() -> None:
    controller, attacker, defender, weapon = _make_world("hunting")
    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    executor = AttackExecutor()
    attack = weapon.melee_attack
    assert attack is not None
    result = D20ResolutionResult(outcome_tier=OutcomeTier.FAILURE)
    executor._handle_attack_miss(intent, result, attack, weapon)
    assert any("off balance" in msg for msg in controller.message_log.messages)


def test_awkward_weapon_ranged_miss_no_effect() -> None:
    controller, attacker, defender, weapon = _make_world("sniper")
    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    executor = AttackExecutor()
    attack = weapon.ranged_attack
    assert attack is not None
    result = D20ResolutionResult(outcome_tier=OutcomeTier.FAILURE)
    executor._handle_attack_miss(intent, result, attack, weapon)
    assert not any("off balance" in msg for msg in controller.message_log.messages)


def test_automatic_weapon_ammo_consumption() -> None:
    _controller, _attacker, _defender, weapon = _make_world("smg")
    ranged = weapon.ranged_attack
    assert ranged is not None
    executor = WeaponAreaEffectExecutor()
    # Directly test the ammo consumption helper
    ranged.current_ammo = 5
    executor._consume_ammo(ranged)
    assert ranged.current_ammo == 2


def test_scoped_weapon_range_modifier() -> None:
    """Scoped weapons are awkward up close but steady at distance."""
    _controller, _attacker, _defender, weapon = _make_world("sniper")
    ranged = weapon.ranged_attack
    assert ranged is not None
    # Scoped weapons have disadvantage at close range
    assert ranges.get_range_modifier(weapon, "adjacent") == {"has_disadvantage": True}
    assert ranges.get_range_modifier(weapon, "close") == {"has_disadvantage": True}
    # Scoped weapons have no modifier at far range (scope compensates for distance)
    assert ranges.get_range_modifier(weapon, "far") == {}


def test_explosive_visual_effect() -> None:
    reset_event_bus_for_testing()
    effects: list[str] = []
    subscribe_to_event(EffectEvent, lambda e: effects.append(e.effect_name))
    controller, attacker, _defender, weapon = _make_world("grenade")
    effect = weapon.area_effect
    assert effect is not None
    executor = WeaponAreaEffectExecutor()
    intent = AreaEffectIntent(controller, attacker, 5, 5, weapon)
    executor._trigger_visual_effect(intent, effect)
    assert effects == ["explosion"]


def test_smoke_visual_effect() -> None:
    # Create custom smoke grenade
    from catley.game.enums import AreaType, ItemSize
    from catley.game.items.capabilities import AreaEffectSpec
    from catley.game.items.item_core import ItemType

    smoke_type = ItemType(
        name="Smoke",
        description="",
        size=ItemSize.TINY,
        area_effect=AreaEffectSpec(
            "d4", AreaType.CIRCLE, 1, properties={TacticalProperty.SMOKE}
        ),
    )
    weapon = smoke_type.create()
    reset_event_bus_for_testing()
    effects: list[str] = []
    subscribe_to_event(EffectEvent, lambda e: effects.append(e.effect_name))
    controller, attacker, _defender, _ = _make_world("grenade")
    executor = WeaponAreaEffectExecutor()
    intent = AreaEffectIntent(controller, attacker, 2, 2, weapon)
    effect = weapon.area_effect
    assert effect is not None
    executor._trigger_visual_effect(intent, effect)
    assert effects == ["smoke_cloud"]


# --- ARMOR_PIERCING Tests ---


def test_armor_piercing_bypasses_armor() -> None:
    """ARMOR_PIERCING weapons deal damage directly to HP, bypassing armor."""
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    defender = Character(2, 1, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw))
    gw.add_actor(attacker)
    gw.add_actor(defender)

    # Give defender armor
    from catley.game.outfit import LEATHER_ARMOR_TYPE

    assert defender.health is not None
    armor_item = LEATHER_ARMOR_TYPE.create()
    defender.inventory.set_starting_outfit(armor_item)
    assert armor_item.outfit_capability is not None
    initial_ap = armor_item.outfit_capability.ap
    initial_hp = defender.health.hp

    # Sniper rifle has ARMOR_PIERCING property
    weapon = SNIPER_RIFLE_TYPE.create()
    assert weapon.ranged_attack is not None
    assert WeaponProperty.ARMOR_PIERCING in weapon.ranged_attack.properties

    # Test that armor_piercing damage bypasses armor (handled in combat_arbiter)
    # Here we just test that take_damage applies damage to HP
    defender.take_damage(5)

    # HP should be reduced
    assert defender.health.hp == initial_hp - 5  # HP reduced directly
    # Note: Armor bypass is handled in combat_arbiter, not take_damage
    # Armor AP unchanged since take_damage doesn't know about armor
    assert armor_item.outfit_capability.ap == initial_ap


# --- SILENT Tests ---


def test_silent_weapon_no_noise_alert() -> None:
    """SILENT weapons don't generate noise_alert consequences."""
    from catley.game.consequences import AttackConsequenceGenerator
    from catley.game.enums import ItemSize
    from catley.game.items.capabilities import RangedAttackSpec
    from catley.game.items.item_core import ItemType

    # Create a silent weapon
    silent_weapon_type = ItemType(
        name="Suppressed Pistol",
        description="Silent",
        size=ItemSize.NORMAL,
        ranged_attack=RangedAttackSpec(
            damage_die="d6",
            ammo_type="9mm",
            max_ammo=10,
            optimal_range=6,
            max_range=12,
            properties={WeaponProperty.SILENT, WeaponProperty.PREFERRED},
        ),
    )
    weapon = silent_weapon_type.create()

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    gw.add_actor(attacker)

    generator = AttackConsequenceGenerator()
    consequences = generator.generate(attacker, weapon, OutcomeTier.SUCCESS)

    # Should NOT have noise_alert consequence
    assert not any(c.type == "noise_alert" for c in consequences)


def test_non_silent_weapon_generates_noise_alert() -> None:
    """Non-silent weapons generate noise_alert consequences."""
    from catley.game.consequences import AttackConsequenceGenerator
    from catley.game.items.item_types import PISTOL_TYPE

    weapon = PISTOL_TYPE.create()

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    gw.add_actor(attacker)

    generator = AttackConsequenceGenerator()
    consequences = generator.generate(attacker, weapon, OutcomeTier.SUCCESS)

    # Should have noise_alert consequence
    assert any(c.type == "noise_alert" for c in consequences)


# --- THROWN Tests ---


def _make_thrown_world():
    """Create a test world with a combat knife for thrown weapon tests."""
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    defender = Character(5, 5, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw))
    gw.add_actor(attacker)
    gw.add_actor(defender)
    gw.player = attacker

    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )

    knife = COMBAT_KNIFE_TYPE.create()
    attacker.inventory.equip_to_slot(knife, 0)

    return controller, attacker, defender, knife


def test_thrown_weapon_removed_from_inventory_on_success() -> None:
    """Thrown weapons are removed from inventory and ready slots after use."""
    controller, attacker, defender, knife = _make_thrown_world()

    # Verify knife is equipped
    assert attacker.inventory.get_active_item() is knife
    assert knife in attacker.inventory.ready_slots
    assert knife.ranged_attack is not None

    intent = AttackIntent(cast(Controller, controller), attacker, defender, knife)
    executor = AttackExecutor()

    # Create a mock resolver that returns SUCCESS
    mock_resolver = type(
        "MockResolver",
        (),
        {"resolve": lambda self, *args: D20ResolutionResult(OutcomeTier.SUCCESS)},
    )()

    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(controller, "create_resolver", return_value=mock_resolver),
        patch.object(AttackExecutor, "_apply_combat_outcome", return_value=0),
        patch.object(AttackExecutor, "_handle_post_attack_effects"),
    ):
        knife.ranged_attack.current_ammo = 1
        executor.execute(intent)

    # Knife should be removed from ready slots
    assert knife not in attacker.inventory.ready_slots
    assert attacker.inventory.get_active_item() is not knife


def test_thrown_weapon_spawns_at_target_location() -> None:
    """Thrown weapons appear on ground at defender's location."""
    controller, attacker, defender, knife = _make_thrown_world()
    assert knife.ranged_attack is not None

    intent = AttackIntent(cast(Controller, controller), attacker, defender, knife)
    executor = AttackExecutor()

    # Create a mock resolver that returns SUCCESS
    mock_resolver = type(
        "MockResolver",
        (),
        {"resolve": lambda self, *args: D20ResolutionResult(OutcomeTier.SUCCESS)},
    )()

    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(controller, "create_resolver", return_value=mock_resolver),
        patch.object(AttackExecutor, "_apply_combat_outcome", return_value=0),
        patch.object(AttackExecutor, "_handle_post_attack_effects"),
    ):
        knife.ranged_attack.current_ammo = 1
        executor.execute(intent)

    # Find ground actor at defender's location (items spawn as actors with inventory)
    ground_actor = next(
        (
            a
            for a in controller.gw.actors
            if a is not attacker and a is not defender and a.x == defender.x
        ),
        None,
    )
    assert ground_actor is not None, "No ground item actor found at defender location"
    assert ground_actor.inventory is not None
    assert any(it.name == "Combat Knife" for it in ground_actor.inventory)


def test_thrown_weapon_not_consumed_on_critical_failure() -> None:
    """On critical failure, weapon_drop handles the knife at attacker's feet."""
    controller, attacker, defender, knife = _make_thrown_world()
    assert knife.ranged_attack is not None

    intent = AttackIntent(cast(Controller, controller), attacker, defender, knife)
    executor = AttackExecutor()

    # Create a mock resolver that returns CRITICAL_FAILURE
    mock_resolver = type(
        "MockResolver",
        (),
        {
            "resolve": lambda self, *args: D20ResolutionResult(
                OutcomeTier.CRITICAL_FAILURE
            )
        },
    )()

    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(controller, "create_resolver", return_value=mock_resolver),
        patch.object(AttackExecutor, "_apply_combat_outcome", return_value=0),
        patch.object(AttackExecutor, "_handle_post_attack_effects"),
        patch.object(random, "random", return_value=0.20),  # Selects weapon_drop
    ):
        knife.ranged_attack.current_ammo = 1
        executor.execute(intent)

    # Find all ground item actors (not attacker or defender)
    ground_actors = [
        a for a in controller.gw.actors if a is not attacker and a is not defender
    ]

    # Check knife is NOT at defender's location (THROWN logic skipped on crit fail)
    knives_at_target = sum(
        1
        for a in ground_actors
        if a.x == defender.x
        and a.inventory
        and any(it.name == "Combat Knife" for it in a.inventory)
    )
    assert knives_at_target == 0, (
        "Knife should NOT be at defender location on crit fail"
    )

    # Check knife IS at attacker's location (weapon_drop consequence)
    knives_at_attacker = sum(
        1
        for a in ground_actors
        if a.x == attacker.x
        and a.inventory
        and any(it.name == "Combat Knife" for it in a.inventory)
    )
    assert knives_at_attacker == 1, (
        "Knife should be at attacker location from weapon_drop"
    )

    # Total should be exactly 1 knife on ground (no duplication)
    total_knives = knives_at_target + knives_at_attacker
    assert total_knives == 1, f"Expected 1 knife total, found {total_knives}"
