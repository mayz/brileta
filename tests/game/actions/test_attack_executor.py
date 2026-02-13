"""Tests for AttackExecutor.

Covers the _fire_weapon method (ammo consumption, inventory revision,
muzzle flash, gunfire sound, dry-fire click), thrown weapon handling,
attack verb resolution, and on-hit status effect application.
"""

from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock, patch

from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.executors.combat import AttackExecutor
from brileta.game.actors import Character
from brileta.game.actors.conditions import Sickness
from brileta.game.enums import ActionBlockReason, ItemSize
from brileta.game.game_world import GameWorld
from brileta.game.items.capabilities import (
    MeleeAttack,
    MeleeAttackSpec,
    RangedAttackSpec,
)
from brileta.game.items.item_core import Item, ItemType
from brileta.game.items.item_types import SCORPION_STING_TYPE
from brileta.game.items.properties import StatusProperty, WeaponProperty
from brileta.game.turn_manager import TurnManager
from brileta.view.presentation import PresentationEvent
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    """Minimal controller for attack executor tests."""

    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(cast(Controller, self))
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100


def make_ranged_weapon(
    name: str = "Test Pistol",
    ammo_type: str = "pistol",
    max_ammo: int = 6,
    current_ammo: int | None = None,
) -> Item:
    """Create a ranged weapon for testing."""
    item_type = ItemType(
        name=name,
        description="A test weapon",
        size=ItemSize.NORMAL,
        ranged_attack=RangedAttackSpec(
            damage_die="1d6",
            ammo_type=ammo_type,
            max_ammo=max_ammo,
            optimal_range=10,
            max_range=20,
        ),
    )
    weapon = Item(item_type)
    # Set current ammo if specified (defaults to max_ammo)
    if current_ammo is not None and weapon.ranged_attack is not None:
        weapon.ranged_attack.current_ammo = current_ammo
    return weapon


def make_melee_weapon(name: str = "Test Baton") -> Item:
    """Create a melee weapon for testing."""
    item_type = ItemType(
        name=name,
        description="A test melee weapon",
        size=ItemSize.NORMAL,
        melee_attack=MeleeAttackSpec(damage_die="1d4"),
    )
    return Item(item_type)


def make_attacker_with_weapon(weapon: Item) -> tuple[DummyController, Character]:
    """Create a world with an attacker who has the weapon equipped."""
    gw = DummyGameWorld()

    # Create attacker with decent stats for combat
    attacker = Character(
        5,
        5,
        "@",
        colors.WHITE,
        "Attacker",
        game_world=cast(GameWorld, gw),
        strength=10,
        agility=10,
    )
    gw.player = attacker
    gw.add_actor(attacker)

    # Equip weapon to the character's inventory
    attacker.inventory.equip_to_slot(weapon, 0)

    controller = DummyController(gw)
    return controller, attacker


class TestFireWeaponAmmoConsumption:
    """Tests for ammo consumption in _fire_weapon."""

    def test_fire_weapon_decrements_ammo(self) -> None:
        """Firing a weapon decreases current_ammo by 1."""
        weapon = make_ranged_weapon(current_ammo=6)
        controller, attacker = make_attacker_with_weapon(weapon)

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
        ):
            result = executor._fire_weapon(intent, weapon, 10, 5)

        assert result is True
        assert weapon.ranged_attack is not None
        assert weapon.ranged_attack.current_ammo == 5

    def test_fire_weapon_returns_false_when_empty(self) -> None:
        """Firing fails when weapon has no ammo."""
        weapon = make_ranged_weapon(current_ammo=0)
        controller, attacker = make_attacker_with_weapon(weapon)

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        published_events: list = []
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            result = executor._fire_weapon(intent, weapon, 10, 5)

        assert result is False
        # Should publish "empty" message
        messages = [e.text for e in published_events if hasattr(e, "text")]
        assert any("empty" in msg.lower() for msg in messages)

    def test_fire_weapon_returns_false_when_no_ranged_attack(self) -> None:
        """Firing fails when weapon has no ranged capability."""
        item_type = ItemType(
            name="Sword", description="A melee weapon", size=ItemSize.NORMAL
        )
        weapon = Item(item_type)
        # No ranged_attack set

        controller, attacker = make_attacker_with_weapon(weapon)

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
        ):
            result = executor._fire_weapon(intent, weapon, 10, 5)

        assert result is False


class TestFireWeaponInventoryRevision:
    """Tests for inventory revision updates in _fire_weapon."""

    def test_fire_weapon_increments_inventory_revision(self) -> None:
        """Firing a weapon increments inventory revision for UI cache invalidation."""
        weapon = make_ranged_weapon(current_ammo=6)
        controller, attacker = make_attacker_with_weapon(weapon)

        initial_revision = attacker.inventory.revision

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
        ):
            executor._fire_weapon(intent, weapon, 10, 5)

        # Revision should have increased
        assert attacker.inventory.revision > initial_revision

    def test_fire_weapon_revision_not_incremented_on_failure(self) -> None:
        """Inventory revision is not incremented when firing fails."""
        weapon = make_ranged_weapon(current_ammo=0)  # Empty
        controller, attacker = make_attacker_with_weapon(weapon)

        initial_revision = attacker.inventory.revision

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
        ):
            executor._fire_weapon(intent, weapon, 10, 5)

        # Revision should not have changed
        assert attacker.inventory.revision == initial_revision


class TestFireWeaponEffects:
    """Tests for muzzle flash and sound effects in _fire_weapon."""

    def test_fire_weapon_emits_muzzle_flash(self) -> None:
        """Firing a weapon emits a muzzle flash effect."""
        weapon = make_ranged_weapon(current_ammo=6)
        controller, attacker = make_attacker_with_weapon(weapon)

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        published_events: list = []
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            executor._fire_weapon(intent, weapon, 10, 5)

        # Find presentation event with muzzle flash
        presentation_events = [
            e for e in published_events if isinstance(e, PresentationEvent)
        ]
        assert len(presentation_events) >= 1

        # Check for muzzle flash effect
        has_muzzle_flash = any(
            any(
                getattr(eff, "effect_name", None) == "muzzle_flash"
                for eff in (pe.effect_events or [])
            )
            for pe in presentation_events
        )
        assert has_muzzle_flash

    def test_fire_weapon_emits_gunfire_sound(self) -> None:
        """Firing a weapon emits a gunfire sound."""
        weapon = make_ranged_weapon(current_ammo=6, ammo_type="pistol")
        controller, attacker = make_attacker_with_weapon(weapon)

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        published_events: list = []
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            executor._fire_weapon(intent, weapon, 10, 5)

        # Find presentation event with sounds
        presentation_events = [
            e for e in published_events if isinstance(e, PresentationEvent)
        ]
        assert len(presentation_events) >= 1

        # Check for gunfire sound
        has_sound = any(len(pe.sound_events or []) > 0 for pe in presentation_events)
        assert has_sound

    def test_fire_weapon_emits_dry_fire_when_empty(self) -> None:
        """Firing the last round emits a dry fire click."""
        weapon = make_ranged_weapon(current_ammo=1)  # Last round
        controller, attacker = make_attacker_with_weapon(weapon)

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        published_events: list = []
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            executor._fire_weapon(intent, weapon, 10, 5)

        # Weapon should now be empty
        assert weapon.ranged_attack is not None
        assert weapon.ranged_attack.current_ammo == 0

        # Should have two presentation events: muzzle flash + dry fire
        presentation_events = [
            e for e in published_events if isinstance(e, PresentationEvent)
        ]
        assert len(presentation_events) == 2

        # Check for dry fire sound
        has_dry_fire = any(
            any(
                getattr(sound, "sound_id", None) == "gun_dry_fire"
                for sound in (pe.sound_events or [])
            )
            for pe in presentation_events
        )
        assert has_dry_fire


class TestFireWeaponDirection:
    """Tests for muzzle flash direction calculation."""

    def test_fire_weapon_calculates_direction_from_target(self) -> None:
        """Muzzle flash direction is calculated from attacker to target."""
        weapon = make_ranged_weapon(current_ammo=6)
        controller, attacker = make_attacker_with_weapon(weapon)
        # Attacker is at (5, 5)

        intent = AttackIntent(
            cast(Controller, controller),
            attacker,
            defender=None,
            weapon=weapon,
        )

        executor = AttackExecutor()
        published_events: list = []
        with patch(
            "brileta.game.actions.executors.combat.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            # Target at (10, 5) - directly to the right
            executor._fire_weapon(intent, weapon, target_x=10, target_y=5)

        # Find muzzle flash effect
        presentation_events = [
            e for e in published_events if isinstance(e, PresentationEvent)
        ]
        muzzle_flash = None
        for pe in presentation_events:
            for eff in pe.effect_events or []:
                if getattr(eff, "effect_name", None) == "muzzle_flash":
                    muzzle_flash = eff
                    break

        assert muzzle_flash is not None
        # Direction should be (10-5, 5-5) = (5, 0)
        assert muzzle_flash.direction_x == 5
        assert muzzle_flash.direction_y == 0


# Note: Thrown weapon handling (removal from inventory, spawning at target)
# is comprehensively tested in tests/game/items/test_property_game_logic.py
# Those tests cover the full end-to-end attack resolution flow including:
# - Weapon removal on successful throw
# - Spawning at target location
# - Critical failure behavior (weapon_drop consequence)


def test_execute_melee_out_of_range_returns_not_adjacent_block_reason() -> None:
    """Out-of-range melee attacks should return the not_adjacent block reason."""
    weapon = make_melee_weapon()
    controller, attacker = make_attacker_with_weapon(weapon)
    defender = Character(
        9,
        5,
        "r",
        colors.RED,
        "Defender",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(defender)

    intent = AttackIntent(
        cast(Controller, controller),
        attacker,
        defender=defender,
        weapon=weapon,
        attack_mode="melee",
    )

    executor = AttackExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded is False
    assert result.block_reason == ActionBlockReason.NOT_ADJACENT


# --- Attack verb tests ---


class TestAttackVerb:
    """Tests for _get_attack_verb with the actual attack handler."""

    def test_verb_from_melee_attack_spec(self) -> None:
        """Attack verb is derived from the melee attack spec, not the intent mode."""
        sting_spec = MeleeAttackSpec("d6", {WeaponProperty.UNARMED}, verb="sting")
        attack = MeleeAttack(sting_spec)
        weapon = SCORPION_STING_TYPE.create()

        executor = AttackExecutor()
        # attack_mode is None (like NPC AI intents).
        verb = executor._get_attack_verb(attack, weapon, attack_mode=None)

        assert verb == "sting"

    def test_verb_fallback_to_hit_when_spec_lacks_verb(self) -> None:
        """Falls back to 'hit' when the attack spec has no verb attribute."""
        # Create a mock attack whose spec has no verb attribute.
        mock_attack = MagicMock()
        mock_attack._spec = object()  # Plain object, no verb attr
        mock_attack.properties = set()

        weapon = SCORPION_STING_TYPE.create()
        executor = AttackExecutor()
        verb = executor._get_attack_verb(mock_attack, weapon, attack_mode=None)

        assert verb == "hit"


# --- On-hit status effect tests ---


def _make_adjacent_combatants() -> tuple[DummyController, Character, Character]:
    """Create attacker at (5,5) and defender at (6,5) - adjacent."""
    gw = DummyGameWorld()
    attacker = Character(
        5, 5, "@", colors.WHITE, "Attacker", game_world=cast(GameWorld, gw)
    )
    defender = Character(
        6, 5, "D", colors.RED, "Defender", game_world=cast(GameWorld, gw)
    )
    gw.player = attacker
    gw.add_actor(attacker)
    gw.add_actor(defender)
    controller = DummyController(gw)
    return controller, attacker, defender


class TestOnHitStatusEffects:
    """Tests for on-hit status effect application."""

    def test_natural_weapon_poison_applies_venom(self) -> None:
        """A natural weapon (UNARMED + POISONING) inflicts Venom on failed save."""
        controller, attacker, defender = _make_adjacent_combatants()
        weapon = SCORPION_STING_TYPE.create()
        attack = weapon.melee_attack
        assert attack is not None

        intent = AttackIntent(
            cast(Controller, controller), attacker, defender=defender, weapon=weapon
        )

        executor = AttackExecutor()

        # Force the RNG: save roll succeeds, duration samples to 5.
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.0
        mock_rng.gauss.return_value = 5.0

        with (
            patch("brileta.game.actions.executors.combat._status_effect_rng", mock_rng),
            patch("brileta.game.actions.executors.combat.publish_event"),
        ):
            executor._apply_on_hit_effects(intent, attack)

        # Defender should have a Venom sickness condition with a duration.
        sickness_conditions = [
            c for c in defender.conditions if isinstance(c, Sickness)
        ]
        assert len(sickness_conditions) == 1
        assert sickness_conditions[0].sickness_type == "Venom"
        assert sickness_conditions[0].remaining_turns == 5

    def test_no_status_effect_on_non_poisonous_weapon(self) -> None:
        """A weapon without status properties does not inflict sickness."""
        controller, attacker, defender = _make_adjacent_combatants()
        # Plain melee weapon with no status properties.
        weapon = make_melee_weapon()
        attack = weapon.melee_attack
        assert attack is not None

        intent = AttackIntent(
            cast(Controller, controller), attacker, defender=defender, weapon=weapon
        )

        executor = AttackExecutor()
        with patch("brileta.game.actions.executors.combat.publish_event"):
            executor._apply_on_hit_effects(intent, attack)

        sickness_conditions = [
            c for c in defender.conditions if isinstance(c, Sickness)
        ]
        assert len(sickness_conditions) == 0

    def test_high_toughness_resists_venom(self) -> None:
        """A defender with high toughness can resist the status effect."""
        controller, attacker, defender = _make_adjacent_combatants()
        weapon = SCORPION_STING_TYPE.create()
        attack = weapon.melee_attack
        assert attack is not None

        intent = AttackIntent(
            cast(Controller, controller), attacker, defender=defender, weapon=weapon
        )

        executor = AttackExecutor()

        # Force the RNG to roll just above the apply threshold for max toughness.
        # Toughness 5 -> apply_chance = 0.5 - 5*0.05 = 0.25. Roll 0.3 > 0.25.
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.3
        defender.stats.strength = 0
        defender.stats.toughness = 5

        with (
            patch("brileta.game.actions.executors.combat._status_effect_rng", mock_rng),
            patch("brileta.game.actions.executors.combat.publish_event"),
        ):
            executor._apply_on_hit_effects(intent, attack)

        # Defender resisted - no sickness.
        sickness_conditions = [
            c for c in defender.conditions if isinstance(c, Sickness)
        ]
        assert len(sickness_conditions) == 0

    def test_poisoning_property_applies_poisoned_sickness(self) -> None:
        """A POISONING weapon inflicts Sickness('Poisoned')."""
        controller, attacker, defender = _make_adjacent_combatants()

        # Create a weapon with POISONING property.
        item_type = ItemType(
            name="Poison Blade",
            description="A poisoned knife.",
            size=ItemSize.NORMAL,
            melee_attack=MeleeAttackSpec("d4", {StatusProperty.POISONING}, verb="stab"),
        )
        weapon = Item(item_type)
        attack = weapon.melee_attack
        assert attack is not None

        intent = AttackIntent(
            cast(Controller, controller), attacker, defender=defender, weapon=weapon
        )

        executor = AttackExecutor()
        mock_rng = MagicMock()
        mock_rng.random.return_value = 0.0
        mock_rng.gauss.return_value = 5.0

        with (
            patch("brileta.game.actions.executors.combat._status_effect_rng", mock_rng),
            patch("brileta.game.actions.executors.combat.publish_event"),
        ):
            executor._apply_on_hit_effects(intent, attack)

        sickness_conditions = [
            c for c in defender.conditions if isinstance(c, Sickness)
        ]
        assert len(sickness_conditions) == 1
        assert sickness_conditions[0].sickness_type == "Poisoned"
        assert sickness_conditions[0].remaining_turns == 5
