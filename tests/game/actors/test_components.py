"""Tests for actor component classes.

This module provides comprehensive unit tests for the component system defined
in components.py. Tests are organized by component class and focus on behaviors
not already covered by existing test files.

Existing Coverage (not duplicated here):
- test_inventory.py: InventoryComponent add/remove, equip/unequip, try_remove_item
- test_status_effects.py: StatusEffectsComponent expiration, condition management
- test_idle_animation.py: VisualEffectsComponent idle animation features
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from catley import colors
from catley.config import DEFAULT_ACTOR_SPEED, DEFAULT_MAX_ARMOR
from catley.constants.movement import MovementConstants
from catley.game.actors import conditions
from catley.game.actors.components import (
    CharacterInventory,
    ConditionsComponent,
    EnergyComponent,
    HealthComponent,
    ModifiersComponent,
    StatsComponent,
    StatusEffectsComponent,
    VisualEffectsComponent,
)
from catley.game.actors.status_effects import (
    EncumberedEffect,
    FocusedEffect,
    OffBalanceEffect,
    StatusEffect,
    StrengthBoostEffect,
)
from catley.game.enums import InjuryLocation, ItemSize
from catley.game.items.item_core import Item, ItemType

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def make_stats(
    strength: int = 3,
    toughness: int = 5,
    agility: int = 0,
    observation: int = 0,
    intelligence: int = 0,
    demeanor: int = 0,
    weirdness: int = 0,
) -> StatsComponent:
    """Create a StatsComponent with specified or default ability scores."""
    return StatsComponent(
        strength=strength,
        toughness=toughness,
        agility=agility,
        observation=observation,
        intelligence=intelligence,
        demeanor=demeanor,
        weirdness=weirdness,
    )


def make_item(name: str = "Test Item", size: ItemSize = ItemSize.NORMAL) -> Item:
    """Create a test item with the given name and size."""
    return Item(ItemType(name=name, description="A test item", size=size))


def make_mock_actor(
    stats: StatsComponent | None = None,
    with_inventory: bool = True,
) -> MagicMock:
    """Create a mock Actor with the necessary component structure.

    This is used for testing components that need an actor back-reference
    but don't need full Character functionality.
    """
    actor: MagicMock = MagicMock()
    actor.name = "Test Actor"

    stats = stats or make_stats()
    actor.stats = stats

    if with_inventory:
        inventory = CharacterInventory(stats, actor=actor)
        actor.inventory = inventory
        actor.conditions = ConditionsComponent(inventory)
    else:
        actor.inventory = None
        actor.conditions = None

    actor.status_effects = StatusEffectsComponent(actor)
    actor.modifiers = ModifiersComponent(actor)

    return actor


# =============================================================================
# STATS COMPONENT TESTS
# =============================================================================


class TestStatsComponent:
    """Tests for StatsComponent ability score management and derived values."""

    def test_initialization_with_all_ability_scores(self) -> None:
        """All seven ability scores should be stored correctly."""
        stats = StatsComponent(
            strength=1,
            toughness=2,
            agility=3,
            observation=4,
            intelligence=5,
            demeanor=6,
            weirdness=7,
        )

        assert stats.strength == 1
        assert stats.toughness == 2
        assert stats.agility == 3
        assert stats.observation == 4
        assert stats.intelligence == 5
        assert stats.demeanor == 6
        assert stats.weirdness == 7

    def test_default_ability_scores_are_zero(self) -> None:
        """All ability scores should default to zero if not specified."""
        stats = StatsComponent()

        assert stats.strength == 0
        assert stats.toughness == 0
        assert stats.agility == 0
        assert stats.observation == 0
        assert stats.intelligence == 0
        assert stats.demeanor == 0
        assert stats.weirdness == 0

    def test_max_hp_derived_from_toughness(self) -> None:
        """max_hp should be toughness + 5."""
        stats = StatsComponent(toughness=10)
        assert stats.max_hp == 15

    def test_max_hp_with_zero_toughness(self) -> None:
        """max_hp should be 5 when toughness is zero."""
        stats = StatsComponent(toughness=0)
        assert stats.max_hp == 5

    def test_max_hp_with_negative_toughness(self) -> None:
        """max_hp calculation works with negative toughness (edge case)."""
        stats = StatsComponent(toughness=-3)
        assert stats.max_hp == 2

    def test_inventory_slots_derived_from_strength(self) -> None:
        """inventory_slots should be strength + 5."""
        stats = StatsComponent(strength=7)
        assert stats.inventory_slots == 12

    def test_inventory_slots_with_zero_strength(self) -> None:
        """inventory_slots should be 5 when strength is zero."""
        stats = StatsComponent(strength=0)
        assert stats.inventory_slots == 5


# =============================================================================
# HEALTH COMPONENT TESTS
# =============================================================================


class TestHealthComponent:
    """Tests for HealthComponent HP/AP management and damage handling."""

    def test_initialization_derives_hp_from_stats(self) -> None:
        """HP should be initialized to max_hp from stats."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats)

        assert health.hp == 15  # toughness + 5
        assert health.max_hp == 15

    def test_initialization_with_custom_max_armor(self) -> None:
        """AP should be initialized to the specified max_ap."""
        stats = make_stats()
        health = HealthComponent(stats, max_ap=5)

        assert health.ap == 5
        assert health.max_ap == 5

    def test_initialization_with_default_max_armor(self) -> None:
        """AP should default to DEFAULT_MAX_ARMOR from config."""
        stats = make_stats()
        health = HealthComponent(stats)

        assert health.ap == DEFAULT_MAX_ARMOR
        assert health.max_ap == DEFAULT_MAX_ARMOR

    def test_take_damage_normal_reduces_armor_first(self) -> None:
        """Normal damage should reduce AP before HP."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats, max_ap=3)

        health.take_damage(2, "normal")

        assert health.ap == 1
        assert health.hp == 15

    def test_take_damage_normal_overflow_to_hp(self) -> None:
        """Normal damage exceeding AP should overflow to HP."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats, max_ap=2)

        health.take_damage(5, "normal")

        assert health.ap == 0
        assert health.hp == 12  # 15 - (5 - 2) = 12

    def test_take_damage_normal_with_no_armor(self) -> None:
        """Normal damage with zero AP should go directly to HP."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats, max_ap=0)

        health.take_damage(3, "normal")

        assert health.ap == 0
        assert health.hp == 12

    def test_take_damage_radiation_bypasses_armor(self) -> None:
        """Radiation damage should bypass armor entirely."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats, max_ap=3)

        health.take_damage(4, "radiation")

        assert health.ap == 3  # Unchanged
        assert health.hp == 11

    def test_take_damage_armor_piercing_bypasses_armor(self) -> None:
        """Armor-piercing damage should bypass armor entirely."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats, max_ap=3)

        health.take_damage(4, "armor_piercing")

        assert health.ap == 3  # Unchanged
        assert health.hp == 11

    def test_take_damage_hp_cannot_go_below_zero(self) -> None:
        """HP should be clamped to zero, never negative."""
        stats = make_stats(toughness=5)  # max_hp = 10
        health = HealthComponent(stats, max_ap=0)

        health.take_damage(100, "normal")

        assert health.hp == 0

    def test_heal_increases_hp(self) -> None:
        """heal() should increase HP by the specified amount."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats)
        health.hp = 5

        health.heal(3)

        assert health.hp == 8

    def test_heal_capped_at_max_hp(self) -> None:
        """heal() should not increase HP beyond max_hp."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats)
        health.hp = 14

        health.heal(10)

        assert health.hp == 15  # Capped at max_hp

    def test_heal_exact_to_max(self) -> None:
        """Healing that exactly reaches max_hp works correctly."""
        stats = make_stats(toughness=10)
        health = HealthComponent(stats)
        health.hp = 10

        health.heal(5)

        assert health.hp == 15

    def test_is_alive_true_when_hp_positive(self) -> None:
        """is_alive() should return True when HP > 0."""
        stats = make_stats()
        health = HealthComponent(stats)
        health.hp = 1

        assert health.is_alive() is True

    def test_is_alive_false_when_hp_zero(self) -> None:
        """is_alive() should return False when HP == 0."""
        stats = make_stats()
        health = HealthComponent(stats)
        health.hp = 0

        assert health.is_alive() is False


# =============================================================================
# INVENTORY COMPONENT TESTS (gaps only)
# =============================================================================


class TestInventoryComponentGaps:
    """Tests for CharacterInventory behaviors not covered in test_inventory.py.

    Existing coverage in test_inventory.py:
    - get_used_inventory_slots() with various sizes
    - can_add_to_inventory() capacity checks
    - equip_from_inventory() / unequip_to_inventory()
    - try_remove_item() from equipped and stored
    """

    def test_add_condition_to_full_inventory_causes_encumbrance(self) -> None:
        """Adding a condition to full inventory causes encumbrance, not item drops.

        This verifies the intended design: injuries push you over capacity rather
        than silently dropping items. Players must manually decide what to drop.
        """
        stats = make_stats(strength=0)  # 5 slots
        actor = make_mock_actor(stats=stats)
        inv = CharacterInventory(stats, actor=actor)
        actor.inventory = inv

        # Fill inventory to capacity
        for i in range(5):
            inv.add_voluntary_item(make_item(f"item{i}"))
        assert inv.get_used_inventory_slots() == 5
        assert not inv.is_encumbered()

        # Add a condition - should NOT drop any items
        injury = conditions.Injury(InjuryLocation.HEAD, "Wound")
        with patch("catley.events.publish_event"):
            dropped, message = inv.add_condition(injury)

        # Verify no items were dropped
        assert len(dropped) == 0
        assert "Dropped" not in message

        # Verify condition was added and inventory is over capacity
        assert injury in inv
        assert inv.get_used_inventory_slots() == 6  # 5 items + 1 condition

        # Verify player is now encumbered
        assert inv.is_encumbered()
        assert "encumbered" in message.lower()

    def test_add_condition_preserves_all_items(self) -> None:
        """All items remain in inventory when a condition is added."""
        stats = make_stats(strength=0)  # 5 slots
        inv = CharacterInventory(stats)

        # Fill with 5 named items
        item_names = [f"item{i}" for i in range(5)]
        for name in item_names:
            inv.add_voluntary_item(make_item(name))

        injury = conditions.Injury(InjuryLocation.HEAD, "Wound")
        dropped, _ = inv.add_condition(injury)

        # All original items should still be present
        assert len(dropped) == 0
        stored_names = [
            item.name for item in inv._stored_items if hasattr(item, "name")
        ]
        for name in item_names:
            assert name in stored_names

    def test_update_encumbrance_status_adds_effect(self) -> None:
        """Becoming encumbered should add EncumberedEffect to the actor."""
        actor = make_mock_actor()
        stats = actor.stats
        inv = CharacterInventory(stats, actor=actor)
        actor.inventory = inv

        # Fill beyond capacity
        for i in range(stats.inventory_slots + 1):
            inv._stored_items.append(make_item(f"item{i}"))

        with patch("catley.events.publish_event"):
            inv._update_encumbrance_status()

        assert actor.status_effects.has_status_effect(EncumberedEffect)

    def test_update_encumbrance_status_removes_effect(self) -> None:
        """Dropping below encumbrance threshold removes EncumberedEffect."""
        actor = make_mock_actor()
        stats = actor.stats
        inv = CharacterInventory(stats, actor=actor)
        actor.inventory = inv

        # Start encumbered
        for i in range(stats.inventory_slots + 1):
            inv._stored_items.append(make_item(f"item{i}"))

        with patch("catley.events.publish_event"):
            inv._update_encumbrance_status()

        # Now drop below threshold
        inv._stored_items.pop()
        inv._stored_items.pop()

        with patch("catley.events.publish_event"):
            inv._update_encumbrance_status()

        assert not actor.status_effects.has_status_effect(EncumberedEffect)

    def test_get_inventory_slot_colors_reflects_item_sizes(self) -> None:
        """Slot colors should have one entry per used slot."""
        stats = make_stats(strength=5)
        inv = CharacterInventory(stats)

        inv.add_voluntary_item(make_item("normal", ItemSize.NORMAL))
        inv.add_voluntary_item(make_item("big", ItemSize.BIG))

        slot_colors = inv.get_inventory_slot_colors()

        # 1 slot for NORMAL + 2 slots for BIG = 3 colors
        assert len(slot_colors) == 3

    def test_get_inventory_slot_colors_includes_conditions(self) -> None:
        """Condition slots should use the condition's display_color."""
        actor = make_mock_actor()
        inv = actor.inventory

        injury = conditions.Injury(InjuryLocation.HEAD, "Wound")
        inv.add_to_inventory(injury)

        slot_colors = inv.get_inventory_slot_colors()

        assert len(slot_colors) == 1
        assert slot_colors[0] == colors.RED  # Injury display_color

    def test_get_available_attacks_returns_all_slots(self) -> None:
        """get_available_attacks() should return all attack slots."""
        stats = make_stats()
        inv = CharacterInventory(stats, num_attack_slots=3)

        weapon = make_item("sword")
        inv.equip_to_slot(weapon, 1)

        attacks = inv.get_available_attacks()

        assert len(attacks) == 3
        assert attacks[0] == (None, 0, "Primary")
        assert attacks[1] == (weapon, 1, "Secondary")
        assert attacks[2] == (None, 2, "Tertiary")

    def test_revision_counter_increments_on_add(self) -> None:
        """Revision counter should increment when items are added."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        initial_revision = inv.revision

        inv.add_voluntary_item(make_item("item"))

        assert inv.revision == initial_revision + 1

    def test_revision_counter_increments_on_remove(self) -> None:
        """Revision counter should increment when items are removed."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        item = make_item("item")
        inv.add_voluntary_item(item)
        initial_revision = inv.revision

        inv.remove_from_inventory(item)

        assert inv.revision == initial_revision + 1

    def test_revision_counter_increments_on_equip(self) -> None:
        """Revision counter should increment on equipment changes."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        initial_revision = inv.revision

        inv.equip_to_slot(make_item("weapon"), 0)

        assert inv.revision == initial_revision + 1

    def test_get_slots_over_capacity_zero_when_under(self) -> None:
        """get_slots_over_capacity() returns 0 when under capacity."""
        stats = make_stats(strength=5)  # 10 slots
        inv = CharacterInventory(stats)

        # Add fewer items than capacity
        for i in range(5):
            inv.add_voluntary_item(make_item(f"item{i}"))

        assert inv.get_slots_over_capacity() == 0

    def test_get_slots_over_capacity_zero_when_at_capacity(self) -> None:
        """get_slots_over_capacity() returns 0 when exactly at capacity."""
        stats = make_stats(strength=0)  # 5 slots
        inv = CharacterInventory(stats)

        # Fill to exactly capacity
        for i in range(5):
            inv._stored_items.append(make_item(f"item{i}"))

        assert inv.get_slots_over_capacity() == 0

    def test_get_slots_over_capacity_correct_when_over(self) -> None:
        """get_slots_over_capacity() returns correct count when over capacity."""
        stats = make_stats(strength=0)  # 5 slots
        inv = CharacterInventory(stats)

        # Fill to 3 over capacity (8 items in 5 slots)
        for i in range(8):
            inv._stored_items.append(make_item(f"item{i}"))

        assert inv.get_slots_over_capacity() == 3

    def test_get_slots_over_capacity_with_big_items(self) -> None:
        """get_slots_over_capacity() accounts for item sizes correctly."""
        stats = make_stats(strength=0)  # 5 slots
        inv = CharacterInventory(stats)

        # Add 2 BIG items (2 slots each = 4 slots)
        inv._stored_items.append(make_item("big1", ItemSize.BIG))
        inv._stored_items.append(make_item("big2", ItemSize.BIG))

        assert inv.get_slots_over_capacity() == 0

        # Add 1 more NORMAL item = 5 slots total, still at capacity
        inv._stored_items.append(make_item("normal"))
        assert inv.get_slots_over_capacity() == 0

        # Add 1 more NORMAL item = 6 slots total, 1 over
        inv._stored_items.append(make_item("normal2"))
        assert inv.get_slots_over_capacity() == 1


# =============================================================================
# VISUAL EFFECTS COMPONENT TESTS (gaps only)
# =============================================================================


class TestVisualEffectsComponentGaps:
    """Tests for VisualEffectsComponent behaviors not covered in test_idle_animation.py.

    Existing coverage in test_idle_animation.py:
    - idle_profile lazy creation
    - set_idle_profile()
    - get_idle_drift_offset() with various profiles
    - Desynchronization between actors
    """

    def test_flash_sets_color_and_duration(self) -> None:
        """flash() should set the flash color and duration."""
        component = VisualEffectsComponent()

        component.flash(colors.RED, duration_frames=10)

        assert component._flash_color == colors.RED
        assert component._flash_duration_frames == 10

    def test_get_flash_color_returns_color_during_flash(self) -> None:
        """get_flash_color() should return the flash color when active."""
        component = VisualEffectsComponent()
        component.flash(colors.BLUE, duration_frames=5)

        assert component.get_flash_color() == colors.BLUE

    def test_get_flash_color_returns_none_when_no_flash(self) -> None:
        """get_flash_color() should return None when no flash is active."""
        component = VisualEffectsComponent()

        assert component.get_flash_color() is None

    def test_get_flash_color_returns_none_after_expiry(self) -> None:
        """get_flash_color() should return None after flash duration expires."""
        component = VisualEffectsComponent()
        component.flash(colors.RED, duration_frames=2)

        component.update(0.016)  # Frame 1
        component.update(0.016)  # Frame 2 - expires

        assert component.get_flash_color() is None

    def test_update_decrements_flash_frames(self) -> None:
        """update() should decrement flash duration frames."""
        component = VisualEffectsComponent()
        component.flash(colors.RED, duration_frames=5)

        component.update(0.016)

        assert component._flash_duration_frames == 4

    def test_update_clears_flash_color_at_zero(self) -> None:
        """update() should clear flash color when duration reaches zero."""
        component = VisualEffectsComponent()
        component.flash(colors.RED, duration_frames=1)

        component.update(0.016)

        assert component._flash_color is None

    def test_add_continuous_effect(self) -> None:
        """add_continuous_effect() should add effect to the list."""
        component = VisualEffectsComponent()
        effect = MagicMock()

        component.add_continuous_effect(effect)

        assert effect in component.continuous_effects

    def test_remove_continuous_effect(self) -> None:
        """remove_continuous_effect() should remove effect and call stop()."""
        component = VisualEffectsComponent()
        effect = MagicMock()
        component.add_continuous_effect(effect)

        component.remove_continuous_effect(effect)

        assert effect not in component.continuous_effects
        effect.stop.assert_called_once()

    def test_remove_continuous_effect_missing(self) -> None:
        """remove_continuous_effect() should be safe if effect not present."""
        component = VisualEffectsComponent()
        effect = MagicMock()

        # Should not raise
        component.remove_continuous_effect(effect)

    def test_has_continuous_effects_true(self) -> None:
        """has_continuous_effects() returns True when effects exist."""
        component = VisualEffectsComponent()
        component.add_continuous_effect(MagicMock())

        assert component.has_continuous_effects() is True

    def test_has_continuous_effects_false(self) -> None:
        """has_continuous_effects() returns False when no effects."""
        component = VisualEffectsComponent()

        assert component.has_continuous_effects() is False

    def test_update_calls_continuous_effect_update(self) -> None:
        """update() should call update() on continuous effects."""
        component = VisualEffectsComponent()
        effect = MagicMock()
        component.add_continuous_effect(effect)

        component.update(0.016)

        effect.update.assert_called_once_with(0.016)


# =============================================================================
# MODIFIERS COMPONENT TESTS
# =============================================================================


class TestModifiersComponent:
    """Tests for ModifiersComponent facade for querying effects/conditions."""

    def test_get_all_status_effects_delegates_to_actor(self) -> None:
        """get_all_status_effects() should delegate to StatusEffectsComponent."""
        actor = make_mock_actor()
        actor.status_effects.apply_status_effect(OffBalanceEffect())

        effects = actor.modifiers.get_all_status_effects()

        assert len(effects) == 1
        assert isinstance(effects[0], OffBalanceEffect)

    def test_get_all_conditions_delegates_to_actor(self) -> None:
        """get_all_conditions() should delegate to ConditionsComponent."""
        actor = make_mock_actor()
        injury = conditions.Injury(InjuryLocation.HEAD, "Wound")
        actor.conditions.add_condition(injury)

        result = actor.modifiers.get_all_conditions()

        assert len(result) == 1
        assert result[0] is injury

    def test_get_all_conditions_returns_empty_without_conditions(self) -> None:
        """get_all_conditions() returns empty list if actor has no conditions."""
        actor = make_mock_actor(with_inventory=False)
        actor.conditions = None

        result = actor.modifiers.get_all_conditions()

        assert result == []

    def test_get_all_active_effects_combines_both(self) -> None:
        """get_all_active_effects() should combine status effects and conditions."""
        actor = make_mock_actor()
        actor.status_effects.apply_status_effect(FocusedEffect())
        injury = conditions.Injury(InjuryLocation.HEAD, "Wound")
        actor.conditions.add_condition(injury)

        all_effects = actor.modifiers.get_all_active_effects()

        assert len(all_effects) == 2
        types = {type(e) for e in all_effects}
        assert FocusedEffect in types
        assert conditions.Injury in types

    def test_get_resolution_modifiers_aggregates_status_effects(self) -> None:
        """get_resolution_modifiers() should aggregate from status effects."""
        actor = make_mock_actor()
        actor.status_effects.apply_status_effect(OffBalanceEffect())

        modifiers = actor.modifiers.get_resolution_modifiers("strength")

        assert modifiers.get("has_disadvantage") is True

    def test_get_resolution_modifiers_aggregates_conditions(self) -> None:
        """get_resolution_modifiers() should aggregate from conditions."""
        actor = make_mock_actor()
        injury = conditions.Injury(InjuryLocation.LEFT_ARM, "Wound")
        actor.conditions.add_condition(injury)

        # Arm injury affects strength
        modifiers = actor.modifiers.get_resolution_modifiers("strength")

        assert modifiers.get("has_disadvantage") is True

    def test_get_movement_speed_multiplier_default(self) -> None:
        """get_movement_speed_multiplier() returns 1.0 with no conditions."""
        actor = make_mock_actor()

        multiplier = actor.modifiers.get_movement_speed_multiplier()

        assert multiplier == 1.0

    def test_get_movement_speed_multiplier_with_leg_injury(self) -> None:
        """Leg injury should reduce movement speed."""
        actor = make_mock_actor()
        injury = conditions.Injury(InjuryLocation.LEFT_LEG, "Wound")
        actor.conditions.add_condition(injury)

        multiplier = actor.modifiers.get_movement_speed_multiplier()

        assert multiplier == 0.75

    def test_get_movement_speed_multiplier_with_exhaustion(self) -> None:
        """Exhaustion should reduce movement speed."""
        actor = make_mock_actor()
        actor.conditions.add_condition(conditions.Exhaustion())

        multiplier = actor.modifiers.get_movement_speed_multiplier()

        expected = MovementConstants.EXHAUSTION_SPEED_REDUCTION_PER_STACK
        assert multiplier == pytest.approx(expected)

    def test_get_exhaustion_count_zero(self) -> None:
        """get_exhaustion_count() returns 0 with no exhaustion."""
        actor = make_mock_actor()

        assert actor.modifiers.get_exhaustion_count() == 0

    def test_get_exhaustion_count_multiple(self) -> None:
        """get_exhaustion_count() counts all exhaustion conditions."""
        actor = make_mock_actor()
        actor.conditions.add_condition(conditions.Exhaustion())
        actor.conditions.add_condition(conditions.Exhaustion())
        actor.conditions.add_condition(conditions.Exhaustion())

        assert actor.modifiers.get_exhaustion_count() == 3

    def test_has_disadvantage_from_exhaustion_below_threshold(self) -> None:
        """has_disadvantage_from_exhaustion() returns False with < 2 exhaustion."""
        actor = make_mock_actor()
        actor.conditions.add_condition(conditions.Exhaustion())

        assert actor.modifiers.has_disadvantage_from_exhaustion() is False

    def test_has_disadvantage_from_exhaustion_at_threshold(self) -> None:
        """has_disadvantage_from_exhaustion() returns True with >= 2 exhaustion."""
        actor = make_mock_actor()
        actor.conditions.add_condition(conditions.Exhaustion())
        actor.conditions.add_condition(conditions.Exhaustion())

        assert actor.modifiers.has_disadvantage_from_exhaustion() is True

    def test_get_exhaustion_energy_multiplier_no_exhaustion(self) -> None:
        """get_exhaustion_energy_multiplier() returns 1.0 with no exhaustion."""
        actor = make_mock_actor()

        assert actor.modifiers.get_exhaustion_energy_multiplier() == 1.0

    def test_get_exhaustion_energy_multiplier_stacks_exponentially(self) -> None:
        """get_exhaustion_energy_multiplier() applies exponentially per stack."""
        actor = make_mock_actor()
        actor.conditions.add_condition(conditions.Exhaustion())
        actor.conditions.add_condition(conditions.Exhaustion())

        multiplier = actor.modifiers.get_exhaustion_energy_multiplier()
        expected = MovementConstants.EXHAUSTION_ENERGY_REDUCTION_PER_STACK**2

        assert multiplier == pytest.approx(expected)

    def test_get_movement_speed_multiplier_with_encumbrance_1_over(self) -> None:
        """Encumbrance reduces speed by 0.85^1 when 1 slot over capacity."""
        actor = make_mock_actor()
        stats = actor.stats
        inv = actor.inventory

        # Fill to exactly 1 slot over capacity
        for i in range(stats.inventory_slots + 1):
            inv._stored_items.append(make_item(f"item{i}"))

        multiplier = actor.modifiers.get_movement_speed_multiplier()
        expected = MovementConstants.ENCUMBRANCE_SPEED_BASE**1

        assert multiplier == pytest.approx(expected)

    def test_get_movement_speed_multiplier_with_encumbrance_2_over(self) -> None:
        """Encumbrance reduces speed by 0.85^2 when 2 slots over capacity."""
        actor = make_mock_actor()
        stats = actor.stats
        inv = actor.inventory

        # Fill to exactly 2 slots over capacity
        for i in range(stats.inventory_slots + 2):
            inv._stored_items.append(make_item(f"item{i}"))

        multiplier = actor.modifiers.get_movement_speed_multiplier()
        expected = MovementConstants.ENCUMBRANCE_SPEED_BASE**2

        assert multiplier == pytest.approx(expected)

    def test_get_movement_speed_multiplier_with_encumbrance_4_over(self) -> None:
        """Encumbrance reduces speed by 0.85^4 when 4 slots over capacity."""
        actor = make_mock_actor()
        stats = actor.stats
        inv = actor.inventory

        # Fill to exactly 4 slots over capacity
        for i in range(stats.inventory_slots + 4):
            inv._stored_items.append(make_item(f"item{i}"))

        multiplier = actor.modifiers.get_movement_speed_multiplier()
        expected = MovementConstants.ENCUMBRANCE_SPEED_BASE**4

        assert multiplier == pytest.approx(expected)

    def test_get_movement_speed_multiplier_encumbrance_stacks_with_leg_injury(
        self,
    ) -> None:
        """Encumbrance and leg injury penalties stack multiplicatively."""
        actor = make_mock_actor()
        stats = actor.stats
        inv = actor.inventory

        # Add leg injury (0.75x) - also takes 1 inventory slot
        injury = conditions.Injury(InjuryLocation.LEFT_LEG, "Wound")
        actor.conditions.add_condition(injury)

        # Fill to 2 slots over capacity total. Injury takes 1 slot, so adding
        # (capacity + 1) items gives us (capacity + 1) + 1 = capacity + 2 slots.
        for i in range(stats.inventory_slots + 1):
            inv._stored_items.append(make_item(f"item{i}"))

        multiplier = actor.modifiers.get_movement_speed_multiplier()
        # 0.75 (leg injury) * 0.85^2 (2 slots over) = 0.75 * 0.7225 = 0.541875
        expected = 0.75 * (MovementConstants.ENCUMBRANCE_SPEED_BASE**2)

        assert multiplier == pytest.approx(expected)

    def test_get_movement_speed_multiplier_no_encumbrance_penalty_at_capacity(
        self,
    ) -> None:
        """No encumbrance penalty when exactly at capacity."""
        actor = make_mock_actor()
        stats = actor.stats
        inv = actor.inventory

        # Fill to exactly capacity (not over)
        for i in range(stats.inventory_slots):
            inv._stored_items.append(make_item(f"item{i}"))

        multiplier = actor.modifiers.get_movement_speed_multiplier()

        assert multiplier == 1.0


# =============================================================================
# STATUS EFFECTS COMPONENT TESTS (gaps only)
# =============================================================================


class TestStatusEffectsComponentGaps:
    """Tests for StatusEffectsComponent behaviors not covered elsewhere.

    Existing coverage in test_status_effects.py:
    - Effect expiration via update_turn()
    - Multiple effects coexisting
    - Duration decrement
    """

    def test_apply_status_effect_prevents_stacking_when_disabled(self) -> None:
        """Non-stackable effects should not be applied twice."""
        actor = make_mock_actor()

        actor.status_effects.apply_status_effect(OffBalanceEffect())
        actor.status_effects.apply_status_effect(OffBalanceEffect())

        # OffBalanceEffect has can_stack=False by default
        assert len(actor.status_effects.get_all_status_effects()) == 1

    def test_apply_status_effect_calls_apply_on_start_hook(self) -> None:
        """apply_status_effect() should call the effect's apply_on_start()."""
        actor = make_mock_actor()
        initial_strength = actor.stats.strength

        actor.status_effects.apply_status_effect(StrengthBoostEffect(duration=2))

        # StrengthBoostEffect adds 2 to strength on start
        assert actor.stats.strength == initial_strength + 2

    def test_remove_status_effect_calls_remove_effect_hook(self) -> None:
        """remove_status_effect() should call the effect's remove_effect()."""
        actor = make_mock_actor()
        initial_strength = actor.stats.strength
        actor.status_effects.apply_status_effect(StrengthBoostEffect(duration=2))

        actor.status_effects.remove_status_effect(StrengthBoostEffect)

        # StrengthBoostEffect removes 2 from strength on removal
        assert actor.stats.strength == initial_strength

    def test_get_status_effects_by_type_filters_correctly(self) -> None:
        """get_status_effects_by_type() should return only matching effects."""
        actor = make_mock_actor()
        actor.status_effects.apply_status_effect(OffBalanceEffect())
        actor.status_effects.apply_status_effect(FocusedEffect())

        off_balance = actor.status_effects.get_status_effects_by_type(OffBalanceEffect)
        focused = actor.status_effects.get_status_effects_by_type(FocusedEffect)

        assert len(off_balance) == 1
        assert isinstance(off_balance[0], OffBalanceEffect)
        assert len(focused) == 1
        assert isinstance(focused[0], FocusedEffect)

    def test_get_status_effects_by_type_empty_result(self) -> None:
        """get_status_effects_by_type() returns empty list when no matches."""
        actor = make_mock_actor()
        actor.status_effects.apply_status_effect(OffBalanceEffect())

        result = actor.status_effects.get_status_effects_by_type(FocusedEffect)

        assert result == []

    def test_update_turn_calls_apply_turn_effect(self) -> None:
        """update_turn() should call apply_turn_effect on each effect."""
        actor = make_mock_actor()
        effect = MagicMock(spec=StatusEffect)
        effect.duration = 3
        effect.can_stack = False
        effect.should_remove.return_value = False
        actor.status_effects._status_effects.append(effect)

        actor.status_effects.update_turn()

        effect.apply_turn_effect.assert_called_once_with(actor)

    def test_update_turn_decrements_positive_duration(self) -> None:
        """update_turn() should decrement duration when > 0."""
        actor = make_mock_actor()
        effect = StrengthBoostEffect(duration=3)
        actor.status_effects.apply_status_effect(effect)

        actor.status_effects.update_turn()

        assert effect.duration == 2

    def test_update_turn_respects_indefinite_duration(self) -> None:
        """update_turn() should not decrement duration when == -1."""
        actor = make_mock_actor()
        effect = EncumberedEffect()  # duration = -1
        actor.status_effects.apply_status_effect(effect)

        actor.status_effects.update_turn()

        # Duration stays at -1 (not decremented)
        assert effect.duration == -1

    def test_update_turn_removes_when_should_remove_true(self) -> None:
        """update_turn() removes effects when should_remove() returns True."""
        actor = make_mock_actor()
        actor.status_effects.apply_status_effect(OffBalanceEffect())

        # OffBalanceEffect has duration=1, so after one update it expires
        actor.status_effects.update_turn()

        assert not actor.status_effects.has_status_effect(OffBalanceEffect)


# =============================================================================
# CONDITIONS COMPONENT TESTS (gaps only)
# =============================================================================


class TestConditionsComponentGaps:
    """Tests for ConditionsComponent behaviors not covered elsewhere.

    Existing coverage in test_status_effects.py:
    - add_condition() / remove_condition()
    - has_condition() / get_conditions_by_type()
    """

    def test_apply_turn_effects_calls_condition_hooks(self) -> None:
        """apply_turn_effects() should call apply_turn_effect on conditions."""
        actor = make_mock_actor()

        # Sickness with Poisoned type deals 1 damage per turn
        sickness = conditions.Sickness(sickness_type="Poisoned")
        actor.conditions.add_condition(sickness)

        # Create a mock for the actor's take_damage method
        actor.take_damage = MagicMock()

        with patch("catley.game.actors.conditions.publish_event"):
            actor.conditions.apply_turn_effects(actor)

        # Poisoned deals 1 damage per turn
        actor.take_damage.assert_called_once_with(1)

    def test_iter_yields_all_conditions(self) -> None:
        """__iter__ should yield all conditions."""
        actor = make_mock_actor()
        exhaustion = conditions.Exhaustion()
        injury = conditions.Injury(InjuryLocation.HEAD, "Wound")
        actor.conditions.add_condition(exhaustion)
        actor.conditions.add_condition(injury)

        result = list(actor.conditions)

        assert len(result) == 2
        assert exhaustion in result
        assert injury in result

    def test_len_returns_condition_count(self) -> None:
        """__len__ should return the number of conditions."""
        actor = make_mock_actor()
        actor.conditions.add_condition(conditions.Exhaustion())
        actor.conditions.add_condition(conditions.Exhaustion())

        assert len(actor.conditions) == 2


# =============================================================================
# ENERGY COMPONENT TESTS
# =============================================================================


class TestEnergyComponent:
    """Tests for EnergyComponent action economy management."""

    def test_initialization_with_default_speed(self) -> None:
        """EnergyComponent should initialize with default speed."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)

        assert energy.speed == DEFAULT_ACTOR_SPEED

    def test_initialization_with_custom_speed(self) -> None:
        """EnergyComponent should accept custom speed."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor, speed=150)

        assert energy.speed == 150

    def test_initialization_sets_initial_energy_to_speed(self) -> None:
        """__post_init__ should set accumulated_energy to speed."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor, speed=120)

        assert energy.accumulated_energy == 120

    def test_accumulate_energy_adds_to_current(self) -> None:
        """accumulate_energy() should add to accumulated_energy."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 50

        energy.accumulate_energy(30)

        assert energy.accumulated_energy == 80

    def test_accumulate_energy_capped_at_max(self) -> None:
        """accumulate_energy() should not exceed max_energy."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 190

        energy.accumulate_energy(50)

        assert energy.accumulated_energy == energy.max_energy

    def test_accumulate_energy_ignores_negative_amount(self) -> None:
        """accumulate_energy() should ignore negative amounts."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 100

        energy.accumulate_energy(-50)

        assert energy.accumulated_energy == 100

    def test_accumulate_energy_ignores_zero_amount(self) -> None:
        """accumulate_energy() should ignore zero amount."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 100

        energy.accumulate_energy(0)

        assert energy.accumulated_energy == 100

    def test_get_speed_based_energy_amount_baseline(self) -> None:
        """get_speed_based_energy_amount() returns ~100 for speed=100."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor, speed=100)

        amount = energy.get_speed_based_energy_amount()

        # Base is 100, normalized speed is 1.0, no modifiers
        assert amount == 100.0

    def test_get_speed_based_energy_amount_scales_with_speed(self) -> None:
        """Faster actors should get proportionally more energy."""
        actor = make_mock_actor()

        energy_slow = EnergyComponent(actor, speed=50)
        energy_fast = EnergyComponent(actor, speed=200)

        slow_amount = energy_slow.get_speed_based_energy_amount()
        fast_amount = energy_fast.get_speed_based_energy_amount()

        assert fast_amount == slow_amount * 4  # 200/50 = 4x

    def test_get_speed_based_energy_amount_with_exhaustion(self) -> None:
        """Exhaustion should reduce energy amount."""
        actor = make_mock_actor()
        actor.conditions.add_condition(conditions.Exhaustion())
        energy = EnergyComponent(actor, speed=100)

        amount = energy.get_speed_based_energy_amount()

        expected = 100.0 * MovementConstants.EXHAUSTION_SPEED_REDUCTION_PER_STACK
        expected *= MovementConstants.EXHAUSTION_ENERGY_REDUCTION_PER_STACK
        assert amount == pytest.approx(expected)

    def test_can_afford_true_when_sufficient_energy(self) -> None:
        """can_afford() returns True when energy >= cost."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 100

        assert energy.can_afford(100) is True
        assert energy.can_afford(50) is True

    def test_can_afford_false_when_insufficient_energy(self) -> None:
        """can_afford() returns False when energy < cost."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 50

        assert energy.can_afford(100) is False

    def test_spend_reduces_accumulated_energy(self) -> None:
        """spend() should reduce accumulated_energy by cost."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 100

        energy.spend(60)

        assert energy.accumulated_energy == 40

    def test_spend_cannot_go_below_zero(self) -> None:
        """spend() should clamp energy to 0."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 50

        energy.spend(100)

        assert energy.accumulated_energy == 0

    def test_energy_property_alias(self) -> None:
        """energy property should alias accumulated_energy."""
        actor = make_mock_actor()
        energy = EnergyComponent(actor)
        energy.accumulated_energy = 75

        assert energy.energy == 75

        energy.energy = 120
        assert energy.accumulated_energy == 120
