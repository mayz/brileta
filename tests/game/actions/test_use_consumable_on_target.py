"""Tests for UseConsumableOnTargetIntent and executor.

UseConsumableOnTargetIntent allows using a consumable item on another character,
such as healing an ally. The executor validates adjacency and applies the effect
to the target.
"""

from __future__ import annotations

from catley import colors
from catley.game.actions.executors.recovery import UseConsumableOnTargetExecutor
from catley.game.actions.recovery import UseConsumableOnTargetIntent
from catley.game.actors import Character
from catley.game.enums import ConsumableEffectType, ItemSize
from catley.game.items.capabilities import ConsumableEffectSpec
from catley.game.items.item_core import Item, ItemType
from tests.helpers import get_controller_with_player_and_map


def make_healing_potion() -> Item:
    """Create a test healing potion."""
    consumable_spec = ConsumableEffectSpec(
        effect_type=ConsumableEffectType.HEAL,
        effect_value=10,
    )
    item_type = ItemType(
        name="Health Potion",
        description="A healing potion",
        size=ItemSize.TINY,
        consumable_effect=consumable_spec,
    )
    return item_type.create()


def test_use_consumable_on_adjacent_target() -> None:
    """Using a consumable on an adjacent target succeeds."""
    controller = get_controller_with_player_and_map()
    gw = controller.gw
    player = gw.player

    # Create an adjacent NPC
    npc = Character(player.x + 1, player.y, "N", colors.WHITE, "NPC", game_world=gw)
    gw.add_actor(npc)

    # Give player a healing potion
    potion = make_healing_potion()
    player.inventory.add_to_inventory(potion)

    # Damage the NPC
    npc.health.hp = 5  # Less than max

    # Create and execute the intent
    intent = UseConsumableOnTargetIntent(controller, player, potion, npc)
    executor = UseConsumableOnTargetExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded is True
    # Potion should have been consumed (removed from inventory)
    assert potion not in player.inventory


def test_use_consumable_on_distant_target_fails() -> None:
    """Using a consumable on a distant target fails."""
    controller = get_controller_with_player_and_map()
    gw = controller.gw
    player = gw.player

    # Create a distant NPC
    npc = Character(player.x + 5, player.y + 5, "N", colors.WHITE, "NPC", game_world=gw)
    gw.add_actor(npc)

    # Give player a healing potion
    potion = make_healing_potion()
    player.inventory.add_to_inventory(potion)

    # Create and execute the intent
    intent = UseConsumableOnTargetIntent(controller, player, potion, npc)
    executor = UseConsumableOnTargetExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded is False
    # Potion should NOT have been consumed
    assert potion in player.inventory


def test_use_consumable_on_dead_target_fails() -> None:
    """Using a consumable on a dead target fails."""
    controller = get_controller_with_player_and_map()
    gw = controller.gw
    player = gw.player

    # Create an adjacent but dead NPC
    npc = Character(player.x + 1, player.y, "N", colors.WHITE, "NPC", game_world=gw)
    gw.add_actor(npc)
    npc.health.hp = 0  # Dead

    # Give player a healing potion
    potion = make_healing_potion()
    player.inventory.add_to_inventory(potion)

    # Create and execute the intent
    intent = UseConsumableOnTargetIntent(controller, player, potion, npc)
    executor = UseConsumableOnTargetExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded is False


def test_use_non_consumable_item_fails() -> None:
    """Using a non-consumable item on a target fails."""
    controller = get_controller_with_player_and_map()
    gw = controller.gw
    player = gw.player

    # Create an adjacent NPC
    npc = Character(player.x + 1, player.y, "N", colors.WHITE, "NPC", game_world=gw)
    gw.add_actor(npc)

    # Create a non-consumable item
    item_type = ItemType(
        name="Rock",
        description="A rock",
        size=ItemSize.TINY,
        # No consumable_effect
    )
    rock = item_type.create()
    player.inventory.add_to_inventory(rock)

    # Create and execute the intent
    intent = UseConsumableOnTargetIntent(controller, player, rock, npc)
    executor = UseConsumableOnTargetExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded is False


def test_use_consumable_not_in_inventory_fails() -> None:
    """Using a consumable not in inventory fails."""
    controller = get_controller_with_player_and_map()
    gw = controller.gw
    player = gw.player

    # Create an adjacent NPC
    npc = Character(player.x + 1, player.y, "N", colors.WHITE, "NPC", game_world=gw)
    gw.add_actor(npc)

    # Create a potion but don't add it to inventory
    potion = make_healing_potion()

    # Create and execute the intent
    intent = UseConsumableOnTargetIntent(controller, player, potion, npc)
    executor = UseConsumableOnTargetExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert result.succeeded is False


def test_use_consumable_intent_stores_target() -> None:
    """UseConsumableOnTargetIntent correctly stores the target."""
    controller = get_controller_with_player_and_map()
    gw = controller.gw
    player = gw.player

    npc = Character(player.x + 1, player.y, "N", colors.WHITE, "NPC", game_world=gw)
    gw.add_actor(npc)

    potion = make_healing_potion()

    intent = UseConsumableOnTargetIntent(controller, player, potion, npc)

    assert intent.actor is player
    assert intent.item is potion
    assert intent.target is npc


def test_use_equipped_consumable_on_adjacent_target() -> None:
    """Using an equipped consumable (in ready slot) on an adjacent target succeeds.

    This tests the fix for the "item not found in inventory" bug where equipped
    consumables couldn't be used because the removal code only checked stored
    inventory, not ready slots.
    """
    controller = get_controller_with_player_and_map()
    gw = controller.gw
    player = gw.player

    # Create an adjacent NPC
    npc = Character(player.x + 1, player.y, "N", colors.WHITE, "NPC", game_world=gw)
    gw.add_actor(npc)

    # Create and EQUIP a healing potion (put it in ready slot, not stored inventory)
    potion = make_healing_potion()
    player.inventory.equip_to_slot(potion, slot_index=0)

    # Verify it's equipped, not in stored inventory
    assert potion in player.inventory.ready_slots
    assert potion not in player.inventory._stored_items

    # Damage the NPC
    npc.health.hp = 5  # Less than max

    # Create and execute the intent
    intent = UseConsumableOnTargetIntent(controller, player, potion, npc)
    executor = UseConsumableOnTargetExecutor()
    result = executor.execute(intent)

    # Should succeed - the fix uses try_remove_item which checks ready_slots
    assert result is not None
    assert result.succeeded is True
    # Potion should have been consumed (removed from ready slot)
    assert potion not in player.inventory.ready_slots
    assert potion not in player.inventory
