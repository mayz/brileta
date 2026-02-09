"""Tests for EquipmentView click-to-use interaction.

The equipment view now supports:
- Click inactive slot: switch to it (existing behavior)
- Click active slot: use the item (enter combat/targeting mode)
- Click active slot while in combat/targeting: exit that mode
"""

from __future__ import annotations

from unittest.mock import MagicMock

from brileta.controller import Controller
from brileta.game.enums import ConsumableEffectType, ItemSize
from brileta.game.items.capabilities import (
    ConsumableEffectSpec,
    MeleeAttackSpec,
)
from brileta.game.items.item_core import Item, ItemType
from brileta.view.render.graphics import GraphicsContext
from brileta.view.views.equipment_view import EquipmentView


def make_equipment_view(controller: Controller) -> EquipmentView:
    """Create an EquipmentView wired to the given controller."""
    renderer = MagicMock(spec=GraphicsContext)
    renderer.create_canvas = MagicMock(return_value=MagicMock())
    # PillowImageCanvas requires tile_dimensions to return actual integers
    renderer.tile_dimensions = (16, 16)
    view = EquipmentView(controller, renderer)

    # Initialize slot pixel bounds as if it were rendered.
    # With tile_dimensions (16, 16), row 0 center is at pixel 8, row 1 center at 24.
    # Each slot spans one line_height (approx tile_height=16 pixels).
    view._slot_pixel_bounds = {0: (0, 16), 1: (16, 32)}

    return view


def make_melee_weapon() -> Item:
    """Create a test melee weapon."""
    melee_spec = MeleeAttackSpec(damage_die="1d6")
    item_type = ItemType(
        name="Sword",
        description="A steel sword",
        size=ItemSize.NORMAL,
        melee_attack=melee_spec,
    )
    return item_type.create()


def make_consumable_item() -> Item:
    """Create a test consumable item."""
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


def test_click_inactive_slot_switches_weapon(controller: Controller) -> None:
    """Clicking an inactive slot switches to it via intent system."""
    view = make_equipment_view(controller)
    player = controller.gw.player

    assert player.inventory.active_slot == 0

    # Click row 1 which is slot 1 (inactive)
    result = view.handle_click(1)

    assert result is True

    # Process the queued SwitchWeaponIntent
    action = controller.turn_manager.dequeue_player_action()
    assert action is not None
    controller.turn_manager.execute_player_intent(action)

    assert player.inventory.active_slot == 1


def test_click_active_slot_enters_combat_mode(controller: Controller) -> None:
    """Clicking active slot with weapon enters combat mode."""
    view = make_equipment_view(controller)
    player = controller.gw.player

    # Equip a melee weapon in active slot
    weapon = make_melee_weapon()
    player.inventory.ready_slots[0] = weapon

    assert not controller.is_combat_mode()

    # Click row 0 which is in the active slot
    result = view.handle_click(0)

    assert result is True
    assert controller.is_combat_mode()


def test_click_active_slot_exits_combat_mode(controller: Controller) -> None:
    """Clicking active slot while in combat exits combat mode."""
    view = make_equipment_view(controller)
    player = controller.gw.player

    # Equip a melee weapon and enter combat mode
    weapon = make_melee_weapon()
    player.inventory.ready_slots[0] = weapon
    controller.enter_combat_mode()

    assert controller.is_combat_mode()

    # Click active slot again to exit
    result = view.handle_click(0)

    assert result is True
    assert not controller.is_combat_mode()


def test_click_active_consumable_slot_enters_targeting(controller: Controller) -> None:
    """Clicking active slot with consumable enters picker mode for targeting."""
    view = make_equipment_view(controller)
    player = controller.gw.player

    # Equip a consumable in active slot
    consumable = make_consumable_item()
    player.inventory.ready_slots[0] = consumable

    # Click active slot
    result = view.handle_click(0)

    assert result is True

    # Should have entered picker mode for targeting
    assert controller.picker_mode.active


def test_click_active_consumable_slot_exits_targeting(controller: Controller) -> None:
    """Clicking active consumable slot while targeting exits targeting mode."""
    view = make_equipment_view(controller)
    player = controller.gw.player

    # Equip a consumable
    consumable = make_consumable_item()
    player.inventory.ready_slots[0] = consumable

    # Enter targeting mode
    controller.start_consumable_targeting(consumable)

    assert controller.picker_mode.active

    # Click active slot again to exit
    result = view.handle_click(0)

    assert result is True
    # Should have exited targeting mode
    assert not controller.picker_mode.active


def test_click_empty_active_slot_enters_combat_mode(controller: Controller) -> None:
    """Clicking active slot when empty (Fists) enters combat mode."""
    view = make_equipment_view(controller)
    player = controller.gw.player

    # Ensure active slot is empty (Fists)
    player.inventory.ready_slots[0] = None

    assert not controller.is_combat_mode()

    # Click active slot - should enter combat mode for unarmed combat
    result = view.handle_click(0)

    assert result is True
    assert controller.is_combat_mode()


def test_is_row_in_active_slot(controller: Controller) -> None:
    """is_row_in_active_slot correctly identifies active slot rows."""
    view = make_equipment_view(controller)

    # With slot 0 active: row 0 is active, row 1 is inactive.
    # Pixel bounds use 16px per slot with tile_dimensions (16, 16).
    view._slot_pixel_bounds = {0: (0, 16), 1: (16, 32)}

    assert view.is_row_in_active_slot(0) is True
    assert view.is_row_in_active_slot(1) is False
    assert view.is_row_in_active_slot(2) is False


def test_get_slot_at_row(controller: Controller) -> None:
    """_get_slot_at_row returns correct slot index for each row."""
    view = make_equipment_view(controller)

    # Pixel bounds use 16px per slot with tile_dimensions (16, 16).
    view._slot_pixel_bounds = {0: (0, 16), 1: (16, 32)}

    assert view._get_slot_at_row(0) == 0
    assert view._get_slot_at_row(1) == 1
    assert view._get_slot_at_row(2) is None
