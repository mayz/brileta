from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock, patch

from catley import colors
from catley.controller import Controller
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.items.item_core import Item, ItemSize, ItemType
from catley.view.render.graphics import GraphicsContext
from catley.view.views.equipment_view import EquipmentView
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    graphics: Any = None


def make_equipment_view() -> tuple[DummyController, Character, EquipmentView]:
    """Create a test equipment view with a player character."""
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    renderer = MagicMock(spec=GraphicsContext)
    renderer.create_canvas = MagicMock(return_value=MagicMock())
    # PillowImageCanvas requires tile_dimensions to return actual integers
    renderer.tile_dimensions = (16, 16)
    controller = DummyController(gw=gw, graphics=renderer)
    view = EquipmentView(cast(Controller, controller), renderer)

    return controller, player, view


def test_switch_to_slot_changes_active_weapon() -> None:
    """switch_to_slot should change the active weapon slot."""
    _controller, player, view = make_equipment_view()

    # Start at slot 0
    assert player.inventory.active_weapon_slot == 0

    # Switch to slot 1
    view.switch_to_slot(1)
    assert player.inventory.active_weapon_slot == 1

    # Switch back to slot 0
    view.switch_to_slot(0)
    assert player.inventory.active_weapon_slot == 0


def test_switch_to_slot_publishes_message() -> None:
    """switch_to_slot should publish a message with the weapon name."""
    _controller, player, view = make_equipment_view()

    # Add a weapon to slot 1
    weapon_type = ItemType(
        name="Plasma Rifle", description="A plasma weapon", size=ItemSize.NORMAL
    )
    weapon = Item(weapon_type)
    player.inventory.attack_slots[1] = weapon

    with patch("catley.view.views.equipment_view.publish_event") as mock_publish:
        view.switch_to_slot(1)

        mock_publish.assert_called_once()
        message_event = mock_publish.call_args[0][0]
        assert "Plasma Rifle" in message_event.text
        assert "secondary" in message_event.text


def test_switch_to_slot_shows_empty_for_no_weapon() -> None:
    """switch_to_slot should show 'Empty' when slot has no weapon."""
    _controller, player, view = make_equipment_view()

    # Slot 1 is empty by default
    assert player.inventory.attack_slots[1] is None

    with patch("catley.view.views.equipment_view.publish_event") as mock_publish:
        view.switch_to_slot(1)

        mock_publish.assert_called_once()
        message_event = mock_publish.call_args[0][0]
        assert "Empty" in message_event.text


def test_handle_click_on_inactive_slot_switches() -> None:
    """Clicking on an inactive weapon slot should switch to that weapon."""
    _controller, player, view = make_equipment_view()

    assert player.inventory.active_weapon_slot == 0

    # Simulate the slot bounds that would be set during rendering:
    # Each slot takes exactly one row (implementation uses 1 row per slot)
    view._slot_row_bounds = {0: (0, 1), 1: (1, 2)}

    # Click row 1 which is slot 1 (inactive)
    result = view.handle_click(1)

    assert result is True
    assert player.inventory.active_weapon_slot == 1


def test_handle_click_outside_slots_does_nothing() -> None:
    """Clicking outside the slot bounds should not switch weapons."""
    _controller, player, view = make_equipment_view()

    assert player.inventory.active_weapon_slot == 0

    # Simulate the slot bounds (1 row per slot)
    view._slot_row_bounds = {0: (0, 1), 1: (1, 2)}

    # Click row 2 which is outside the slot bounds
    result = view.handle_click(2)

    assert result is False
    assert player.inventory.active_weapon_slot == 0
