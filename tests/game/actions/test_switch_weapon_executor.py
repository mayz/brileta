"""Tests for SwitchWeaponExecutor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.game.actions.executors.misc import SwitchWeaponExecutor
from catley.game.actions.misc import SwitchWeaponIntent
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.items.item_core import Item, ItemSize, ItemType
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    """Minimal controller for switch weapon tests."""

    gw: DummyGameWorld


def _make_player_with_weapons() -> tuple[DummyController, Character]:
    """Create a player with weapons in both slots."""
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Add weapons to slots
    pistol_type = ItemType(name="Pistol", description="A pistol", size=ItemSize.NORMAL)
    rifle_type = ItemType(
        name="Plasma Rifle", description="A plasma weapon", size=ItemSize.NORMAL
    )
    player.inventory.ready_slots[0] = Item(pistol_type)
    player.inventory.ready_slots[1] = Item(rifle_type)

    controller = DummyController(gw=gw)
    return controller, player


def test_switch_weapon_executor_changes_active_slot() -> None:
    """SwitchWeaponExecutor should change the active weapon slot."""
    controller, player = _make_player_with_weapons()

    assert player.inventory.active_slot == 0

    intent = SwitchWeaponIntent(cast(Controller, controller), player, slot=1)
    executor = SwitchWeaponExecutor()
    executor.execute(intent)

    assert player.inventory.active_slot == 1


def test_switch_weapon_executor_publishes_message_with_weapon_name() -> None:
    """SwitchWeaponExecutor should publish a message with the weapon name."""
    controller, player = _make_player_with_weapons()

    intent = SwitchWeaponIntent(cast(Controller, controller), player, slot=1)
    executor = SwitchWeaponExecutor()

    with patch("catley.game.actions.executors.misc.publish_event") as mock_publish:
        executor.execute(intent)

        mock_publish.assert_called_once()
        message_event = mock_publish.call_args[0][0]
        assert "Plasma Rifle" in message_event.text
        assert message_event.color == colors.GREEN


def test_switch_weapon_executor_shows_fists_for_empty_slot() -> None:
    """SwitchWeaponExecutor should show 'Fists' when slot has no weapon."""
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw=gw)

    # Slot 1 is empty by default
    assert player.inventory.ready_slots[1] is None

    intent = SwitchWeaponIntent(cast(Controller, controller), player, slot=1)
    executor = SwitchWeaponExecutor()

    with patch("catley.game.actions.executors.misc.publish_event") as mock_publish:
        executor.execute(intent)

        mock_publish.assert_called_once()
        message_event = mock_publish.call_args[0][0]
        assert "Fists" in message_event.text


def test_switch_weapon_executor_no_message_if_already_active() -> None:
    """SwitchWeaponExecutor should not publish a message if slot is already active."""
    controller, player = _make_player_with_weapons()

    assert player.inventory.active_slot == 0

    # Try to switch to the already-active slot
    intent = SwitchWeaponIntent(cast(Controller, controller), player, slot=0)
    executor = SwitchWeaponExecutor()

    with patch("catley.game.actions.executors.misc.publish_event") as mock_publish:
        executor.execute(intent)

        # No message should be published since we didn't actually switch
        mock_publish.assert_not_called()
