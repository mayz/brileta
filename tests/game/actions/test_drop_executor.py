"""Tests for DropItemExecutor.

Verifies that items are properly dropped from inventory to the ground.
"""

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.game.actions.executors.misc import DropItemExecutor
from catley.game.actions.misc import DropItemIntent
from catley.game.actors import Actor, Character
from catley.game.enums import ItemSize
from catley.game.game_world import GameWorld
from catley.game.items.item_core import Item, ItemType
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    """Minimal controller for drop tests."""

    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(cast(Controller, self))
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100


def make_test_item(name: str = "Test Item") -> Item:
    """Create a simple test item."""
    item_type = ItemType(name=name, description="A test item", size=ItemSize.NORMAL)
    return Item(item_type)


def make_player_with_item() -> tuple[DummyController, Character, Item]:
    """Create a world with a player holding an item in inventory.

    Returns:
        (controller, player, item)
    """
    gw = DummyGameWorld()

    # Create player with inventory
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Create item and add to player's inventory
    item = make_test_item("Rusty Knife")
    player.inventory.add_to_inventory(item)

    controller = DummyController(gw)
    return controller, player, item


class TestDropItemExecutorBasicFunctionality:
    """Test basic drop functionality."""

    def test_drop_removes_item_from_stored_inventory(self) -> None:
        """Test that dropping an item removes it from stored inventory."""
        controller, player, item = make_player_with_item()

        # Verify item is in player's inventory
        assert item in player.inventory

        # Create and execute drop intent
        intent = DropItemIntent(cast(Controller, controller), player, item)
        executor = DropItemExecutor()
        result = executor.execute(intent)

        # Item should be removed from inventory
        assert result is not None
        assert result.succeeded
        assert item not in player.inventory

    def test_drop_removes_equipped_item_from_attack_slot(self) -> None:
        """Test that dropping an equipped item removes it from the slot."""
        controller, player, item = make_player_with_item()

        # Move item from stored inventory to equipped slot
        player.inventory.remove_from_inventory(item)
        player.inventory.equip_to_slot(item, 0)

        # Verify item is equipped
        assert player.inventory.attack_slots[0] == item

        # Create and execute drop intent
        intent = DropItemIntent(cast(Controller, controller), player, item)
        executor = DropItemExecutor()
        result = executor.execute(intent)

        # Item should be removed from equipped slot
        assert result is not None
        assert result.succeeded
        assert player.inventory.attack_slots[0] is None

    def test_drop_spawns_item_on_ground_at_player_location(self) -> None:
        """Test that dropped item appears on ground at player's position."""
        controller, player, item = make_player_with_item()

        # Execute drop
        intent = DropItemIntent(cast(Controller, controller), player, item)
        executor = DropItemExecutor()
        executor.execute(intent)

        # Find ground containers at player's location
        actors_at_player = [
            a
            for a in controller.gw.actors
            if a.x == player.x and a.y == player.y and a is not player
        ]

        # Should have spawned a ground container with the item
        assert len(actors_at_player) >= 1
        ground_container = actors_at_player[0]
        assert ground_container.inventory is not None
        assert item in ground_container.inventory


class TestDropItemExecutorFeedback:
    """Test that DropItemExecutor provides feedback messages."""

    def test_drop_publishes_success_message(self) -> None:
        """Test that dropping an item publishes a success message."""
        controller, player, item = make_player_with_item()

        intent = DropItemIntent(cast(Controller, controller), player, item)
        executor = DropItemExecutor()

        # Capture published events
        published_events: list = []
        with patch(
            "catley.game.actions.executors.misc.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            executor.execute(intent)

        # Should have published a message about dropping the item
        assert len(published_events) >= 1
        messages = [e.text for e in published_events if hasattr(e, "text")]
        assert any("drop" in msg.lower() and "Rusty Knife" in msg for msg in messages)

    def test_drop_missing_item_publishes_error_message(self) -> None:
        """Test that dropping a missing item publishes error message."""
        controller, player, _item = make_player_with_item()

        # Create a different item that's not in inventory
        missing_item = make_test_item("Ghost Item")

        intent = DropItemIntent(cast(Controller, controller), player, missing_item)
        executor = DropItemExecutor()

        published_events: list = []
        with patch(
            "catley.game.actions.executors.misc.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            result = executor.execute(intent)

        # Should fail and publish error message
        assert result is not None
        assert not result.succeeded
        messages = [e.text for e in published_events if hasattr(e, "text")]
        assert any("not found" in msg.lower() for msg in messages)


class TestDropItemExecutorEdgeCases:
    """Test edge cases in DropItemExecutor."""

    def test_drop_returns_failure_when_actor_has_no_inventory(self) -> None:
        """Test that drop fails gracefully when actor has no inventory."""
        gw = DummyGameWorld()

        # Create actor without inventory
        actor = Actor(
            0, 0, "@", colors.WHITE, "No Inventory", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(actor)

        controller = DummyController(gw)
        item = make_test_item()

        # Note: DropItemIntent expects Character but we're testing Actor edge case
        intent = DropItemIntent(
            cast(Controller, controller), cast(Character, actor), item
        )
        executor = DropItemExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert not result.succeeded

    def test_drop_item_at_different_player_positions(self) -> None:
        """Test that items drop at correct location when player moves."""
        gw = DummyGameWorld()

        # Create player at non-origin position
        player = Character(
            10, 15, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        item = make_test_item("Position Test Item")
        player.inventory.add_to_inventory(item)

        controller = DummyController(gw)

        # Execute drop
        intent = DropItemIntent(cast(Controller, controller), player, item)
        executor = DropItemExecutor()
        executor.execute(intent)

        # Find ground container - should be at player's position (10, 15)
        actors_at_player = [
            a for a in gw.actors if a.x == 10 and a.y == 15 and a is not player
        ]

        assert len(actors_at_player) >= 1
        ground_container = actors_at_player[0]
        assert ground_container.inventory is not None
        assert item in ground_container.inventory

    def test_drop_frees_inventory_slot(self) -> None:
        """Test that dropping an item frees up inventory space."""
        controller, player, item = make_player_with_item()

        # Get initial inventory count
        initial_count = len(list(player.inventory))

        # Execute drop
        intent = DropItemIntent(cast(Controller, controller), player, item)
        executor = DropItemExecutor()
        executor.execute(intent)

        # Inventory should have one less item
        final_count = len(list(player.inventory))
        assert final_count == initial_count - 1

    def test_drop_equipped_item_increments_revision(self) -> None:
        """Test that dropping an equipped item increments the inventory revision.

        The revision counter is used by the view caching system to detect changes.
        If not incremented, the UI won't update to show the slot as empty.
        """
        controller, player, item = make_player_with_item()

        # Move item from stored inventory to equipped slot
        player.inventory.remove_from_inventory(item)
        player.inventory.equip_to_slot(item, 0)

        # Record revision before drop
        revision_before = player.inventory.revision

        # Execute drop
        intent = DropItemIntent(cast(Controller, controller), player, item)
        executor = DropItemExecutor()
        executor.execute(intent)

        # Revision should have incremented so views know to redraw
        assert player.inventory.revision > revision_before
