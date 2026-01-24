"""Tests for PickupExecutor.

Verifies that items are properly removed from source actors and feedback is given.
"""

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.game.actions.executors.misc import PickupExecutor
from catley.game.actions.misc import PickupIntent
from catley.game.actors import Actor, Character, ItemPile
from catley.game.enums import ItemSize
from catley.game.game_world import GameWorld
from catley.game.items.item_core import Item, ItemType
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    """Minimal controller for pickup tests."""

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


def make_world_with_ground_item() -> tuple[DummyController, Character, Item, Character]:
    """Create a world with a player and an item on a dead body.

    Returns:
        (controller, player, item, dead_character_holding_item)
    """
    gw = DummyGameWorld()

    # Create player
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Create item
    item = make_test_item("Rusty Knife")

    # Create dead character holding the item at player's location
    dead_char = Character(
        x=0,
        y=0,
        ch="x",
        color=colors.DEAD,
        name="Dead Body",
        game_world=cast(GameWorld, gw),
    )
    # Kill the character so it's recognized as a corpse
    dead_char.take_damage(dead_char.health.max_hp)
    assert dead_char.inventory is not None
    dead_char.inventory.add_to_inventory(item)
    gw.add_actor(dead_char)

    controller = DummyController(gw)
    return controller, player, item, dead_char


class TestPickupExecutorRemovesItemsFromSource:
    """Test that PickupExecutor removes items from source actors."""

    def test_pickup_removes_item_from_dead_actor_inventory(self) -> None:
        """Test that picking up an item removes it from the source inventory."""
        controller, player, item, dead_actor = make_world_with_ground_item()

        # Verify item is in dead actor's inventory
        assert dead_actor.inventory is not None
        assert item in dead_actor.inventory

        # Create and execute pickup intent
        intent = PickupIntent(cast(Controller, controller), player, [item])
        executor = PickupExecutor()
        result = executor.execute(intent)

        # Item should be removed from dead actor and added to player
        assert result is not None
        assert result.succeeded
        assert item not in dead_actor.inventory
        assert item in player.inventory

    def test_pickup_removes_equipped_item_from_dead_actor(self) -> None:
        """Test that picking up an equipped item unequips it from source."""
        controller, player, item, dead_actor = make_world_with_ground_item()

        # Move item from inventory to equipped slot
        assert dead_actor.inventory is not None
        dead_actor.inventory.remove_from_inventory(item)
        dead_actor.inventory.equip_to_slot(item, 0)

        # Verify item is equipped
        assert dead_actor.inventory.ready_slots[0] == item

        # Create and execute pickup intent
        intent = PickupIntent(cast(Controller, controller), player, [item])
        executor = PickupExecutor()
        result = executor.execute(intent)

        # Item should be removed from dead actor's equipment
        assert result is not None
        assert result.succeeded
        assert dead_actor.inventory.ready_slots[0] is None
        assert item in player.inventory

    def test_pickup_multiple_items_removes_all_from_source(self) -> None:
        """Test that picking up multiple items removes all from source."""
        controller, player, item1, dead_actor = make_world_with_ground_item()

        # Add a second item
        assert dead_actor.inventory is not None
        item2 = make_test_item("Bent Fork")
        dead_actor.inventory.add_to_inventory(item2)

        # Verify both items are in dead actor's inventory
        assert item1 in dead_actor.inventory
        assert item2 in dead_actor.inventory

        # Create and execute pickup intent for both
        intent = PickupIntent(cast(Controller, controller), player, [item1, item2])
        executor = PickupExecutor()
        result = executor.execute(intent)

        # Both items should be removed from dead actor and added to player
        assert result is not None
        assert result.succeeded
        assert item1 not in dead_actor.inventory
        assert item2 not in dead_actor.inventory
        assert item1 in player.inventory
        assert item2 in player.inventory


class TestPickupExecutorFeedback:
    """Test that PickupExecutor provides feedback messages."""

    def test_pickup_publishes_single_item_message(self) -> None:
        """Test that picking up one item publishes appropriate message."""
        controller, player, item, _dead_actor = make_world_with_ground_item()

        intent = PickupIntent(cast(Controller, controller), player, [item])
        executor = PickupExecutor()

        # Capture published events
        published_events: list = []
        with patch(
            "catley.game.actions.executors.misc.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            executor.execute(intent)

        # Should have published a message about picking up the item
        assert len(published_events) >= 1
        messages = [e.text for e in published_events if hasattr(e, "text")]
        assert any("Rusty Knife" in msg for msg in messages)

    def test_pickup_publishes_multiple_items_message(self) -> None:
        """Test that picking up multiple items publishes count message."""
        controller, player, item1, dead_actor = make_world_with_ground_item()

        # Add more items
        assert dead_actor.inventory is not None
        item2 = make_test_item("Bent Fork")
        item3 = make_test_item("Broken Glass")
        dead_actor.inventory.add_to_inventory(item2)
        dead_actor.inventory.add_to_inventory(item3)

        intent = PickupIntent(
            cast(Controller, controller), player, [item1, item2, item3]
        )
        executor = PickupExecutor()

        published_events: list = []
        with patch(
            "catley.game.actions.executors.misc.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            executor.execute(intent)

        # Should mention picking up multiple items
        messages = [e.text for e in published_events if hasattr(e, "text")]
        assert any("3 items" in msg for msg in messages)

    def test_pickup_publishes_inventory_full_message(self) -> None:
        """Test that inventory full condition publishes error message."""
        controller, player, item, _dead_actor = make_world_with_ground_item()

        # Fill player's inventory
        player.stats.strength = 0  # Minimal slots (5)
        for i in range(player.inventory.total_inventory_slots + 1):
            filler = make_test_item(f"Filler {i}")
            player.inventory.add_to_inventory(filler)

        intent = PickupIntent(cast(Controller, controller), player, [item])
        executor = PickupExecutor()

        published_events: list = []
        with patch(
            "catley.game.actions.executors.misc.publish_event",
            side_effect=lambda e: published_events.append(e),
        ):
            executor.execute(intent)

        # Should have published inventory full message
        messages = [e.text for e in published_events if hasattr(e, "text")]
        assert any("full" in msg.lower() for msg in messages)


class TestPickupExecutorEdgeCases:
    """Test edge cases in PickupExecutor."""

    def test_pickup_returns_failure_when_actor_has_no_inventory(self) -> None:
        """Test that pickup fails gracefully when actor has no inventory."""
        gw = DummyGameWorld()

        # Create actor without inventory
        actor = Actor(
            0, 0, "@", colors.WHITE, "No Inventory", game_world=cast(GameWorld, gw)
        )
        gw.add_actor(actor)

        controller = DummyController(gw)
        item = make_test_item()

        # Note: PickupIntent expects Character but we're testing Actor edge case
        intent = PickupIntent(
            cast(Controller, controller), cast(Character, actor), [item]
        )
        executor = PickupExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert not result.succeeded

    def test_pickup_skips_items_that_cant_fit(self) -> None:
        """Test that items that can't fit are skipped without error."""
        controller, player, item1, dead_actor = make_world_with_ground_item()

        # Fill player's inventory almost completely
        player.stats.strength = 0  # Minimal slots (5)
        for i in range(4):  # Fill 4 of 5 slots
            filler = make_test_item(f"Filler {i}")
            player.inventory.add_to_inventory(filler)

        # Try to pick up two items when only one can fit
        assert dead_actor.inventory is not None
        item2 = make_test_item("Second Item")
        dead_actor.inventory.add_to_inventory(item2)

        intent = PickupIntent(cast(Controller, controller), player, [item1, item2])
        executor = PickupExecutor()
        result = executor.execute(intent)

        # Should succeed (picked up at least one item)
        assert result is not None
        assert result.succeeded

        # First item should be picked up, second should remain
        assert item1 in player.inventory
        # Second item should still be in dead actor's inventory
        assert item2 in dead_actor.inventory

    def test_pickup_handles_already_missing_item(self) -> None:
        """Test pickup handles items removed between intent creation and execute."""
        controller, player, item, dead_actor = make_world_with_ground_item()

        # Remove the item before executing (simulates race condition)
        assert dead_actor.inventory is not None
        dead_actor.inventory.remove_from_inventory(item)

        intent = PickupIntent(cast(Controller, controller), player, [item])
        executor = PickupExecutor()
        result = executor.execute(intent)

        # Should not crash, but also not succeed
        assert result is not None
        assert not result.succeeded

    def test_pickup_removes_empty_ground_container(self) -> None:
        """Test that empty item piles are removed after pickup."""
        gw = DummyGameWorld()

        player = Character(
            0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        item = make_test_item()

        # Create an item pile
        ground_container = ItemPile(
            x=0,
            y=0,
            ch="!",
            name="Item Pile",
            items=[item],
            game_world=cast(GameWorld, gw),
        )
        gw.add_actor(ground_container)

        # Verify container is in the world
        assert ground_container in gw.actors

        controller = DummyController(gw)
        intent = PickupIntent(cast(Controller, controller), player, [item])
        executor = PickupExecutor()
        executor.execute(intent)

        # Container should be removed since it's now empty
        assert ground_container not in gw.actors
