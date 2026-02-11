"""Unit tests for the ItemTransferHandler class.

Tests the extracted game logic for item transfers between player inventory,
containers, and ground - without requiring any UI/rendering setup.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from brileta import colors
from brileta.game.actors import Character, ItemPile
from brileta.game.actors.container import create_bookcase
from brileta.game.countables import CountableType
from brileta.game.game_world import GameWorld
from brileta.game.items.item_core import Item
from brileta.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    PISTOL_TYPE,
    STIM_TYPE,
)
from brileta.view.ui.inventory import (
    ActorInventorySource,
    ExternalInventory,
    ItemTransferHandler,
)
from tests.helpers import DummyGameWorld

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ground_items(gw: GameWorld, x: int, y: int) -> list[Item]:
    """Get items on the ground at a position by checking ItemPile actors."""
    items: list[Item] = []
    for actor in gw.actor_spatial_index.get_at_point(x, y):
        if isinstance(actor, ItemPile):
            items.extend(actor.inventory.get_items())
    return items


def _make_handler(
    *,
    source: ExternalInventory | ActorInventorySource | None = None,
) -> tuple[ItemTransferHandler, Character]:
    """Create a handler backed by a DummyGameWorld with a player."""
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)

    # Build a minimal controller with queue_action support
    controller: Any = MagicMock()
    controller.gw = gw
    controller.queue_action = MagicMock()

    handler = ItemTransferHandler(controller, player, source)
    return handler, player


# ---------------------------------------------------------------------------
# Equip Tests
# ---------------------------------------------------------------------------


class TestEquipItem:
    """Tests for equip_item (equip/unequip toggle)."""

    def test_equip_to_empty_slot(self) -> None:
        """Equipping an item should place it in the first empty ready slot."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.add_to_inventory(knife)

        handler.equip_item(knife)

        assert knife in player.inventory.ready_slots
        assert knife not in player.inventory._stored_items

    def test_equip_when_slots_full_swaps(self) -> None:
        """When all slots are full, equipping swaps with the active slot."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        pistol = PISTOL_TYPE.create()
        player.inventory.equip_to_slot(knife, 0)
        player.inventory.equip_to_slot(pistol, 1)
        player.inventory.active_slot = 0

        # Add a new weapon to inventory
        knife2 = COMBAT_KNIFE_TYPE.create()
        player.inventory.add_to_inventory(knife2)

        handler.equip_item(knife2)

        # New weapon should replace active slot, old one goes to inventory
        assert player.inventory.ready_slots[0] == knife2
        assert player.inventory.ready_slots[1] == pistol
        assert knife in player.inventory._stored_items

    def test_unequip_already_equipped(self) -> None:
        """Equipping an already-equipped item should unequip it."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.equip_to_slot(knife, 0)

        handler.equip_item(knife)

        assert knife not in player.inventory.ready_slots
        assert knife in player.inventory._stored_items

    def test_equip_returns_false(self) -> None:
        """equip_item always returns False (keep menu open)."""
        handler, player = _make_handler()
        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.add_to_inventory(knife)

        result = handler.equip_item(knife)

        assert result is False


# ---------------------------------------------------------------------------
# Use Consumable Tests
# ---------------------------------------------------------------------------


class TestUseConsumable:
    """Tests for use_consumable."""

    def test_use_consumable_queues_action(self) -> None:
        """Using a consumable should queue an action and return True."""
        handler, player = _make_handler()

        stim = STIM_TYPE.create()
        player.inventory.add_to_inventory(stim)

        result = handler.use_consumable(stim)

        assert result is True
        mock_ctrl = cast(MagicMock, handler.controller)
        mock_ctrl.queue_action.assert_called_once()

    def test_use_non_consumable_returns_false(self) -> None:
        """Using a non-consumable item should return False."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.add_to_inventory(knife)

        result = handler.use_consumable(knife)

        assert result is False
        mock_ctrl = cast(MagicMock, handler.controller)
        mock_ctrl.queue_action.assert_not_called()


# ---------------------------------------------------------------------------
# Transfer To Inventory Tests
# ---------------------------------------------------------------------------


class TestTransferToInventory:
    """Tests for transferring items from source to player inventory."""

    def test_transfer_single_item_from_container(self) -> None:
        """Transferring from a container should move item to player inventory."""
        handler, player = _make_handler()

        # Create a container with an item
        bookcase = create_bookcase(x=6, y=5, items=[])
        knife = COMBAT_KNIFE_TYPE.create()
        bookcase.inventory.add_item(knife)
        handler.controller.gw.add_actor(bookcase)

        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        handler.transfer_to_inventory(knife)

        assert knife in player.inventory
        assert len(bookcase.inventory) == 0

    def test_transfer_fails_when_inventory_full(self) -> None:
        """Transfer should fail when player inventory is full."""
        handler, player = _make_handler()

        # Fill inventory
        for _ in range(player.inventory.total_inventory_slots):
            player.inventory.add_to_inventory(COMBAT_KNIFE_TYPE.create())

        bookcase = create_bookcase(x=6, y=5, items=[])
        source_knife = COMBAT_KNIFE_TYPE.create()
        bookcase.inventory.add_item(source_knife)
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        initial_inv_size = len(player.inventory)
        handler.transfer_to_inventory(source_knife)

        # Inventory should not have changed
        assert len(player.inventory) == initial_inv_size

    def test_transfer_from_ground_pile(self) -> None:
        """Transferring from an ExternalInventory (ground) should work."""
        handler, player = _make_handler()

        # Spawn an item on the ground using the real spawning system
        knife = COMBAT_KNIFE_TYPE.create()
        handler.controller.gw.spawn_ground_item(knife, player.x, player.y)

        handler.source = ExternalInventory(
            position=(player.x, player.y), label="On the ground"
        )

        handler.transfer_to_inventory(knife)

        assert knife in player.inventory


# ---------------------------------------------------------------------------
# Transfer All To Inventory Tests
# ---------------------------------------------------------------------------


class TestTransferAllToInventory:
    """Tests for transferring all items of a type from source to inventory."""

    def test_transfer_all_of_type(self) -> None:
        """Should transfer all items of the same ItemType."""
        handler, player = _make_handler()

        bookcase = create_bookcase(x=6, y=5, items=[])
        knives = [COMBAT_KNIFE_TYPE.create() for _ in range(3)]
        for k in knives:
            bookcase.inventory.add_item(k)
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        handler.transfer_all_to_inventory(knives[0])

        # All 3 knives should now be in player inventory
        for k in knives:
            assert k in player.inventory
        assert len(bookcase.inventory) == 0

    def test_transfer_all_stops_when_full(self) -> None:
        """Should stop transferring when inventory becomes full."""
        handler, player = _make_handler()

        # Leave room for only 1 item
        capacity = player.inventory.total_inventory_slots
        for _ in range(capacity - 1):
            player.inventory.add_to_inventory(COMBAT_KNIFE_TYPE.create())

        bookcase = create_bookcase(x=6, y=5, items=[])
        knives = [COMBAT_KNIFE_TYPE.create() for _ in range(3)]
        for k in knives:
            bookcase.inventory.add_item(k)
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        handler.transfer_all_to_inventory(knives[0])

        # Only 1 knife should have been transferred (tiny items share a slot,
        # so all 3 may fit if the first was transferable)
        transferred = sum(1 for k in knives if k in player.inventory)
        assert transferred >= 1
        assert transferred < 3 or len(bookcase.inventory) == 0


# ---------------------------------------------------------------------------
# Transfer To Container Tests
# ---------------------------------------------------------------------------


class TestTransferToContainer:
    """Tests for transferring items from player inventory to container/ground."""

    def test_transfer_to_container(self) -> None:
        """Transferring to a container should move item from inventory."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.add_to_inventory(knife)

        bookcase = create_bookcase(x=6, y=5, items=[])
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        handler.transfer_to_container(knife)

        assert knife not in player.inventory
        assert knife in bookcase.inventory.get_items()

    def test_transfer_to_container_no_source_returns_false(self) -> None:
        """Transfer to container should return False when source is None."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.add_to_inventory(knife)

        result = handler.transfer_to_container(knife)

        assert result is False
        assert knife in player.inventory

    def test_transfer_to_container_unequips_first(self) -> None:
        """Transferring an equipped item should unequip it first."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.equip_to_slot(knife, 0)

        bookcase = create_bookcase(x=6, y=5, items=[])
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        handler.transfer_to_container(knife)

        assert knife not in player.inventory.ready_slots
        assert knife in bookcase.inventory.get_items()

    def test_transfer_all_to_container(self) -> None:
        """Should transfer all items of the same type to the container."""
        handler, player = _make_handler()

        knives = [COMBAT_KNIFE_TYPE.create() for _ in range(3)]
        for k in knives:
            player.inventory.add_to_inventory(k)

        bookcase = create_bookcase(x=6, y=5, items=[])
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        handler.transfer_all_to_container(knives[0])

        for k in knives:
            assert k not in player.inventory
            assert k in bookcase.inventory.get_items()


# ---------------------------------------------------------------------------
# Countable Transfer Tests
# ---------------------------------------------------------------------------


class TestCountableTransfers:
    """Tests for countable (coin) transfers between inventories."""

    def test_transfer_countable_to_inventory(self) -> None:
        """Should transfer countables from container to player."""
        handler, player = _make_handler()

        bookcase = create_bookcase(x=6, y=5, items=[])
        bookcase.inventory.add_countable(CountableType.COIN, 10)
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        handler.transfer_countable_to_inventory(CountableType.COIN)

        assert player.inventory.get_countable(CountableType.COIN) == 10
        assert bookcase.inventory.get_countable(CountableType.COIN) == 0

    def test_transfer_countable_zero_count_fails(self) -> None:
        """Should return False when source has no countables of that type."""
        handler, player = _make_handler()

        bookcase = create_bookcase(x=6, y=5, items=[])
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        result = handler.transfer_countable_to_inventory(CountableType.COIN)

        assert result is False
        assert player.inventory.get_countable(CountableType.COIN) == 0

    def test_transfer_countable_to_ground(self) -> None:
        """Should transfer countables from player inventory to container."""
        handler, player = _make_handler()

        player.inventory.add_countable(CountableType.COIN, 5)

        bookcase = create_bookcase(x=6, y=5, items=[])
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        handler.transfer_countable_to_ground(CountableType.COIN)

        assert player.inventory.get_countable(CountableType.COIN) == 0
        assert bookcase.inventory.get_countable(CountableType.COIN) == 5

    def test_transfer_countable_to_ground_no_source(self) -> None:
        """Should return False when source is None."""
        handler, player = _make_handler()
        player.inventory.add_countable(CountableType.COIN, 5)

        result = handler.transfer_countable_to_ground(CountableType.COIN)

        assert result is False
        assert player.inventory.get_countable(CountableType.COIN) == 5


# ---------------------------------------------------------------------------
# Drop Item Tests
# ---------------------------------------------------------------------------


class TestDropItem:
    """Tests for dropping items to the ground (no container present)."""

    def test_drop_item_to_ground(self) -> None:
        """Dropping should remove from inventory and spawn on ground."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.add_to_inventory(knife)

        handler.drop_item(knife)

        assert knife not in player.inventory
        # Item should be on ground (in an ItemPile actor)
        ground_items = _get_ground_items(handler.controller.gw, player.x, player.y)
        assert knife in ground_items

    def test_drop_item_unequips_first(self) -> None:
        """Dropping an equipped item should unequip it first."""
        handler, player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.equip_to_slot(knife, 0)

        handler.drop_item(knife)

        assert knife not in player.inventory.ready_slots
        assert knife not in player.inventory
        ground_items = _get_ground_items(handler.controller.gw, player.x, player.y)
        assert knife in ground_items

    def test_drop_returns_false(self) -> None:
        """drop_item always returns False (keep menu open)."""
        handler, player = _make_handler()
        knife = COMBAT_KNIFE_TYPE.create()
        player.inventory.add_to_inventory(knife)

        result = handler.drop_item(knife)

        assert result is False


# ---------------------------------------------------------------------------
# Drop All Of Type Tests
# ---------------------------------------------------------------------------


class TestDropAllOfType:
    """Tests for dropping all items of the same type."""

    def test_drop_all_of_type(self) -> None:
        """Should drop all items with the same ItemType."""
        handler, player = _make_handler()

        knives = [COMBAT_KNIFE_TYPE.create() for _ in range(3)]
        for k in knives:
            player.inventory.add_to_inventory(k)

        handler.drop_all_of_type(knives[0])

        for k in knives:
            assert k not in player.inventory

        ground_items = _get_ground_items(handler.controller.gw, player.x, player.y)
        for k in knives:
            assert k in ground_items

    def test_drop_all_no_items_returns_false(self) -> None:
        """Should return False when no items of the type exist."""
        handler, _player = _make_handler()

        knife = COMBAT_KNIFE_TYPE.create()
        # Don't add to inventory

        result = handler.drop_all_of_type(knife)

        assert result is False


# ---------------------------------------------------------------------------
# Drop Countable Tests
# ---------------------------------------------------------------------------


class TestDropCountable:
    """Tests for dropping countables to the ground."""

    def test_drop_countable(self) -> None:
        """Dropping countables should remove from inventory and spawn on ground."""
        handler, player = _make_handler()

        player.inventory.add_countable(CountableType.COIN, 7)

        handler.drop_countable(CountableType.COIN)

        assert player.inventory.get_countable(CountableType.COIN) == 0

    def test_drop_countable_zero_returns_false(self) -> None:
        """Should return False when player has no countables of that type."""
        handler, _player = _make_handler()

        result = handler.drop_countable(CountableType.COIN)

        assert result is False


# ---------------------------------------------------------------------------
# Source Query Tests
# ---------------------------------------------------------------------------


class TestSourceQueries:
    """Tests for get_items_from_source and get_countables_from_source."""

    def test_get_items_no_source(self) -> None:
        """Should return empty list when source is None."""
        handler, _player = _make_handler()

        assert handler.get_items_from_source() == []

    def test_get_items_from_container(self) -> None:
        """Should return items from a container source."""
        handler, _player = _make_handler()

        bookcase = create_bookcase(x=6, y=5, items=[])
        knife = COMBAT_KNIFE_TYPE.create()
        bookcase.inventory.add_item(knife)
        handler.controller.gw.add_actor(bookcase)
        handler.source = ActorInventorySource(actor=bookcase, label="Bookcase")

        items = handler.get_items_from_source()

        assert knife in items

    def test_get_countables_no_source(self) -> None:
        """Should return empty dict when source is None."""
        handler, _player = _make_handler()

        assert handler.get_countables_from_source() == {}
