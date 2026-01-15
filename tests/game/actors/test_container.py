"""Tests for Container actor and ContainerStorage component.

This module tests the container system including:
- ContainerStorage: Fixed-capacity item storage for containers
- Container: Actor class for searchable containers in the game world
- Factory functions for creating different container types
"""

from __future__ import annotations

from catley.game.actors.components import CharacterInventory, ContainerStorage
from catley.game.actors.container import (
    Container,
    create_barrel,
    create_crate,
    create_footlocker,
    create_locker,
)
from catley.game.enums import ItemSize
from catley.game.items.item_core import Item, ItemType

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def make_item(name: str = "Test Item", size: ItemSize = ItemSize.NORMAL) -> Item:
    """Create a test item with the given name and size."""
    return Item(ItemType(name=name, description="A test item", size=size))


# =============================================================================
# CONTAINER STORAGE TESTS
# =============================================================================


class TestContainerStorage:
    """Tests for ContainerStorage component."""

    def test_initialization_with_default_capacity(self) -> None:
        """ContainerStorage should default to capacity of 10."""
        storage = ContainerStorage()
        assert storage.capacity == 10

    def test_initialization_with_custom_capacity(self) -> None:
        """ContainerStorage should accept custom capacity."""
        storage = ContainerStorage(capacity=5)
        assert storage.capacity == 5

    def test_add_item_success(self) -> None:
        """Adding an item should succeed when space is available."""
        storage = ContainerStorage(capacity=5)
        item = make_item("Sword")

        success, message = storage.add_item(item)

        assert success is True
        assert "Added" in message
        assert item in storage

    def test_add_item_failure_when_full(self) -> None:
        """Adding an item should fail when container is full."""
        storage = ContainerStorage(capacity=2)
        storage.add_item(make_item("Item1"))
        storage.add_item(make_item("Item2"))

        success, message = storage.add_item(make_item("Item3"))

        assert success is False
        assert "full" in message.lower()

    def test_remove_item_success(self) -> None:
        """Removing an item that exists should succeed."""
        storage = ContainerStorage()
        item = make_item("Sword")
        storage.add_item(item)

        result = storage.remove_item(item)

        assert result is True
        assert item not in storage

    def test_remove_item_failure_when_not_present(self) -> None:
        """Removing an item that doesn't exist should return False."""
        storage = ContainerStorage()
        item = make_item("Sword")

        result = storage.remove_item(item)

        assert result is False

    def test_get_items_returns_copy(self) -> None:
        """get_items should return a copy, not the internal list."""
        storage = ContainerStorage()
        item = make_item("Sword")
        storage.add_item(item)

        items = storage.get_items()
        items.clear()

        assert item in storage  # Original still has the item

    def test_len_returns_item_count(self) -> None:
        """__len__ should return the number of items."""
        storage = ContainerStorage()
        storage.add_item(make_item("Item1"))
        storage.add_item(make_item("Item2"))
        storage.add_item(make_item("Item3"))

        assert len(storage) == 3

    def test_contains_checks_item_presence(self) -> None:
        """__contains__ should check if item is in storage."""
        storage = ContainerStorage()
        item1 = make_item("Item1")
        item2 = make_item("Item2")
        storage.add_item(item1)

        assert item1 in storage
        assert item2 not in storage

    def test_iter_yields_all_items(self) -> None:
        """__iter__ should yield all items."""
        storage = ContainerStorage()
        item1 = make_item("Item1")
        item2 = make_item("Item2")
        storage.add_item(item1)
        storage.add_item(item2)

        items = list(storage)

        assert item1 in items
        assert item2 in items

    def test_is_empty_true_when_no_items(self) -> None:
        """is_empty should return True when container has no items."""
        storage = ContainerStorage()
        assert storage.is_empty() is True

    def test_is_empty_false_when_has_items(self) -> None:
        """is_empty should return False when container has items."""
        storage = ContainerStorage()
        storage.add_item(make_item())
        assert storage.is_empty() is False

    def test_has_items_false_when_empty(self) -> None:
        """has_items should return False when container is empty."""
        storage = ContainerStorage()
        assert storage.has_items() is False

    def test_has_items_true_when_not_empty(self) -> None:
        """has_items should return True when container has items."""
        storage = ContainerStorage()
        storage.add_item(make_item())
        assert storage.has_items() is True

    def test_revision_increments_on_add(self) -> None:
        """Revision counter should increment when item is added."""
        storage = ContainerStorage()
        initial_revision = storage.revision

        storage.add_item(make_item())

        assert storage.revision == initial_revision + 1

    def test_revision_increments_on_remove(self) -> None:
        """Revision counter should increment when item is removed."""
        storage = ContainerStorage()
        item = make_item()
        storage.add_item(item)
        initial_revision = storage.revision

        storage.remove_item(item)

        assert storage.revision == initial_revision + 1


# =============================================================================
# CONTAINER ACTOR TESTS
# =============================================================================


class TestContainer:
    """Tests for Container actor class."""

    def test_initialization_creates_storage(self) -> None:
        """Container should be created with ContainerStorage inventory."""
        container = Container(x=5, y=10)

        assert container.inventory is not None
        assert isinstance(container.inventory, ContainerStorage)

    def test_initialization_with_custom_capacity(self) -> None:
        """Container should respect custom capacity."""
        container = Container(x=0, y=0, capacity=20)

        assert container.inventory.capacity == 20

    def test_initialization_with_items(self) -> None:
        """Container should be initialized with provided items."""
        items = [make_item("Item1"), make_item("Item2")]
        container = Container(x=0, y=0, items=items)

        assert len(container.inventory) == 2
        assert items[0] in container.inventory
        assert items[1] in container.inventory

    def test_initialization_with_name(self) -> None:
        """Container should use provided name."""
        container = Container(x=0, y=0, name="Treasure Chest")

        assert container.name == "Treasure Chest"

    def test_initialization_blocks_movement_by_default(self) -> None:
        """Containers should block movement by default."""
        container = Container(x=0, y=0)

        assert container.blocks_movement is True

    def test_initialization_can_allow_movement(self) -> None:
        """Container can be configured to not block movement."""
        container = Container(x=0, y=0, blocks_movement=False)

        assert container.blocks_movement is False

    def test_position_is_set(self) -> None:
        """Container should be at specified position."""
        container = Container(x=15, y=25)

        assert container.x == 15
        assert container.y == 25


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestContainerFactories:
    """Tests for container factory functions."""

    def test_create_crate_returns_container(self) -> None:
        """create_crate should return a Container with appropriate properties."""
        crate = create_crate(x=5, y=10)

        assert isinstance(crate, Container)
        assert crate.name == "Wooden Crate"
        assert crate.blocks_movement is True
        assert crate.x == 5
        assert crate.y == 10

    def test_create_crate_with_items(self) -> None:
        """create_crate should accept initial items."""
        items = [make_item("Loot")]
        crate = create_crate(x=0, y=0, items=items)

        assert len(crate.inventory) == 1

    def test_create_locker_returns_container(self) -> None:
        """create_locker should return a Container with appropriate properties."""
        locker = create_locker(x=3, y=7)

        assert isinstance(locker, Container)
        assert locker.name == "Metal Locker"
        assert locker.x == 3
        assert locker.y == 7

    def test_create_locker_has_higher_capacity(self) -> None:
        """create_locker should have higher default capacity than crates."""
        crate = create_crate(x=0, y=0)
        locker = create_locker(x=0, y=0)

        assert locker.inventory.capacity > crate.inventory.capacity

    def test_create_barrel_returns_container(self) -> None:
        """create_barrel should return a Container with appropriate properties."""
        barrel = create_barrel(x=2, y=4)

        assert isinstance(barrel, Container)
        assert barrel.name == "Barrel"
        assert barrel.x == 2
        assert barrel.y == 4

    def test_create_footlocker_returns_container(self) -> None:
        """create_footlocker should return a Container with appropriate properties."""
        footlocker = create_footlocker(x=1, y=1)

        assert isinstance(footlocker, Container)
        assert footlocker.name == "Footlocker"

    def test_create_footlocker_does_not_block_movement(self) -> None:
        """create_footlocker should not block movement (can step over)."""
        footlocker = create_footlocker(x=0, y=0)

        assert footlocker.blocks_movement is False


# =============================================================================
# INVENTORY POLYMORPHISM TESTS
# =============================================================================


class TestInventoryPolymorphism:
    """Tests verifying both inventory types share a common interface."""

    def test_both_have_get_items(self) -> None:
        """Both inventory types should support get_items()."""
        from catley.game.actors.components import StatsComponent

        char_inv = CharacterInventory(StatsComponent())
        container_inv = ContainerStorage()

        item1 = make_item("CharItem")
        item2 = make_item("ContainerItem")
        char_inv.add_item(item1)
        container_inv.add_item(item2)

        # get_items() should return the items we added
        char_items = char_inv.get_items()
        container_items = container_inv.get_items()

        assert item1 in char_items
        assert item2 in container_items
        assert len(char_items) == 1
        assert len(container_items) == 1

    def test_both_have_add_item(self) -> None:
        """Both inventory types should support add_item()."""
        from catley.game.actors.components import StatsComponent

        char_inv = CharacterInventory(StatsComponent())
        container_inv = ContainerStorage()

        item1 = make_item("Item1")
        item2 = make_item("Item2")

        success1, _ = char_inv.add_item(item1)
        success2, _ = container_inv.add_item(item2)

        assert success1 is True
        assert success2 is True

    def test_both_have_remove_item(self) -> None:
        """Both inventory types should support remove_item()."""
        from catley.game.actors.components import StatsComponent

        char_inv = CharacterInventory(StatsComponent())
        container_inv = ContainerStorage()

        item1 = make_item("Item1")
        item2 = make_item("Item2")
        char_inv.add_item(item1)
        container_inv.add_item(item2)

        result1 = char_inv.remove_item(item1)
        result2 = container_inv.remove_item(item2)

        assert result1 is True
        assert result2 is True

    def test_both_have_capacity_property(self) -> None:
        """Both inventory types should have capacity property."""
        from catley.game.actors.components import StatsComponent

        stats = StatsComponent(strength=5)  # 10 slots
        char_inv = CharacterInventory(stats)
        container_inv = ContainerStorage(capacity=15)

        assert char_inv.capacity == 10
        assert container_inv.capacity == 15

    def test_both_support_len(self) -> None:
        """Both inventory types should support len()."""
        from catley.game.actors.components import StatsComponent

        char_inv = CharacterInventory(StatsComponent())
        container_inv = ContainerStorage()

        char_inv.add_item(make_item())
        container_inv.add_item(make_item())
        container_inv.add_item(make_item())

        assert len(char_inv) >= 1
        assert len(container_inv) == 2

    def test_both_support_contains(self) -> None:
        """Both inventory types should support 'in' operator."""
        from catley.game.actors.components import StatsComponent

        char_inv = CharacterInventory(StatsComponent())
        container_inv = ContainerStorage()

        item1 = make_item("Item1")
        item2 = make_item("Item2")
        char_inv.add_item(item1)
        container_inv.add_item(item2)

        assert item1 in char_inv
        assert item2 in container_inv

    def test_both_support_iter(self) -> None:
        """Both inventory types should support iteration."""
        from catley.game.actors.components import StatsComponent

        char_inv = CharacterInventory(StatsComponent())
        container_inv = ContainerStorage()

        item1 = make_item("Item1")
        item2 = make_item("Item2")
        char_inv.add_item(item1)
        container_inv.add_item(item2)

        char_items = list(char_inv)
        container_items = list(container_inv)

        assert item1 in char_items
        assert item2 in container_items
