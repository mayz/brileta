"""Tests for the countables system and item stacking display.

This module tests:
- CountableType and CountableDefinition
- Countable storage on InventoryComponent
- ItemPile is_empty() and display_name with countables
- group_items_for_display() for UI stacking
- PickupCountableExecutor and DropCountableExecutor
- ItemSpawner.spawn_ground_countable()
"""

from __future__ import annotations

import pytest

from catley.game.actors.components import (
    CharacterInventory,
    ContainerStorage,
    StatsComponent,
)
from catley.game.actors.container import ItemPile
from catley.game.countables import (
    COUNTABLE_DEFINITIONS,
    CountableDefinition,
    CountableType,
    get_countable_definition,
    get_countable_display_name,
)
from catley.game.enums import ItemSize
from catley.game.items.item_core import Item, ItemType
from catley.view.ui.dual_pane_menu import group_items_for_display

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def make_stats(strength: int = 5, toughness: int = 5) -> StatsComponent:
    """Create a StatsComponent with specified or default stats."""
    return StatsComponent(strength=strength, toughness=toughness)


def make_item(name: str = "Test Item", size: ItemSize = ItemSize.NORMAL) -> Item:
    """Create a test item with the given name and size."""
    item_type = ItemType(name=name, description="A test item", size=size)
    return Item(item_type)


def make_tiny_item(name: str = "Shell") -> Item:
    """Create a tiny (stackable) test item."""
    item_type = ItemType(name=name, description="A tiny item", size=ItemSize.TINY)
    return Item(item_type)


# Shared ItemType for testing stacking (items of same type should stack)
_SHELL_TYPE = ItemType(name="Shotgun Shell", description="12 gauge", size=ItemSize.TINY)
_BANDAGE_TYPE = ItemType(name="Bandage", description="First aid", size=ItemSize.TINY)


def make_shell() -> Item:
    """Create a shell item that shares ItemType with other shells."""
    return Item(_SHELL_TYPE)


def make_bandage() -> Item:
    """Create a bandage item that shares ItemType with other bandages."""
    return Item(_BANDAGE_TYPE)


# =============================================================================
# COUNTABLE DEFINITIONS TESTS
# =============================================================================


class TestCountableDefinitions:
    """Tests for CountableType and CountableDefinition."""

    def test_coin_type_exists(self) -> None:
        """COIN should be a valid CountableType."""
        assert CountableType.COIN is not None
        assert CountableType.COIN in COUNTABLE_DEFINITIONS

    def test_coin_definition_properties(self) -> None:
        """COIN definition should have expected properties."""
        defn = COUNTABLE_DEFINITIONS[CountableType.COIN]

        assert defn.countable_type == CountableType.COIN
        assert defn.name == "Coin"
        assert defn.plural_name == "Coins"
        assert defn.glyph == "\U0000e706"
        assert defn.color == (255, 215, 0)  # gold

    def test_countable_definition_is_frozen(self) -> None:
        """CountableDefinition should be immutable (frozen dataclass)."""
        defn = COUNTABLE_DEFINITIONS[CountableType.COIN]

        with pytest.raises(AttributeError):  # FrozenInstanceError
            defn.name = "Modified"

    def test_get_countable_definition_returns_correct_definition(self) -> None:
        """get_countable_definition() should return the matching definition."""
        defn = get_countable_definition(CountableType.COIN)

        assert isinstance(defn, CountableDefinition)
        assert defn.countable_type == CountableType.COIN

    def test_get_countable_display_name_singular(self) -> None:
        """Display name should use singular form for quantity 1."""
        name = get_countable_display_name(CountableType.COIN, 1)

        assert name == "1 Coin"

    def test_get_countable_display_name_plural(self) -> None:
        """Display name should use plural form for quantity > 1."""
        name = get_countable_display_name(CountableType.COIN, 47)

        assert name == "47 Coins"

    def test_get_countable_display_name_zero(self) -> None:
        """Display name should use plural form for quantity 0."""
        name = get_countable_display_name(CountableType.COIN, 0)

        assert name == "0 Coins"


# =============================================================================
# INVENTORY COMPONENT COUNTABLE TESTS
# =============================================================================


class TestInventoryComponentCountables:
    """Tests for countable storage on InventoryComponent."""

    def test_get_countable_returns_zero_when_empty(self) -> None:
        """get_countable() should return 0 for countables not present."""
        stats = make_stats()
        inv = CharacterInventory(stats)

        assert inv.get_countable(CountableType.COIN) == 0

    def test_add_countable_stores_amount(self) -> None:
        """add_countable() should store the specified amount."""
        stats = make_stats()
        inv = CharacterInventory(stats)

        inv.add_countable(CountableType.COIN, 50)

        assert inv.get_countable(CountableType.COIN) == 50

    def test_add_countable_accumulates(self) -> None:
        """add_countable() should add to existing amount."""
        stats = make_stats()
        inv = CharacterInventory(stats)

        inv.add_countable(CountableType.COIN, 30)
        inv.add_countable(CountableType.COIN, 20)

        assert inv.get_countable(CountableType.COIN) == 50

    def test_add_countable_ignores_zero(self) -> None:
        """add_countable() should ignore zero amounts."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        initial_revision = inv.revision

        inv.add_countable(CountableType.COIN, 0)

        assert inv.get_countable(CountableType.COIN) == 0
        assert inv.revision == initial_revision  # No revision bump

    def test_add_countable_ignores_negative(self) -> None:
        """add_countable() should ignore negative amounts."""
        stats = make_stats()
        inv = CharacterInventory(stats)

        inv.add_countable(CountableType.COIN, 50)
        inv.add_countable(CountableType.COIN, -20)

        assert inv.get_countable(CountableType.COIN) == 50

    def test_add_countable_increments_revision(self) -> None:
        """add_countable() should increment revision counter."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        initial_revision = inv.revision

        inv.add_countable(CountableType.COIN, 10)

        assert inv.revision == initial_revision + 1

    def test_remove_countable_reduces_amount(self) -> None:
        """remove_countable() should reduce the stored amount."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        inv.add_countable(CountableType.COIN, 50)

        result = inv.remove_countable(CountableType.COIN, 20)

        assert result is True
        assert inv.get_countable(CountableType.COIN) == 30

    def test_remove_countable_removes_key_when_zero(self) -> None:
        """remove_countable() should remove key when amount reaches 0."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        inv.add_countable(CountableType.COIN, 50)

        result = inv.remove_countable(CountableType.COIN, 50)

        assert result is True
        assert inv.get_countable(CountableType.COIN) == 0
        assert CountableType.COIN not in inv._countables

    def test_remove_countable_fails_insufficient_amount(self) -> None:
        """remove_countable() should return False if amount is insufficient."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        inv.add_countable(CountableType.COIN, 30)

        result = inv.remove_countable(CountableType.COIN, 50)

        assert result is False
        assert inv.get_countable(CountableType.COIN) == 30  # Unchanged

    def test_remove_countable_fails_when_empty(self) -> None:
        """remove_countable() should return False when none present."""
        stats = make_stats()
        inv = CharacterInventory(stats)

        result = inv.remove_countable(CountableType.COIN, 10)

        assert result is False

    def test_remove_countable_increments_revision(self) -> None:
        """remove_countable() should increment revision counter on success."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        inv.add_countable(CountableType.COIN, 50)
        initial_revision = inv.revision

        inv.remove_countable(CountableType.COIN, 10)

        assert inv.revision == initial_revision + 1

    def test_transfer_countable_to_moves_amount(self) -> None:
        """transfer_countable_to() should move countables between inventories."""
        stats = make_stats()
        source = CharacterInventory(stats)
        target = CharacterInventory(stats)
        source.add_countable(CountableType.COIN, 100)

        result = source.transfer_countable_to(target, CountableType.COIN, 30)

        assert result is True
        assert source.get_countable(CountableType.COIN) == 70
        assert target.get_countable(CountableType.COIN) == 30

    def test_transfer_countable_to_fails_insufficient(self) -> None:
        """transfer_countable_to() should fail if source has insufficient."""
        stats = make_stats()
        source = CharacterInventory(stats)
        target = CharacterInventory(stats)
        source.add_countable(CountableType.COIN, 30)

        result = source.transfer_countable_to(target, CountableType.COIN, 50)

        assert result is False
        assert source.get_countable(CountableType.COIN) == 30  # Unchanged
        assert target.get_countable(CountableType.COIN) == 0

    def test_countables_property_returns_copy(self) -> None:
        """countables property should return a copy, not the internal dict."""
        stats = make_stats()
        inv = CharacterInventory(stats)
        inv.add_countable(CountableType.COIN, 50)

        countables = inv.countables
        countables[CountableType.COIN] = 9999  # Modify the copy

        # Internal state should be unchanged
        assert inv.get_countable(CountableType.COIN) == 50


class TestContainerStorageCountables:
    """Tests for countable storage on ContainerStorage."""

    def test_container_storage_supports_countables(self) -> None:
        """ContainerStorage should inherit countable methods."""
        storage = ContainerStorage(capacity=10)

        storage.add_countable(CountableType.COIN, 100)

        assert storage.get_countable(CountableType.COIN) == 100

    def test_container_storage_transfer_to_character(self) -> None:
        """Countables can transfer from container to character inventory."""
        container = ContainerStorage(capacity=10)
        character = CharacterInventory(make_stats())
        container.add_countable(CountableType.COIN, 50)

        result = container.transfer_countable_to(character, CountableType.COIN, 50)

        assert result is True
        assert container.get_countable(CountableType.COIN) == 0
        assert character.get_countable(CountableType.COIN) == 50


# =============================================================================
# ITEM PILE TESTS
# =============================================================================


class TestItemPileCountables:
    """Tests for ItemPile with countables."""

    def test_is_empty_true_when_no_items_and_no_countables(self) -> None:
        """is_empty() should return True when pile has nothing."""
        pile = ItemPile(x=5, y=5)

        assert pile.is_empty() is True

    def test_is_empty_false_when_has_items(self) -> None:
        """is_empty() should return False when pile has items."""
        pile = ItemPile(x=5, y=5, items=[make_item()])

        assert pile.is_empty() is False

    def test_is_empty_false_when_has_countables(self) -> None:
        """is_empty() should return False when pile has countables."""
        pile = ItemPile(x=5, y=5)
        pile.inventory.add_countable(CountableType.COIN, 10)

        assert pile.is_empty() is False

    def test_is_empty_false_when_has_both(self) -> None:
        """is_empty() should return False when pile has items and countables."""
        pile = ItemPile(x=5, y=5, items=[make_item()])
        pile.inventory.add_countable(CountableType.COIN, 10)

        assert pile.is_empty() is False

    def test_display_name_single_item(self) -> None:
        """display_name should show item name when pile has one item."""
        item = make_item("Sword")
        pile = ItemPile(x=5, y=5, items=[item])

        assert pile.display_name == "Sword"

    def test_display_name_multiple_items(self) -> None:
        """display_name should show count when pile has multiple items."""
        pile = ItemPile(x=5, y=5, items=[make_item(), make_item()])

        assert pile.display_name == "Item pile (2 items)"

    def test_display_name_countables_only(self) -> None:
        """display_name should show countable info when only countables."""
        pile = ItemPile(x=5, y=5)
        pile.inventory.add_countable(CountableType.COIN, 47)

        assert pile.display_name == "47 Coins"

    def test_display_name_countables_singular(self) -> None:
        """display_name should use singular for 1 countable."""
        pile = ItemPile(x=5, y=5)
        pile.inventory.add_countable(CountableType.COIN, 1)

        assert pile.display_name == "1 Coin"

    def test_display_name_items_and_countables(self) -> None:
        """display_name should show both when pile has items and countables."""
        pile = ItemPile(x=5, y=5, items=[make_item()])
        pile.inventory.add_countable(CountableType.COIN, 25)

        assert pile.display_name == "Items and 25 Coins"


# =============================================================================
# ITEM STACKING DISPLAY TESTS
# =============================================================================


class TestGroupItemsForDisplay:
    """Tests for group_items_for_display() UI grouping function."""

    def test_empty_list_returns_empty(self) -> None:
        """Empty input should return empty output."""
        result = group_items_for_display([])

        assert result == []

    def test_single_normal_item_count_one(self) -> None:
        """Single normal item should have count 1."""
        item = make_item("Sword", ItemSize.NORMAL)

        result = group_items_for_display([item])

        assert len(result) == 1
        assert result[0] == (item, 1)

    def test_normal_items_not_stacked(self) -> None:
        """Normal size items should not stack even with same type."""
        item_type = ItemType(name="Sword", description="", size=ItemSize.NORMAL)
        item1 = Item(item_type)
        item2 = Item(item_type)

        result = group_items_for_display([item1, item2])

        assert len(result) == 2
        assert result[0] == (item1, 1)
        assert result[1] == (item2, 1)

    def test_big_items_not_stacked(self) -> None:
        """Big items should not stack."""
        item = make_item("Rifle", ItemSize.BIG)

        result = group_items_for_display([item])

        assert result == [(item, 1)]

    def test_huge_items_not_stacked(self) -> None:
        """Huge items should not stack."""
        item = make_item("Cannon", ItemSize.HUGE)

        result = group_items_for_display([item])

        assert result == [(item, 1)]

    def test_tiny_items_same_type_stack(self) -> None:
        """Tiny items of the same ItemType should stack."""
        shell1 = make_shell()
        shell2 = make_shell()
        shell3 = make_shell()

        result = group_items_for_display([shell1, shell2, shell3])

        assert len(result) == 1
        assert result[0][0] is shell1  # First one is representative
        assert result[0][1] == 3

    def test_tiny_items_different_types_separate(self) -> None:
        """Tiny items of different ItemTypes should not stack."""
        shell = make_shell()
        bandage = make_bandage()

        result = group_items_for_display([shell, bandage])

        assert len(result) == 2
        # Order preserved from input
        assert result[0] == (shell, 1)
        assert result[1] == (bandage, 1)

    def test_mixed_sizes_correct_grouping(self) -> None:
        """Mixed sizes should group correctly - only tiny stacks."""
        sword = make_item("Sword", ItemSize.NORMAL)
        shell1 = make_shell()
        shell2 = make_shell()
        bandage = make_bandage()
        rifle = make_item("Rifle", ItemSize.BIG)

        result = group_items_for_display([sword, shell1, shell2, bandage, rifle])

        # sword: count 1 (normal)
        # shells: grouped to count 2 (tiny, same type)
        # bandage: count 1 (tiny, different type)
        # rifle: count 1 (big)
        assert len(result) == 4
        assert result[0] == (sword, 1)
        assert result[1] == (shell1, 2)
        assert result[2] == (bandage, 1)
        assert result[3] == (rifle, 1)

    def test_interleaved_tiny_items_stack(self) -> None:
        """Tiny items should stack even if not adjacent in input list."""
        shell1 = make_shell()
        bandage = make_bandage()
        shell2 = make_shell()

        result = group_items_for_display([shell1, bandage, shell2])

        # Shells should be grouped together (first shell is representative)
        assert len(result) == 2
        assert result[0] == (shell1, 2)
        assert result[1] == (bandage, 1)

    def test_representative_item_is_first_of_type(self) -> None:
        """The representative item should be the first one encountered."""
        shell1 = make_shell()
        shell2 = make_shell()

        result = group_items_for_display([shell1, shell2])

        assert result[0][0] is shell1
        assert result[0][0] is not shell2


# =============================================================================
# INVENTORY DISPLAY TESTS
# =============================================================================


class TestDualPaneMenuCountableDisplay:
    """Tests for countable display in the inventory menu."""

    def test_generate_item_detail_handles_countable_type(self) -> None:
        """_generate_item_detail() should handle CountableType without crashing."""
        from unittest.mock import MagicMock

        from catley.view.ui.dual_pane_menu import DualPaneMenu

        # Create a minimal mock controller with required attributes
        mock_controller = MagicMock()
        mock_controller.gw.player.inventory = CharacterInventory(make_stats())
        mock_controller.gw.player.inventory.add_countable(CountableType.COIN, 50)

        menu = DualPaneMenu(mock_controller, source=None)

        # This should not raise an AttributeError
        header, description, stats = menu._generate_item_detail(CountableType.COIN)

        assert len(header) == 2
        assert header[0] == "Currency"
        assert "50" in header[1]
        assert "Coin" in header[1]
        assert "Tiny (shares slot)" in header[1]
        assert description == ["Money. You know how it works."]
        assert stats is None


# =============================================================================
# EXECUTOR TESTS
# =============================================================================


class TestPickupCountableExecutor:
    """Tests for PickupCountableExecutor."""

    def test_pickup_transfers_countables_to_actor(self) -> None:
        """Executor should transfer countables from source to actor inventory."""
        from unittest.mock import MagicMock

        from catley.game.actions.executors.misc import PickupCountableExecutor

        # Set up source and actor inventories
        source_inv = ContainerStorage(capacity=10)
        source_inv.add_countable(CountableType.COIN, 100)

        actor_inv = CharacterInventory(make_stats())

        # Create mock intent
        intent = MagicMock()
        intent.actor.inventory = actor_inv
        intent.source = source_inv
        intent.countable_type = CountableType.COIN
        intent.amount = 30

        executor = PickupCountableExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert result.succeeded is True
        assert source_inv.get_countable(CountableType.COIN) == 70
        assert actor_inv.get_countable(CountableType.COIN) == 30

    def test_pickup_negative_one_takes_all(self) -> None:
        """Amount of -1 should take all countables from source."""
        from unittest.mock import MagicMock

        from catley.game.actions.executors.misc import PickupCountableExecutor

        source_inv = ContainerStorage(capacity=10)
        source_inv.add_countable(CountableType.COIN, 75)

        actor_inv = CharacterInventory(make_stats())

        intent = MagicMock()
        intent.actor.inventory = actor_inv
        intent.source = source_inv
        intent.countable_type = CountableType.COIN
        intent.amount = -1  # Take all

        executor = PickupCountableExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert result.succeeded is True
        assert source_inv.get_countable(CountableType.COIN) == 0
        assert actor_inv.get_countable(CountableType.COIN) == 75

    def test_pickup_fails_when_actor_has_no_inventory(self) -> None:
        """Executor should fail if actor has no inventory."""
        from unittest.mock import MagicMock

        from catley.game.actions.executors.misc import PickupCountableExecutor

        intent = MagicMock()
        intent.actor.inventory = None
        intent.source = ContainerStorage(capacity=10)
        intent.countable_type = CountableType.COIN
        intent.amount = 10

        executor = PickupCountableExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert result.succeeded is False

    def test_pickup_fails_when_source_empty(self) -> None:
        """Executor should fail if source has no countables of that type."""
        from unittest.mock import MagicMock

        from catley.game.actions.executors.misc import PickupCountableExecutor

        source_inv = ContainerStorage(capacity=10)  # No coins
        actor_inv = CharacterInventory(make_stats())

        intent = MagicMock()
        intent.actor.inventory = actor_inv
        intent.source = source_inv
        intent.countable_type = CountableType.COIN
        intent.amount = -1  # Take all (but there are none)

        executor = PickupCountableExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert result.succeeded is False

    def test_pickup_fails_when_insufficient_amount(self) -> None:
        """Executor should fail if source has insufficient countables."""
        from unittest.mock import MagicMock

        from catley.game.actions.executors.misc import PickupCountableExecutor

        source_inv = ContainerStorage(capacity=10)
        source_inv.add_countable(CountableType.COIN, 20)

        actor_inv = CharacterInventory(make_stats())

        intent = MagicMock()
        intent.actor.inventory = actor_inv
        intent.source = source_inv
        intent.countable_type = CountableType.COIN
        intent.amount = 50  # More than available

        executor = PickupCountableExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert result.succeeded is False
        # Source should be unchanged
        assert source_inv.get_countable(CountableType.COIN) == 20


class TestDropCountableExecutor:
    """Tests for DropCountableExecutor."""

    def test_drop_removes_from_inventory_and_spawns(self) -> None:
        """Executor should remove from inventory and spawn on ground."""
        from unittest.mock import MagicMock

        from catley.game.actions.executors.misc import DropCountableExecutor

        actor_inv = CharacterInventory(make_stats())
        actor_inv.add_countable(CountableType.COIN, 100)

        # Track what was spawned
        spawned_countables: list[tuple[tuple[int, int], CountableType, int]] = []

        def mock_spawn(pos: tuple[int, int], ct: CountableType, amt: int) -> MagicMock:
            spawned_countables.append((pos, ct, amt))
            return MagicMock()

        intent = MagicMock()
        intent.actor.inventory = actor_inv
        intent.actor.x = 5
        intent.actor.y = 10
        intent.countable_type = CountableType.COIN
        intent.amount = 30
        intent.controller.gw.item_spawner.spawn_ground_countable = mock_spawn

        executor = DropCountableExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert result.succeeded is True
        assert actor_inv.get_countable(CountableType.COIN) == 70
        assert len(spawned_countables) == 1
        assert spawned_countables[0] == ((5, 10), CountableType.COIN, 30)

    def test_drop_fails_when_actor_has_no_inventory(self) -> None:
        """Executor should fail if actor has no inventory."""
        from unittest.mock import MagicMock

        from catley.game.actions.executors.misc import DropCountableExecutor

        intent = MagicMock()
        intent.actor.inventory = None
        intent.countable_type = CountableType.COIN
        intent.amount = 10

        executor = DropCountableExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert result.succeeded is False

    def test_drop_fails_when_insufficient_countables(self) -> None:
        """Executor should fail if inventory has insufficient countables."""
        from unittest.mock import MagicMock

        from catley.game.actions.executors.misc import DropCountableExecutor

        actor_inv = CharacterInventory(make_stats())
        actor_inv.add_countable(CountableType.COIN, 20)

        intent = MagicMock()
        intent.actor.inventory = actor_inv
        intent.countable_type = CountableType.COIN
        intent.amount = 50  # More than available

        executor = DropCountableExecutor()
        result = executor.execute(intent)

        assert result is not None
        assert result.succeeded is False
        # Inventory should be unchanged
        assert actor_inv.get_countable(CountableType.COIN) == 20


# =============================================================================
# ITEM SPAWNER TESTS
# =============================================================================


class TestSpawnGroundCountable:
    """Tests for ItemSpawner.spawn_ground_countable()."""

    def test_spawn_creates_new_pile_when_no_existing_pile(self) -> None:
        """Should create a new ItemPile when no pile exists at location."""
        from unittest.mock import MagicMock

        from catley.game.item_spawner import ItemSpawner

        # Mock game world with no actor at location
        mock_gw = MagicMock()
        mock_gw.actor_spatial_index.get_at_point.return_value = []

        added_actors: list[ItemPile] = []
        mock_gw.add_actor = lambda a: added_actors.append(a)

        spawner = ItemSpawner(mock_gw)
        result = spawner.spawn_ground_countable((5, 10), CountableType.COIN, 50)

        assert isinstance(result, ItemPile)
        assert result.x == 5
        assert result.y == 10
        assert result.inventory.get_countable(CountableType.COIN) == 50
        assert len(added_actors) == 1
        assert added_actors[0] is result

    def test_spawn_consolidates_with_existing_pile(self) -> None:
        """Should add countables to existing ItemPile at location."""
        from unittest.mock import MagicMock

        from catley.game.item_spawner import ItemSpawner

        # Create an existing pile with some coins
        existing_pile = ItemPile(x=5, y=10)
        existing_pile.inventory.add_countable(CountableType.COIN, 30)

        mock_gw = MagicMock()
        # Mock spatial index to return existing pile
        mock_gw.actor_spatial_index.get_at_point.return_value = [existing_pile]

        spawner = ItemSpawner(mock_gw)
        result = spawner.spawn_ground_countable((5, 10), CountableType.COIN, 20)

        assert result is existing_pile
        assert existing_pile.inventory.get_countable(CountableType.COIN) == 50
        # add_actor should NOT be called since we used existing pile
        mock_gw.add_actor.assert_not_called()

    def test_spawn_creates_new_pile_when_non_pile_actor_exists(self) -> None:
        """Should create new pile when existing actor is not an ItemPile."""
        from unittest.mock import MagicMock

        from catley.game.item_spawner import ItemSpawner

        # Mock a non-ItemPile actor at location (e.g., a character)
        other_actor = MagicMock()
        other_actor.__class__ = type("Character", (), {})  # Not an ItemPile

        mock_gw = MagicMock()
        # Spatial index returns non-pile actor
        mock_gw.actor_spatial_index.get_at_point.return_value = [other_actor]

        added_actors: list[ItemPile] = []
        mock_gw.add_actor = lambda a: added_actors.append(a)

        spawner = ItemSpawner(mock_gw)
        result = spawner.spawn_ground_countable((5, 10), CountableType.COIN, 25)

        assert isinstance(result, ItemPile)
        assert result.inventory.get_countable(CountableType.COIN) == 25
        assert len(added_actors) == 1

    def test_spawn_pile_has_correct_appearance(self) -> None:
        """New pile should have correct glyph and name."""
        from unittest.mock import MagicMock

        from catley.game.item_spawner import ItemSpawner

        mock_gw = MagicMock()
        mock_gw.actor_spatial_index.get_at_point.return_value = []
        mock_gw.add_actor = MagicMock()

        spawner = ItemSpawner(mock_gw)
        result = spawner.spawn_ground_countable((5, 10), CountableType.COIN, 50)

        assert result.ch == "%"
        assert result.name == "Countables"
