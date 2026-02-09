"""Tests for junk item type definitions."""

from brileta.game.enums import ItemCategory
from brileta.game.items.junk_item_types import JUNK_ITEM_TYPES, get_random_junk_type


def test_junk_item_count() -> None:
    """Ensure we have the expected number of junk items."""
    assert len(JUNK_ITEM_TYPES) == 100


def test_all_junk_items_have_junk_category() -> None:
    """All junk items should be tagged with ItemCategory.JUNK."""
    for item_type in JUNK_ITEM_TYPES:
        assert item_type.category == ItemCategory.JUNK, (
            f"{item_type.name} has wrong category"
        )


def test_all_junk_items_have_names_and_descriptions() -> None:
    """All junk items should have non-empty names and descriptions."""
    for item_type in JUNK_ITEM_TYPES:
        assert item_type.name, "Junk item has empty name"
        assert item_type.description, f"{item_type.name} has empty description"


def test_junk_item_names_are_unique() -> None:
    """All junk items should have unique names."""
    names = [item_type.name for item_type in JUNK_ITEM_TYPES]
    assert len(names) == len(set(names)), "Duplicate junk item names found"


def test_get_random_junk_type_returns_valid_item() -> None:
    """get_random_junk_type should return an item from the list."""
    for _ in range(10):  # Test multiple times due to randomness
        junk = get_random_junk_type()
        assert junk in JUNK_ITEM_TYPES
