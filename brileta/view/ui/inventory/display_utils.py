"""Display utilities for item rendering in inventory menus.

Pure functions for grouping, categorizing, and labeling items for UI display.
Shared by pane classes and the item detail panel.
"""

from __future__ import annotations

from brileta import colors
from brileta.game.enums import ItemCategory, ItemSize
from brileta.game.items.item_core import Item


def group_items_for_display(items: list[Item]) -> list[tuple[Item, int]]:
    """Group stackable (Tiny) items by ItemType for display.

    Returns a list of (representative_item, count) tuples.
    Non-tiny items always have count=1.

    This is a UI-only transformation - items remain individual objects in storage.
    It enables compact display like "Shotgun Shell x 5" while preserving the
    ability to drop/transfer items individually.

    Args:
        items: List of items to group.

    Returns:
        List of (item, count) tuples where item is a representative of its group.
    """
    groups: list[tuple[Item, int]] = []

    for item in items:
        # Only tiny items can stack
        if item.size != ItemSize.TINY:
            groups.append((item, 1))
            continue

        # Check if we can add to existing group (same ItemType)
        found = False
        for i, (rep, count) in enumerate(groups):
            if rep.item_type is item.item_type:
                groups[i] = (rep, count + 1)
                found = True
                break

        if not found:
            groups.append((item, 1))

    return groups


def get_category_prefix(item: Item) -> tuple[str, colors.Color | None]:
    """Return the category prefix symbol and color for an item.

    Returns a tuple of (prefix_string, color) where prefix_string is
    a colored dot indicator. Returns ("", None) for MISC category items.
    """
    category_info: dict[ItemCategory, tuple[str, colors.Color]] = {
        ItemCategory.WEAPON: ("\u25cf ", colors.CATEGORY_WEAPON),
        ItemCategory.OUTFIT: ("\u25cf ", colors.CATEGORY_OUTFIT),
        ItemCategory.CONSUMABLE: ("\u25cf ", colors.CATEGORY_CONSUMABLE),
        ItemCategory.JUNK: ("\u25cf ", colors.CATEGORY_JUNK),
        ItemCategory.MUNITIONS: ("\u25cf ", colors.CATEGORY_MUNITIONS),
    }

    return category_info.get(item.category, ("", None))


def get_category_name(category: ItemCategory) -> str:
    """Return the display name for an item category."""
    names: dict[ItemCategory, str] = {
        ItemCategory.WEAPON: "Weapon",
        ItemCategory.OUTFIT: "Outfit",
        ItemCategory.CONSUMABLE: "Consumable",
        ItemCategory.JUNK: "Junk",
        ItemCategory.MUNITIONS: "Munition",
        ItemCategory.MISC: "Misc",
    }
    return names.get(category, "")
