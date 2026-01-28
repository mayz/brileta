"""Countables system for currency and other fungible, countable items.

Countables are stackable, weightless items like coins, bottle caps, screws,
ribbons, etc. Unlike regular items which have individual identity, countables
are fungible - they only track quantity by type.

Design decisions:
- Storage: dict[CountableType, int] on all inventories (weightless)
- Ground: Stored in ItemPile's ContainerStorage
- Default pickup behavior: Take all
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from catley.colors import Color


class CountableType(Enum):
    """Types of countable, fungible items."""

    COIN = auto()
    # Future: BOTTLE_CAP, SCREW, RIBBON, etc.


@dataclass(frozen=True)
class CountableDefinition:
    """Template defining a countable type's display properties."""

    countable_type: CountableType
    name: str
    plural_name: str
    glyph: str
    color: Color


# Registry of all countable type definitions
COUNTABLE_DEFINITIONS: dict[CountableType, CountableDefinition] = {
    CountableType.COIN: CountableDefinition(
        countable_type=CountableType.COIN,
        name="Coin",
        plural_name="Coins",
        glyph="\u25cf",  # solid circle
        color=(255, 215, 0),  # gold
    ),
}


def get_countable_definition(countable_type: CountableType) -> CountableDefinition:
    """Get the definition for a countable type.

    Args:
        countable_type: The type to look up.

    Returns:
        The CountableDefinition for this type.

    Raises:
        KeyError: If the countable type is not registered.
    """
    return COUNTABLE_DEFINITIONS[countable_type]


def get_countable_display_name(countable_type: CountableType, quantity: int) -> str:
    """Return display string like '1 Coin' or '47 Coins'.

    Args:
        countable_type: The type of countable.
        quantity: The quantity to display.

    Returns:
        A formatted string with quantity and appropriate singular/plural name.
    """
    defn = COUNTABLE_DEFINITIONS[countable_type]
    if quantity == 1:
        return f"1 {defn.name}"
    return f"{quantity} {defn.plural_name}"
