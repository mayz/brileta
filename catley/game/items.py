"""Item and equipment classes for the game."""

from __future__ import annotations

import copy
from enum import Enum, auto

from catley.util import dice


class ItemSize(Enum):
    """Size categories for items, used for inventory management."""

    TINY = auto()
    NORMAL = auto()
    BIG = auto()
    HUGE = auto()


# Type alias for weapon properties
type Property = str


class Item:
    """Base class for all items in the game."""

    def __init__(
        self,
        name: str,
        description: str = "",
        size: ItemSize = ItemSize.NORMAL,
        equippable: bool = False,
    ) -> None:
        """Initialize an item.

        Args:
            name: The name of the item
            description: Optional description of the item
            size: Physical size category of the item
            equippable: Whether the item can be equipped
        """
        self.name = name
        self.description = description
        self.size = size
        self.equippable = equippable

    def __str__(self) -> str:
        return self.name


class Weapon(Item):
    """A weapon that can be used in combat."""

    def __init__(
        self,
        name: str,
        damage_die: str,
        melee: bool,
        properties: set[Property] | None = None,
        description: str = "",
        size: ItemSize = ItemSize.NORMAL,
    ) -> None:
        """Initialize a weapon.

        Args:
            name: The name of the weapon
            damage_die: The damage die (e.g., "d6", "d8")
            skill_used: The ability score used for attack rolls
                        (e.g., "strength" for melee, "observation" for ranged attack)
            properties: List of weapon properties (e.g., ["awkward", "big", "up-close"])
            description: Optional description of the weapon
            size: Physical size category of the weapon
        """
        super().__init__(
            name=name,
            description=description,
            size=size,
            equippable=True,  # Weapons are always equippable
        )
        self.damage_dice = dice.Dice(damage_die)
        self.melee = melee
        self.properties = properties or {}

    def clone(self) -> Weapon:
        # Shallow copy for most attributes.
        cloned = copy.copy(self)
        # Ensure 'properties' is a new set to avoid shared mutable state.
        if self.properties is not None:
            cloned.properties = set(self.properties)
        return cloned

    def __str__(self) -> str:
        properties = ", ".join(self.properties) if self.properties else "none"
        return (
            f"{self.name} ({self.damage_dice.dice_str}, "
            f"melee: {self.melee}, "
            f"properties: {properties})"
        )


# Common weapon instances
# FIXME: Need to make sure this doesn't take inventory space.
FISTS = Weapon(
    name="Fists",
    damage_die="d4",
    properties={"unarmed"},
    melee=True,
    description=("Your bare hands. Better than nothing, but not by much."),
)

# Weapon categories.
# Don't use directly - use one of the specific weapons instead.
_JUNK_WEAPON = Weapon(
    name="Junk Weapon",
    damage_die="d6",
    properties={"awkward"},
    melee=True,
)
_SMALL_MELEE_WEAPON = Weapon(
    name="Small Melee Weapon",
    damage_die="d6",
    melee=True,
)
_LARGE_MELEE_WEAPON = Weapon(
    name="Large Melee Weapon",
    damage_die="d8",
    melee=True,
    size=ItemSize.BIG,
)

LEAD_PIPE = _JUNK_WEAPON.clone()
LEAD_PIPE.name = "Lead Pipe"
LEAD_PIPE.description = (
    "A heavy, unwieldy pipe. It's seen better days but still gets the job done."
)

BRASS_KNUCKLES = _SMALL_MELEE_WEAPON.clone()
BRASS_KNUCKLES.name = "Brass Knuckles"
BRASS_KNUCKLES.description = (
    "A pair of brass knuckles. They're not very effective, but they're better "
    "than nothing."
)

DAGGER = _SMALL_MELEE_WEAPON.clone()
DAGGER.name = "Dagger"
DAGGER.description = "A dagger. It's not very effective, but it's better than nothing."

SLEDGEHAMMER = _LARGE_MELEE_WEAPON.clone()
SLEDGEHAMMER.name = "Sledgehammer"
SLEDGEHAMMER.description = (
    "A sledgehammer. It's not very effective, but it's better than nothing."
)

SWORD = _LARGE_MELEE_WEAPON.clone()
SWORD.name = "Sword"
SWORD.description = "A sword. It's not very effective, but it's better than nothing."
