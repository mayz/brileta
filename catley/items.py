"""Item and equipment classes for the game."""

from __future__ import annotations

from enum import Enum, auto


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
        skill_used: str,
        properties: list[Property] | None = None,
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
        self.damage_die = damage_die
        self.skill_used = skill_used
        self.properties = properties or []

    def __str__(self) -> str:
        properties = ", ".join(self.properties) if self.properties else "none"
        return (
            f"{self.name} ({self.damage_die}, "
            f"uses {self.skill_used}, "
            f"properties: {properties})"
        )


# Common weapon instances
FISTS = Weapon(
    name="Fists",
    damage_die="d4",
    properties=["unarmed"],
    skill_used="strength",
    description=("Your bare hands. Better than nothing, but not by much."),
)

LEAD_PIPE = Weapon(
    name="Lead Pipe",
    damage_die="d6",
    properties=["awkward"],
    skill_used="strength",
    description=(
        "A heavy, unwieldy pipe. It's seen better days but still gets the job done."
    ),
)
