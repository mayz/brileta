from dataclasses import dataclass
from typing import cast

from catley.game.enums import ItemSize
from catley.game.items.capabilities import (
    Ammo,
    AmmoSpec,
    AreaEffect,
    AreaEffectSpec,
    ConsumableEffect,
    ConsumableEffectSpec,
    MeleeAttack,
    MeleeAttackSpec,
    RangedAttack,
    RangedAttackSpec,
)


@dataclass
class ItemType:
    """Defines what kind of item this is - shared template holding specs."""

    name: str
    description: str
    size: ItemSize

    # References to spec objects
    melee_attack: MeleeAttackSpec | None = None
    ranged_attack: RangedAttackSpec | None = None
    area_effect: AreaEffectSpec | None = None
    consumable_effect: ConsumableEffectSpec | None = None
    ammo: AmmoSpec | None = None

    def create(self) -> "Item":
        """Create a new Item instance of this type."""
        return Item(self)


class Item:
    """An actual item instance in the world/inventory, holding component objects."""

    def __init__(self, item_type: ItemType) -> None:
        """Create a new item instance from a type definition."""
        self.item_type = item_type

        self.melee_attack: MeleeAttack | None = None
        if item_type.melee_attack:
            spec = cast("MeleeAttackSpec", item_type.melee_attack)
            self.melee_attack = MeleeAttack(spec=spec)

        self.ranged_attack: RangedAttack | None = None
        if item_type.ranged_attack:
            spec = cast("RangedAttackSpec", item_type.ranged_attack)
            self.ranged_attack = RangedAttack(spec=spec)

        self.area_effect: AreaEffect | None = None
        if item_type.area_effect:
            spec = cast("AreaEffectSpec", item_type.area_effect)
            self.area_effect = AreaEffect(spec=spec)

        self.consumable_effect: ConsumableEffect | None = None
        if item_type.consumable_effect:
            spec = cast("ConsumableEffectSpec", item_type.consumable_effect)
            self.consumable_effect = ConsumableEffect(spec=spec)

        self.ammo: Ammo | None = None
        if item_type.ammo:
            spec = cast("AmmoSpec", item_type.ammo)
            self.ammo = Ammo(spec=spec)

    # Convenience properties
    @property
    def name(self) -> str:
        return self.item_type.name

    @property
    def description(self) -> str:
        return self.item_type.description

    @property
    def size(self) -> ItemSize:
        return self.item_type.size

    @property
    def equippable(self) -> bool:
        """Return True if this item can be equipped (has any attack capabilities)."""
        return self.melee_attack is not None or self.ranged_attack is not None
