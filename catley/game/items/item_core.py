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
from catley.game.items.properties import (
    StatusProperty,
    TacticalProperty,
    WeaponProperty,
)

AttackHandler = MeleeAttack | RangedAttack | AreaEffect


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

    # Whether this item can exist as a physical object in the world.
    # Unarmed attack placeholders like Fists should not be materialized
    # into an Actor that can be picked up.
    can_materialize: bool = True

    def create(self) -> "Item":
        """Create a new Item instance of this type."""
        return Item(self)


class Item:
    """An actual item instance in the world/inventory, holding component objects."""

    def __init__(self, item_type: ItemType) -> None:
        """Create a new item instance from a type definition."""
        self.item_type = item_type

        # Convenience flag for quick checks without reaching into the type
        self.can_materialize: bool = item_type.can_materialize

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

    def get_preferred_attack_mode(self, distance: int) -> AttackHandler | None:
        """
        Get the attack mode this item prefers at the given distance.

        Considers item design intent and situational factors to determine
        which attack mode (melee, ranged, area) is most appropriate.

        Args:
            distance: Distance to target in tiles

        Returns:
            The preferred attack handler, or None if no attacks available
        """
        available_attacks = []

        # Collect available attacks for this distance
        if self.melee_attack and distance == 1:
            available_attacks.append(self.melee_attack)
        # Ranged attacks are ignored at point blank range to avoid
        # unintentionally pistol-whipping instead of shooting.
        if self.ranged_attack and distance > 1:
            available_attacks.append(self.ranged_attack)
        if self.area_effect:  # Area effects work at various ranges
            available_attacks.append(self.area_effect)

        if not available_attacks:
            return None

        # First priority: attacks marked as PREFERRED
        preferred_attacks = [
            attack
            for attack in available_attacks
            if WeaponProperty.PREFERRED in attack.properties
        ]
        if preferred_attacks:
            return preferred_attacks[0]

        # Second priority: non-improvised attacks
        designed_attacks = [
            attack
            for attack in available_attacks
            if WeaponProperty.IMPROVISED not in attack.properties
        ]
        if designed_attacks:
            return designed_attacks[0]

        # Fall back to any available attack
        return available_attacks[0]

    def is_improvised_weapon(self) -> bool:
        """
        Check if this item is being used as an improvised weapon.

        An item is considered improvised if any of its attack capabilities
        are marked with the IMPROVISED property.

        Returns:
            True if any attack mode is improvised
        """
        attack_handlers = [
            handler
            for handler in [self.melee_attack, self.ranged_attack, self.area_effect]
            if handler is not None
        ]

        return any(
            WeaponProperty.IMPROVISED in handler.properties
            for handler in attack_handlers
        )

    def is_designed_weapon(self) -> bool:
        """
        Check if this item is a purpose-built weapon.

        An item is considered a designed weapon if it has at least one
        attack capability that is NOT marked as improvised.

        Returns:
            True if any attack mode is purpose-built
        """
        attack_handlers = [
            handler
            for handler in [self.melee_attack, self.ranged_attack, self.area_effect]
            if handler is not None
        ]

        return any(
            WeaponProperty.IMPROVISED not in handler.properties
            for handler in attack_handlers
        )

    def get_weapon_properties(self) -> set[WeaponProperty | TacticalProperty]:
        """Return weapon and tactical properties across all attack modes."""
        all_properties: set[WeaponProperty | TacticalProperty] = set()

        for attack_handler in [self.melee_attack, self.ranged_attack, self.area_effect]:
            if attack_handler:
                relevant_props = {
                    prop
                    for prop in attack_handler.properties
                    if isinstance(prop, WeaponProperty | TacticalProperty)
                }
                all_properties.update(relevant_props)

        return all_properties

    def get_status_effects(self) -> set[StatusProperty]:
        """
        Get all status effect properties from all attack modes.

        Returns:
            Set of all StatusProperty enums found across all attack modes
        """
        all_properties = set()

        for attack_handler in [self.melee_attack, self.ranged_attack, self.area_effect]:
            if attack_handler:
                status_props = {
                    prop
                    for prop in attack_handler.properties
                    if isinstance(prop, StatusProperty)
                }
                all_properties.update(status_props)

        return all_properties

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
