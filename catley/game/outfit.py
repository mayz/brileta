"""
Outfit system for protection and armor points.

This module implements the Protection + Armor Points model where:
- **Protection (PR)** reduces incoming damage by a flat amount
- **Armor Points (AP)** tracks how many hits have penetrated the armor

Outfit is an **inventory item** (clothing/armor), not an abstract stat. All outfit
logic lives in this module for future expansion (glyphs, factions, appearance).

Core mechanics:
    1. Armor reduces damage by Protection. Remainder goes to HP.
    2. If damage exceeds Protection (hit penetrates), armor loses 1 AP.
    3. At 0 AP, armor provides no protection (broken state).
    4. Rest recovers 1 AP if armor was damaged since last rest (not if broken).
    5. Broken armor (0 AP) requires repair to restore.

Why AP tracks penetrations, not damage:
    AP represents structural integrity - how many times the armor can be breached
    before it fails. A bullet that barely punches through does the same structural
    damage to the armor as one that punches through with force to spare. The extra
    force goes into the wearer (more HP damage), not into further degrading the
    armor. This means heavy hits are dangerous because they hurt YOU more, not
    because they break your armor faster.

Example - Leather Armor (PR: 2, Max AP: 4):
    | Attack Damage | After Protection | To Player | AP Change |
    |---------------|------------------|-----------|-----------|
    | 1 (blocked)   | 0                | 0 HP      | None      |
    | 3 (penetrates)| 1                | 1 HP      | -1 AP     |
    | 6 (penetrates)| 4                | 4 HP      | -1 AP     |

After 4 penetrating hits, armor breaks and provides 0 protection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from catley.game.enums import ItemCategory, ItemSize
from catley.game.items.capabilities import OutfitSpec
from catley.game.items.item_core import ItemType


class OutfitCapability:
    """Runtime capability for an outfit item.

    This is the mutable instance attached to an Item, tracking current
    AP and damage state. Similar to how RangedAttack tracks current_ammo.

    Attributes:
        protection: Flat damage reduction (PR) from the spec.
        max_ap: Maximum armor points.
        ap: Current armor points (changes during gameplay).
        damaged_since_rest: Flag for rest recovery - True if hit since last rest.
    """

    def __init__(self, spec: OutfitSpec) -> None:
        """Create an outfit capability from a spec.

        Args:
            spec: The outfit specification defining protection and max AP.
        """
        self.spec = spec
        self.protection = spec.protection
        self.max_ap = spec.max_ap
        self.ap = spec.max_ap
        self.damaged_since_rest = False

    @property
    def is_broken(self) -> bool:
        """Return True if the outfit is broken (has protection but 0 AP)."""
        return self.max_ap > 0 and self.ap <= 0

    @property
    def has_protection(self) -> bool:
        """Return True if the outfit provides any protection."""
        return self.protection > 0

    @property
    def effective_protection(self) -> int:
        """Return the current effective protection (0 if broken)."""
        if self.is_broken:
            return 0
        return self.protection


@dataclass
class DamageResult:
    """Result of applying damage through an outfit.

    Attributes:
        damage_to_hp: Amount of damage that reaches the character's HP.
        penetrated: True if the attack penetrated (damage exceeded protection).
        blocked: True if all damage was absorbed by armor.
    """

    damage_to_hp: int
    penetrated: bool
    blocked: bool = field(init=False)

    def __post_init__(self) -> None:
        self.blocked = self.damage_to_hp == 0 and not self.penetrated


def calculate_damage(raw_damage: int, outfit: OutfitCapability | None) -> DamageResult:
    """Calculate damage after outfit protection.

    This is the core damage calculation for the outfit system. It applies
    the outfit's protection value to reduce incoming damage.

    Args:
        raw_damage: The incoming damage amount before armor reduction.
        outfit: The outfit capability, or None if no outfit equipped.

    Returns:
        DamageResult with damage to HP and whether the attack penetrated.

    Examples:
        >>> # No outfit - full damage
        >>> result = calculate_damage(5, None)
        >>> result.damage_to_hp
        5
        >>> result.penetrated
        True

        >>> # Outfit blocks all damage
        >>> spec = OutfitSpec(protection=3, max_ap=4)
        >>> outfit = OutfitCapability(spec)
        >>> result = calculate_damage(2, outfit)
        >>> result.damage_to_hp
        0
        >>> result.penetrated
        False

        >>> # Damage exceeds protection
        >>> result = calculate_damage(5, outfit)
        >>> result.damage_to_hp
        2
        >>> result.penetrated
        True
    """
    if outfit is None:
        # No outfit - all damage goes to HP
        return DamageResult(damage_to_hp=raw_damage, penetrated=True)

    effective_pr = outfit.effective_protection

    if raw_damage <= effective_pr:
        # Armor blocks all damage
        return DamageResult(damage_to_hp=0, penetrated=False)

    # Damage exceeds protection - remainder goes to HP
    damage_to_hp = raw_damage - effective_pr
    return DamageResult(damage_to_hp=damage_to_hp, penetrated=True)


def apply_hit(outfit: OutfitCapability, penetrated: bool) -> bool:
    """Apply hit effects to an outfit (AP loss if penetrated).

    Call this after damage calculation to update the outfit's AP.
    Only reduces AP if the hit penetrated AND the outfit has AP to track.

    Args:
        outfit: The outfit capability to update.
        penetrated: Whether the attack penetrated the armor.

    Returns:
        True if AP was reduced, False otherwise.
    """
    if not penetrated:
        return False

    if outfit.max_ap <= 0:
        # No AP to track (e.g., regular clothes)
        return False

    if outfit.ap <= 0:
        # Already broken
        return False

    outfit.ap -= 1
    outfit.damaged_since_rest = True
    return True


def rest_recovery(outfit: OutfitCapability) -> bool:
    """Attempt to recover 1 AP during rest.

    Recovery only happens if:
    - Outfit has AP to track (max_ap > 0)
    - Outfit was damaged since last rest
    - Outfit is not broken (ap > 0)

    Broken armor (0 AP) cannot be recovered through rest - it requires repair.

    Args:
        outfit: The outfit capability to recover.

    Returns:
        True if AP was recovered, False otherwise.
    """
    if outfit.max_ap <= 0:
        # No AP to track
        return False

    if not outfit.damaged_since_rest:
        # No damage to recover
        return False

    if outfit.ap <= 0:
        # Broken - requires repair, not rest
        outfit.damaged_since_rest = False
        return False

    if outfit.ap >= outfit.max_ap:
        # Already at max
        outfit.damaged_since_rest = False
        return False

    outfit.ap += 1
    outfit.damaged_since_rest = False
    return True


def repair(outfit: OutfitCapability, amount: int | None = None) -> int:
    """Repair an outfit, restoring AP.

    This is used for repairing broken armor (or topping up damaged armor)
    using resources like junk + tools or a tinker service.

    Args:
        outfit: The outfit capability to repair.
        amount: Amount of AP to restore. If None, fully repairs.

    Returns:
        The amount of AP actually restored.
    """
    if outfit.max_ap <= 0:
        # No AP to repair
        return 0

    old_ap = outfit.ap

    if amount is None:
        # Full repair
        outfit.ap = outfit.max_ap
    else:
        outfit.ap = min(outfit.max_ap, outfit.ap + amount)

    # Clear damaged flag after repair
    if outfit.ap > 0:
        outfit.damaged_since_rest = False

    return outfit.ap - old_ap


# =============================================================================
# Outfit Item Types
# =============================================================================

RAGS_TYPE = ItemType(
    name="Rags",
    description="Tattered scraps of cloth. Offers no real protection.",
    size=ItemSize.NORMAL,
    category=ItemCategory.ARMOR,
    outfit=OutfitSpec(protection=0, max_ap=0),
)

SHELTER_JUMPSUIT_TYPE = ItemType(
    name="Shelter Jumpsuit",
    description="Standard-issue vault dweller attire. Comfortable but not protective.",
    size=ItemSize.NORMAL,
    category=ItemCategory.ARMOR,
    outfit=OutfitSpec(protection=0, max_ap=0),
)

LEATHER_ARMOR_TYPE = ItemType(
    name="Leather Armor",
    description=(
        "Hardened leather plates. Basic but reliable protection. "
        "Covers the torso and shoulders while allowing freedom of movement."
    ),
    size=ItemSize.NORMAL,
    category=ItemCategory.ARMOR,
    outfit=OutfitSpec(protection=2, max_ap=4),
)

SCRAP_ARMOR_TYPE = ItemType(
    name="Scrap Armor",
    description="Cobbled together from salvaged metal. Decent protection.",
    size=ItemSize.BIG,
    category=ItemCategory.ARMOR,
    outfit=OutfitSpec(protection=4, max_ap=6),
)

COMBAT_ARMOR_TYPE = ItemType(
    name="Combat Armor",
    description="Pre-war military grade protection. Hard to find, harder to break.",
    size=ItemSize.BIG,
    category=ItemCategory.ARMOR,
    outfit=OutfitSpec(protection=6, max_ap=8),
)

POWER_ARMOR_TYPE = ItemType(
    name="Power Armor",
    description="Powered exoskeleton with armor plating. The best protection.",
    size=ItemSize.HUGE,
    category=ItemCategory.ARMOR,
    outfit=OutfitSpec(protection=8, max_ap=12),
)


def get_outfit_spec(item_type: ItemType) -> OutfitSpec | None:
    """Get the outfit spec for an item type, if it's an outfit.

    Args:
        item_type: The item type to check.

    Returns:
        The OutfitSpec if this is an outfit type, None otherwise.
    """
    return item_type.outfit


def is_outfit_type(item_type: ItemType) -> bool:
    """Check if an item type is an outfit.

    Args:
        item_type: The item type to check.

    Returns:
        True if this is an outfit type.
    """
    return item_type.outfit is not None
