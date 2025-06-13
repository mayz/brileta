"""
Property enums for item capabilities.

Defines standardized properties that can be applied to attack specs
and other item components to modify their behavior.
"""

from enum import Enum, auto


class WeaponProperty(Enum):
    """Properties that affect weapon behavior and UI display."""

    PREFERRED = auto()  # Default attack mode for this weapon
    IMPROVISED = auto()  # Not designed as weapon, might have penalties
    THROWN = auto()  # Single-use ranged attack (consumes item)
    SCOPED = auto()  # Advantage at long range instead of disadvantage
    AUTOMATIC = auto()  # Can spray multiple shots
    CONTINUOUS = auto()  # Sustained stream rather than discrete shots
    ARMOR_PIERCING = auto()  # Ignores armor points
    AWKWARD = auto()  # Causes off-balance effect on miss
    SILENT = auto()  # Doesn't alert nearby enemies
    TWO_HANDED = auto()  # Requires both hands to use effectively
    UNARMED = auto()  # Bare hands, fists, natural weapons


class StatusProperty(Enum):
    """Properties that apply status effects to targets."""

    BLINDING = auto()  # Temporary vision loss
    POISONING = auto()  # Damage over time from toxins
    BURNING = auto()  # Fire damage over time
    STUNNING = auto()  # Skip next turn from shock/impact
    BLEEDING = auto()  # Gradual health loss from wounds


class TacticalProperty(Enum):
    """Properties affecting tactical considerations and environment."""

    CHEMICAL = auto()  # Affected by gas masks, ventilation
    EXPLOSIVE = auto()  # Triggers on impact/timer
    SMOKE = auto()  # Creates concealment cloud
    FIRE = auto()  # Spreads to flammable objects
    SPRAY = auto()  # Area effect for automatic weapons
    RADIATION = auto()  # Causes radiation damage


# Type alias for convenience
ItemProperty = WeaponProperty | StatusProperty | TacticalProperty
