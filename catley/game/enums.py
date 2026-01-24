from enum import Enum, auto


class ItemSize(Enum):
    """Size categories for items, used for inventory management."""

    TINY = auto()
    NORMAL = auto()
    BIG = auto()
    HUGE = auto()


class ItemCategory(Enum):
    """Item categories for UI display and game mechanics.

    Used for inventory display and for mechanics that care about item type
    (e.g., repair uses junk).
    """

    WEAPON = auto()  # Melee and ranged weapons
    OUTFIT = auto()  # Clothing and protective equipment
    CONSUMABLE = auto()  # Food, medicine, chems
    JUNK = auto()  # Scavenged parts for repair/crafting
    MUNITIONS = auto()  # Ammo, magazines, batteries
    MISC = auto()  # Uncategorized items


class CreatureSize(Enum):
    """Creature size categories.

    Size affects animation parameters and may influence combat, movement,
    and detection mechanics.
    """

    TINY = auto()  # Insects, rats
    SMALL = auto()  # Dogs, children
    MEDIUM = auto()  # Humans - baseline
    LARGE = auto()  # Bears, mutants
    HUGE = auto()  # Giants, vehicles


class Disposition(Enum):
    HOSTILE = auto()  # Will attack/flee.
    UNFRIENDLY = auto()
    WARY = auto()
    APPROACHABLE = auto()
    FRIENDLY = auto()
    ALLY = auto()


class OutcomeTier(Enum):
    """Represents the broad outcome tiers for action resolution.

    This 5-tier system provides granular results for all game actions,
    enabling varied and interesting consequences beyond simple pass/fail.

    From best to worst outcome:
    - CRITICAL_SUCCESS: Exceptional success with additional benefits
    - SUCCESS: Full successful outcome as intended
    - PARTIAL_SUCCESS: Limited success with complications or reduced effectiveness
    - FAILURE: Standard unsuccessful outcome
    - CRITICAL_FAILURE: Catastrophic failure with additional negative consequences

    This system allows actions to have nuanced outcomes
    while remaining universal across all resolution mechanics.
    """

    CRITICAL_FAILURE = auto()
    FAILURE = auto()
    PARTIAL_SUCCESS = auto()
    SUCCESS = auto()
    CRITICAL_SUCCESS = auto()


class AreaType(Enum):
    CIRCLE = "circle"
    LINE = "line"
    CONE = "cone"
    CROSS = "cross"


class BlendMode(Enum):
    """Background blending behavior for particles."""

    # Integer values are stored directly in NumPy arrays for efficiency.
    TINT = 0
    OVERLAY = 1
    REPLACE = 2


class ConsumableEffectType(Enum):
    HEAL = "heal"  # Heals HP: specific amount if effect_value set, full restore if None
    BOOST_STRENGTH = "boost_strength"
    POISON = "poison"
    CURE_ALL_CONDITIONS = "cure_all_conditions"
    CURE_EXHAUSTION = "cure_exhaustion"
    CURE_INJURIES = "cure_injuries"


class InjuryLocation(Enum):
    """Possible body locations that can be injured."""

    HEAD = auto()
    TORSO = auto()
    LEFT_ARM = auto()
    RIGHT_ARM = auto()
    LEFT_LEG = auto()
    RIGHT_LEG = auto()


class ImpactMaterial(Enum):
    """Material types for impact sound selection.

    Used by the audio system to play appropriate impact sounds when
    projectiles or melee attacks hit actors or environmental surfaces.
    """

    FLESH = auto()  # Biological targets (humans, animals, mutants)
    METAL = auto()  # Metallic armor, robots, metal surfaces
    STONE = auto()  # Walls, boulders, concrete, rocky ground
    WOOD = auto()  # Doors, wooden structures, furniture
