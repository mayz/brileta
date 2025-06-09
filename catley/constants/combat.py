"""Constants for combat calculations and mechanics."""


class CombatConstants:
    """Constants for combat calculations and mechanics."""

    # Area effects
    CONE_SPREAD_COSINE = 0.707  # ~45 degrees spread

    # AI behavior
    DEFAULT_AGGRO_RADIUS = 8

    # Screen shake - melee attacks
    MELEE_SHAKE_INTENSITY_MULT = 0.15  # 0.15-0.8 probability per damage
    MELEE_SHAKE_INTENSITY_CAP = 0.8  # Maximum shake probability
    MELEE_SHAKE_DURATION_BASE = 0.2  # Base duration in seconds
    MELEE_SHAKE_DURATION_MULT = 0.03  # Additional duration per damage

    # Screen shake - ranged attacks
    RANGED_SHAKE_INTENSITY_MULT = 0.08  # 0.08-0.5 probability per damage
    RANGED_SHAKE_INTENSITY_CAP = 0.5
    RANGED_SHAKE_DURATION_BASE = 0.1
    RANGED_SHAKE_DURATION_MULT = 0.02

    # Extra intensity for critical hits
    CRIT_SHAKE_INTENSITY_MULT = 1.5  # Higher probability for crits
    CRIT_SHAKE_DURATION_MULT = 1.3
