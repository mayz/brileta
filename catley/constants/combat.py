"""Constants for combat calculations and mechanics."""


class CombatConstants:
    """Constants for combat calculations and mechanics."""

    # Area effects
    CONE_SPREAD_COSINE = 0.707  # ~45 degrees spread

    # AI behavior
    DEFAULT_AGGRO_RADIUS = 8

    # Screen shake - melee attacks (amplitude in tiles)
    MELEE_SHAKE_INTENSITY_MULT = 0.125  # Amplitude per point of damage
    MELEE_SHAKE_INTENSITY_CAP = 0.75  # Maximum amplitude in tiles
    MELEE_SHAKE_DURATION_BASE = 0.25  # Base duration in seconds
    MELEE_SHAKE_DURATION_MULT = 0.05  # Additional duration per damage

    # Screen shake - ranged attacks (amplitude in tiles)
    RANGED_SHAKE_INTENSITY_MULT = 0.1  # Lower amplitude than melee
    RANGED_SHAKE_INTENSITY_CAP = 0.625
    RANGED_SHAKE_DURATION_BASE = 0.2
    RANGED_SHAKE_DURATION_MULT = 0.0375

    # Extra intensity for critical hits
    CRIT_SHAKE_INTENSITY_MULT = 1.5  # Multiplier for crit amplitude
    CRIT_SHAKE_DURATION_MULT = 1.3
