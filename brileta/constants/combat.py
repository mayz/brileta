"""Constants for combat calculations and mechanics."""


class CombatConstants:
    """Constants for combat calculations and mechanics."""

    # --- d20 Resolution ---
    # Difficulty class base: roll_to_exceed = defender_stat + D20_DC_BASE
    D20_DC_BASE = 10

    # --- Action presentation durations (milliseconds) ---
    # How long the game pauses after each action type to let the
    # player read the outcome before the next action fires.
    MELEE_DURATION_MS = 300
    RANGED_DURATION_MS = 350
    STUNT_DURATION_MS = 400
    RELOAD_DURATION_MS = 300
    HOLSTER_DURATION_MS = 300
    PUNCH_DURATION_MS = 350

    # --- Damage-to-VFX intensity ---
    # Divides raw damage to produce a 0..~1 intensity for blood splatter, etc.
    # Typical damage 1-20 becomes intensity 0.05-1.0.
    DAMAGE_INTENSITY_DIVISOR = 20.0

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
