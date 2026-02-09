"""Constants for visual rendering and UI behavior."""


class ViewConstants:
    """Constants for visual effects and interface layout."""

    # Particle system resolution
    SUB_TILE_SUBDIVISION = 3  # 3x3 sub-tiles per tile
    PARTICLE_SPREAD = 0.5  # +/-0.5 sub-pixels around emission point

    # Radial burst defaults (blood, debris)
    RADIAL_SPEED_MIN = 0.5
    RADIAL_SPEED_MAX = 2.0
    RADIAL_LIFETIME_MIN = 0.3
    RADIAL_LIFETIME_MAX = 0.8

    # Muzzle flash
    MUZZLE_CONE_SPREAD = 0.3  # Fairly tight cone (~17 degrees)
    MUZZLE_SPEED_MIN = 3.0  # Fast-moving flash particles
    MUZZLE_SPEED_MAX = 6.0
    MUZZLE_LIFE_MIN = 0.05  # Very brief flash effect
    MUZZLE_LIFE_MAX = 0.15
    MUZZLE_ORIGIN_OFFSET = 0.4  # Muzzle flash originates slightly forward

    # Explosion debris and flash
    EXPLOSION_FLASH_LIFE_MIN = 0.1  # Very brief flash
    EXPLOSION_FLASH_LIFE_MAX = 0.2
    EXPLOSION_DEBRIS_SPEED_MIN = 1.0  # Medium to fast speeds
    EXPLOSION_DEBRIS_SPEED_MAX = 3.0
    EXPLOSION_DEBRIS_LIFE_MIN = 0.4  # Longer-lasting than flash
    EXPLOSION_DEBRIS_LIFE_MAX = 1.0
    EXPLOSION_DEBRIS_GRAVITY = 2.0  # Debris falls due to gravity

    # Smoke cloud
    SMOKE_DRIFT_MIN = 0.2  # Slow to medium drift
    SMOKE_DRIFT_MAX = 1.0
    SMOKE_LIFE_MIN = 1.0  # Long-lasting smoke
    SMOKE_LIFE_MAX = 2.0
    SMOKE_UPWARD_DRIFT = -0.3  # Negative = upward (smoke rises)

    # Poison gas
    POISON_DRIFT_MIN = 0.1  # Slower spread than smoke
    POISON_DRIFT_MAX = 0.8
    POISON_LIFE_MIN = 2.0  # Very long-lasting gas
    POISON_LIFE_MAX = 4.0
    POISON_UPWARD_DRIFT = 0.0  # Heavy gas doesn't rise like smoke

    # UI constants
    FPS_UPDATE_INTERVAL = 0.5
    EQUIPMENT_MAX_SLOTS = 2
    EQUIPMENT_RELOAD_HINT_OFFSET = 3
    MESSAGE_LOG_MIN_FONT_SIZE = 8
