"""Constants for visual effects and rendering."""


class RenderingConstants:
    """Constants for visual effects and rendering."""

    # Particle system resolution
    SUB_TILE_SUBDIVISION_DEFAULT = 3
    PARTICLE_SPREAD_SUBPIXELS = 0.5

    # Radial burst defaults (used by blood splatter and other effects)
    RADIAL_SPEED_MIN = 0.5
    RADIAL_SPEED_MAX = 2.0
    RADIAL_LIFETIME_MIN = 0.3
    RADIAL_LIFETIME_MAX = 0.8

    # Muzzle flash
    MUZZLE_CONE_SPREAD = 0.3
    MUZZLE_SPEED_MIN = 3.0
    MUZZLE_SPEED_MAX = 6.0
    MUZZLE_LIFETIME_MIN = 0.05
    MUZZLE_LIFETIME_MAX = 0.15
    MUZZLE_ORIGIN_OFFSET_TILES = 0.4

    # Explosion debris and flash
    EXPLOSION_FLASH_LIFETIME_MIN = 0.1
    EXPLOSION_FLASH_LIFETIME_MAX = 0.2
    EXPLOSION_DEBRIS_SPEED_MIN = 1.0
    EXPLOSION_DEBRIS_SPEED_MAX = 3.0
    EXPLOSION_DEBRIS_LIFETIME_MIN = 0.4
    EXPLOSION_DEBRIS_LIFETIME_MAX = 1.0
    EXPLOSION_DEBRIS_GRAVITY = 2.0

    # Smoke cloud
    SMOKE_DRIFT_SPEED_MIN = 0.2
    SMOKE_DRIFT_SPEED_MAX = 1.0
    SMOKE_LIFETIME_MIN = 1.0
    SMOKE_LIFETIME_MAX = 2.0
    SMOKE_UPWARD_DRIFT = -0.3

    # Poison gas
    POISON_DRIFT_SPEED_MIN = 0.1
    POISON_DRIFT_SPEED_MAX = 0.8
    POISON_LIFETIME_MIN = 2.0
    POISON_LIFETIME_MAX = 4.0
    POISON_UPWARD_DRIFT = 0.0
