"""Idle animation profiles for making characters feel alive.

This module provides a configurable system for idle animations with sensible
biological defaults. Characters exhibit subtle life-like behaviors based on
their physical characteristics - smaller creatures move faster and fidget
more, larger creatures move slowly and deliberately.

Features:
- Sub-tile weight-shifting (drift) scaled to creature size

Example usage:
    # Use size-based defaults
    profile = create_profile_for_size(CreatureSize.MEDIUM)

    # Or create a custom profile
    profile = IdleAnimationProfile(
        drift_amplitude=0.08,
        drift_speed=0.4,
    )

    # Or use a preset
    profile = PROFILE_MECHANICAL  # For robots
"""

from dataclasses import dataclass

from brileta.game.enums import CreatureSize


@dataclass
class IdleAnimationProfile:
    """Configuration for an actor's idle animations.

    All parameters have sensible defaults for a medium-sized humanoid.
    Use create_profile_for_size() for size-appropriate defaults, or
    override individual values as needed.

    Attributes:
        drift_enabled: Whether sub-tile weight-shifting movement is enabled.
        drift_amplitude: Maximum drift distance as fraction of tile (±).
        drift_speed: How fast the drift cycles (cycles per second).
    """

    # === Weight Shifting (oscillation along a random direction) ===
    drift_enabled: bool = True
    drift_amplitude: float = 0.025  # Fraction of tile
    drift_speed: float = (
        0.3  # Oscillations per second (one back-and-forth every ~3 sec)
    )


def create_profile_for_size(size: CreatureSize) -> IdleAnimationProfile:
    """Create an IdleAnimationProfile with size-appropriate defaults.

    This factory function generates profiles with sensible defaults based on
    creature size. Smaller creatures move faster with smaller amplitude;
    larger creatures move slower with larger amplitude.

    Args:
        size: The creature's size category.

    Returns:
        An IdleAnimationProfile configured for the given size.
    """
    match size:
        case CreatureSize.TINY:
            return IdleAnimationProfile(
                drift_amplitude=0.012,  # Small movements - tiny body
                drift_speed=0.5,  # Fast, twitchy
            )

        case CreatureSize.SMALL:
            return IdleAnimationProfile(
                drift_amplitude=0.018,  # Modest movements
                drift_speed=0.4,  # Quick
            )

        case CreatureSize.MEDIUM:
            # Human baseline - use all defaults
            return IdleAnimationProfile()

        case CreatureSize.LARGE:
            return IdleAnimationProfile(
                drift_amplitude=0.035,  # Bigger body, wider stance
                drift_speed=0.2,  # Slower, more deliberate
            )

        case CreatureSize.HUGE:
            return IdleAnimationProfile(
                drift_amplitude=0.045,  # Large weight shifts, wide stance
                drift_speed=0.12,  # Ponderous movement
            )


def scale_for_size(size: CreatureSize) -> float:
    """Map CreatureSize to a visual_scale factor for rendering.

    This controls how large the actor's glyph appears on screen, relative
    to the standard tile size. Values are relative to MEDIUM (1.0).

    Collision and pathfinding remain tile-based at 1x1 regardless of
    visual_scale. This function only affects rendering.

    Args:
        size: The creature's size category.

    Returns:
        A float scale factor for rendering (0.6 to 1.3 range).

    Raises:
        NotImplementedError: If size is HUGE, which requires multi-tile
            collision (future tile_footprint feature).
    """
    match size:
        case CreatureSize.TINY:
            return 0.6  # Small, compact (insects, rats)
        case CreatureSize.SMALL:
            return 0.8  # Noticeably smaller (dogs, children)
        case CreatureSize.MEDIUM:
            return 1.0  # Human baseline
        case CreatureSize.LARGE:
            return 1.3  # Clearly larger (bears, mutants)
        case CreatureSize.HUGE:
            raise NotImplementedError(
                "CreatureSize.HUGE requires multi-tile collision (tile_footprint). "
                "Use LARGE for the largest single-tile creatures."
            )


# =============================================================================
# Preset Profiles for Special Cases
# =============================================================================

PROFILE_MECHANICAL = IdleAnimationProfile(
    drift_enabled=True,
    drift_amplitude=0.005,  # Minimal - precise mechanical vibration
    drift_speed=0.05,  # Slow, steady
)
"""Profile for mechanical/robotic entities. Minimal movement."""

PROFILE_UNDEAD = IdleAnimationProfile(
    drift_enabled=True,
    drift_amplitude=0.01,  # Slight unnatural swaying
    drift_speed=0.08,  # Slow, unsettling
)
"""Profile for undead creatures. Eerie, slow movement."""

PROFILE_STATIC = IdleAnimationProfile(
    drift_enabled=False,
)
"""Profile for completely static entities (turrets, statues, etc.)."""

PROFILE_NERVOUS = IdleAnimationProfile(
    drift_enabled=True,
    drift_amplitude=0.025,  # More movement
    drift_speed=0.25,  # Faster, jittery
)
"""Profile for nervous or anxious characters. Faster fidgeting."""

PROFILE_CALM = IdleAnimationProfile(
    drift_enabled=True,
    drift_amplitude=0.01,  # Minimal movement
    drift_speed=0.1,  # Slow, composed
)
"""Profile for calm, composed characters. Slow, deliberate movements."""
