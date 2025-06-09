"""
Unified Effect System for Game Visual Effects

This module provides a clean, extensible system for creating complex visual
effects by composing simple, reusable components. Effects are triggered
through a consistent API that coordinates particles, decals (future),
sound (future), and other visual systems.

The design separates game logic (what effects to create) from technical
implementation (how particles work), making it easy to add new effects
and modify existing ones without touching low-level systems.

Usage Example:
    # Initialize the effect system
    effect_library = EffectLibrary()

    # Create an effect context with the systems and parameters needed
    context = EffectContext(
        particle_system=renderer.particle_system,
        x=target.x,
        y=target.y,
        intensity=damage / 20.0,
        direction_x=1.0,
        direction_y=0.0
    )

    # Trigger effects by name - consistent API for all effects
    effect_library.trigger("blood_splatter", context)
    effect_library.trigger("muzzle_flash", context)
    effect_library.trigger("explosion", context)

Effect Types:
    - Blood Splatter: Radial burst of blood particles
    - Muzzle Flash: Directional cone of bright particles
    - Explosion: Area flash + debris particles
    - Smoke Cloud: Background tinting with drifting particles
    - Poison Gas: Green background tinting with slow drift

Future Expansion:
    When decals are added, effects will automatically use both particles
    and decals without changing the API. Sound effects can be added the
    same way by expanding EffectContext and individual effect classes.
"""

import abc
import math
import random
from dataclasses import dataclass

from catley import colors
from catley.constants.view import ViewConstants as View
from catley.game.enums import BlendMode

from .particles import SubTileParticleSystem


@dataclass
class EffectContext:
    """
    Provides access to all systems and parameters that effects might need.

    This context object is passed to every effect, giving it access to
    the rendering systems it needs to create visual effects. By centralizing
    all the parameters and systems here, we keep effect implementations
    clean and focused.

    Attributes:
        particle_system: The particle system for creating flying particles
        x: X position in tile coordinates where effect should occur
        y: Y position in tile coordinates where effect should occur
        intensity: Effect intensity from 0.0 to 1.0+
                   (affects particle count, brightness, etc.)
        direction_x: X component of direction vector for directional effects
        direction_y: Y component of direction vector for directional effects

    Future attributes to add:
        decal_system: For persistent visual effects on tiles
        sound_system: For audio feedback
        screen_shake: For impact feedback
        lighting_system: For temporary lighting effects
    """

    particle_system: "SubTileParticleSystem"
    x: int
    y: int
    intensity: float = 1.0
    direction_x: float = 0.0  # For directional effects like muzzle flash
    direction_y: float = 0.0


class Effect(abc.ABC):
    """
    Abstract base class for all visual effects.

    Effects are the building blocks of the visual system. Each effect
    implements a specific visual phenomenon (explosion, blood splatter, etc.)
    by using the systems provided in EffectContext.

    The single execute() method keeps the interface simple while allowing
    unlimited complexity in the implementation.
    """

    @abc.abstractmethod
    def execute(self, context: EffectContext) -> None:
        """
        Execute this effect using the provided context.

        Args:
            context: EffectContext containing systems and parameters
        """
        pass


# =============================================================================
# COMBAT EFFECTS
# =============================================================================
# Effects related to combat actions: hits, misses, weapon fire, etc.


class BloodSplatterEffect(Effect):
    """
    Creates a blood splatter effect when something takes damage.

    Uses a radial burst pattern to simulate blood flying in all directions
    from the impact point. The number of particles and their properties
    scale with the intensity parameter.

    Visual Characteristics:
        - Dark red particles with slight color variation
        - Radial spray pattern from impact point
        - Medium speed with no gravity (blood droplets)
        - Short to medium lifetime (0.3-0.8 seconds)
        - Mix of characters for varied visual texture
    """

    def execute(self, context: EffectContext) -> None:
        """Create blood splatter particles at the impact location."""

        # Define blood color variations - darker reds with some randomness
        # Each tuple is (color, character) for visual variety
        blood_colors_chars = [
            ((120 + random.randint(0, 40), 0, 0), "*"),  # Bright red asterisk
            ((100 + random.randint(0, 40), 0, 0), "."),  # Medium red dot
            ((80 + random.randint(0, 40), 0, 0), ","),  # Dark red comma
        ]

        # Scale particle count with intensity - more damage = more blood
        # Minimum of 5 particles even for low damage to ensure visibility
        count = max(5, int(15 * context.intensity))

        # Create the radial burst of blood particles
        context.particle_system.emit_radial_burst(
            context.x,
            context.y,
            count=count,
            speed_range=(
                View.RADIAL_SPEED_MIN,
                View.RADIAL_SPEED_MAX,
            ),
            lifetime_range=(
                View.RADIAL_LIFETIME_MIN,
                View.RADIAL_LIFETIME_MAX,
            ),
            colors_and_chars=blood_colors_chars,
        )


class MuzzleFlashEffect(Effect):
    """
    Creates a muzzle flash effect when ranged weapons are fired.

    Uses a directional cone pattern to simulate the bright flash and
    hot gases expelled from a weapon barrel. The effect is brief and
    bright to represent the explosive nature of gunpowder.

    Visual Characteristics:
        - Bright yellow/orange colors with white highlights
        - Directional cone extending from weapon position
        - Very high speed particles (fast-moving gases)
        - Very short lifetime (brief flash)
        - Angular characters for explosive appearance
    """

    def execute(self, context: EffectContext) -> None:
        """Create muzzle flash particles in the firing direction."""

        flash_colors_chars = [
            ((255, random.randint(200, 255), random.randint(0, 100)), "+"),
            ((255, random.randint(200, 255), random.randint(0, 100)), "*"),
            ((255, random.randint(200, 255), random.randint(0, 100)), "x"),
        ]

        # Create directional cone of flash particles
        context.particle_system.emit_directional_cone(
            context.x,
            context.y,
            context.direction_x,
            context.direction_y,  # Direction weapon is pointing
            count=8,  # Fixed count for consistent flash appearance
            cone_spread=View.MUZZLE_CONE_SPREAD,
            speed_range=(
                View.MUZZLE_SPEED_MIN,
                View.MUZZLE_SPEED_MAX,
            ),
            lifetime_range=(
                View.MUZZLE_LIFE_MIN,
                View.MUZZLE_LIFE_MAX,
            ),
            colors_and_chars=flash_colors_chars,
            origin_offset_tiles=View.MUZZLE_ORIGIN_OFFSET,
        )


# =============================================================================
# EXPLOSIVE EFFECTS
# =============================================================================
# Effects for explosions, grenades, destructive magic, etc.


class ExplosionEffect(Effect):
    """
    Creates a complex explosion effect with both flash and debris.

    Combines two particle patterns to create a convincing explosion:
    1. Bright area flash for the initial blast of light and heat
    2. Radial debris burst for flying fragments and smoke

    The effect scales with intensity to represent different explosion sizes.

    Visual Characteristics:
        - Bright yellow-white flash covering circular area
        - Flying debris in fire colors (orange, yellow, red)
        - Gravity affects debris particles (they fall)
        - Flash is very brief, debris lasts longer
        - Intensity affects both flash radius and debris count
    """

    def execute(self, context: EffectContext) -> None:
        """Create explosion flash and debris at the blast location."""

        # Calculate explosion radius based on intensity
        # Minimum radius of 1, scales up with intensity
        radius = max(1, int(3 * context.intensity))

        # 1. Create the initial bright flash across the explosion area
        # This represents the intense light and heat of the blast
        context.particle_system.emit_area_flash(
            context.x,
            context.y,
            radius=radius,
            flash_color=(255, 255, 200),  # Bright yellow-white
            lifetime_range=(
                View.EXPLOSION_FLASH_LIFE_MIN,
                View.EXPLOSION_FLASH_LIFE_MAX,
            ),
        )

        # 2. Create flying debris particles
        # These represent fragments, sparks, and smoke from the explosion
        debris_colors_chars = [
            ((255, 100, 0), "*"),  # Orange sparks
            ((255, 200, 0), "+"),  # Yellow flames
            ((200, 50, 0), "."),  # Dark red embers
            ((100, 100, 100), ","),  # Gray smoke particles
        ]

        # More debris for larger explosions
        particle_count = radius * 8

        context.particle_system.emit_radial_burst(
            context.x,
            context.y,
            count=particle_count,
            speed_range=(
                View.EXPLOSION_DEBRIS_SPEED_MIN,
                View.EXPLOSION_DEBRIS_SPEED_MAX,
            ),
            lifetime_range=(
                View.EXPLOSION_DEBRIS_LIFE_MIN,
                View.EXPLOSION_DEBRIS_LIFE_MAX,
            ),
            colors_and_chars=debris_colors_chars,
            gravity=View.EXPLOSION_DEBRIS_GRAVITY,
        )


# =============================================================================
# ATMOSPHERIC EFFECTS
# =============================================================================
# Effects for environmental conditions: smoke, gas, dust, etc.


class SmokeCloudEffect(Effect):
    """
    Creates a smoke cloud that provides background tinting.

    Simulates smoke by creating particles that primarily affect the
    background color of tiles rather than displaying characters.
    The smoke drifts slowly and rises due to heat.

    Visual Characteristics:
        - Gray particles with color variation
        - Slow drift with upward tendency (hot smoke rises)
        - Background tinting effect to fill areas
        - Long lifetime for persistent clouds
        - Mix of small characters and spaces
    """

    def execute(self, context: EffectContext) -> None:
        """Create drifting smoke particles with background tinting."""

        # Calculate smoke area based on intensity
        radius = max(1, int(2 * context.intensity))
        count = radius * 6  # More particles for larger smoke clouds

        # Generate a random gray color for this smoke cloud
        # Varies from medium gray to light gray
        smoke_gray = random.randint(60, 120)
        tint_color: colors.Color = (smoke_gray, smoke_gray, smoke_gray)

        # Create the background tinting smoke particles
        context.particle_system.emit_background_tint(
            context.x,
            context.y,
            count=count,
            drift_speed=(
                View.SMOKE_DRIFT_MIN,
                View.SMOKE_DRIFT_MAX,
            ),
            lifetime_range=(
                View.SMOKE_LIFE_MIN,
                View.SMOKE_LIFE_MAX,
            ),
            tint_color=tint_color,
            blend_mode=BlendMode.TINT,  # Subtle blending with background
            chars=[".", ",", " "],  # Small characters and spaces
            upward_drift=View.SMOKE_UPWARD_DRIFT,
        )


class PoisonGasEffect(Effect):
    """
    Creates a poison gas cloud with green background tinting.

    Similar to smoke but with different visual properties to represent
    toxic gas. Spreads more slowly than smoke and doesn't rise as much
    (heavier gas that stays low).

    Visual Characteristics:
        - Green particles with intensity variation
        - Very slow drift (heavier than air)
        - Strong background tinting for contamination feel
        - Very long lifetime (persistent hazard)
        - Wavy characters to suggest gas movement
    """

    def execute(self, context: EffectContext) -> None:
        """Create drifting poison gas with green background tinting."""

        # Calculate gas area based on intensity
        radius = max(1, int(2 * context.intensity))
        count = radius * 4  # Fewer particles than smoke (denser gas)

        # Generate a random green color for the poison gas
        # Varies in intensity but stays green
        green_intensity = random.randint(40, 100)
        tint_color: colors.Color = (0, green_intensity, 0)

        # Create the background tinting gas particles
        context.particle_system.emit_background_tint(
            context.x,
            context.y,
            count=count,
            drift_speed=(
                View.POISON_DRIFT_MIN,
                View.POISON_DRIFT_MAX,
            ),
            lifetime_range=(
                View.POISON_LIFE_MIN,
                View.POISON_LIFE_MAX,
            ),
            tint_color=tint_color,
            blend_mode=BlendMode.TINT,  # Tints areas with poison color
            chars=[".", ",", "~"],  # Includes wavy character for gas movement
            upward_drift=View.POISON_UPWARD_DRIFT,
        )


# =============================================================================
# EFFECT REGISTRY AND MANAGEMENT
# =============================================================================
# Central system for registering, organizing, and triggering effects.


class EffectLibrary:
    """
    Central registry and manager for all game effects.

    This class provides a unified interface for triggering effects by name,
    making it easy to create effects from anywhere in the game code without
    needing to know the specific implementation details.

    The library comes pre-loaded with standard effects but can be extended
    with custom effects for specific game needs.

    Benefits:
        - Consistent API: All effects use effect_library.trigger()
        - Easy to extend: Register new effects without changing existing code
        - Centralized: All effect definitions in one place
        - Flexible: Effects can be reconfigured or replaced at runtime

    Usage:
        effect_library = EffectLibrary()
        context = EffectContext(particle_system, x=10, y=10, intensity=0.8)
        effect_library.trigger("blood_splatter", context)
    """

    def __init__(self) -> None:
        """Initialize the effect library with standard game effects."""
        # Dictionary mapping effect names to effect instances
        # Using instances rather than classes allows for effect configuration
        self.effects: dict[str, Effect] = {}

        # Load all the standard effects that most games will need
        self._register_default_effects()

    def register_effect(self, name: str, effect: Effect) -> None:
        """
        Register a custom effect in the library.

        This allows game-specific effects to be added without modifying
        the core effect system. Effects can also be reconfigured by
        registering a new effect with an existing name.

        Args:
            name: Unique identifier for the effect
            effect: Effect instance to register

        Example:
            # Register a custom explosion variant
            big_explosion = ExplosionEffect()  # With custom parameters
            effect_library.register_effect("big_explosion", big_explosion)
        """
        self.effects[name] = effect

    def trigger(self, effect_name: str, context: EffectContext) -> None:
        """
        Trigger an effect by name using the provided context.

        This is the main method used throughout the game to create visual
        effects. It provides a consistent interface regardless of the
        complexity of the underlying effect.

        Args:
            effect_name: Name of the effect to trigger
            context: EffectContext with systems and parameters

        Example:
            # In combat system
            context = EffectContext(
                particle_system=self.particle_system,
                x=target.x, y=target.y,
                intensity=damage / 20.0
            )
            self.effect_library.trigger("blood_splatter", context)
        """
        if effect_name in self.effects:
            # Execute the effect with the provided context
            self.effects[effect_name].execute(context)
        else:
            # Log warning for debugging - don't crash the game
            print(f"Warning: Unknown effect '{effect_name}' requested")

    def list_effects(self) -> list[str]:
        """
        Get a list of all registered effect names.

        Useful for debugging, configuration UIs, or dynamic effect selection.

        Returns:
            list[str]: Sorted list of all registered effect names
        """
        return sorted(self.effects.keys())

    def _register_default_effects(self) -> None:
        """
        Register the standard effects that most games will need.

        This method is called during initialization to populate the library
        with commonly used effects. Games can override these by registering
        new effects with the same names.
        """

        # Combat effects - triggered when things take damage or attack
        self.effects["blood_splatter"] = BloodSplatterEffect()
        self.effects["muzzle_flash"] = MuzzleFlashEffect()

        # Explosive effects - triggered by grenades, etc.
        self.effects["explosion"] = ExplosionEffect()

        # Atmospheric effects - triggered by environmental conditions
        self.effects["smoke_cloud"] = SmokeCloudEffect()
        self.effects["poison_gas"] = PoisonGasEffect()

        # Future effects to add:
        # self.effects["sparks"] = SparksEffect()  # For electrical damage
        # self.effects["frost"] = FrostEffect()    # For cold damage
        # self.effects["fire"] = FireEffect()      # For burning
        # self.effects["dust"] = DustEffect()      # For movement/impacts


# =============================================================================
# HELPER FUNCTIONS AND UTILITIES
# =============================================================================
# Convenience functions for common effect operations.


def create_combat_effect_context(
    particle_system: "SubTileParticleSystem",
    attacker_x: int,
    attacker_y: int,
    target_x: int,
    target_y: int,
    damage: int,
) -> EffectContext:
    """
    Create an EffectContext configured for combat effects.

    This helper function simplifies creating contexts for combat-related
    effects by calculating common parameters like direction and intensity.

    Args:
        particle_system: The particle system for rendering
        attacker_x: X position of the attacker
        attacker_y: Y position of the attacker
        target_x: X position of the target
        target_y: Y position of the target
        damage: Amount of damage dealt (for intensity calculation)

    Returns:
        EffectContext: Configured context ready for combat effects

    Example:
        context = create_combat_effect_context(
            particle_system, attacker.x, attacker.y,
            target.x, target.y, damage_amount
        )
        effect_library.trigger("blood_splatter", context)

        # For directional effects like muzzle flash, use the same context
        effect_library.trigger("muzzle_flash", context)
    """
    # Calculate direction vector from attacker to target
    dx = target_x - attacker_x
    dy = target_y - attacker_y

    # Normalize direction vector (handle zero-length case)
    distance = math.sqrt(dx * dx + dy * dy)
    if distance > 0:
        direction_x = dx / distance
        direction_y = dy / distance
    else:
        # If attacker and target are in same position, default to right
        direction_x = 1.0
        direction_y = 0.0

    # Scale damage to reasonable intensity range (0.1 to 1.0+)
    # Typical damage values of 1-20 become intensities of 0.1-1.0
    intensity = max(0.1, min(2.0, damage / 20.0))

    return EffectContext(
        particle_system=particle_system,
        x=target_x,  # Effects appear at target location
        y=target_y,
        intensity=intensity,
        direction_x=direction_x,
        direction_y=direction_y,
    )


def create_environmental_effect_context(
    particle_system: "SubTileParticleSystem", x: int, y: int, intensity: float = 1.0
) -> EffectContext:
    """
    Create an EffectContext for environmental effects.

    Simplified context creation for non-directional effects like smoke,
    gas, dust, and other environmental phenomena.

    Args:
        particle_system: The particle system for rendering
        x: X position for the effect
        y: Y position for the effect
        intensity: Effect intensity (0.0 to 1.0+)

    Returns:
        EffectContext: Configured context for environmental effects

    Example:
        # Create smoke at a specific location
        context = create_environmental_effect_context(
            particle_system, smoke_x, smoke_y, intensity=0.8
        )
        effect_library.trigger("smoke_cloud", context)
    """
    return EffectContext(
        particle_system=particle_system,
        x=x,
        y=y,
        intensity=intensity,
        direction_x=0.0,  # No direction for environmental effects
        direction_y=0.0,
    )
