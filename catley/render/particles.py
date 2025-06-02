import math
import random

import numpy as np
import tcod.console

from catley import colors

# Usage example:
# particle_system = SubTileParticleSystem(map_width, map_height)
#
# # NEW: Using the effect system (recommended)
# from catley.render.effects import EffectLibrary, EffectContext
# effect_library = EffectLibrary()
# context = EffectContext(particle_system, x=enemy.x, y=enemy.y, intensity=0.8)
# effect_library.trigger("blood_splatter", context)
#
# Generic particle creation methods:
#   - emit_radial_burst()        # Particles radiating from a point
#   - emit_directional_cone()    # Particles in a directional cone
#   - emit_area_flash()          # Background flash effect over area
#   - emit_background_tint()     # Background tinting particles (smoke, gas)
#
# # In render loop:
# particle_system.update(delta_time)
# particle_system.render_to_console(self.game_map_console)


class SubParticle:
    """
    Represents a single particle that exists at sub-tile resolution.

    Particles have position, velocity, lifetime, and visual properties.
    They can render as foreground characters and/or background colors.
    """

    def __init__(
        self,
        sub_x: float,
        sub_y: float,
        vel_x: float,
        vel_y: float,
        lifetime: float,
        char: str = "*",
        color: colors.Color = (255, 0, 0),
        bg_color: colors.Color | None = None,
        bg_blend_mode: str = "tint",  # "tint", "overlay", "replace"
    ):
        # Position and movement (in sub-tile coordinates)
        self.sub_x = sub_x  # Position in sub-tile coordinates
        self.sub_y = sub_y
        self.vel_x = vel_x  # Velocity in sub-tiles per second
        self.vel_y = vel_y

        # Lifetime tracking
        self.lifetime = lifetime
        self.max_lifetime = lifetime

        # Visual properties
        self.char = char
        self.color = color
        self.bg_color = bg_color
        self.bg_blend_mode = bg_blend_mode

        # Physics properties
        self.gravity = 0.0  # Optional gravity in sub-tiles/sec^2

        # Special properties for explosion flash effects
        self._flash_intensity: float | None = None  # For explosion flash effects

    def update(self, delta_time: float) -> bool:
        """
        Update particle physics and lifetime.

        Args:
            delta_time: Time elapsed since last update in seconds

        Returns:
            bool: True if particle is still alive, False if it should be removed
        """
        # Update position based on velocity
        self.sub_x += self.vel_x * delta_time
        self.sub_y += self.vel_y * delta_time

        # Apply gravity to vertical velocity
        self.vel_y += self.gravity * delta_time

        # Update lifetime
        self.lifetime -= delta_time
        return self.lifetime > 0

    @property
    def intensity(self) -> float:
        """
        Get current intensity based on remaining lifetime.

        Returns:
            float: Intensity from 0.0 (about to die) to 1.0 (just created)
        """
        return self.lifetime / self.max_lifetime

    @property
    def tile_pos(self) -> tuple[int, int]:
        """
        Get the tile position this particle occupies.

        Returns:
            tuple[int, int]: (x, y) tile coordinates
        """
        return int(self.sub_x), int(self.sub_y)


class SubTileParticleSystem:
    """
    Manages particles that exist at sub-tile resolution for smooth movement.

    This system provides both generic particle creation methods (recommended)
    and game-specific convenience methods (deprecated). New code should use
    the effect system (catley.render.effects) which uses the generic methods.

    The subdivision parameter controls sub-tile resolution. A subdivision of 3
    means each tile is divided into a 3x3 grid for particle positioning.
    """

    def __init__(self, map_width: int, map_height: int, subdivision: int = 3):
        """
        Initialize the particle system.

        Args:
            map_width: Width of the game map in tiles
            map_height: Height of the game map in tiles
            subdivision: Sub-tile subdivision factor (3 = 3x3 sub-tiles per tile)
        """
        self.subdivision = subdivision  # 3x3 sub-tiles per tile
        self.sub_width = map_width * subdivision
        self.sub_height = map_height * subdivision
        self.map_width = map_width
        self.map_height = map_height
        self.particles: list[SubParticle] = []

    def add_particle(
        self,
        tile_x: float,
        tile_y: float,
        vel_x: float,
        vel_y: float,
        lifetime: float = 1.0,
        char: str = "*",
        color: colors.Color = (255, 0, 0),
        bg_color: colors.Color | None = None,
        bg_blend_mode: str = "tint",
    ) -> None:
        """
        Add a single particle to the system.

        This is the fundamental method for creating particles. All other
        particle creation methods ultimately use this.

        Args:
            tile_x: X position in tile coordinates
            tile_y: Y position in tile coordinates
            vel_x: X velocity in tiles per second
            vel_y: Y velocity in tiles per second
            lifetime: How long particle lives in seconds
            char: Character to display (empty string for background-only)
            color: Foreground color RGB tuple
            bg_color: Background color RGB tuple (None for no background)
            bg_blend_mode: How to blend background ("tint", "overlay", "replace")
        """
        # Convert tile coordinates to sub-tile coordinates with slight randomization
        # for more natural spread
        sub_x = tile_x * self.subdivision + random.uniform(-0.5, 0.5)
        sub_y = tile_y * self.subdivision + random.uniform(-0.5, 0.5)

        # Convert velocity from tiles/sec to sub-tiles/sec
        sub_vel_x = vel_x * self.subdivision
        sub_vel_y = vel_y * self.subdivision

        particle = SubParticle(
            sub_x,
            sub_y,
            sub_vel_x,
            sub_vel_y,
            lifetime,
            char,
            color,
            bg_color,
            bg_blend_mode,
        )
        self.particles.append(particle)

    # =============================================================================
    # GENERIC PARTICLE CREATION METHODS
    # =============================================================================
    # These methods provide flexible, reusable particle patterns that can be
    # combined to create complex effects. The effect system uses these methods.

    def emit_radial_burst(
        self,
        tile_x: float,
        tile_y: float,
        count: int,
        speed_range: tuple[float, float] = (0.5, 2.0),
        lifetime_range: tuple[float, float] = (0.3, 0.8),
        colors_and_chars: list[tuple[colors.Color, str]] | None = None,
        gravity: float = 0.0,
        bg_color: colors.Color | None = None,
        bg_blend_mode: str = "tint",
    ) -> None:
        """
        Emit particles in all directions from a central point.

        Perfect for explosions, impacts, splashes, and burst effects.
        Each particle gets a random direction, speed, and lifetime.

        Args:
            tile_x: X center position in tile coordinates
            tile_y: Y center position in tile coordinates
            count: Number of particles to create
            speed_range: (min_speed, max_speed) in tiles per second
            lifetime_range: (min_lifetime, max_lifetime) in seconds
            colors_and_chars: List of (color, char) tuples to randomly choose from
            gravity: Gravity acceleration in tiles/sec^2 (0 = no gravity)
            bg_color: Background color for all particles (None for no background)
            bg_blend_mode: Background blend mode if bg_color is set
        """
        # Default to red asterisks if no colors specified
        if colors_and_chars is None:
            colors_and_chars = [((255, 0, 0), "*")]

        for _ in range(count):
            # Random direction (angle in radians)
            angle = random.uniform(0, 2 * math.pi)

            # Random speed within specified range
            speed = random.uniform(*speed_range)

            # Convert polar coordinates to cartesian velocity
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed

            # Random lifetime within specified range
            lifetime = random.uniform(*lifetime_range)

            # Random color and character from the provided list
            color, char = random.choice(colors_and_chars)

            # Create the particle using our core add_particle method
            particle = SubParticle(
                tile_x * self.subdivision,
                tile_y * self.subdivision,
                vel_x * self.subdivision,
                vel_y * self.subdivision,
                lifetime,
                char,
                color,
                bg_color,
                bg_blend_mode,
            )

            # Apply gravity if specified
            particle.gravity = gravity * self.subdivision

            self.particles.append(particle)

    def emit_directional_cone(
        self,
        tile_x: float,
        tile_y: float,
        direction_x: float,
        direction_y: float,
        count: int,
        cone_spread: float = 0.3,
        speed_range: tuple[float, float] = (3.0, 6.0),
        lifetime_range: tuple[float, float] = (0.05, 0.15),
        colors_and_chars: list[tuple[colors.Color, str]] | None = None,
        gravity: float = 0.0,
    ) -> None:
        """
        Emit particles in a cone shape in a specific direction.

        Perfect for muzzle flashes, spray effects, breath weapons, and
        any directional effect that spreads outward.

        Args:
            tile_x: X source position in tile coordinates
            tile_y: Y source position in tile coordinates
            direction_x: X component of direction vector (will be normalized)
            direction_y: Y component of direction vector (will be normalized)
            count: Number of particles to create
            cone_spread: Cone spread in radians (0.3 ≈ 17 degrees)
            speed_range: (min_speed, max_speed) in tiles per second
            lifetime_range: (min_lifetime, max_lifetime) in seconds
            colors_and_chars: List of (color, char) tuples to randomly choose from
            gravity: Gravity acceleration in tiles/sec^2
        """
        # Default to bright yellow plus signs if no colors specified
        if colors_and_chars is None:
            colors_and_chars = [((255, 255, 0), "+")]

        # Calculate the base angle from the direction vector
        base_angle = math.atan2(direction_y, direction_x)

        for _ in range(count):
            # Add random spread to the base angle
            spread = random.uniform(-cone_spread, cone_spread)
            angle = base_angle + spread

            # Random speed within specified range
            speed = random.uniform(*speed_range)

            # Convert to velocity components
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed

            # Random lifetime
            lifetime = random.uniform(*lifetime_range)

            # Random color and character
            color, char = random.choice(colors_and_chars)

            # Create and add the particle
            particle = SubParticle(
                tile_x * self.subdivision,
                tile_y * self.subdivision,
                vel_x * self.subdivision,
                vel_y * self.subdivision,
                lifetime,
                char,
                color,
            )
            particle.gravity = gravity * self.subdivision
            self.particles.append(particle)

    def emit_area_flash(
        self,
        tile_x: float,
        tile_y: float,
        radius: int,
        flash_color: colors.Color = (255, 255, 200),
        lifetime_range: tuple[float, float] = (0.1, 0.2),
        intensity_falloff: bool = True,
    ) -> None:
        """
        Create a bright background flash effect over a circular area.

        Perfect for explosions, lightning, magic effects. Creates stationary
        particles that only affect background color for a brief flash.

        Args:
            tile_x: X center position in tile coordinates
            tile_y: Y center position in tile coordinates
            radius: Flash radius in tiles
            flash_color: RGB color of the flash
            lifetime_range: (min_lifetime, max_lifetime) in seconds
            intensity_falloff: If True, flash intensity decreases with distance
        """
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # Calculate distance from center
                distance = (dx * dx + dy * dy) ** 0.5

                # Skip tiles outside the circular radius
                if distance <= radius:
                    # Calculate flash intensity based on distance
                    if intensity_falloff:
                        flash_intensity = 1.0 - (distance / radius)
                    else:
                        flash_intensity = 1.0

                    # Random lifetime within range
                    lifetime = random.uniform(*lifetime_range)

                    # Create a background-only particle (no character)
                    flash_particle = SubParticle(
                        (tile_x + dx) * self.subdivision,
                        (tile_y + dy) * self.subdivision,
                        0,
                        0,  # No movement - stationary flash
                        lifetime,
                        char="",  # No character, just background effect
                        color=(0, 0, 0),  # No foreground color
                        bg_color=flash_color,
                        bg_blend_mode="replace",  # Replace background completely
                    )

                    # Store the intensity for special flash rendering
                    flash_particle._flash_intensity = flash_intensity
                    self.particles.append(flash_particle)

    def emit_background_tint(
        self,
        tile_x: float,
        tile_y: float,
        count: int,
        drift_speed: tuple[float, float] = (0.2, 1.0),
        lifetime_range: tuple[float, float] = (1.0, 2.0),
        tint_color: colors.Color = (100, 100, 100),
        blend_mode: str = "tint",
        chars: list[str] | None = None,
        upward_drift: float = -0.3,
    ) -> None:
        """
        Create particles that primarily affect background color (smoke, gas, etc.).

        Perfect for smoke clouds, poison gas, dust, and other atmospheric effects
        that should tint or color the background while drifting slowly.

        Args:
            tile_x: X center position in tile coordinates
            tile_y: Y center position in tile coordinates
            count: Number of particles to create
            drift_speed: (min_speed, max_speed) for random drift in tiles/sec
            lifetime_range: (min_lifetime, max_lifetime) in seconds
            tint_color: Base RGB color for the tint effect
            blend_mode: How to blend background ("tint", "overlay", "replace")
            chars: List of characters to use (None for default smoke chars)
            upward_drift: Constant upward velocity component (negative = up)
        """
        # Default smoke/gas characters
        if chars is None:
            chars = [".", ",", " "]

        for _ in range(count):
            # Random drift direction and speed
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(*drift_speed)
            vel_x = math.cos(angle) * speed

            # Vertical velocity includes both random drift and upward component
            vel_y = math.sin(angle) * speed + upward_drift

            # Random lifetime
            lifetime = random.uniform(*lifetime_range)

            # Random character from the list
            char = random.choice(chars)

            # Vary the tint color slightly for more natural effect
            varied_tint: colors.Color = (
                max(0, min(255, tint_color[0] + random.randint(-20, 20))),
                max(0, min(255, tint_color[1] + random.randint(-20, 20))),
                max(0, min(255, tint_color[2] + random.randint(-20, 20))),
            )

            # Background tint is dimmer than foreground
            bg_tint: colors.Color = (
                varied_tint[0] // 2,
                varied_tint[1] // 2,
                varied_tint[2] // 2,
            )

            # Create the particle
            self.add_particle(
                tile_x,
                tile_y,
                vel_x,
                vel_y,
                lifetime,
                char,
                varied_tint,
                bg_tint,
                blend_mode,
            )

    # =============================================================================
    # CORE SYSTEM METHODS
    # =============================================================================
    # These methods handle the fundamental particle system operations and
    # should not be changed.

    def update(self, delta_time: float):
        """
        Update all particles and remove dead/out-of-bounds ones.

        This method:
        1. Updates each particle's physics and lifetime
        2. Removes particles that have expired (lifetime <= 0)
        3. Removes particles that have moved outside the map bounds

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        # Update particles and remove dead ones
        # List comprehension filters out particles where update() returns False
        self.particles = [p for p in self.particles if p.update(delta_time)]

        # Remove particles that are out of bounds
        # This prevents particles from affecting rendering outside the map
        self.particles = [
            p
            for p in self.particles
            if 0 <= p.sub_x < self.sub_width and 0 <= p.sub_y < self.sub_height
        ]

    def render_to_console(self, console: tcod.console.Console) -> None:
        """
        Render particles to the console using sub-tile blending.

        This method handles the complex process of converting sub-tile particle
        positions into tile-based console effects. Multiple particles in the same
        tile are blended together for the final visual result.

        The rendering process:
        1. Accumulate all particle effects per tile
        2. Blend multiple particles in the same tile
        3. Apply background effects first, then foreground effects
        4. Choose the most intense particle for character display

        Args:
            console: TCOD console to render particles onto
        """
        # Create arrays to track particle effects per tile
        particle_intensity = np.zeros((self.map_width, self.map_height))
        particle_chars = {}  # tile_pos -> list[(char, intensity, color)]
        particle_bg_effects = {}  # tile_pos -> list[(bg_color, intensity, blend_mode)]

        # Accumulate particle effects by converting sub-tile positions to tiles
        for particle in self.particles:
            # Convert sub-tile position to tile position
            tile_x = int(particle.sub_x // self.subdivision)
            tile_y = int(particle.sub_y // self.subdivision)

            # Skip particles that are somehow outside the map bounds
            if 0 <= tile_x < self.map_width and 0 <= tile_y < self.map_height:
                # Use flash intensity if available (for explosion effects),
                # otherwise use normal lifetime-based intensity
                intensity = (
                    particle._flash_intensity
                    if particle._flash_intensity is not None
                    else particle.intensity
                )

                # Accumulate total intensity for this tile
                particle_intensity[tile_x, tile_y] += intensity

                # Track character and color info for foreground rendering
                tile_pos = (tile_x, tile_y)
                if tile_pos not in particle_chars:
                    particle_chars[tile_pos] = []

                # Only track particles that have actual characters
                if particle.char:  # Skip empty characters (background-only particles)
                    particle_chars[tile_pos].append(
                        (particle.char, intensity, particle.color)
                    )

                # Track background effects
                if particle.bg_color:
                    if tile_pos not in particle_bg_effects:
                        particle_bg_effects[tile_pos] = []
                    particle_bg_effects[tile_pos].append(
                        (particle.bg_color, intensity, particle.bg_blend_mode)
                    )

        # Render accumulated effects to console
        for x in range(self.map_width):
            for y in range(self.map_height):
                # Apply background effects first (smoke, flashes, etc.)
                if (x, y) in particle_bg_effects:
                    self._apply_background_effects(
                        console, x, y, particle_bg_effects[(x, y)]
                    )

                # Then apply foreground effects (characters and colors)
                total_intensity = particle_intensity[x, y]
                if total_intensity > 0 and (x, y) in particle_chars:
                    # Choose the most intense particle for this tile's character
                    char_data = particle_chars[(x, y)]

                    # Skip if no particles with actual characters
                    # (e.g., explosion flash only)
                    if not char_data:
                        continue

                    # Sort by intensity to find the most prominent particle
                    char_data.sort(
                        key=lambda x: x[1], reverse=True
                    )  # Sort by intensity

                    best_char, best_intensity, best_color = char_data[0]

                    # Scale intensity (multiple particles can make it brighter)
                    # Cap at 1.0 to prevent over-brightening
                    display_intensity = min(total_intensity, 1.0)

                    # Blend color based on intensity
                    final_color = tuple(int(c * display_intensity) for c in best_color)

                    # Don't overwrite important map elements, blend instead
                    current_char = chr(console.rgb["ch"][x, y])
                    if current_char == " ":  # Empty space, can overwrite safely
                        console.rgb["ch"][x, y] = ord(best_char)
                        console.rgb["fg"][x, y] = final_color
                    else:  # Blend with existing content
                        # Just modify the foreground color for blend effect
                        # This preserves map content while showing particle influence
                        current_fg = console.rgb["fg"][x, y]
                        blended_fg = tuple(
                            min(255, int(current_fg[i] + final_color[i] * 0.3))
                            for i in range(3)
                        )
                        console.rgb["fg"][x, y] = blended_fg

    def _apply_background_effects(
        self, console: tcod.console.Console, x: int, y: int, bg_effects
    ) -> None:
        """
        Apply background color effects to a single tile.

        This method handles the different background blending modes:
        - "tint": Subtle additive blending with existing background
        - "overlay": Stronger blending that replaces more of the original
        - "replace": Complete replacement of background color

        Multiple background effects on the same tile are applied sequentially.

        Args:
            console: TCOD console to modify
            x: Tile X coordinate
            y: Tile Y coordinate
            bg_effects: List of (bg_color, intensity, blend_mode) tuples
        """
        current_bg = console.rgb["bg"][x, y]

        # Apply each background effect in sequence
        for bg_color, intensity, blend_mode in bg_effects:
            if blend_mode == "tint":
                # Subtle blend with existing background
                # Adds some of the effect color to the existing color
                blended_bg = tuple(
                    min(255, int(current_bg[i] + bg_color[i] * intensity * 0.4))
                    for i in range(3)
                )
                console.rgb["bg"][x, y] = blended_bg

            elif blend_mode == "overlay":
                # Stronger overlay effect
                # Blends between original and effect color based on intensity
                blend_factor = intensity * 0.6
                blended_bg = tuple(
                    int(current_bg[i] * (1 - blend_factor) + bg_color[i] * blend_factor)
                    for i in range(3)
                )
                console.rgb["bg"][x, y] = blended_bg

            elif blend_mode == "replace":
                # Full replacement (for bright flashes)
                # Completely replaces background with effect color
                console.rgb["bg"][x, y] = tuple(int(c * intensity) for c in bg_color)

            # Update current_bg for next effect in the list
            # This allows multiple effects to compound properly
            current_bg = console.rgb["bg"][x, y]
