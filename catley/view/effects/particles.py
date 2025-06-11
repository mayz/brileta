import math
import random

import numpy as np
import tcod.console

from catley import colors
from catley.constants.view import ViewConstants as View
from catley.game.enums import AreaType, BlendMode, ConsumableEffectType  # noqa: F401

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


class SubTileParticleSystem:
    """
    Manages particles that exist at sub-tile resolution for smooth movement.

    This system provides both generic particle creation methods (recommended)
    and game-specific convenience methods (deprecated). New code should use
    the effect system (catley.render.effects) which uses the generic methods.

    The subdivision parameter controls sub-tile resolution. A subdivision of 3
    means each tile is divided into a 3x3 grid for particle positioning.
    """

    def __init__(
        self,
        map_width: int,
        map_height: int,
        subdivision: int = View.SUB_TILE_SUBDIVISION,
        max_particles: int = 1000,
    ) -> None:
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

        self.max_particles = max_particles
        self.active_count = 0

        self.positions = np.zeros((max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((max_particles, 2), dtype=np.float32)
        self.lifetimes = np.zeros(max_particles, dtype=np.float32)
        self.max_lifetimes = np.zeros(max_particles, dtype=np.float32)
        self.colors = np.zeros((max_particles, 3), dtype=np.uint8)
        self.chars = np.full(max_particles, "*", dtype="<U1")
        self.bg_colors = np.zeros((max_particles, 3), dtype=np.uint8)
        self.has_bg_color = np.zeros(max_particles, dtype=bool)
        self.bg_blend_mode = np.full(max_particles, BlendMode.TINT.value, dtype=np.int8)
        self.gravity = np.zeros(max_particles, dtype=np.float32)
        # NaN sentinel means no explicit flash intensity supplied.
        self.flash_intensity = np.full(max_particles, np.nan, dtype=np.float32)

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
        bg_blend_mode: BlendMode = BlendMode.TINT,
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
        # Calculate the center of the specified tile_x, tile_y in sub-pixel coordinates
        center_sub_x_of_tile = (tile_x + 0.5) * self.subdivision
        center_sub_y_of_tile = (tile_y + 0.5) * self.subdivision

        # Spread particles slightly around this center point.
        # A small fixed sub-pixel spread (e.g., +/- 0.5 sub-pixels) is often good
        # to give a bit of volume to the emission point without precise alignment.
        spread_sub_pixels = View.PARTICLE_SPREAD
        final_particle_sub_x = center_sub_x_of_tile + random.uniform(
            -spread_sub_pixels, spread_sub_pixels
        )
        final_particle_sub_y = center_sub_y_of_tile + random.uniform(
            -spread_sub_pixels, spread_sub_pixels
        )

        if self.active_count >= self.max_particles:
            return

        index = self.active_count

        # Convert velocity from tiles/sec to sub-tiles/sec
        sub_vel_x = vel_x * self.subdivision
        sub_vel_y = vel_y * self.subdivision

        self.positions[index] = (final_particle_sub_x, final_particle_sub_y)
        self.velocities[index] = (sub_vel_x, sub_vel_y)
        self.lifetimes[index] = lifetime
        self.max_lifetimes[index] = lifetime
        self.colors[index] = color
        self.chars[index] = char
        if bg_color is not None:
            self.bg_colors[index] = bg_color
            self.has_bg_color[index] = True
            self.bg_blend_mode[index] = bg_blend_mode.value
        else:
            self.has_bg_color[index] = False
        self.gravity[index] = 0.0
        self.flash_intensity[index] = np.nan

        self.active_count += 1

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
        speed_range: tuple[float, float] = (
            View.RADIAL_SPEED_MIN,
            View.RADIAL_SPEED_MAX,
        ),
        lifetime_range: tuple[float, float] = (
            View.RADIAL_LIFETIME_MIN,
            View.RADIAL_LIFETIME_MAX,
        ),
        colors_and_chars: list[tuple[colors.Color, str]] | None = None,
        gravity: float = 0.0,
        bg_color: colors.Color | None = None,
        bg_blend_mode: BlendMode = BlendMode.TINT,
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
            if self.active_count >= self.max_particles:
                break
            # Random direction (angle in radians)
            angle = random.uniform(0, 2 * math.pi)

            # Random speed within specified range
            min_speed, max_speed = speed_range
            speed = random.uniform(min_speed, max_speed)

            # Convert polar coordinates to cartesian velocity
            vel_x = math.cos(angle) * speed
            vel_y = math.sin(angle) * speed

            # Random lifetime within specified range
            min_lifetime, max_lifetime = lifetime_range
            lifetime = random.uniform(min_lifetime, max_lifetime)

            # Random color and character from the provided list
            color, char = random.choice(colors_and_chars)

            center_sub_x = tile_x * self.subdivision + self.subdivision / 2.0
            center_sub_y = tile_y * self.subdivision + self.subdivision / 2.0

            idx = self.active_count
            self.positions[idx] = (center_sub_x, center_sub_y)
            self.velocities[idx] = (
                vel_x * self.subdivision,
                vel_y * self.subdivision,
            )
            self.lifetimes[idx] = lifetime
            self.max_lifetimes[idx] = lifetime
            self.colors[idx] = color
            self.chars[idx] = char
            if bg_color is not None:
                self.bg_colors[idx] = bg_color
                self.has_bg_color[idx] = True
                self.bg_blend_mode[idx] = bg_blend_mode.value
            else:
                self.has_bg_color[idx] = False
            self.gravity[idx] = gravity * self.subdivision
            self.flash_intensity[idx] = np.nan

            self.active_count += 1

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
        origin_offset_tiles: float = 0.0,
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
            origin_offset_tiles: How far (in tiles) to offset the particle origin
                        from (tile_x, tile_y) in the direction of the cone.
                        Default is 0.0 (no offset, origin at tile center)
        """
        # Default to bright yellow plus signs if no colors specified
        if colors_and_chars is None:
            colors_and_chars = [((255, 255, 0), "+")]

        # Calculate the base angle from the direction vector
        base_angle = math.atan2(direction_y, direction_x)

        # Calculate the actual origin point for particles, including any offset
        center_sub_x = tile_x * self.subdivision + self.subdivision / 2.0
        center_sub_y = tile_y * self.subdivision + self.subdivision / 2.0

        if origin_offset_tiles > 0.0:
            offset_distance_sub_pixels = origin_offset_tiles * self.subdivision
            initial_particle_sub_x = (
                center_sub_x + math.cos(base_angle) * offset_distance_sub_pixels
            )
            initial_particle_sub_y = (
                center_sub_y + math.sin(base_angle) * offset_distance_sub_pixels
            )
        else:
            initial_particle_sub_x = center_sub_x
            initial_particle_sub_y = center_sub_y

        for _ in range(count):
            if self.active_count >= self.max_particles:
                break

            # Add random spread to the base angle for velocity
            spread = random.uniform(-cone_spread, cone_spread)
            angle_for_velocity = base_angle + spread  # This angle is for velocity

            # Random speed within specified range
            min_speed, max_speed = speed_range
            speed = random.uniform(min_speed, max_speed)

            # Convert to velocity components
            vel_x = math.cos(angle_for_velocity) * speed
            vel_y = math.sin(angle_for_velocity) * speed

            # Random lifetime
            min_lifetime, max_lifetime = lifetime_range
            lifetime = random.uniform(min_lifetime, max_lifetime)

            # Random color and character
            color, char = random.choice(colors_and_chars)

            idx = self.active_count
            self.positions[idx] = (
                initial_particle_sub_x,
                initial_particle_sub_y,
            )
            self.velocities[idx] = (
                vel_x * self.subdivision,
                vel_y * self.subdivision,
            )
            self.lifetimes[idx] = lifetime
            self.max_lifetimes[idx] = lifetime
            self.colors[idx] = color
            self.chars[idx] = char
            self.has_bg_color[idx] = False
            self.gravity[idx] = gravity * self.subdivision
            self.flash_intensity[idx] = np.nan
            self.active_count += 1

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
                    min_lifetime, max_lifetime = lifetime_range
                    lifetime = random.uniform(min_lifetime, max_lifetime)

                    if self.active_count >= self.max_particles:
                        return

                    idx = self.active_count
                    self.positions[idx] = (
                        (tile_x + dx) * self.subdivision,
                        (tile_y + dy) * self.subdivision,
                    )
                    self.velocities[idx] = (0.0, 0.0)
                    self.lifetimes[idx] = lifetime
                    self.max_lifetimes[idx] = lifetime
                    self.colors[idx] = (0, 0, 0)
                    self.chars[idx] = ""
                    self.bg_colors[idx] = flash_color
                    self.has_bg_color[idx] = True
                    self.bg_blend_mode[idx] = BlendMode.REPLACE.value
                    self.gravity[idx] = 0.0
                    self.flash_intensity[idx] = flash_intensity
                    self.active_count += 1

    def emit_background_tint(
        self,
        tile_x: float,
        tile_y: float,
        count: int,
        drift_speed: tuple[float, float] = (
            View.SMOKE_DRIFT_MIN,
            View.SMOKE_DRIFT_MAX,
        ),
        lifetime_range: tuple[float, float] = (
            View.SMOKE_LIFE_MIN,
            View.SMOKE_LIFE_MAX,
        ),
        tint_color: colors.Color = (100, 100, 100),
        blend_mode: BlendMode = BlendMode.TINT,
        chars: list[str] | None = None,
        upward_drift: float = View.SMOKE_UPWARD_DRIFT,
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
            if self.active_count >= self.max_particles:
                break

            # Random drift direction and speed
            angle = random.uniform(0, 2 * math.pi)
            min_speed, max_speed = drift_speed
            speed = random.uniform(min_speed, max_speed)
            vel_x = math.cos(angle) * speed

            # Vertical velocity includes both random drift and upward component
            vel_y = math.sin(angle) * speed + upward_drift

            # Random lifetime
            min_lifetime, max_lifetime = lifetime_range
            lifetime = random.uniform(min_lifetime, max_lifetime)

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

    def update(self, delta_time: float) -> None:
        """
        Update all particles and remove dead/out-of-bounds ones.

        This method:
        1. Updates each particle's physics and lifetime
        2. Removes particles that have expired (lifetime <= 0)
        3. Removes particles that have moved outside the map bounds

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        if self.active_count == 0:
            return

        active = slice(0, self.active_count)

        # Vectorized physics update for all active particles
        self.positions[active] += self.velocities[active] * delta_time
        self.velocities[active, 1] += self.gravity[active] * delta_time
        self.lifetimes[active] -= delta_time

        # Determine which particles remain alive and within bounds
        in_bounds_x = (self.positions[active, 0] >= 0) & (
            self.positions[active, 0] < self.sub_width
        )
        in_bounds_y = (self.positions[active, 1] >= 0) & (
            self.positions[active, 1] < self.sub_height
        )
        alive_mask = (self.lifetimes[active] > 0) & in_bounds_x & in_bounds_y

        # Compact arrays in a single pass
        write_idx = 0
        for read_idx in range(self.active_count):
            if alive_mask[read_idx]:
                if write_idx != read_idx:
                    self.positions[write_idx] = self.positions[read_idx]
                    self.velocities[write_idx] = self.velocities[read_idx]
                    self.lifetimes[write_idx] = self.lifetimes[read_idx]
                    self.max_lifetimes[write_idx] = self.max_lifetimes[read_idx]
                    self.colors[write_idx] = self.colors[read_idx]
                    self.chars[write_idx] = self.chars[read_idx]
                    self.bg_colors[write_idx] = self.bg_colors[read_idx]
                    self.has_bg_color[write_idx] = self.has_bg_color[read_idx]
                    self.bg_blend_mode[write_idx] = self.bg_blend_mode[read_idx]
                    self.gravity[write_idx] = self.gravity[read_idx]
                    self.flash_intensity[write_idx] = self.flash_intensity[read_idx]
                write_idx += 1

        self.active_count = write_idx

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

        for i in range(self.active_count):
            sub_x, sub_y = self.positions[i]
            tile_x = int(sub_x // self.subdivision)
            tile_y = int(sub_y // self.subdivision)

            if 0 <= tile_x < self.map_width and 0 <= tile_y < self.map_height:
                if not np.isnan(self.flash_intensity[i]):
                    intensity = self.flash_intensity[i]
                else:
                    intensity = self.lifetimes[i] / self.max_lifetimes[i]

                particle_intensity[tile_x, tile_y] += intensity

                tile_pos = (tile_x, tile_y)
                if tile_pos not in particle_chars:
                    particle_chars[tile_pos] = []

                if self.chars[i]:
                    particle_chars[tile_pos].append(
                        (self.chars[i], intensity, tuple(self.colors[i]))
                    )

                if self.has_bg_color[i]:
                    if tile_pos not in particle_bg_effects:
                        particle_bg_effects[tile_pos] = []
                    blend_mode = BlendMode(self.bg_blend_mode[i])
                    particle_bg_effects[tile_pos].append(
                        (tuple(self.bg_colors[i]), intensity, blend_mode)
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
            if blend_mode == BlendMode.TINT:
                # Subtle blend with existing background
                # Adds some of the effect color to the existing color
                blended_bg = tuple(
                    min(255, int(current_bg[i] + bg_color[i] * intensity * 0.4))
                    for i in range(3)
                )
                console.rgb["bg"][x, y] = blended_bg

            elif blend_mode == BlendMode.OVERLAY:
                # Stronger overlay effect
                # Blends between original and effect color based on intensity
                blend_factor = intensity * 0.6
                blended_bg = tuple(
                    int(current_bg[i] * (1 - blend_factor) + bg_color[i] * blend_factor)
                    for i in range(3)
                )
                console.rgb["bg"][x, y] = blended_bg

            elif blend_mode == BlendMode.REPLACE:
                # Full replacement (for bright flashes)
                # Completely replaces background with effect color
                console.rgb["bg"][x, y] = tuple(int(c * intensity) for c in bg_color)

            # Update current_bg for next effect in the list
            # This allows multiple effects to compound properly
            current_bg = console.rgb["bg"][x, y]
