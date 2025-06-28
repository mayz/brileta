import math
import random
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from catley import colors
from catley.constants.view import ViewConstants as View
from catley.game.enums import BlendMode
from catley.types import DeltaTime
from catley.util.coordinates import Rect

if TYPE_CHECKING:
    from catley.view.render.graphics import GraphicsContext


class ParticleLayer(Enum):
    """Render layer for particles."""

    UNDER_ACTORS = 0
    OVER_ACTORS = 1


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
# renderer.render_particles(
#     particle_system,
#     ParticleLayer.OVER_ACTORS,
#     viewport_bounds,
#     (0, 0),
# )


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
        self.layers = np.full(
            max_particles, ParticleLayer.UNDER_ACTORS.value, dtype=np.int8
        )

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
        layer: ParticleLayer = ParticleLayer.OVER_ACTORS,
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
        self.layers[index] = layer.value

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
        layer: ParticleLayer = ParticleLayer.OVER_ACTORS,
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
            self.layers[idx] = layer.value

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
        layer: ParticleLayer = ParticleLayer.OVER_ACTORS,
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
            self.layers[idx] = layer.value
            self.active_count += 1

    def emit_area_flash(
        self,
        tile_x: float,
        tile_y: float,
        radius: int,
        flash_color: colors.Color = (255, 255, 200),
        lifetime_range: tuple[float, float] = (0.1, 0.2),
        intensity_falloff: bool = True,
        layer: ParticleLayer = ParticleLayer.OVER_ACTORS,
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
                    self.layers[idx] = layer.value
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
        layer: ParticleLayer = ParticleLayer.OVER_ACTORS,
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
                layer,
            )

    # =============================================================================
    # CORE SYSTEM METHODS
    # =============================================================================
    # These methods handle the fundamental particle system operations and
    # should not be changed.

    def update(self, delta_time: DeltaTime) -> None:
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

        if alive_mask.all():
            # All particles are still alive
            return

        # Count alive particles for new active_count
        new_active_count = int(np.sum(alive_mask))
        if new_active_count == 0:
            self.active_count = 0
            return

        # Use boolean indexing to compact all arrays in vectorized operations
        self.positions[:new_active_count] = self.positions[active][alive_mask]
        self.velocities[:new_active_count] = self.velocities[active][alive_mask]
        self.lifetimes[:new_active_count] = self.lifetimes[active][alive_mask]
        self.max_lifetimes[:new_active_count] = self.max_lifetimes[active][alive_mask]
        self.colors[:new_active_count] = self.colors[active][alive_mask]
        self.chars[:new_active_count] = self.chars[active][alive_mask]
        self.bg_colors[:new_active_count] = self.bg_colors[active][alive_mask]
        self.has_bg_color[:new_active_count] = self.has_bg_color[active][alive_mask]
        self.bg_blend_mode[:new_active_count] = self.bg_blend_mode[active][alive_mask]
        self.gravity[:new_active_count] = self.gravity[active][alive_mask]
        self.flash_intensity[:new_active_count] = self.flash_intensity[active][
            alive_mask
        ]
        self.layers[:new_active_count] = self.layers[active][alive_mask]

        self.active_count = new_active_count

    def _convert_particle_to_screen_coords(
        self,
        particle_index: int,
        viewport_bounds: Rect,
        view_offset: tuple[int, int],
        renderer: "GraphicsContext",
    ) -> tuple[float, float] | None:
        """Convert particle index to screen coordinates, or None if off-screen."""
        if not (0 <= particle_index < self.active_count):
            return None

        sub_x, sub_y = self.positions[particle_index]
        vp_x = sub_x / self.subdivision
        vp_y = sub_y / self.subdivision

        if (
            vp_x < viewport_bounds.x1
            or vp_x > viewport_bounds.x2
            or vp_y < viewport_bounds.y1
            or vp_y > viewport_bounds.y2
        ):
            return None

        root_x = view_offset[0] + vp_x
        root_y = view_offset[1] + vp_y
        return renderer.console_to_screen_coords(root_x, root_y)
