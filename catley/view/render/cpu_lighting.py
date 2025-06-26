"""CPU-based implementation of the lighting system.

This class is responsible for all lighting calculations performed on the CPU,
including light intensity, color blending, flicker effects, and shadow casting.
It maintains internal caches to optimize performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import tcod

from catley.config import (
    AMBIENT_LIGHT_LEVEL,
    DEFAULT_MAX_BRIGHTNESS,
    DEFAULT_MIN_BRIGHTNESS,
    SHADOW_FALLOFF,
    SHADOW_INTENSITY,
    SHADOW_MAX_LENGTH,
)
from catley.environment.map import TileCoord
from catley.util.caching import ResourceCache
from catley.util.coordinates import (
    Rect,
    ViewportTilePos,
    WorldTileCoord,
    WorldTilePos,
)

from .lighting_system import LightingSystem

if TYPE_CHECKING:
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource


@dataclass
class LightingConfig:
    """Configuration for the lighting system"""

    fov_radius: TileCoord = 15
    ambient_light: float = AMBIENT_LIGHT_LEVEL


class CPULightingSystem(LightingSystem):
    """CPU-based implementation of the lighting system."""

    def __init__(
        self, game_world: GameWorld, config: LightingConfig | None = None
    ) -> None:
        """Initialize the CPU lighting system.

        Args:
            game_world: The game world to query for lighting data
            config: Optional lighting configuration
        """
        super().__init__(game_world)

        self.config = config if config is not None else LightingConfig()

        # Noise generator for flickering effects
        self.noise = tcod.noise.Noise(
            dimensions=2,  # 2D noise for position and time
            algorithm=tcod.noise.Algorithm.SIMPLEX,
            implementation=tcod.noise.Implementation.SIMPLE,
        )

        # Internal cache for static light layer optimization
        # This cache will be used to store pre-computed static lighting
        # so we only need to compute dynamic lights each frame
        self._static_light_cache = ResourceCache[tuple, np.ndarray](
            name="CPULightingStaticCache", max_size=5
        )

        # Full lighting cache for complete viewport computations (like the old system)
        self._lighting_cache = ResourceCache[tuple, np.ndarray](
            name="CPULightingFullCache", max_size=15
        )

        # Track time for dynamic effects like flickering
        self._time = 0.0
        self.revision = 0  # Increments each time light map changes

    def update(self, delta_time: float) -> None:
        """Update internal time-based state for dynamic effects.

        Args:
            delta_time: Time elapsed since last frame in seconds
        """
        self._time += delta_time

    def compute_lightmap(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Compute the final lightmap for the visible area.

        This method computes lighting with optional viewport culling and actor shadows.
        Uses an LRU cache for performance.

        Args:
            viewport_bounds: The visible area to compute lighting for

        Returns:
            A (width, height, 3) NumPy array of float RGB intensity values,
            or None if no lighting data is available
        """
        from catley.config import SHADOWS_ENABLED

        map_width = viewport_bounds.width
        map_height = viewport_bounds.height
        viewport_offset = (viewport_bounds.x1, viewport_bounds.y1)

        # Get relevant actors for shadow casting
        actors = self.game_world.actor_spatial_index.get_in_bounds(
            viewport_bounds.x1,
            viewport_bounds.y1,
            viewport_bounds.x2,
            viewport_bounds.y2,
        )

        # Generate cache key for this lighting computation
        cache_key = self._generate_cache_key(
            viewport_bounds, actors, map_width, map_height
        )

        # Check the cache for a pre-computed lighting map
        cached_result = self._lighting_cache.get(cache_key)
        if cached_result is not None:
            return cached_result.copy()

        # Compute base lighting for the viewport
        base_lighting = self._compute_base_lighting(viewport_bounds)

        # Apply shadows if enabled
        if not SHADOWS_ENABLED:
            final_lighting = base_lighting
        else:
            offset_x = viewport_offset[0]
            offset_y = viewport_offset[1]
            final_lighting = self._apply_actor_shadows(
                base_lighting, offset_x, offset_y
            )

        # Store result in cache (make a copy to prevent mutation)
        self._lighting_cache.store(cache_key, final_lighting.copy())

        # Finished a fresh lighting pass - notify views
        self.revision += 1

        return final_lighting

    def _compute_base_lighting(self, viewport_bounds: Rect) -> np.ndarray:
        """Compute lighting only for the viewport currently in view."""
        light_map = np.full(
            (viewport_bounds.width, viewport_bounds.height, 3),
            self.config.ambient_light,
            dtype=np.float32,
        )

        # Apply each light in the game world
        for light in self.game_world.lights:
            lx, ly = light.position
            if (
                viewport_bounds.x1 - light.radius
                <= lx
                < viewport_bounds.x1 + viewport_bounds.width + light.radius
                and viewport_bounds.y1 - light.radius
                <= ly
                < viewport_bounds.y1 + viewport_bounds.height + light.radius
            ):
                self._apply_light_to_map(
                    light,
                    light_map,
                    viewport_bounds.width,
                    viewport_bounds.height,
                    (viewport_bounds.x1, viewport_bounds.y1),
                )

        return np.clip(light_map, 0.0, 1.0)

    def on_light_added(self, light: LightSource) -> None:
        """Handle notification that a light has been added.

        Args:
            light: The light source that was added
        """
        # Invalidate lighting caches since lighting conditions changed
        self._lighting_cache.clear()
        if light.is_static():
            self._static_light_cache.clear()

    def on_light_removed(self, light: LightSource) -> None:
        """Handle notification that a light has been removed.

        Args:
            light: The light source that was removed
        """
        # Invalidate lighting caches since lighting conditions changed
        self._lighting_cache.clear()
        if light.is_static():
            self._static_light_cache.clear()

    def on_light_moved(self, light: LightSource) -> None:
        """Handle notification that a light has moved.

        Args:
            light: The light source that moved
        """
        # Invalidate lighting caches since lighting conditions changed
        self._lighting_cache.clear()
        if light.is_static():
            # Static lights shouldn't move by definition, but clear cache if they do
            self._static_light_cache.clear()

    def _get_static_cache_key(self, viewport_bounds: Rect) -> tuple:
        """Generate a cache key for the static light layer.

        Args:
            viewport_bounds: The visible area

        Returns:
            A hashable tuple representing the static lighting state
        """
        return (
            viewport_bounds.x1,
            viewport_bounds.y1,
            viewport_bounds.width,
            viewport_bounds.height,
        )

    def _calculate_light_flicker(self, light: LightSource) -> float:
        """Calculate the current flicker multiplier for dynamic lights."""
        flicker = 1.0
        if not light.is_static() and getattr(light, "flicker_enabled", False):
            # Sample a single noise value for uniform flicker
            flicker_grid = tcod.noise.grid(
                shape=(1, 1),
                scale=0.5,  # Lower scale = smoother transitions
                # The type hint for `tcod.noise.grid` is ignored here because
                # passing an int quantizes the value and results in jerky flicker.
                # Moreover, the function's own docstring shows that it accepts
                # a float for `origin`.
                origin=(self._time * getattr(light, "flicker_speed", 1.0), 0),  # type: ignore[arg-type]
                indexing="ij",
            )
            flicker_noise = self.noise[flicker_grid][0, 0]
            # Convert from [-1, 1] to [min_brightness, max_brightness]
            min_brightness = getattr(light, "min_brightness", DEFAULT_MIN_BRIGHTNESS)
            max_brightness = getattr(light, "max_brightness", DEFAULT_MAX_BRIGHTNESS)
            flicker = min_brightness + (
                (flicker_noise + 1.0) * 0.5 * (max_brightness - min_brightness)
            )
        return flicker

    def _compute_light_bounds(
        self,
        pos_x: float,
        pos_y: float,
        radius: int,
        map_width: TileCoord,
        map_height: TileCoord,
    ) -> Rect | None:
        """Compute and validate the bounding box for a light's influence."""
        # Determine the bounding box for the light's influence,
        # clamped to map boundaries. Ensure integer results for slicing.
        bounds: Rect = Rect.from_bounds(
            max(0, int(np.floor(pos_x - radius))),
            max(0, int(np.floor(pos_y - radius))),
            min(map_width - 1, int(np.ceil(pos_x + radius))),
            min(map_height - 1, int(np.ceil(pos_y + radius))),
        )

        # If the bounding box is invalid (e.g., light entirely off-map or zero
        # effective area), skip.
        if bounds.x1 > bounds.x2 or bounds.y1 > bounds.y2:
            return None

        return bounds

    def _calculate_light_intensity(
        self,
        pos_x: TileCoord,
        pos_y: TileCoord,
        radius: TileCoord,
        bounds: Rect,
        flicker_multiplier: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Calculate light intensity values within the bounding box."""
        # Create coordinate grid for the bounding box only
        # np.mgrid uses [inclusive:exclusive] for the stop value in slices, so +1
        tile_x_coords, tile_y_coords = np.mgrid[
            bounds.x1 : bounds.x2 + 1, bounds.y1 : bounds.y2 + 1
        ]

        # Calculate distances from the light source to each tile in the bounding box
        dx = tile_x_coords - pos_x
        dy = tile_y_coords - pos_y

        # Calculate squared distance first
        distance_sq_in_bounds = dx * dx + dy * dy

        # Create a mask for tiles within the circular radius
        # (tiles outside this are not actually lit by this light,
        # even if in bounding box).
        radius_sq = radius * radius
        in_circle_mask = distance_sq_in_bounds <= radius_sq

        # Only proceed with tiles that are actually within the circle
        if not np.any(in_circle_mask):
            return None

        # Calculate actual distance only for relevant tiles within the circle
        # Add a small epsilon to distance_sq_in_bounds before sqrt to avoid sqrt(0)
        # issues if needed, though for light falloff, sqrt(0) giving 0 is fine.
        distance_in_bounds = np.sqrt(distance_sq_in_bounds[in_circle_mask])

        # Calculate light intensity with normalized distance for these tiles
        # Intensity falls off linearly to the edge of the radius
        # Ensure radius isn't zero if we got here
        # (already checked, but good practice)
        norm_distance = np.clip(1.0 - (distance_in_bounds / radius), 0.0, 1.0)
        scalar_intensity_values = (
            norm_distance * flicker_multiplier
        )  # 1D array of scalar intensities

        return in_circle_mask, scalar_intensity_values

    def _blend_light_contribution(
        self,
        light: LightSource,
        light_map: np.ndarray,
        bounds: Rect,
        in_circle_mask: np.ndarray,
        scalar_intensity_values: np.ndarray,
    ) -> None:
        """Blend a light's contribution into the main light map."""
        # Get the slice of the main light_map we'll be updating
        # We need to apply the 1D scalar_intensity_values to the 2D masked region
        # of the slice.
        light_map_slice = light_map[
            bounds.x1 : bounds.x2 + 1, bounds.y1 : bounds.y2 + 1, :
        ]

        # Convert integer color to normalized (0.0-1.0) range for lighting calculations
        light_color_array = np.array(light.color, dtype=np.float32) / 255.0

        colored_intensity_contribution = (
            scalar_intensity_values[:, np.newaxis] * light_color_array
        )

        # Apply to the specific (masked) elements within the slice
        # This directly modifies the `light_map_slice` (and thus `light_map`)
        # Get current values in the masked part of the slice
        current_values_in_masked_slice = light_map_slice[in_circle_mask]
        # Combine by taking the maximum
        updated_values = np.maximum(
            current_values_in_masked_slice, colored_intensity_contribution
        )
        # Assign back to the masked part of the slice
        light_map_slice[in_circle_mask] = updated_values

    def _cast_gradual_shadow(
        self, light_pos: WorldTilePos, actor_pos: WorldTilePos, max_distance: TileCoord
    ) -> list[tuple[WorldTileCoord, WorldTileCoord, float]]:
        """Cast a shadow with gradual falloff and softer edges."""
        lx, ly = light_pos
        ax, ay = actor_pos

        dx, dy = ax - lx, ay - ly
        distance = max(abs(dx), abs(dy))

        if distance == 0:
            return []

        # Calculate shadow direction (more nuanced)
        shadow_dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        shadow_dy = 1 if dy > 0 else (-1 if dy < 0 else 0)

        shadow_data: list[tuple[WorldTileCoord, WorldTileCoord, float]] = []
        shadow_length = min(SHADOW_MAX_LENGTH, max_distance - distance + 2)

        for i in range(1, shadow_length + 1):
            # Falloff: shadows get lighter with distance
            distance_falloff = 1.0 - (i - 1) / shadow_length if SHADOW_FALLOFF else 1.0

            center_x: WorldTileCoord = ax + shadow_dx * i
            center_y: WorldTileCoord = ay + shadow_dy * i

            # Core shadow (strongest)
            core_intensity = SHADOW_INTENSITY * distance_falloff
            shadow_data.append((center_x, center_y, core_intensity))

            # Soft edges (weaker shadow around core)
            edge_intensity = core_intensity * 0.4  # Much lighter edges

            # Add softer edge tiles
            if i <= 2:  # Only close shadows have soft edges
                edge_offsets = [
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),  # Adjacent
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),  # Diagonal
                ]

                for ox, oy in edge_offsets:
                    edge_x = center_x + ox
                    edge_y = center_y + oy
                    # Only add edge if it's not the core shadow direction
                    if not (ox == shadow_dx and oy == shadow_dy):
                        shadow_data.append((edge_x, edge_y, edge_intensity))

        return shadow_data

    def _get_shadow_casting_tiles(
        self,
        light_x: WorldTileCoord,
        light_y: WorldTileCoord,
        light_radius: TileCoord,
    ) -> list[WorldTilePos]:
        """Find tiles that cast shadows within light radius."""
        from catley.environment import tile_types

        game_map = self.game_world.game_map
        if game_map is None:
            return []

        shadow_casters: list[WorldTilePos] = []
        for dx in range(-light_radius, light_radius + 1):
            for dy in range(-light_radius, light_radius + 1):
                world_x: WorldTileCoord = light_x + dx
                world_y: WorldTileCoord = light_y + dy
                if 0 <= world_x < game_map.width and 0 <= world_y < game_map.height:
                    tile_id = game_map.tiles[world_x, world_y]
                    tile_data = tile_types.get_tile_type_data_by_id(int(tile_id))
                    if tile_data["casts_shadows"]:
                        shadow_casters.append((world_x, world_y))

        return shadow_casters

    def _apply_actor_shadows(
        self, light_map: np.ndarray, offset_x: TileCoord, offset_y: TileCoord
    ) -> np.ndarray:
        """Apply shadows from actors and shadow-casting tiles."""

        # Get lights from the new system
        lights = self.game_world.lights

        shadow_map = light_map.copy()

        for light_source in lights:
            lx, ly = light_source.position
            light_radius = light_source.radius

            # Use the spatial index for an efficient query
            nearby_actors = [
                actor
                for actor in self.game_world.actor_spatial_index.get_in_radius(
                    lx, ly, light_radius
                )
                if actor.blocks_movement
            ]

            for actor in nearby_actors:
                ax_vp, ay_vp = actor.x - offset_x, actor.y - offset_y
                lx_vp, ly_vp = lx - offset_x, ly - offset_y

                if not (
                    0 <= ax_vp < light_map.shape[0] and 0 <= ay_vp < light_map.shape[1]
                ):
                    continue

                shadow_data = self._cast_gradual_shadow(
                    (lx_vp, ly_vp), (ax_vp, ay_vp), light_radius
                )

                for sx, sy, intensity in shadow_data:
                    if 0 <= sx < light_map.shape[0] and 0 <= sy < light_map.shape[1]:
                        shadow_map[sx, sy] *= 1.0 - intensity

            shadow_casting_positions = self._get_shadow_casting_tiles(
                lx, ly, light_radius
            )

            for tile_x, tile_y in shadow_casting_positions:
                ax_vp, ay_vp = tile_x - offset_x, tile_y - offset_y
                lx_vp, ly_vp = lx - offset_x, ly - offset_y

                shadow_data = self._cast_gradual_shadow(
                    (lx_vp, ly_vp), (ax_vp, ay_vp), light_radius
                )

                for sx, sy, intensity in shadow_data:
                    if 0 <= sx < light_map.shape[0] and 0 <= sy < light_map.shape[1]:
                        shadow_map[sx, sy] *= 1.0 - intensity

        return shadow_map

    def _apply_light_to_map(
        self,
        light: LightSource,
        light_map: np.ndarray,
        map_width: TileCoord,
        map_height: TileCoord,
        viewport_offset: ViewportTilePos = (0, 0),
    ) -> None:
        """Apply a single light source's contribution to the light map."""
        pos_x, pos_y = light.position
        pos_x -= viewport_offset[0]
        pos_y -= viewport_offset[1]
        radius = light.radius

        # Skip lights with zero or negative radius, as they illuminate nothing.
        if radius <= 0:
            return

        # Calculate flicker multiplier for dynamic lights
        flicker_multiplier = self._calculate_light_flicker(light)

        # Compute bounding box and validate
        bounds: Rect | None = self._compute_light_bounds(
            pos_x, pos_y, radius, map_width, map_height
        )
        if not bounds:
            return  # Light is completely off-map or invalid

        # Calculate light intensity within bounds
        intensity_data = self._calculate_light_intensity(
            pos_x,
            pos_y,
            radius,
            bounds,
            flicker_multiplier,
        )
        if not intensity_data:
            return  # No valid light contribution

        in_circle_mask, scalar_intensity_values = intensity_data

        # Blend the light contribution into the main light map
        self._blend_light_contribution(
            light,
            light_map,
            bounds,
            in_circle_mask,
            scalar_intensity_values,
        )

    def _generate_cache_key(
        self,
        viewport_bounds: Rect,
        actors: list,
        map_width: int,
        map_height: int,
    ) -> tuple:
        """Generate a hashable key capturing the lighting state."""

        # Viewport bounds are part of the key
        bounds_key = (
            viewport_bounds.x1,
            viewport_bounds.y1,
            map_width,
            map_height,
        )

        # Include all light source properties that affect lighting
        light_sources_key = tuple(
            (
                light.position[0],
                light.position[1],
                light.radius,
                light.color,
                light.is_static(),
                getattr(light, "min_brightness", DEFAULT_MIN_BRIGHTNESS)
                if not light.is_static()
                else None,
                getattr(light, "max_brightness", DEFAULT_MAX_BRIGHTNESS)
                if not light.is_static()
                else None,
                round(self._time, 1)
                if not light.is_static() and getattr(light, "flicker_enabled", False)
                else None,
            )
            for light in self.game_world.lights
        )

        # Shadow-casting actors within the viewport
        shadow_actors_key = tuple(
            (actor.x, actor.y)
            for actor in actors
            if (
                hasattr(actor, "blocks_movement")
                and actor.blocks_movement
                and viewport_bounds.x1 <= actor.x <= viewport_bounds.x2
                and viewport_bounds.y1 <= actor.y <= viewport_bounds.y2
            )
        )

        return (bounds_key, light_sources_key, shadow_actors_key)
