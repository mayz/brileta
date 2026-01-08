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
    SKY_EXPOSURE_POWER,
    SUN_AZIMUTH_DEGREES,
    SUN_COLOR,
    SUN_ELEVATION_DEGREES,
    SUN_ENABLED,
    SUN_INTENSITY,
)
from catley.environment.map import TileCoord
from catley.types import FixedTimestep, ViewportTilePos, WorldTileCoord, WorldTilePos
from catley.util.caching import ResourceCache
from catley.util.coordinates import Rect

from .base import LightingSystem

if TYPE_CHECKING:
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource


@dataclass
class LightingConfig:
    """Configuration for the lighting system"""

    fov_radius: TileCoord = 15
    ambient_light: float = AMBIENT_LIGHT_LEVEL

    # Global sunlight configuration
    sun_enabled: bool = SUN_ENABLED
    sun_color: tuple[int, int, int] = SUN_COLOR
    sun_elevation_degrees: float = SUN_ELEVATION_DEGREES
    sun_azimuth_degrees: float = SUN_AZIMUTH_DEGREES
    sun_intensity: float = SUN_INTENSITY
    sky_exposure_power: float = SKY_EXPOSURE_POWER


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

    def update(self, fixed_timestep: FixedTimestep) -> None:
        """Update internal time-based state for dynamic effects.

        Args:
            fixed_timestep: Fixed time step duration for deterministic lighting timing
        """
        self._time += fixed_timestep

    def compute_lightmap(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Compute the final lightmap for the visible area using two-layer optimization.

        This method uses a two-layer approach for optimal performance:
        1. Static lighting layer (cached) - ambient + static lights only
        2. Dynamic lighting layer (computed fresh) - flickering/moving lights

        Shadows are applied after combining both layers.

        Args:
            viewport_bounds: The visible area to compute lighting for

        Returns:
            A (width, height, 3) NumPy array of float RGB intensity values,
            or None if no lighting data is available
        """
        from catley.config import SHADOWS_ENABLED

        viewport_offset = (viewport_bounds.x1, viewport_bounds.y1)

        # Layer 1: Get static lighting (from cache if possible)
        static_cache_key = self._get_static_cache_key(viewport_bounds)
        static_lighting = self._static_light_cache.get(static_cache_key)

        if static_lighting is None:
            # Cache miss - compute static lighting and store it
            static_lighting = self._compute_static_lights(viewport_bounds)
            self._static_light_cache.store(static_cache_key, static_lighting.copy())

        # Layer 2: Compute dynamic lighting (always fresh for flicker/movement)
        final_lighting = self._compute_dynamic_lights(viewport_bounds, static_lighting)

        # Apply shadows if enabled
        if SHADOWS_ENABLED:
            offset_x = viewport_offset[0]
            offset_y = viewport_offset[1]
            final_lighting = self._apply_actor_shadows(
                final_lighting, offset_x, offset_y
            )

        # Apply light spillover after all lighting (including dynamic) is computed
        # This creates smooth transitions between outdoor and indoor areas
        self._apply_final_spillover(final_lighting, viewport_bounds)

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

    def _compute_static_lights(self, viewport_bounds: Rect) -> np.ndarray:
        """Compute lighting contribution from static lights only.

        This method creates the base lighting layer that can be cached since
        static lights never change position, color, or intensity.

        Args:
            viewport_bounds: The visible area to compute lighting for

        Returns:
            A (width, height, 3) NumPy array with ambient + static contributions
        """
        light_map = np.full(
            (viewport_bounds.width, viewport_bounds.height, 3),
            self.config.ambient_light,
            dtype=np.float32,
        )

        # Apply global directional lights first (sun/moon)
        for light in self.game_world.lights:
            from catley.game.lights import DirectionalLight

            if isinstance(light, DirectionalLight):
                self._apply_directional_light(light, light_map, viewport_bounds)

        # Apply static point lights to the map
        for light in self.game_world.lights:
            from catley.game.lights import DirectionalLight

            if not light.is_static() or isinstance(light, DirectionalLight):
                continue  # Skip dynamic lights and directional lights

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

    def _compute_dynamic_lights(
        self, viewport_bounds: Rect, static_base: np.ndarray
    ) -> np.ndarray:
        """Compute lighting contribution from dynamic lights and blend with static base.

        This method takes the pre-computed static lighting layer and adds
        the contribution
        of all dynamic lights (flickering torches, actor-carried lights, etc.).

        Args:
            viewport_bounds: The visible area to compute lighting for
            static_base: Pre-computed static lighting layer to build upon

        Returns:
            A (width, height, 3) NumPy array with static + dynamic light contributions
        """
        # Start with the static lighting base
        light_map = static_base.copy()

        # Apply only dynamic lights to the map
        for light in self.game_world.lights:
            if light.is_static():
                continue  # Skip static lights (already in base layer)

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
        self.revision += 1

    def on_light_removed(self, light: LightSource) -> None:
        """Handle notification that a light has been removed.

        Args:
            light: The light source that was removed
        """
        # Invalidate lighting caches since lighting conditions changed
        self._lighting_cache.clear()
        if light.is_static():
            self._static_light_cache.clear()
        self.revision += 1

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
        self.revision += 1

    def on_global_light_changed(self) -> None:
        """Handle notification that global lighting has changed (e.g., time of day).

        This invalidates the static light cache since global lights are cached
        with the static layer.
        """
        self._lighting_cache.clear()
        self._static_light_cache.clear()

    def _get_static_cache_key(self, viewport_bounds: Rect) -> tuple:
        """Generate a cache key for the static light layer.

        Args:
            viewport_bounds: The visible area

        Returns:
            A hashable tuple representing the static lighting state
        """
        from catley.game.lights import DirectionalLight

        # Collect static light properties for the cache key
        static_lights = []
        global_lights = []

        for light in self.game_world.lights:
            if isinstance(light, DirectionalLight):
                # Global directional lights affect all tiles
                global_lights.append(
                    (
                        light.direction.x,
                        light.direction.y,
                        light.intensity,
                        light.color[0],
                        light.color[1],
                        light.color[2],
                    )
                )
            elif light.is_static():
                lx, ly = light.position
                # Only include lights that could affect this viewport
                if (
                    viewport_bounds.x1 - light.radius
                    <= lx
                    < viewport_bounds.x1 + viewport_bounds.width + light.radius
                    and viewport_bounds.y1 - light.radius
                    <= ly
                    < viewport_bounds.y1 + viewport_bounds.height + light.radius
                ):
                    static_lights.append(
                        (
                            lx,
                            ly,
                            light.radius,
                            light.color[0],
                            light.color[1],
                            light.color[2],
                        )
                    )

        # Sort to ensure consistent key ordering
        static_lights.sort()
        global_lights.sort()

        return (
            viewport_bounds.x1,
            viewport_bounds.y1,
            viewport_bounds.width,
            viewport_bounds.height,
            tuple(static_lights),  # Include static light properties
            tuple(global_lights),  # Include global light properties
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
        light: LightSource,
        viewport_offset: ViewportTilePos,
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

        # Apply daylight reduction based on the light SOURCE position
        # This ensures uniform reduction across all affected tiles
        source_world_x = int(pos_x + viewport_offset[0])
        source_world_y = int(pos_y + viewport_offset[1])
        source_daylight_reduction = self._calculate_tile_daylight_reduction(
            light, source_world_x, source_world_y
        )

        scalar_intensity_values = (
            norm_distance * flicker_multiplier * source_daylight_reduction
        )  # 1D array of scalar intensities

        return in_circle_mask, scalar_intensity_values

    def _blend_light_contribution(
        self,
        light: LightSource,
        light_map: np.ndarray,
        bounds: Rect,
        in_circle_mask: np.ndarray,
        scalar_intensity_values: np.ndarray,
        viewport_offset: ViewportTilePos,
    ) -> None:
        """Blend light contribution into the light map."""
        # Get the slice of the main light_map we'll be updating
        light_map_slice = light_map[
            bounds.x1 : bounds.x2 + 1, bounds.y1 : bounds.y2 + 1, :
        ]

        # Convert integer color to normalized (0.0-1.0) range for lighting calculations
        light_color_array = np.array(light.color, dtype=np.float32) / 255.0

        # Daylight reduction is now applied uniformly in _calculate_light_intensity
        # So we can directly use the scalar_intensity_values
        colored_intensity_contribution = (
            scalar_intensity_values[:, np.newaxis] * light_color_array
        )

        # Apply to the specific (masked) elements within the slice
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

    def _calculate_tile_daylight_reduction(
        self, light: LightSource, world_x: int, world_y: int
    ) -> float:
        """Calculate how much to reduce light effectiveness at a tile based on daylight.

        Returns a multiplier (0.0 to 1.0) where:
        - 1.0 = full light effect (dark areas)
        - 0.0 = no light effect (bright daylight)
        """
        from catley.game.lights import DynamicLight, GlobalLight

        # Only reduce point lights, not global lights
        if isinstance(light, GlobalLight):
            return 1.0

        # Check if this is a player torch (gets special treatment)
        is_player_torch = (
            isinstance(light, DynamicLight)
            and hasattr(light, "owner")
            and light.owner == self.game_world.player
        )

        if not self.game_world.game_map:
            return 1.0

        # Get the region at this specific tile position
        region = self.game_world.game_map.get_region_at((world_x, world_y))
        if not region:
            return 1.0

        # Calculate reduction based on this tile's sky exposure
        sky_exposure = region.sky_exposure

        # Smooth exponential transition for more natural feel:
        # sky_exposure 0.0 -> reduction 1.0 (full effect)
        # sky_exposure 0.3 -> reduction 0.7 (gradual reduction)
        # sky_exposure 0.7 -> reduction 0.3 (noticeable reduction)
        # sky_exposure 1.0 -> reduction 0.15-0.25 (minimal but visible)

        import math

        if is_player_torch:
            # Player torch: more visible for gameplay (25% minimum)
            reduction = 0.25 + 0.75 * math.exp(-3.0 * sky_exposure)
            return max(0.25, reduction)
        # Other lights: can be more heavily reduced (15% minimum)
        reduction = 0.15 + 0.85 * math.exp(-3.0 * sky_exposure)
        return max(0.15, reduction)

    def _apply_directional_light(
        self, light: LightSource, light_map: np.ndarray, viewport_bounds: Rect
    ) -> None:
        """Apply directional lighting (sun/moon) to the light map.

        Args:
            light: The DirectionalLight to apply
            light_map: The light map to modify in-place
            viewport_bounds: The viewport bounds for coordinate conversion
        """
        from catley.game.lights import DirectionalLight

        if not isinstance(light, DirectionalLight):
            return

        game_map = self.game_world.game_map
        if game_map is None:
            return

        # Iterate through each tile in the viewport
        for y in range(viewport_bounds.height):
            for x in range(viewport_bounds.width):
                world_x = viewport_bounds.x1 + x
                world_y = viewport_bounds.y1 + y

                # Skip tiles outside the map
                if not (
                    0 <= world_x < game_map.width and 0 <= world_y < game_map.height
                ):
                    continue

                # Get sky exposure for this tile's region
                region = game_map.get_region_at((world_x, world_y))
                sky_exposure = region.sky_exposure if region else 0.0

                # Apply direct sunlight only to tiles with sky exposure
                if sky_exposure <= 0:
                    continue  # No sky exposure, no direct sunlight

                # Apply non-linear sky exposure curve for more realistic lighting
                effective_exposure = sky_exposure**self.config.sky_exposure_power

                # Calculate terrain shadows (simplified)
                shadow_factor = 1.0
                if (
                    0 <= world_x < game_map.width
                    and 0 <= world_y < game_map.height
                    and not game_map.transparent[world_x, world_y]
                ):
                    shadow_factor = 0.1  # Walls block most sunlight

                # Combine all factors
                final_intensity = light.intensity * effective_exposure * shadow_factor

                # Apply to RGB channels with additive blending
                sunlight_contribution = (
                    np.array(light.color, dtype=np.float32) / 255.0 * final_intensity
                )
                light_map[x, y] = np.minimum(
                    1.0, light_map[x, y] + sunlight_contribution
                )

    def _apply_final_spillover(
        self, light_map: np.ndarray, viewport_bounds: Rect
    ) -> None:
        """Apply basic light spillover from bright outdoor to adjacent indoor areas."""
        game_map = self.game_world.game_map
        if not game_map:
            return

        # Simple spillover: outdoor areas spill light to adjacent indoor areas
        for y in range(viewport_bounds.height):
            for x in range(viewport_bounds.width):
                world_x = viewport_bounds.x1 + x
                world_y = viewport_bounds.y1 + y

                # Skip if outside map bounds
                if not (
                    0 <= world_x < game_map.width and 0 <= world_y < game_map.height
                ):
                    continue

                current_region = game_map.get_region_at((world_x, world_y))
                if not current_region:
                    continue

                # Only apply spillover to indoor areas (low sky exposure)
                if current_region.sky_exposure > 0.1:
                    continue  # Skip outdoor areas

                # Skip walls - they should block light, not transmit it
                from catley.environment.tile_types import TileTypeID

                tile_id = game_map.tiles[world_x, world_y]
                if tile_id in (
                    TileTypeID.DOOR_CLOSED,
                    TileTypeID.WALL,
                    TileTypeID.OUTDOOR_WALL,
                ):
                    continue

                # Check immediate neighbors for bright outdoor areas
                found_bright_neighbor = False
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    neighbor_x = world_x + dx
                    neighbor_y = world_y + dy

                    if (
                        0 <= neighbor_x < game_map.width
                        and 0 <= neighbor_y < game_map.height
                    ):
                        neighbor_region = game_map.get_region_at(
                            (neighbor_x, neighbor_y)
                        )
                        if neighbor_region and neighbor_region.sky_exposure > 0.8:
                            found_bright_neighbor = True
                            break

                # Apply basic spillover if adjacent to bright outdoor area
                if found_bright_neighbor:
                    # Simple spillover: add 20% of sun intensity
                    sun_color = (
                        np.array(self.config.sun_color, dtype=np.float32) / 255.0
                    )
                    spillover_intensity = self.config.sun_intensity * 0.2
                    spillover_contribution = sun_color * spillover_intensity

                    light_map[x, y] = np.clip(
                        light_map[x, y] + spillover_contribution, 0.0, 1.0
                    )

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

        # Apply point light shadows
        for light_source in lights:
            from catley.game.lights import DirectionalLight

            # Skip directional lights for point light shadow logic
            if isinstance(light_source, DirectionalLight):
                continue

            lx, ly = light_source.position
            light_radius = light_source.radius

            # Skip shadow casting if light source is heavily reduced
            # Check daylight reduction at the light source position for early exit
            light_world_x, light_world_y = light_source.position
            light_source_reduction = self._calculate_tile_daylight_reduction(
                light_source, light_world_x, light_world_y
            )

            if light_source_reduction < 0.3:  # If torch is less than 30% effective
                continue  # Skip shadow casting for very weak torches in daylight

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
                        # Calculate daylight reduction for this specific shadow tile
                        shadow_world_x = sx + offset_x
                        shadow_world_y = sy + offset_y
                        shadow_tile_reduction = self._calculate_tile_daylight_reduction(
                            light_source, shadow_world_x, shadow_world_y
                        )
                        # Apply per-tile daylight reduction to shadow intensity
                        reduced_intensity = intensity * shadow_tile_reduction
                        shadow_map[sx, sy] *= 1.0 - reduced_intensity

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
                        # Calculate daylight reduction for this specific shadow tile
                        shadow_world_x = sx + offset_x
                        shadow_world_y = sy + offset_y
                        shadow_tile_reduction = self._calculate_tile_daylight_reduction(
                            light_source, shadow_world_x, shadow_world_y
                        )
                        # Apply per-tile daylight reduction to shadow intensity
                        reduced_intensity = intensity * shadow_tile_reduction
                        shadow_map[sx, sy] *= 1.0 - reduced_intensity

        # Apply directional light shadows
        return self._apply_directional_shadows(shadow_map, offset_x, offset_y)

    def _apply_directional_shadows(
        self, light_map: np.ndarray, offset_x: TileCoord, offset_y: TileCoord
    ) -> np.ndarray:
        """Apply shadows from directional lights (sun/moon)."""
        from catley.game.lights import DirectionalLight

        shadow_map = light_map.copy()

        # Find all directional lights
        directional_lights = [
            light
            for light in self.game_world.lights
            if isinstance(light, DirectionalLight) and light.intensity > 0
        ]

        if not directional_lights:
            return shadow_map

        # For simplicity, use the strongest directional light (usually the sun)
        primary_light = max(directional_lights, key=lambda light: light.intensity)

        # Get viewport bounds for actor search
        viewport_width, viewport_height = light_map.shape[0], light_map.shape[1]
        world_x1, world_y1 = offset_x, offset_y
        world_x2 = world_x1 + viewport_width - 1
        world_y2 = world_y1 + viewport_height - 1

        # Get all shadow-casting actors in viewport
        shadow_casting_actors = [
            actor
            for actor in self.game_world.actor_spatial_index.get_in_bounds(
                world_x1, world_y1, world_x2, world_y2
            )
            if actor.blocks_movement
        ]

        # Cast directional shadows from each actor
        for actor in shadow_casting_actors:
            # Only cast shadows in outdoor areas (areas with sky exposure)
            game_map = self.game_world.game_map
            if not game_map:
                continue

            region = game_map.get_region_at((actor.x, actor.y))
            if not region or region.sky_exposure <= 0.1:
                continue  # No directional shadows indoors

            ax_vp, ay_vp = actor.x - offset_x, actor.y - offset_y

            if not (0 <= ax_vp < viewport_width and 0 <= ay_vp < viewport_height):
                continue

            # Cast shadow in the direction of the light
            shadow_data = self._cast_directional_shadow(
                primary_light, (ax_vp, ay_vp), viewport_width, viewport_height
            )

            for sx, sy, intensity in shadow_data:
                if 0 <= sx < viewport_width and 0 <= sy < viewport_height:
                    shadow_map[sx, sy] *= 1.0 - intensity

        # Cast shadows from shadow-casting tiles in outdoor areas
        if self.game_world.game_map:
            for x in range(viewport_width):
                for y in range(viewport_height):
                    world_x, world_y = x + offset_x, y + offset_y

                    if not (
                        0 <= world_x < self.game_world.game_map.width
                        and 0 <= world_y < self.game_world.game_map.height
                    ):
                        continue

                    # Check if this tile casts shadows
                    from catley.environment import tile_types

                    tile_id = self.game_world.game_map.tiles[world_x, world_y]
                    tile_data = tile_types.get_tile_type_data_by_id(int(tile_id))

                    if not tile_data["casts_shadows"]:
                        continue

                    # Only cast shadows in outdoor areas
                    region = self.game_world.game_map.get_region_at((world_x, world_y))
                    if not region or region.sky_exposure <= 0.1:
                        continue

                    # Cast shadow in the direction of the light
                    shadow_data = self._cast_directional_shadow(
                        primary_light, (x, y), viewport_width, viewport_height
                    )

                    for sx, sy, intensity in shadow_data:
                        if 0 <= sx < viewport_width and 0 <= sy < viewport_height:
                            shadow_map[sx, sy] *= 1.0 - intensity

        return shadow_map

    def _cast_directional_shadow(
        self,
        directional_light,
        caster_pos: tuple[int, int],
        viewport_width: int,
        viewport_height: int,
    ) -> list[tuple[int, int, float]]:
        """Cast a shadow from a caster in the direction of a directional light."""
        from catley.game.lights import DirectionalLight

        if not isinstance(directional_light, DirectionalLight):
            return []

        cx, cy = caster_pos

        # Get shadow direction (opposite of light direction)
        shadow_dx = -directional_light.direction.x
        shadow_dy = -directional_light.direction.y

        # Normalize shadow direction for consistent shadow length
        magnitude = (shadow_dx * shadow_dx + shadow_dy * shadow_dy) ** 0.5
        if magnitude == 0:
            return []

        shadow_dx /= magnitude
        shadow_dy /= magnitude

        shadow_data = []
        shadow_length = min(
            SHADOW_MAX_LENGTH, 8
        )  # Reasonable length for directional shadows
        base_intensity = SHADOW_INTENSITY * 0.8  # Slightly weaker than torch shadows

        for i in range(1, shadow_length + 1):
            # Calculate shadow position
            shadow_x = cx + int(shadow_dx * i)
            shadow_y = cy + int(shadow_dy * i)

            # Check bounds
            if not (0 <= shadow_x < viewport_width and 0 <= shadow_y < viewport_height):
                break

            # Falloff with distance
            distance_falloff = 1.0 - (i - 1) / shadow_length if SHADOW_FALLOFF else 1.0
            intensity = base_intensity * distance_falloff

            shadow_data.append((shadow_x, shadow_y, intensity))

            # Add softer edges for first few shadow tiles
            if i <= 2:
                edge_intensity = intensity * 0.3
                edge_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

                for ox, oy in edge_offsets:
                    edge_x, edge_y = shadow_x + ox, shadow_y + oy
                    if (
                        0 <= edge_x < viewport_width
                        and 0 <= edge_y < viewport_height
                        and (edge_x != cx or edge_y != cy)
                    ):  # Don't shadow the caster itself
                        shadow_data.append((edge_x, edge_y, edge_intensity))

        return shadow_data

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
            light,
            viewport_offset,
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
            viewport_offset,
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
