"""
Decal System for Persistent Visual Effects

This module provides a system for creating persistent tile-based visual marks
like blood splatters that remain on the floor after combat. Unlike particles
which are ephemeral and fade quickly, decals persist for an extended period
before gradually fading away.

Design Decisions:
    - Tile-based: Decals snap to tile coordinates (not sub-tile like particles)
    - Sparse storage: Only tiles with decals consume memory
    - LRU eviction: Oldest decals removed when hitting global limits
    - Age-based fade: Decals fade after DECAL_LIFETIME seconds

Usage Example:
    # In an effect that creates blood splatter
    if context.decal_system:
        context.decal_system.add_splatter_cone(
            target_x, target_y,
            direction_x, direction_y,
            intensity=0.8,
            colors_and_chars=[(120, 0, 0), "*"), ((80, 0, 0), ".")]
        )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from catley import colors
from catley.util import rng

_rng = rng.get("effects.decals")

if TYPE_CHECKING:
    pass


@dataclass
class Decal:
    """A single persistent decal mark with sub-tile precision.

    Attributes:
        x: X coordinate in world space (float for sub-tile precision)
        y: Y coordinate in world space (float for sub-tile precision)
        char: ASCII character to display
        color: RGB color tuple
        created_at: Game time when this decal was created (for age-based cleanup)
    """

    x: float
    y: float
    char: str
    color: colors.Color
    created_at: float


@dataclass
class DecalSystem:
    """Manages persistent sub-tile decals like blood splatters.

    The system stores decals with sub-tile (float) coordinates for precise
    visual placement. It enforces a global limit to prevent unbounded growth
    and provides age-based fade-out.

    Attributes:
        MAX_TOTAL_DECALS: Maximum decals across all tiles
        DECAL_LIFETIME: Seconds before fade-out begins
        FADE_DURATION: Duration of fade-out in seconds
    """

    MAX_TOTAL_DECALS: int = 200  # Smaller cap to prevent visual clutter
    DECAL_LIFETIME: float = 60.0  # 1 minute before fade starts
    FADE_DURATION: float = 30.0  # 30 second fade-out
    BASE_ALPHA: float = 0.5  # Lower opacity so decals don't obscure gameplay

    # Ray-based splatter generation parameters
    CONE_SPREAD: float = math.pi / 3  # 60 degrees total spread
    MIN_RAY_COUNT: int = 1  # Now controlled by ray_count parameter
    MAX_RAY_COUNT: int = 4  # Now controlled by ray_count parameter
    MIN_DECALS_PER_RAY: int = 2
    MAX_DECALS_PER_RAY: int = 3  # Fewer decals per ray for cleaner streaks
    # Distance range for splatter rays (in tiles from target)
    MIN_DISTANCE: float = 0.5  # Start slightly away from impact
    MAX_DISTANCE_BASE: float = 2.0  # Base max ray length
    MAX_DISTANCE_INTENSITY_MULT: float = 2.0  # Extra distance per intensity

    # Legacy cone generation parameters (kept for compatibility)
    MIN_DECAL_COUNT: int = 3
    MAX_DECAL_COUNT: int = 10

    # Storage: flat list of all decals (sub-tile positions)
    decals: dict[tuple[int, int], list[Decal]] = field(default_factory=dict)
    # Track total count for efficient limit checking
    _total_count: int = field(default=0, repr=False)
    # Track creation order for LRU eviction
    _creation_order: list[Decal] = field(default_factory=list, repr=False)
    # Current game time (updated each frame)
    _game_time: float = field(default=0.0, repr=False)

    def add_decal(
        self,
        x: float,
        y: float,
        char: str,
        color: colors.Color,
        game_time: float | None = None,
    ) -> None:
        """Add a single decal at the specified position with sub-tile precision.

        Decals are keyed by their integer tile position for efficient spatial
        queries, but store float coordinates for sub-tile rendering.

        Args:
            x: X coordinate (float for sub-tile precision)
            y: Y coordinate (float for sub-tile precision)
            char: ASCII character to display
            color: RGB color tuple
            game_time: Optional game time; uses internal time if not provided
        """
        # Key by integer tile for spatial queries
        key = (int(x), int(y))
        creation_time = game_time if game_time is not None else self._game_time

        # Initialize tile entry if needed
        if key not in self.decals:
            self.decals[key] = []

        # Global limit: evict oldest decals until under limit
        while self._total_count >= self.MAX_TOTAL_DECALS:
            self._evict_oldest()

        # Create and store the new decal with sub-tile position
        decal = Decal(x=x, y=y, char=char, color=color, created_at=creation_time)
        self.decals[key].append(decal)
        self._creation_order.append(decal)
        self._total_count += 1

    def add_splatter_cone(
        self,
        target_x: int,
        target_y: int,
        direction_x: float,
        direction_y: float,
        intensity: float,
        colors_and_chars: list[tuple[colors.Color, str]],
        game_time: float | None = None,
    ) -> None:
        """Create a cone-shaped blood splatter behind the hit target.

        The splatter projects OPPOSITE to the attack direction, simulating
        blood flying away from the impact. The cone's size and density
        scale with the intensity parameter.

        Args:
            target_x: X coordinate of the hit target
            target_y: Y coordinate of the hit target
            direction_x: X component of attack direction (from attacker to target)
            direction_y: Y component of attack direction (from attacker to target)
            intensity: Damage intensity (0.0 to 1.0+), affects size and density
            colors_and_chars: List of (color, char) tuples to randomly choose from
            game_time: Optional game time for decal creation
        """
        if not colors_and_chars:
            return

        # Splatter continues in the attack direction - blood exits the FAR side
        # of the target, away from the attacker. The direction vector points
        # FROM attacker TO target, so blood continues in that same direction.
        splat_dir_x = direction_x
        splat_dir_y = direction_y

        # Handle zero direction (shouldn't happen but be safe)
        if splat_dir_x == 0 and splat_dir_y == 0:
            splat_dir_x = 1.0  # Default to right

        # Base angle of the splatter direction
        base_angle = math.atan2(splat_dir_y, splat_dir_x)

        # Parameters scale with intensity
        max_distance = (
            self.MAX_DISTANCE_BASE + intensity * self.MAX_DISTANCE_INTENSITY_MULT
        )
        decal_count = int(
            self.MIN_DECAL_COUNT
            + intensity * (self.MAX_DECAL_COUNT - self.MIN_DECAL_COUNT)
        )

        for _ in range(decal_count):
            # Random angle within the cone spread
            angle = base_angle + _rng.uniform(
                -self.CONE_SPREAD / 2, self.CONE_SPREAD / 2
            )

            # Random distance - uniform distribution between min and max
            # This ensures decals land BEHIND the target, forming a visible cone
            distance = _rng.uniform(self.MIN_DISTANCE, max_distance)

            # Calculate tile offset
            dx = round(math.cos(angle) * distance)
            dy = round(math.sin(angle) * distance)

            # Random color and character from provided options
            color, char = _rng.choice(colors_and_chars)
            # Vary the color slightly for natural look
            varied_color = self._vary_color(color)

            self.add_decal(target_x + dx, target_y + dy, char, varied_color, game_time)

    def add_splatter_rays(
        self,
        target_x: float,
        target_y: float,
        direction_x: float,
        direction_y: float,
        intensity: float,
        colors_and_chars: list[tuple[colors.Color, str]],
        game_time: float | None = None,
        ray_count: int | None = None,
    ) -> None:
        """Create ray-based blood splatter with sub-tile precision.

        Generates discrete rays radiating from the impact point, with multiple
        decals placed along each ray at sub-tile positions. This creates thin
        streaks rather than scattered marks, producing a more visceral "exit
        wound" splatter effect.

        Args:
            target_x: X coordinate of impact point (can be float)
            target_y: Y coordinate of impact point (can be float)
            direction_x: X component of attack direction (from attacker to target)
            direction_y: Y component of attack direction (from attacker to target)
            intensity: Damage intensity (0.0 to 1.0+), affects ray length
            colors_and_chars: List of (color, char) tuples to randomly choose from
            game_time: Optional game time for decal creation
            ray_count: Number of rays (1 for pistol, 3 for shotgun). Defaults to 1.
        """
        if not colors_and_chars:
            return

        # Splatter continues in the attack direction
        splat_dir_x = direction_x
        splat_dir_y = direction_y

        # Handle zero direction
        if splat_dir_x == 0 and splat_dir_y == 0:
            splat_dir_x = 1.0

        # Base angle of the splatter direction
        base_angle = math.atan2(splat_dir_y, splat_dir_x)

        # Number of rays: explicit count or default to 1
        num_rays = ray_count if ray_count is not None else 1

        # Maximum ray length scales with intensity
        max_ray_length = (
            self.MAX_DISTANCE_BASE + intensity * self.MAX_DISTANCE_INTENSITY_MULT
        )

        # Streak characters - smaller chars for the ray, occasional larger droplets
        streak_chars = [".", ".", ",", "'"]
        droplet_chars = ["*", ";"]

        for _ in range(num_rays):
            # Each ray has a slightly different angle within the cone spread
            ray_angle = base_angle + _rng.uniform(
                -self.CONE_SPREAD / 2, self.CONE_SPREAD / 2
            )

            # Random ray length
            ray_length = _rng.uniform(self.MIN_DISTANCE, max_ray_length)

            # Number of decals along this ray
            num_decals_on_ray = _rng.randint(
                self.MIN_DECALS_PER_RAY, self.MAX_DECALS_PER_RAY
            )

            for i in range(num_decals_on_ray):
                # Position along ray with slight randomness to avoid perfect alignment
                # t ranges from ~0.3 to ~0.7 of each segment
                t = (i + _rng.uniform(0.3, 0.7)) / num_decals_on_ray
                distance = self.MIN_DISTANCE + t * (ray_length - self.MIN_DISTANCE)

                # Sub-tile position (float coordinates)
                x = target_x + math.cos(ray_angle) * distance
                y = target_y + math.sin(ray_angle) * distance

                # Choose character - mostly small streaks, occasionally larger drops
                if _rng.random() < 0.15:
                    char = _rng.choice(droplet_chars)
                else:
                    char = _rng.choice(streak_chars)

                # Get color from provided options and vary slightly
                color, _ = _rng.choice(colors_and_chars)
                varied_color = self._vary_color(color)

                self.add_decal(x, y, char, varied_color, game_time)

    def update(self, delta_time: float, game_time: float) -> None:
        """Update the system and remove fully expired decals.

        Args:
            delta_time: Time since last update (unused, kept for API consistency)
            game_time: Current game time for age calculations
        """
        self._game_time = game_time
        max_age = self.DECAL_LIFETIME + self.FADE_DURATION

        # Find and remove expired decals
        keys_to_remove: list[tuple[int, int]] = []
        for key, decal_list in self.decals.items():
            # Filter out expired decals
            remaining = [d for d in decal_list if (game_time - d.created_at) < max_age]
            removed_count = len(decal_list) - len(remaining)

            if removed_count > 0:
                # Update creation order list
                for d in decal_list:
                    if d not in remaining and d in self._creation_order:
                        self._creation_order.remove(d)
                self._total_count -= removed_count

            if remaining:
                self.decals[key] = remaining
            else:
                keys_to_remove.append(key)

        # Clean up empty tile entries
        for key in keys_to_remove:
            del self.decals[key]

    def get_alpha(self, decal: Decal, game_time: float) -> float:
        """Calculate the alpha (opacity) for a decal based on its age.

        Args:
            decal: The decal to check
            game_time: Current game time

        Returns:
            Alpha value from 0.0 (fully faded) to BASE_ALPHA (max opacity)
        """
        age = game_time - decal.created_at
        if age < self.DECAL_LIFETIME:
            return self.BASE_ALPHA
        fade_progress = (age - self.DECAL_LIFETIME) / self.FADE_DURATION
        return max(0.0, self.BASE_ALPHA * (1.0 - fade_progress))

    def get_decals_at(self, x: int, y: int) -> list[Decal]:
        """Get all decals at a specific tile position.

        Args:
            x: Tile X coordinate
            y: Tile Y coordinate

        Returns:
            List of decals at this position, or empty list if none
        """
        return self.decals.get((x, y), [])

    def clear(self) -> None:
        """Clear all decals. Call this on map change."""
        self.decals.clear()
        self._creation_order.clear()
        self._total_count = 0

    def _evict_oldest(self) -> None:
        """Remove the globally oldest decal to make room for new ones."""
        if not self._creation_order:
            return

        oldest = self._creation_order.pop(0)
        # Key is based on integer tile position
        key = (int(oldest.x), int(oldest.y))

        if key in self.decals:
            try:
                self.decals[key].remove(oldest)
                self._total_count -= 1
                # Clean up empty entries
                if not self.decals[key]:
                    del self.decals[key]
            except ValueError:
                # Decal already removed (shouldn't happen but be safe)
                pass

    def _vary_color(self, color: colors.Color) -> colors.Color:
        """Slightly vary a color for natural-looking splatter.

        Args:
            color: Base RGB color

        Returns:
            Varied RGB color
        """
        r, g, b = color
        # Small random variation in each channel
        variation = 20
        r = max(0, min(255, r + _rng.randint(-variation, variation)))
        g = max(0, min(255, g + _rng.randint(-variation, variation)))
        b = max(0, min(255, b + _rng.randint(-variation, variation)))
        return (r, g, b)

    @property
    def total_count(self) -> int:
        """Return the total number of decals across all tiles."""
        return self._total_count
