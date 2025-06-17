from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from catley import colors
from catley.game.enums import BlendMode
from catley.util.coordinates import Rect

if TYPE_CHECKING:
    from catley.view.render.renderer import Renderer


@dataclass
class EnvironmentalEffect:
    """
    Represents an area-based environmental effect that modifies rendering.

    Environmental effects are different from particles - they modify the appearance
    of areas rather than being discrete flying objects. Examples include smoke clouds,
    explosion flashes, poison gas, and fog.

    Attributes:
        position: (x, y) center position in tile coordinates
        radius: Effect radius in tiles
        effect_type: Type identifier ("smoke", "explosion_flash", "poison", etc.)
        intensity: Base intensity from 0.0 to 1.0+ (affects opacity and color strength)
        lifetime: Remaining lifetime in seconds (decreases over time)
        tint_color: RGB color to apply to the affected area
        blend_mode: How to blend the effect with existing rendering
        max_lifetime: Original lifetime value (set automatically)

    Example:
        # Create a smoke effect
        smoke = EnvironmentalEffect(
            position=(10.0, 15.0),
            radius=3.0,
            effect_type="smoke",
            intensity=0.8,
            lifetime=5.0,
            tint_color=(100, 100, 100),
            blend_mode=BlendMode.TINT
        )
    """

    position: tuple[float, float]
    radius: float
    effect_type: str
    intensity: float
    lifetime: float
    tint_color: colors.Color
    blend_mode: BlendMode
    max_lifetime: float = field(init=False)

    def __post_init__(self) -> None:
        self.max_lifetime = self.lifetime


class EnvironmentalEffectSystem:
    """
    Manages a collection of active environmental effects.

    This system handles the lifecycle of area-based visual effects that modify
    the appearance of regions rather than individual particles. It automatically
    updates effect lifetimes and culls expired effects.

    Environmental effects are rendered after actors and particles to provide
    atmospheric overlays like smoke, fog, explosion flashes, and gas clouds.

    Example:
        # Initialize system
        env_system = EnvironmentalEffectSystem()

        # Add effects
        smoke = EnvironmentalEffect(...)
        env_system.add_effect(smoke)

        # Update and render each frame
        env_system.update(delta_time)
        env_system.render_effects(renderer, viewport_bounds, panel_offset)
    """

    def __init__(self) -> None:
        """Initialize an empty environmental effect system."""
        self.effects: list[EnvironmentalEffect] = []

    def add_effect(self, effect: EnvironmentalEffect) -> None:
        """
        Add a new environmental effect to the system.

        The effect will be automatically updated and rendered until its
        lifetime expires, at which point it will be removed from the system.

        Args:
            effect: The environmental effect to add
        """
        self.effects.append(effect)

    def update(self, delta_time: float) -> None:
        """
        Update all environmental effects and remove expired ones.

        This method:
        1. Decreases the lifetime of each effect by delta_time
        2. Removes effects whose lifetime has reached zero or below
        3. Preserves the order of remaining effects

        Args:
            delta_time: Time elapsed since last update in seconds
        """
        remaining: list[EnvironmentalEffect] = []
        for effect in self.effects:
            effect.lifetime -= delta_time
            if effect.lifetime > 0:
                remaining.append(effect)
        self.effects = remaining

    def render_effects(
        self,
        renderer: Renderer,
        viewport_bounds: Rect,
        panel_offset: tuple[int, int],
    ) -> None:
        """
        Render all active environmental effects using the renderer.

        This method efficiently culls effects that are outside the viewport
        and renders visible effects with intensity falloff based on remaining
        lifetime. Effects are rendered as circular overlays that modify the
        appearance of the affected areas.

        Args:
            renderer: The renderer to use for drawing effects
            viewport_bounds: The currently visible area in tile coordinates
            panel_offset: Offset to convert viewport coordinates to screen coordinates
                         as (panel_x, panel_y) in tiles

        Performance Notes:
            - Effects completely outside the viewport are skipped (culled)
            - Uses efficient texture-based rendering with circular gradients
            - Intensity automatically fades as effects approach expiration
        """
        for effect in self.effects:
            if effect.lifetime <= 0:
                continue
            x, y = effect.position
            if (
                x + effect.radius < viewport_bounds.x1
                or x - effect.radius > viewport_bounds.x2
                or y + effect.radius < viewport_bounds.y1
                or y - effect.radius > viewport_bounds.y2
            ):
                continue
            intensity = effect.intensity * (effect.lifetime / effect.max_lifetime)
            renderer.apply_environmental_effect(
                (panel_offset[0] + x, panel_offset[1] + y),
                effect.radius,
                effect.tint_color,
                intensity,
                effect.blend_mode,
            )
