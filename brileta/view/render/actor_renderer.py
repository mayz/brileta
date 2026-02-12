"""Actor rendering and outline drawing for the world view.

Handles smooth sub-pixel actor positioning, traditional tile-aligned rendering,
character layer composition, and contextual/combat outline drawing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from brileta import colors
from brileta.types import (
    ColorRGBf,
    InterpolationAlpha,
    Opacity,
    ViewOffset,
    WorldTilePos,
)
from brileta.util.coordinates import Rect
from brileta.util.live_vars import record_time_live_variable

from .graphics import GraphicsContext
from .shadow_renderer import ShadowRenderer, compute_actor_screen_position
from .viewport import ViewportSystem

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.environment.map import GameMap
    from brileta.game.actors import Actor
    from brileta.game.actors.core import CharacterLayer
    from brileta.game.game_world import GameWorld


# Rendering effects
PULSATION_PERIOD = 2.0  # Seconds for full pulsation cycle (selected actor)
PULSATION_MAX_BLEND_ALPHA: Opacity = Opacity(0.5)  # Maximum alpha for pulsation
LUMINANCE_THRESHOLD = 127.5  # For determining light vs dark colors

# Combat outline shimmer effect (shimmering glyph outlines on targetable enemies)
COMBAT_OUTLINE_SHIMMER_PERIOD = 2.4  # Seconds for full shimmer cycle
COMBAT_OUTLINE_MIN_ALPHA: Opacity = Opacity(0.4)  # Minimum alpha during shimmer
COMBAT_OUTLINE_MAX_ALPHA: Opacity = Opacity(0.85)  # Maximum alpha during shimmer

# Contextual target outline (exploration mode)
CONTEXTUAL_OUTLINE_ALPHA: Opacity = Opacity(0.70)  # Solid outline opacity


class ActorRenderer:
    """Render actors and actor outlines for a world view.

    Follows the ShadowRenderer pattern: stable references at init, per-frame
    parameters passed as method arguments. No stored per-frame state.
    """

    def __init__(
        self,
        viewport_system: ViewportSystem,
        graphics: GraphicsContext,
        shadow_renderer: ShadowRenderer,
    ) -> None:
        self.viewport_system = viewport_system
        self.graphics = graphics
        self.shadow_renderer = shadow_renderer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_actors(
        self,
        interpolation_alpha: InterpolationAlpha,
        *,
        game_world: GameWorld,
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
        visible_actors: list[Actor] | None = None,
        viewport_bounds: Rect | None = None,
        smooth: bool = True,
        game_time: float = 0.0,
        is_combat: bool = False,
    ) -> None:
        """Main entry point - renders all visible actors.

        Args:
            interpolation_alpha: Interpolation factor for smooth movement.
            game_world: The current game world state.
            camera_frac_offset: Fractional camera offset for smooth scrolling.
            view_origin: Root console origin of the viewport (x, y).
            visible_actors: Pre-filtered list of visible actors, or None to compute.
            viewport_bounds: Viewport bounds for culling, or None to compute.
            smooth: If True, use smooth sub-pixel rendering. Otherwise tile-aligned.
            game_time: Current game time in seconds (for pulsation in traditional mode).
            is_combat: Whether combat mode is active (suppresses pulsation).
        """
        if smooth:
            self._render_actors_smooth(
                interpolation_alpha,
                game_world=game_world,
                camera_frac_offset=camera_frac_offset,
                view_origin=view_origin,
                visible_actors=visible_actors,
                viewport_bounds=viewport_bounds,
            )
        else:
            self._render_actors_traditional(
                interpolation_alpha,
                game_world=game_world,
                view_origin=view_origin,
                game_time=game_time,
                is_combat=is_combat,
            )

    def render_actor_outline(
        self,
        actor: Actor,
        color: colors.Color,
        alpha: Opacity,
        *,
        game_map: GameMap,
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
    ) -> None:
        """Render a glyph-shaped outline for an actor at its current position.

        Used for combat targeting to show a shimmering outline around the
        enemy's glyph shape. The actor must be visible.

        Args:
            actor: The actor to render an outline for.
            color: RGB color for the outline.
            alpha: Opacity of the outline (0.0-1.0).
            game_map: The game map (for visibility checks).
            camera_frac_offset: Fractional camera offset for smooth scrolling.
            view_origin: Root console origin of the viewport (x, y).
        """
        if not game_map.visible[actor.x, actor.y]:
            return

        vs = self.viewport_system
        if not vs.is_visible(actor.x, actor.y):
            return

        # Convert actor position to screen coordinates.
        # Use animation-controlled sub-tile positions when available so outlines
        # stay in sync with moving glyphs.
        if getattr(actor, "_animation_controlled", False):
            vp_x, vp_y = vs.world_to_screen_float(actor.render_x, actor.render_y)
        else:
            vp_x, vp_y = vs.world_to_screen(actor.x, actor.y)

        # Apply camera fractional offset for smooth scrolling alignment
        cam_frac_x, cam_frac_y = camera_frac_offset
        vp_x -= cam_frac_x
        vp_y -= cam_frac_y

        root_x = view_origin[0] + vp_x
        root_y = view_origin[1] + vp_y
        screen_x, screen_y = self.graphics.console_to_screen_coords(root_x, root_y)

        visual_scale = getattr(actor, "visual_scale", 1.0)

        self.graphics.draw_actor_outline(
            actor.ch,
            screen_x,
            screen_y,
            color,
            alpha,
            scale_x=visual_scale,
            scale_y=visual_scale,
        )

    def render_selection_and_hover_outlines(
        self,
        *,
        game_world: GameWorld,
        controller: Controller,
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
    ) -> None:
        """Render contextual outlines for selected/hovered actors.

        Outlines are rendered in priority order:
        1. selected_target (golden) - sticky click-to-select
        2. hovered_actor (subtle grey) - visual feedback only

        Called by explore mode.

        Args:
            game_world: The current game world state.
            controller: The game controller (for selection/hover state).
            camera_frac_offset: Fractional camera offset for smooth scrolling.
            view_origin: Root console origin of the viewport (x, y).
        """
        if controller.is_combat_mode():
            return

        game_map = game_world.game_map

        # Priority 1: Render selected target outline (golden)
        selected = controller.selected_target
        if (
            selected is not None
            and selected in game_world.actors
            and game_map.visible[selected.x, selected.y]
        ):
            self._draw_actor_outline(
                selected,
                colors.SELECTION_OUTLINE,
                CONTEXTUAL_OUTLINE_ALPHA,
                game_map=game_map,
                camera_frac_offset=camera_frac_offset,
                view_origin=view_origin,
            )
            return  # Don't also render hover outline for same actor

        # Priority 2: Render hover outline (white)
        hovered = controller.hovered_actor
        if hovered is None or hovered not in game_world.actors:
            return
        if not game_map.visible[hovered.x, hovered.y]:
            return
        self._draw_actor_outline(
            hovered,
            colors.HOVER_OUTLINE,
            Opacity(0.50),
            game_map=game_map,
            camera_frac_offset=camera_frac_offset,
            view_origin=view_origin,
        )

    def get_shimmer_alpha(
        self,
        game_time: float,
        period: float = COMBAT_OUTLINE_SHIMMER_PERIOD,
    ) -> Opacity:
        """Calculate oscillating alpha for shimmer effect.

        Returns an alpha value that smoothly oscillates between configured
        min and max values over the specified period, creating a breathing
        or pulsing visual effect.

        Args:
            game_time: Current game time in seconds.
            period: Duration in seconds for one complete oscillation cycle.

        Returns:
            Alpha value between COMBAT_OUTLINE_MIN_ALPHA and COMBAT_OUTLINE_MAX_ALPHA.
        """
        t = (game_time % period) / period
        # Sinusoidal oscillation from min to max alpha
        normalized = (math.sin(t * 2 * math.pi) + 1) / 2
        return Opacity(
            COMBAT_OUTLINE_MIN_ALPHA
            + (COMBAT_OUTLINE_MAX_ALPHA - COMBAT_OUTLINE_MIN_ALPHA) * normalized
        )

    def get_sorted_visible_actors(
        self, bounds: Rect, game_world: GameWorld
    ) -> list[Actor]:
        """Return actors in the viewport sorted for painter-style rendering.

        Sorted by Y position (painter's algorithm), then visual_scale (larger on
        top at same Y), then player last (always on top at same Y and scale).
        """
        actors_in_viewport = game_world.actor_spatial_index.get_in_bounds(
            bounds.x1, bounds.y1, bounds.x2, bounds.y2
        )

        return sorted(
            actors_in_viewport,
            key=lambda actor: (
                actor.y,
                getattr(actor, "visual_scale", 1.0),
                actor == game_world.player,
            ),
        )

    def apply_pulsating_effect(
        self,
        input_color: colors.Color,
        base_actor_color: colors.Color,
        game_time: float,
    ) -> colors.Color:
        """Apply a pulsating color blend effect.

        Blends the input color toward a contrast color (light or dark depending
        on the actor's luminance) using a sinusoidal oscillation over time.

        Args:
            input_color: The current display color.
            base_actor_color: The actor's base color (for luminance calculation).
            game_time: Current game time in seconds.

        Returns:
            The blended color.
        """
        alpha_oscillation = (
            math.sin((game_time % PULSATION_PERIOD) / PULSATION_PERIOD * 2 * math.pi)
            + 1
        ) / 2.0
        current_blend_alpha = alpha_oscillation * PULSATION_MAX_BLEND_ALPHA
        r, g, b = base_actor_color
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        target_color = (
            colors.DARK_GREY if luminance > LUMINANCE_THRESHOLD else colors.LIGHT_GREY
        )
        blended = [
            int(
                target_color[i] * current_blend_alpha
                + input_color[i] * (1.0 - current_blend_alpha)
            )
            for i in range(3)
        ]
        return (
            max(0, min(255, blended[0])),
            max(0, min(255, blended[1])),
            max(0, min(255, blended[2])),
        )

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _get_actor_screen_position(
        self,
        actor: Actor,
        interpolation_alpha: InterpolationAlpha,
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute interpolated screen coordinates for an actor."""
        return compute_actor_screen_position(
            actor=actor,
            graphics=self.graphics,
            viewport_system=self.viewport_system,
            interpolation_alpha=interpolation_alpha,
            camera_frac_offset=camera_frac_offset,
            view_origin=view_origin,
        )

    @record_time_live_variable("time.render.actors_smooth_ms")
    def _render_actors_smooth(
        self,
        alpha: InterpolationAlpha,
        *,
        game_world: GameWorld,
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
        visible_actors: list[Actor] | None = None,
        viewport_bounds: Rect | None = None,
    ) -> None:
        """Render all actors with smooth sub-pixel positioning."""
        vs = self.viewport_system
        if viewport_bounds is None:
            viewport_bounds = vs.get_visible_bounds()

        if visible_actors is None:
            visible_actors = [
                actor
                for actor in self.get_sorted_visible_actors(viewport_bounds, game_world)
                if game_world.game_map.visible[actor.x, actor.y]
            ]

        if not visible_actors:
            return

        game_map = game_world.game_map
        for actor in visible_actors:
            self._render_single_actor_smooth(
                actor,
                viewport_bounds,
                vs,
                alpha,
                game_map=game_map,
                camera_frac_offset=camera_frac_offset,
                view_origin=view_origin,
            )

    def _render_single_actor_smooth(
        self,
        actor: Actor,
        bounds: Rect,
        vs: ViewportSystem,
        interpolation_alpha: InterpolationAlpha,
        *,
        game_map: GameMap,
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
    ) -> None:
        """Render a single actor with smooth positioning and lighting.

        Uses linear interpolation between the actor's previous position (from last step)
        and current position (from current logic step) to create fluid movement that's
        independent of visual framerate.

        Args:
            actor: The actor to render.
            bounds: Viewport bounds for culling.
            vs: Viewport system for coordinate conversion.
            interpolation_alpha: Interpolation factor (0.0=previous, 1.0=current).
            game_map: The game map (for per-tile background color sampling).
            camera_frac_offset: Fractional camera offset for smooth scrolling.
            view_origin: Root console origin of the viewport (x, y).
        """
        # Get lighting intensity (reuse existing lighting logic)
        light_rgb = self._get_actor_lighting_intensity(actor, bounds)
        _, _, root_x, root_y, screen_pixel_x, screen_pixel_y = (
            self._get_actor_screen_position(
                actor, interpolation_alpha, camera_frac_offset, view_origin
            )
        )

        # Get actor color with visual effects (reuse existing logic)
        final_color = self._get_actor_display_color(actor)
        visual_scale = getattr(actor, "visual_scale", 1.0)

        # Check for multi-character composition (character_layers)
        if actor.character_layers:
            # Render each layer at its sub-tile offset
            self._render_character_layers(
                actor.character_layers,
                root_x,
                root_y,
                light_rgb,
                interpolation_alpha,
                visual_scale,
                actor_world_pos=(actor.x, actor.y),
            )
        else:
            # Send tile background to GPU shader for actor-vs-tile contrast checks.
            tile_bg_np = game_map.light_appearance_map[actor.x, actor.y]["bg"]
            tile_bg = (
                int(tile_bg_np[0]),
                int(tile_bg_np[1]),
                int(tile_bg_np[2]),
            )

            # Render single character (existing behavior) - uniform scaling
            self.graphics.draw_actor_smooth(
                actor.ch,
                final_color,
                screen_pixel_x,
                screen_pixel_y,
                light_rgb,
                interpolation_alpha,
                scale_x=visual_scale,
                scale_y=visual_scale,
                world_pos=(actor.x, actor.y),
                tile_bg=tile_bg,
            )

    def _render_character_layers(
        self,
        layers: list[CharacterLayer],
        root_x: float,
        root_y: float,
        light_rgb: ColorRGBf,
        interpolation_alpha: InterpolationAlpha,
        visual_scale: float,
        actor_world_pos: WorldTilePos,
    ) -> None:
        """Render multiple character layers at sub-tile offsets.

        Each layer is rendered at its offset position relative to the actor's
        center, creating a rich visual composition from multiple ASCII characters.

        Args:
            layers: List of CharacterLayer defining the composition.
            root_x: Base X position in root console coordinates.
            root_y: Base Y position in root console coordinates.
            light_rgb: Lighting intensity tuple.
            interpolation_alpha: Interpolation factor for smooth rendering.
            visual_scale: Base scale factor for the actor.
            actor_world_pos: World tile position of the actor.
        """
        graphics = self.graphics
        for layer in layers:
            # Calculate this layer's position by adding its offset to the base position
            layer_x = root_x + layer.offset_x
            layer_y = root_y + layer.offset_y

            # Convert to screen pixel coordinates
            pixel_x, pixel_y = graphics.console_to_screen_coords(layer_x, layer_y)

            # Combine actor scale with per-layer scale (non-uniform)
            combined_scale_x = visual_scale * layer.scale_x
            combined_scale_y = visual_scale * layer.scale_y

            # Render this layer
            graphics.draw_actor_smooth(
                layer.char,
                layer.color,
                pixel_x,
                pixel_y,
                light_rgb,
                interpolation_alpha,
                scale_x=combined_scale_x,
                scale_y=combined_scale_y,
                world_pos=actor_world_pos,
            )

    def _get_actor_lighting_intensity(self, actor: Actor, _bounds: Rect) -> ColorRGBf:
        """Get actor lighting multiplier tuple for the screen shader path."""
        receive_scale = self.shadow_renderer.actor_shadow_receive_light_scale.get(
            actor, 1.0
        )
        return (receive_scale, receive_scale, receive_scale)

    def _get_actor_display_color(self, actor: Actor) -> colors.Color:
        """Get actor's final display color with visual effects.

        This applies flash effects (from damage, etc.) when active.
        """
        base_color = actor.color

        # Apply visual effects if present
        visual_effects = actor.visual_effects
        if visual_effects is not None:
            visual_effects.update()

            # Flash effect overrides base color (e.g., damage flash)
            flash_color = visual_effects.get_flash_color()
            if flash_color:
                return flash_color

        return base_color

    @record_time_live_variable("time.render.actors_traditional_ms")
    def _render_actors_traditional(
        self,
        alpha: InterpolationAlpha,
        *,
        game_world: GameWorld,
        view_origin: ViewOffset,
        game_time: float = 0.0,
        is_combat: bool = False,
    ) -> None:
        """Tile-aligned actor rendering, adapted for dynamic rendering."""
        vs = self.viewport_system
        bounds = vs.get_visible_bounds()
        world_left, world_right, world_top, world_bottom = (
            bounds.x1,
            bounds.x2,
            bounds.y1,
            bounds.y2,
        )
        # Get only actors within the viewport using the spatial index, then sort
        # for proper z-order: Y-position primary (painter's algorithm),
        # visual_scale secondary (larger actors on top at same Y), player on top
        actors_in_viewport = game_world.actor_spatial_index.get_in_bounds(
            world_left, world_top, world_right, world_bottom
        )
        sorted_actors = sorted(
            actors_in_viewport,
            key=lambda a: (
                a.y,
                getattr(a, "visual_scale", 1.0),
                a == game_world.player,
            ),
        )
        graphics = self.graphics
        for actor in sorted_actors:
            if game_world.game_map.visible[actor.x, actor.y]:
                # Get lighting intensity
                light_rgb = self._get_actor_lighting_intensity(actor, bounds)

                # Convert actor's TILE position to viewport coordinates
                # Note: We use actor.x/y directly for tile-aligned rendering
                vp_x, vp_y = vs.world_to_screen(actor.x, actor.y)

                # Root console position where this viewport tile ends up
                root_x = view_origin[0] + vp_x
                root_y = view_origin[1] + vp_y

                # Convert to final screen pixel coordinates
                screen_pixel_x, screen_pixel_y = graphics.console_to_screen_coords(
                    root_x, root_y
                )

                # Get final color with pulsating effect if needed
                base_actor_color = self._get_actor_display_color(actor)
                final_fg_color = base_actor_color
                if (
                    game_world.selected_actor == actor
                    and game_world.game_map.visible[actor.x, actor.y]
                    and not is_combat
                ):
                    final_fg_color = self.apply_pulsating_effect(
                        base_actor_color, actor.color, game_time
                    )

                visual_scale = getattr(actor, "visual_scale", 1.0)

                # Check for multi-character composition (character_layers)
                if actor.character_layers:
                    # Render each layer at its sub-tile offset
                    self._render_character_layers(
                        actor.character_layers,
                        float(root_x),
                        float(root_y),
                        light_rgb,
                        alpha,
                        visual_scale,
                        actor_world_pos=(actor.x, actor.y),
                    )
                else:
                    # Send tile background to GPU shader for contrast checks.
                    tile_bg_np = game_world.game_map.light_appearance_map[
                        actor.x, actor.y
                    ]["bg"]
                    tile_bg = (
                        int(tile_bg_np[0]),
                        int(tile_bg_np[1]),
                        int(tile_bg_np[2]),
                    )

                    # Render using the renderer's smooth drawing function
                    graphics.draw_actor_smooth(
                        actor.ch,
                        final_fg_color,
                        screen_pixel_x,
                        screen_pixel_y,
                        light_rgb,
                        alpha,
                        scale_x=visual_scale,
                        scale_y=visual_scale,
                        world_pos=(actor.x, actor.y),
                        tile_bg=tile_bg,
                    )

    def _draw_actor_outline(
        self,
        actor: Actor,
        color: colors.Color,
        alpha: Opacity,
        *,
        game_map: GameMap,
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
    ) -> None:
        """Draw an outline around an actor, handling content layers properly.

        If the actor has content layers (multi-character composition like bookcase),
        outlines the entire tile. Otherwise, outlines the glyph shape.

        Args:
            actor: The actor to outline.
            color: RGB color for the outline.
            alpha: Opacity of the outline (0.0-1.0).
            game_map: The game map (for visibility checks).
            camera_frac_offset: Fractional camera offset for smooth scrolling.
            view_origin: Root console origin of the viewport (x, y).
        """
        if actor.character_layers or actor.has_complex_visuals:
            self._render_layered_tile_outline(
                actor, color, alpha, view_origin=view_origin
            )
        else:
            self.render_actor_outline(
                actor,
                color,
                alpha,
                game_map=game_map,
                camera_frac_offset=camera_frac_offset,
                view_origin=view_origin,
            )

    def _render_layered_tile_outline(
        self,
        actor: Actor,
        color: colors.Color,
        alpha: Opacity,
        *,
        view_origin: ViewOffset,
    ) -> None:
        """Render a full-tile rectangle outline for actors with complex visuals."""
        vs = self.viewport_system
        if not vs.is_visible(actor.x, actor.y):
            return

        vp_x, vp_y = vs.world_to_screen(actor.x, actor.y)
        root_x = view_origin[0] + vp_x
        root_y = view_origin[1] + vp_y
        screen_x, screen_y = self.graphics.console_to_screen_coords(root_x, root_y)

        tile_w, tile_h = self.graphics.tile_dimensions
        self.graphics.draw_rect_outline(
            int(screen_x),
            int(screen_y),
            int(tile_w),
            int(tile_h),
            color,
            alpha,
        )
