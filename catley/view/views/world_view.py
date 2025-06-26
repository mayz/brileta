from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import tcod.sdl.render
from tcod.console import Console

from catley import colors, config
from catley.config import (
    LUMINANCE_THRESHOLD,
    PULSATION_MAX_BLEND_ALPHA,
    PULSATION_PERIOD,
    SELECTION_HIGHLIGHT_ALPHA,
)
from catley.util.caching import ResourceCache
from catley.util.coordinates import (
    PixelCoord,
    Rect,
    RootConsoleTilePos,
)
from catley.util.live_vars import record_time_live_variable
from catley.view.render.effects.effects import EffectLibrary
from catley.view.render.effects.environmental import EnvironmentalEffectSystem
from catley.view.render.effects.particles import (
    ParticleLayer,
    SubTileParticleSystem,
)
from catley.view.render.effects.screen_shake import ScreenShake
from catley.view.render.lighting.base import LightingSystem
from catley.view.render.viewport import ViewportSystem

from .base import View

if TYPE_CHECKING:
    from catley.controller import Controller, FrameManager
    from catley.game.actors import Actor
    from catley.view.render.renderer import Renderer


class WorldView(View):
    """View responsible for rendering the game world (map, actors, effects)."""

    def __init__(
        self,
        controller: Controller,
        screen_shake: ScreenShake,
        lighting_system: LightingSystem | None = None,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.screen_shake = screen_shake
        self.lighting_system = lighting_system
        # Initialize a viewport system sized using configuration defaults.
        # These defaults are replaced once resize() sets the real view bounds.
        self.viewport_system = ViewportSystem(
            config.DEFAULT_VIEWPORT_WIDTH, config.DEFAULT_VIEWPORT_HEIGHT
        )
        # Game map console matches the viewport dimensions rather than the
        # entire map. This keeps rendering fast and memory usage reasonable.
        self.game_map_console = Console(
            config.DEFAULT_VIEWPORT_WIDTH, config.DEFAULT_VIEWPORT_HEIGHT, order="F"
        )
        # Particle system also operates only on the visible viewport area.
        self.particle_system = SubTileParticleSystem(
            config.DEFAULT_VIEWPORT_WIDTH, config.DEFAULT_VIEWPORT_HEIGHT
        )
        self.environmental_system = EnvironmentalEffectSystem()
        self.effect_library = EffectLibrary()
        self.current_light_intensity: np.ndarray | None = None
        self._texture_cache = ResourceCache[tuple, tcod.sdl.render.Texture](
            name="WorldViewCache", max_size=5
        )
        self._active_background_texture: tcod.sdl.render.Texture | None = None
        self._light_overlay_texture: tcod.sdl.render.Texture | None = None

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override set_bounds to update viewport and console dimensions."""
        super().set_bounds(x1, y1, x2, y2)
        # When the window size changes we recreate the viewport and consoles
        # so that rendering operates on the new visible dimensions.
        self.viewport_system = ViewportSystem(self.width, self.height)
        self.game_map_console = Console(self.width, self.height, order="F")
        self.particle_system = SubTileParticleSystem(self.width, self.height)
        self.environmental_system = EnvironmentalEffectSystem()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def highlight_actor(
        self,
        actor: Actor,
        color: colors.Color,
        effect: str = "solid",
        alpha: float = 0.4,
    ) -> None:
        """Highlight an actor if it is visible."""
        if self.controller.gw.game_map.visible[actor.x, actor.y]:
            self.highlight_tile(actor.x, actor.y, color, effect, alpha)

    def highlight_tile(
        self,
        x: int,
        y: int,
        color: colors.Color,
        effect: str = "solid",
        alpha: float = 0.4,
    ) -> None:
        """Highlight a tile with an optional effect using world coordinates."""
        vs = self.viewport_system
        if not vs.is_visible(x, y):
            return
        vp_x, vp_y = vs.world_to_screen(x, y)
        if not (0 <= vp_x < self.width and 0 <= vp_y < self.height):
            return
        final_color = color
        if effect == "pulse":
            final_color = self._apply_pulsating_effect(color, color)
        root_x, root_y = self.x + vp_x, self.y + vp_y
        self.controller.renderer.draw_tile_highlight(root_x, root_y, final_color, alpha)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _get_background_cache_key(self) -> tuple:
        """Generate a hashable key representing the state of the static background."""
        gw = self.controller.gw
        vs = self.viewport_system

        camera_key = (
            round(vs.camera.world_x, 2),
            round(vs.camera.world_y, 2),
            vs.offset_x,
            vs.offset_y,
        )

        map_key = gw.game_map.revision

        return (camera_key, map_key)

    def draw(self, renderer: Renderer, alpha: float) -> None:
        """Main drawing method for the world view."""
        if not self.visible:
            return

        delta_time = self.controller.clock.last_delta_time

        # Update the camera first so the shake is applied on top of the
        # correctly-tracked player position.
        vs = self.viewport_system
        gw = self.controller.gw
        old_cam_x = vs.camera.world_x
        old_cam_y = vs.camera.world_y
        vs.update_camera(gw.player, gw.game_map.width, gw.game_map.height)

        if vs.camera.world_x != old_cam_x or vs.camera.world_y != old_cam_y:
            self._update_mouse_tile_location()

        # Apply screen shake by temporarily offsetting the camera position.
        shake_x, shake_y = self.screen_shake.update(delta_time)
        original_cam_x = vs.camera.world_x
        original_cam_y = vs.camera.world_y
        vs.camera.world_x += shake_x
        vs.camera.world_y += shake_y

        # --- Cache lookup and management ---
        cache_key = self._get_background_cache_key()
        texture = self._texture_cache.get(cache_key)

        if texture is None:
            # CACHE MISS: Re-render the static background
            self._render_map_unlit()  # This populates self.game_map_console

            # Cast the generic renderer to the TCOD-specific one we know we're using in
            # Phase 0 of the Graphics Migration Plan. This is an explicit acknowledgment
            # of the technical debt we will pay off in Phase 2.
            from catley.view.render.backends.tcod.renderer import TCODRenderer

            tcod_renderer = cast(TCODRenderer, renderer)

            texture = tcod_renderer.texture_from_console(self.game_map_console)
            self._texture_cache.store(cache_key, texture)

        self._active_background_texture = texture

        # Generate the light overlay texture every frame for dynamic effects.
        if self.lighting_system is not None:
            self._light_overlay_texture = self._render_light_overlay(renderer)
        else:
            self._light_overlay_texture = None

        # Restore the original camera position so subsequent frames start clean.
        self.viewport_system.camera.set_position(original_cam_x, original_cam_y)

        # Update dynamic systems that need to process every frame
        for actor in self.controller.gw.actors:
            actor.update_render_position(alpha)

        self.particle_system.update(alpha)
        self.environmental_system.update(alpha)

    def present(self, renderer: Renderer, alpha: float) -> None:
        """Composite final frame layers in proper order."""
        if not self.visible:
            return

        # 1. Present the cached unlit background
        if self._active_background_texture:
            with record_time_live_variable("cpu.render.present_background_ms"):
                renderer.present_texture(
                    self._active_background_texture,
                    self.x,
                    self.y,
                    self.width,
                    self.height,
                )

        # 2. Present the dynamic light overlay on top of the background
        if self._light_overlay_texture:
            with record_time_live_variable("cpu.render.present_light_overlay_ms"):
                renderer.present_texture(
                    self._light_overlay_texture, self.x, self.y, self.width, self.height
                )

        # 3. Continue with the rest of the rendering
        viewport_bounds = Rect.from_bounds(0, 0, self.width - 1, self.height - 1)
        view_offset = (self.x, self.y)

        with record_time_live_variable("cpu.render.particles_under_actors_ms"):
            self.controller.renderer.render_particles(
                self.particle_system,
                ParticleLayer.UNDER_ACTORS,
                viewport_bounds,
                view_offset,
            )

        if config.SMOOTH_ACTOR_RENDERING_ENABLED:
            self._render_actors_smooth(renderer, alpha)
        else:
            self._render_actors_traditional(renderer, alpha)

        # Render highlights and mode-specific UI on top of actors
        if self.controller.active_mode:
            with record_time_live_variable("cpu.render.active_mode_world_ms"):
                self.controller.active_mode.render_world()
        else:
            self._render_selected_actor_highlight()

        with record_time_live_variable("cpu.render.particles_over_actors_ms"):
            self.controller.renderer.render_particles(
                self.particle_system,
                ParticleLayer.OVER_ACTORS,
                viewport_bounds,
                view_offset,
            )

        if config.ENVIRONMENTAL_EFFECTS_ENABLED:
            with record_time_live_variable("cpu.render.environmental_effects_ms"):
                self.environmental_system.render_effects(
                    renderer,
                    viewport_bounds,
                    view_offset,
                )

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    @record_time_live_variable("cpu.render.map_unlit_ms")
    def _render_map_unlit(self) -> None:
        gw = self.controller.gw
        vs = self.viewport_system
        # Clear entire console to create pillarbox/letterbox bars
        self.game_map_console.rgb[:] = (ord(" "), (0, 0, 0), (0, 0, 0))

        bounds = vs.get_visible_bounds()
        world_left, world_right, world_top, world_bottom = (
            bounds.x1,
            bounds.x2,
            bounds.y1,
            bounds.y2,
        )
        # Clamp the visible bounds to the actual map size so slicing is safe.
        world_left = max(0, world_left)
        world_top = max(0, world_top)
        world_right = min(gw.game_map.width - 1, world_right)
        world_bottom = min(gw.game_map.height - 1, world_bottom)
        world_slice = (
            slice(world_left, world_right + 1),
            slice(world_top, world_bottom + 1),
        )
        # Offset to center smaller maps within the viewport
        dest_x_start = vs.offset_x
        dest_y_start = vs.offset_y
        # Determine the offset within the viewport-sized console where the world
        # slice will be drawn. This keeps the map centered even when smaller
        # than the viewport.
        dark_app_slice = gw.game_map.dark_appearance_map[world_slice]

        # Use ONLY the explored map for the unlit background.
        explored_mask_slice = gw.game_map.explored[world_slice]

        ex_x, ex_y = np.nonzero(explored_mask_slice)

        if ex_x.size > 0:
            self.game_map_console.rgb[dest_x_start + ex_x, dest_y_start + ex_y] = (
                dark_app_slice[ex_x, ex_y]
            )

        # The visible mask is NOT used here. It is used by the light overlay.

    def _render_actors(self) -> None:
        # Traditional actors are now baked into the cache during the draw phase.
        # This method is now only for presenting dynamic elements on top of the cache.
        # Therefore, we do nothing if smooth rendering is disabled.
        if config.SMOOTH_ACTOR_RENDERING_ENABLED:
            # The present() method calls _render_actors_smooth() directly,
            # so this method can simply be a no-op or pass.
            pass

    @record_time_live_variable("cpu.render.actors_smooth_ms")
    def _render_actors_smooth(self, renderer: Renderer, alpha: float) -> None:
        """Render all actors with smooth sub-pixel positioning."""
        gw = self.controller.gw
        vs = self.viewport_system
        bounds = vs.get_visible_bounds()
        world_left, world_right, world_top, world_bottom = (
            bounds.x1,
            bounds.x2,
            bounds.y1,
            bounds.y2,
        )

        # Get visible actors using existing spatial index
        actors_in_viewport = gw.actor_spatial_index.get_in_bounds(
            world_left, world_top, world_right, world_bottom
        )

        # Sort for proper z-order (existing logic)
        sorted_actors = sorted(
            actors_in_viewport,
            key=lambda a: (
                getattr(a, "blocks_movement", False),
                a == gw.player,
            ),
        )

        for actor in sorted_actors:
            if gw.game_map.visible[actor.x, actor.y]:
                self._render_single_actor_smooth(actor, renderer, bounds, vs, alpha)

    def _render_single_actor_smooth(
        self,
        actor: Actor,
        renderer: Renderer,
        bounds: Rect,
        vs: ViewportSystem,
        alpha: float,
    ) -> None:
        """Render a single actor with smooth positioning and lighting.

        Uses linear interpolation between the actor's previous position (from last step)
        and current position (from current logic step) to create fluid movement that's
        independent of visual framerate.

        Args:
            actor: The actor to render
            renderer: Rendering backend
            bounds: Viewport bounds for culling
            vs: Viewport system for coordinate conversion
            alpha: Interpolation factor (0.0=previous state, 1.0=current state)
        """
        # Get lighting intensity (reuse existing lighting logic)
        light_rgb = self._get_actor_lighting_intensity(actor, bounds)

        # INTERPOLATION MAGIC: Blend between previous and current position
        # When alpha=0.0: Show exactly where actor was last logic step (prev_x/prev_y)
        # When alpha=1.0: Show exactly where actor is now (x/y)
        # When alpha=0.5: Show exactly halfway between - this creates smooth movement!
        # Formula: lerp(prev, current, alpha) = prev * (1-alpha) + current * alpha
        interpolated_x = actor.prev_x * (1.0 - alpha) + actor.x * alpha
        interpolated_y = actor.prev_y * (1.0 - alpha) + actor.y * alpha

        vp_x, vp_y = vs.world_to_screen_float(interpolated_x, interpolated_y)

        # Root console position where this viewport pixel ends up
        root_x = self.x + vp_x
        root_y = self.y + vp_y

        screen_pixel_x, screen_pixel_y = renderer.console_to_screen_coords(
            root_x, root_y
        )

        # Get actor color with visual effects (reuse existing logic)
        final_color = self._get_actor_display_color(actor)

        # Render using the enhanced renderer
        renderer.draw_actor_smooth(
            actor.ch, final_color, screen_pixel_x, screen_pixel_y, light_rgb, alpha
        )

    def _get_actor_lighting_intensity(self, actor: Actor, bounds: Rect) -> tuple:
        """Get lighting intensity for actor (extracted from existing code)."""
        if self.current_light_intensity is None:
            return (1.0, 1.0, 1.0)

        world_left, world_top = bounds.x1, bounds.y1
        light_x, light_y = actor.x - world_left, actor.y - world_top

        if (
            0 <= light_x < self.current_light_intensity.shape[0]
            and 0 <= light_y < self.current_light_intensity.shape[1]
        ):
            return tuple(self.current_light_intensity[light_x, light_y])

        return (1.0, 1.0, 1.0)

    def _get_actor_display_color(self, actor: Actor) -> tuple:
        """Get actor's final display color with visual effects."""
        base_color = actor.color

        # Apply visual effects if present (existing logic)
        visual_effects = actor.visual_effects
        if visual_effects is not None:
            visual_effects.update()
            flash_color = visual_effects.get_flash_color()
            if flash_color:
                base_color = flash_color

        return base_color

    @record_time_live_variable("cpu.render.actors_traditional_ms")
    def _render_actors_traditional(self, renderer: Renderer, alpha: float) -> None:
        """Tile-aligned actor rendering, adapted for dynamic rendering."""
        gw = self.controller.gw
        vs = self.viewport_system
        bounds = vs.get_visible_bounds()
        world_left, world_right, world_top, world_bottom = (
            bounds.x1,
            bounds.x2,
            bounds.y1,
            bounds.y2,
        )
        # Get only actors within the viewport using the spatial index, then sort
        # them so the player appears on top of other blocking actors.
        actors_in_viewport = gw.actor_spatial_index.get_in_bounds(
            world_left, world_top, world_right, world_bottom
        )
        sorted_actors = sorted(
            actors_in_viewport,
            key=lambda a: (
                getattr(a, "blocks_movement", False),
                a == gw.player,
            ),
        )
        for actor in sorted_actors:
            if gw.game_map.visible[actor.x, actor.y]:
                # Get lighting intensity
                light_rgb = self._get_actor_lighting_intensity(actor, bounds)

                # Convert actor's TILE position to viewport coordinates
                # Note: We use actor.x/y directly for tile-aligned rendering
                vp_x, vp_y = vs.world_to_screen(actor.x, actor.y)

                # Root console position where this viewport tile ends up
                root_x = self.x + vp_x
                root_y = self.y + vp_y

                # Convert to final screen pixel coordinates
                screen_pixel_x, screen_pixel_y = renderer.console_to_screen_coords(
                    root_x, root_y
                )

                # Get final color with pulsating effect if needed
                base_actor_color = self._get_actor_display_color(actor)
                final_fg_color = base_actor_color
                if (
                    self.controller.gw.selected_actor == actor
                    and self.controller.gw.game_map.visible[actor.x, actor.y]
                    and not self.controller.is_targeting_mode()
                ):
                    final_fg_color = self._apply_pulsating_effect(
                        base_actor_color, actor.color
                    )

                # Render using the renderer's smooth drawing function
                renderer.draw_actor_smooth(
                    actor.ch,
                    final_fg_color,
                    screen_pixel_x,
                    screen_pixel_y,
                    light_rgb,
                    alpha,
                )

    @record_time_live_variable("cpu.render.selected_actor_highlight_ms")
    def _render_selected_actor_highlight(self) -> None:
        if self.controller.is_targeting_mode():
            return
        actor = self.controller.gw.selected_actor
        if actor and self.controller.gw.game_map.visible[actor.x, actor.y]:
            self.highlight_actor(
                actor,
                colors.SELECTED_HIGHLIGHT,
                effect="solid",
                alpha=SELECTION_HIGHLIGHT_ALPHA,
            )

    def _update_mouse_tile_location(self) -> None:
        """Update the stored world-space mouse tile based on the current camera."""
        fm: FrameManager | None = getattr(self.controller, "frame_manager", None)
        if fm is None:
            return
        px_x: PixelCoord = fm.cursor_manager.mouse_pixel_x
        px_y: PixelCoord = fm.cursor_manager.mouse_pixel_y
        root_tile_pos: RootConsoleTilePos = (
            self.controller.coordinate_converter.pixel_to_tile(px_x, px_y)
        )
        world_tile_pos = fm.get_world_coords_from_root_tile_coords(root_tile_pos)
        self.controller.gw.mouse_tile_location_on_map = world_tile_pos

    @record_time_live_variable("cpu.render.light_overlay_ms")
    def _render_light_overlay(
        self, renderer: Renderer
    ) -> tcod.sdl.render.Texture | None:
        """Render pure light world overlay with GPU alpha blending."""
        if self.lighting_system is None:
            return None

        vs = self.viewport_system
        gw = self.controller.gw

        bounds = vs.get_visible_bounds()
        world_left, world_right = (
            max(0, bounds.x1),
            min(gw.game_map.width - 1, bounds.x2),
        )
        world_top, world_bottom = (
            max(0, bounds.y1),
            min(gw.game_map.height - 1, bounds.y2),
        )

        dest_width = world_right - world_left + 1
        dest_height = world_bottom - world_top + 1

        # Use new lighting system to compute light intensity
        viewport_bounds = Rect(world_left, world_top, dest_width, dest_height)
        self.current_light_intensity = self.lighting_system.compute_lightmap(
            viewport_bounds
        )

        if self.current_light_intensity is None:
            return None
        # ---------------------------------------------------------------------

        # Create temporary console for the light overlay
        light_console = Console(self.width, self.height, order="F")

        # Get viewport bounds and calculate world coordinates
        vs = self.viewport_system
        gw = self.controller.gw
        bounds = vs.get_visible_bounds()
        world_left = max(0, bounds.x1)
        world_top = max(0, bounds.y1)
        world_right = min(gw.game_map.width - 1, bounds.x2)
        world_bottom = min(gw.game_map.height - 1, bounds.y2)

        # Get the world slice that corresponds to current light intensity
        world_slice = (
            slice(world_left, world_right + 1),
            slice(world_top, world_bottom + 1),
        )
        visible_mask_slice = gw.game_map.visible[world_slice]

        # Initialize light console with fully transparent
        light_console.rgba[:] = (ord(" "), (0, 0, 0, 0), (0, 0, 0, 0))

        if np.any(visible_mask_slice):
            # Get pure light appearance for overlay
            light_app_slice = gw.game_map.light_appearance_map[world_slice]
            light_intensities = self.current_light_intensity[visible_mask_slice]

            # Map world coordinates to viewport coordinates
            vis_x, vis_y = np.nonzero(visible_mask_slice)
            viewport_x = vs.offset_x + vis_x
            viewport_y = vs.offset_y + vis_y

            # Ensure coordinates are within bounds
            valid_mask = (
                (viewport_x >= 0)
                & (viewport_x < self.width)
                & (viewport_y >= 0)
                & (viewport_y < self.height)
            )

            if np.any(valid_mask):
                final_vp_x = viewport_x[valid_mask]
                final_vp_y = viewport_y[valid_mask]

                # Get pure light appearance data
                light_chars = light_app_slice["ch"][
                    vis_x[valid_mask], vis_y[valid_mask]
                ]
                light_fg = light_app_slice["fg"][vis_x[valid_mask], vis_y[valid_mask]]
                light_bg = light_app_slice["bg"][vis_x[valid_mask], vis_y[valid_mask]]
                dark_bg = gw.game_map.dark_appearance_map[world_slice]["bg"][
                    vis_x[valid_mask], vis_y[valid_mask]
                ]

                # Per-channel scaling: warm colours stay warm
                light_intensity_valid = light_intensities[valid_mask]  # shape (N,3)

                # --- exact CPU blend to match pre-refactor appearance ---
                # light_intensity_valid shape: (N, 3) already carries warm torch colours
                scaled_bg = (
                    light_bg.astype(np.float32) * light_intensity_valid
                    + dark_bg.astype(np.float32) * (1.0 - light_intensity_valid)
                ).astype(np.uint8)
                alpha_values = np.full(
                    len(scaled_bg), 255, dtype=np.uint8
                )  # fully opaque

                # Fully vectorized assignment to RGBA console
                light_console.rgba["ch"][final_vp_x, final_vp_y] = light_chars
                light_console.rgba["fg"][final_vp_x, final_vp_y] = np.column_stack(
                    (
                        light_fg,
                        np.full(len(alpha_values), 255, dtype=np.uint8),  # opaque FG
                    )
                )
                light_console.rgba["bg"][final_vp_x, final_vp_y] = np.column_stack(
                    (scaled_bg, alpha_values)  # RGB already pre-multiplied by alpha
                )

        # Convert console to texture with alpha blending
        from catley.view.render.backends.tcod.renderer import TCODRenderer

        tcod_renderer = cast(TCODRenderer, renderer)
        return tcod_renderer.texture_from_console(light_console, transparent=True)

    def _apply_pulsating_effect(
        self, input_color: colors.Color, base_actor_color: colors.Color
    ) -> colors.Color:
        game_time = self.controller.clock.last_time
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
