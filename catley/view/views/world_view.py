from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np

from catley import colors, config
from catley.config import (
    LUMINANCE_THRESHOLD,
    PULSATION_MAX_BLEND_ALPHA,
    PULSATION_PERIOD,
    SELECTION_HIGHLIGHT_ALPHA,
)
from catley.types import InterpolationAlpha, Opacity
from catley.util.caching import ResourceCache
from catley.util.coordinates import (
    PixelCoord,
    Rect,
    RootConsoleTilePos,
)
from catley.util.glyph_buffer import GlyphBuffer
from catley.util.live_vars import record_time_live_variable
from catley.view.render.effects.decals import DecalSystem
from catley.view.render.effects.effects import EffectLibrary
from catley.view.render.effects.environmental import EnvironmentalEffectSystem
from catley.view.render.effects.particles import (
    ParticleLayer,
    SubTileParticleSystem,
)
from catley.view.render.effects.screen_shake import ScreenShake
from catley.view.render.graphics import GraphicsContext
from catley.view.render.lighting.base import LightingSystem
from catley.view.render.viewport import ViewportSystem

from .base import View

if TYPE_CHECKING:
    from catley.controller import Controller, FrameManager
    from catley.game.actors import Actor
    from catley.game.actors.core import CharacterLayer


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
        self.graphics = controller.graphics
        self.screen_shake = screen_shake
        self.lighting_system = lighting_system
        # Initialize a viewport system sized using configuration defaults.
        # These defaults are replaced once resize() sets the real view bounds.
        self.viewport_system = ViewportSystem(
            config.DEFAULT_VIEWPORT_WIDTH, config.DEFAULT_VIEWPORT_HEIGHT
        )
        # Game map console matches the viewport dimensions rather than the
        # entire map. This keeps rendering fast and memory usage reasonable.
        self.map_glyph_buffer = GlyphBuffer(
            config.DEFAULT_VIEWPORT_WIDTH, config.DEFAULT_VIEWPORT_HEIGHT
        )
        # Particle system also operates only on the visible viewport area.
        self.particle_system = SubTileParticleSystem(
            config.DEFAULT_VIEWPORT_WIDTH, config.DEFAULT_VIEWPORT_HEIGHT
        )
        self.environmental_system = EnvironmentalEffectSystem()
        self.decal_system = DecalSystem()
        self.effect_library = EffectLibrary()
        self.current_light_intensity: np.ndarray | None = None
        self._texture_cache = ResourceCache[tuple, Any](
            name="WorldViewCache", max_size=5
        )
        self._active_background_texture: Any | None = None
        self._light_overlay_texture: Any | None = None
        # Screen shake offset in tiles for sub-tile rendering
        self._shake_offset: tuple[float, float] = (0.0, 0.0)
        # Cumulative game time for decal age tracking
        self._game_time: float = 0.0

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override set_bounds to update viewport and console dimensions."""
        # Only perform resize logic if the dimensions have actually changed.
        if self.width != (x2 - x1) or self.height != (y2 - y1):
            super().set_bounds(x1, y1, x2, y2)
            # Update the existing viewport's size instead of replacing it.
            self.viewport_system.viewport.resize(self.width, self.height)
            self.map_glyph_buffer = GlyphBuffer(self.width, self.height)
            self.particle_system = SubTileParticleSystem(self.width, self.height)
            self.environmental_system = EnvironmentalEffectSystem()
            self.decal_system = DecalSystem()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def highlight_actor(
        self,
        actor: Actor,
        color: colors.Color,
        effect: str = "solid",
        alpha: Opacity = Opacity(0.4),  # noqa: B008
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
        alpha: Opacity = Opacity(0.4),  # noqa: B008
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
        self.graphics.draw_tile_highlight(root_x, root_y, final_color, alpha)

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

        map_key = gw.game_map.structural_revision

        # The background cache must be invalidated whenever the player moves,
        # because movement updates the `explored` map, which changes the appearance
        # of the unlit background. The new `exploration_revision` is the direct
        # source of truth for this.
        exploration_key = gw.game_map.exploration_revision

        return (camera_key, map_key, exploration_key)

    def draw(self, graphics: GraphicsContext, alpha: float) -> None:
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

        # Get screen shake offset for this frame. We apply it as a pixel offset
        # in present() rather than modifying camera position, to avoid cache thrashing.
        shake_x, shake_y = self.screen_shake.update(delta_time)
        self._shake_offset = (shake_x, shake_y)

        # --- Cache lookup and management ---
        cache_key = self._get_background_cache_key()
        texture = self._texture_cache.get(cache_key)

        if texture is None:
            # CACHE MISS: Re-render the static background
            self._render_map_unlit()  # This populates self.game_map_console

            texture = graphics.render_glyph_buffer_to_texture(self.map_glyph_buffer)
            self._texture_cache.store(cache_key, texture)

        self._active_background_texture = texture

        # Generate the light overlay DATA (GlyphBuffer)
        light_overlay_data = self._render_light_overlay(graphics)

        if light_overlay_data is not None:
            # Ask the graphics context to turn this DATA into a RENDERABLE TEXTURE
            self._light_overlay_texture = graphics.render_glyph_buffer_to_texture(
                light_overlay_data
            )
        else:
            self._light_overlay_texture = None

        # Update dynamic systems that need to process every frame.
        # actor.update_render_position is a legacy call from the old rendering
        # system and is no longer needed with fixed-timestep interpolation.

        # Visual effects like particles and environmental effects should update based
        # on the frame's delta_time for maximum smoothness, not the fixed
        # logic timestep or the interpolation alpha.
        self.particle_system.update(delta_time)
        self.environmental_system.update(delta_time)
        # Update decals for age-based cleanup
        self._game_time += delta_time
        self.decal_system.update(delta_time, self._game_time)

        # Update tile animations (color oscillation, glyph flicker)
        self._update_tile_animations()

        # Emit particles from actors with particle emitters
        self._update_actor_particles()

    def present(self, graphics: GraphicsContext, alpha: InterpolationAlpha) -> None:
        """Composite final frame layers in proper order."""
        if not self.visible:
            return

        # Convert shake offset from tiles to pixels for background rendering
        shake_tile_x, shake_tile_y = self._shake_offset
        # Get pixel offset by computing the difference in screen coords
        base_px_x, base_px_y = graphics.console_to_screen_coords(0.0, 0.0)
        shake_px_x, shake_px_y = graphics.console_to_screen_coords(
            shake_tile_x, shake_tile_y
        )
        offset_x_pixels = shake_px_x - base_px_x
        offset_y_pixels = shake_px_y - base_px_y

        # 1. Present the cached unlit background with shake offset
        if self._active_background_texture:
            with record_time_live_variable("cpu.render.present_background_ms"):
                graphics.draw_background(
                    self._active_background_texture,
                    self.x,
                    self.y,
                    self.width,
                    self.height,
                    offset_x_pixels,
                    offset_y_pixels,
                )

        # 2. Present the dynamic light overlay on top of the background
        if self._light_overlay_texture:
            with record_time_live_variable("cpu.render.present_light_overlay_ms"):
                graphics.draw_background(
                    self._light_overlay_texture,
                    self.x,
                    self.y,
                    self.width,
                    self.height,
                    offset_x_pixels,
                    offset_y_pixels,
                )

        # 3. Apply shake to camera for actor/particle rendering
        vs = self.viewport_system
        original_cam_x = vs.camera.world_x
        original_cam_y = vs.camera.world_y
        vs.camera.world_x += shake_tile_x
        vs.camera.world_y += shake_tile_y

        viewport_bounds = Rect.from_bounds(0, 0, self.width - 1, self.height - 1)
        view_offset = (self.x, self.y)

        # Render persistent decals (blood splatters, etc.) on the floor
        with record_time_live_variable("cpu.render.decals_ms"):
            graphics.render_decals(
                self.decal_system,
                viewport_bounds,
                view_offset,
                self.viewport_system,
                self._game_time,
            )

        with record_time_live_variable("cpu.render.particles_under_actors_ms"):
            graphics.render_particles(
                self.particle_system,
                ParticleLayer.UNDER_ACTORS,
                viewport_bounds,
                view_offset,
                self.viewport_system,
            )

        if config.SMOOTH_ACTOR_RENDERING_ENABLED:
            self._render_actors_smooth(graphics, alpha)
        else:
            self._render_actors_traditional(graphics, alpha)

        # Render highlights and mode-specific UI on top of actors
        # The active mode is always set (ExploreMode by default)
        with record_time_live_variable("cpu.render.active_mode_world_ms"):
            self.controller.active_mode.render_world()

        with record_time_live_variable("cpu.render.particles_over_actors_ms"):
            graphics.render_particles(
                self.particle_system,
                ParticleLayer.OVER_ACTORS,
                viewport_bounds,
                view_offset,
                self.viewport_system,
            )

        if config.ENVIRONMENTAL_EFFECTS_ENABLED:
            with record_time_live_variable("cpu.render.environmental_effects_ms"):
                self.environmental_system.render_effects(
                    graphics,
                    viewport_bounds,
                    view_offset,
                )

        # Restore camera position after rendering
        vs.camera.set_position(original_cam_x, original_cam_y)

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    @record_time_live_variable("cpu.render.map_unlit_ms")
    def _render_map_unlit(self) -> None:
        """Renders the static, unlit background of the game world."""
        gw = self.controller.gw
        vs = self.viewport_system

        # Clear the console for this view to a default black background.
        self.map_glyph_buffer.clear()

        # Iterate over every tile in the destination console (the viewport).
        for vp_y in range(self.map_glyph_buffer.height):
            for vp_x in range(self.map_glyph_buffer.width):
                # For each screen tile, find out which world tile it corresponds to.
                world_x, world_y = vs.screen_to_world(vp_x, vp_y)

                # Check if the world tile is within the map bounds.
                if not (
                    0 <= world_x < gw.game_map.width
                    and 0 <= world_y < gw.game_map.height
                ):
                    continue

                # If the world tile has been explored, draw its "dark" appearance.
                if gw.game_map.explored[world_x, world_y]:
                    dark_appearance = gw.game_map.dark_appearance_map[world_x, world_y]
                    fg_rgba = (*dark_appearance["fg"], 255)
                    bg_rgba = (*dark_appearance["bg"], 255)
                    self.map_glyph_buffer.data[vp_x, vp_y] = (
                        dark_appearance["ch"],
                        fg_rgba,
                        bg_rgba,
                    )

    def _render_actors(self) -> None:
        # Traditional actors are now baked into the cache during the draw phase.
        # This method is now only for presenting dynamic elements on top of the cache.
        # Therefore, we do nothing if smooth rendering is disabled.
        if config.SMOOTH_ACTOR_RENDERING_ENABLED:
            # The present() method calls _render_actors_smooth() directly,
            # so this method can simply be a no-op or pass.
            pass

    @record_time_live_variable("cpu.render.actors_smooth_ms")
    def _render_actors_smooth(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
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

        # Sort for proper z-order: Y-position primary (painter's algorithm),
        # visual_scale secondary (larger actors on top at same Y), player on top
        sorted_actors = sorted(
            actors_in_viewport,
            key=lambda a: (
                a.y,
                getattr(a, "visual_scale", 1.0),
                a == gw.player,
            ),
        )

        for actor in sorted_actors:
            if gw.game_map.visible[actor.x, actor.y]:
                self._render_single_actor_smooth(actor, graphics, bounds, vs, alpha)

    def _render_single_actor_smooth(
        self,
        actor: Actor,
        graphics: GraphicsContext,
        bounds: Rect,
        vs: ViewportSystem,
        interpolation_alpha: InterpolationAlpha,
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
            interpolation_alpha: Interpolation factor (0.0=previous state,
                1.0=current state)
        """
        # Get lighting intensity (reuse existing lighting logic)
        light_rgb = self._get_actor_lighting_intensity(actor, bounds)

        # INTERPOLATION MAGIC: Blend between previous and current position
        # When interpolation_alpha=0.0: Show exactly where actor was last logic step
        # (prev_x/prev_y)
        # When interpolation_alpha=1.0: Show exactly where actor is now (x/y)
        # When interpolation_alpha=0.5: Show exactly halfway between - this creates
        # smooth movement!
        # Formula: lerp(prev, current, interpolation_alpha) =
        # prev * (1-interpolation_alpha) + current * interpolation_alpha
        interpolated_x = (
            actor.prev_x * (1.0 - interpolation_alpha) + actor.x * interpolation_alpha
        )
        interpolated_y = (
            actor.prev_y * (1.0 - interpolation_alpha) + actor.y * interpolation_alpha
        )

        # Apply idle animation drift offset (subtle sub-tile movement)
        # Only for living actors - corpses and objects don't shift weight
        if (
            actor.visual_effects is not None
            and actor.health is not None
            and actor.health.is_alive()
        ):
            drift_x, drift_y = actor.visual_effects.get_idle_drift_offset()
            interpolated_x += drift_x
            interpolated_y += drift_y

        vp_x, vp_y = vs.world_to_screen_float(interpolated_x, interpolated_y)

        # Root console position where this viewport pixel ends up
        root_x = self.x + vp_x
        root_y = self.y + vp_y

        screen_pixel_x, screen_pixel_y = graphics.console_to_screen_coords(
            root_x, root_y
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
                graphics,
                light_rgb,
                interpolation_alpha,
                visual_scale,
            )
        else:
            # Render single character (existing behavior) - uniform scaling
            graphics.draw_actor_smooth(
                actor.ch,
                final_color,
                screen_pixel_x,
                screen_pixel_y,
                light_rgb,
                interpolation_alpha,
                scale_x=visual_scale,
                scale_y=visual_scale,
            )

    def _render_character_layers(
        self,
        layers: list[CharacterLayer],
        root_x: float,
        root_y: float,
        graphics: GraphicsContext,
        light_rgb: tuple,
        interpolation_alpha: InterpolationAlpha,
        visual_scale: float,
    ) -> None:
        """Render multiple character layers at sub-tile offsets.

        Each layer is rendered at its offset position relative to the actor's
        center, creating a rich visual composition from multiple ASCII characters.

        Args:
            layers: List of CharacterLayer defining the composition.
            root_x: Base X position in root console coordinates.
            root_y: Base Y position in root console coordinates.
            graphics: Graphics context for rendering.
            light_rgb: Lighting intensity tuple.
            interpolation_alpha: Interpolation factor for smooth rendering.
            visual_scale: Base scale factor for the actor.
        """
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

    def _update_actor_particles(self) -> None:
        """Emit particles from actors with particle emitters."""
        gw = self.controller.gw

        # Get all actors and check for particle emitters
        for actor in gw.actors:
            visual_effects = actor.visual_effects
            if visual_effects is not None and visual_effects.has_continuous_effects():
                # Create an effect context for this actor
                from catley.view.render.effects.effects import EffectContext

                context = EffectContext(
                    particle_system=self.particle_system,
                    environmental_system=self.environmental_system,
                    x=actor.x,
                    y=actor.y,
                )

                # Execute all continuous effects that are ready to emit
                for effect in visual_effects.continuous_effects:
                    if effect.should_emit():
                        effect.execute(context)

    def _update_tile_animations(self, percent_of_cells: int = 3) -> None:
        """Update animation state for a percentage of visible animated tiles.

        Uses a random walk algorithm: each updated tile's RGB modulation values
        are adjusted by a random offset, then clamped to [0, 1000]. This creates
        organic color oscillation.

        Args:
            percent_of_cells: Percentage of visible animated tiles to update
                              each frame (default 3%). At 60 FPS, this means
                              each tile updates roughly every 0.5 seconds.
        """
        gw = self.controller.gw
        vs = self.viewport_system
        game_map = gw.game_map

        # Get viewport bounds
        bounds = vs.get_visible_bounds()
        world_left = max(0, bounds.x1)
        world_top = max(0, bounds.y1)
        world_right = min(game_map.width - 1, bounds.x2)
        world_bottom = min(game_map.height - 1, bounds.y2)

        # Get animation params and state
        animation_params = game_map.animation_params
        animation_state = game_map.animation_state

        # Find all visible animated tiles in the viewport
        animated_tiles = [
            (world_x, world_y)
            for world_x in range(world_left, world_right + 1)
            for world_y in range(world_top, world_bottom + 1)
            if (
                game_map.visible[world_x, world_y]
                and animation_params[world_x, world_y]["animates"]
            )
        ]

        if not animated_tiles:
            return

        # Select a random subset of tiles to update (percent_of_cells %)
        num_to_update = max(1, len(animated_tiles) * percent_of_cells // 100)
        tiles_to_update = random.sample(
            animated_tiles, min(num_to_update, len(animated_tiles))
        )

        # Update each selected tile using random walk
        # Use small step size for subtle, organic color drift
        step_size = 80

        for world_x, world_y in tiles_to_update:
            state = animation_state[world_x, world_y]

            # Random walk for foreground RGB values
            for i in range(3):
                offset = random.randint(-step_size, step_size)
                new_val = int(state["fg_values"][i]) + offset
                state["fg_values"][i] = max(0, min(1000, new_val))

            # Random walk for background RGB values
            for i in range(3):
                offset = random.randint(-step_size, step_size)
                new_val = int(state["bg_values"][i]) + offset
                state["bg_values"][i] = max(0, min(1000, new_val))

            # Glyph is always visible - the "flicker" effect comes naturally
            # when fg and bg colors drift close together, making the glyph
            # blend into the background.

            # Write back the updated state
            animation_state[world_x, world_y] = state

    def _get_actor_display_color(self, actor: Actor) -> tuple:
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

    @record_time_live_variable("cpu.render.actors_traditional_ms")
    def _render_actors_traditional(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
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
        # for proper z-order: Y-position primary (painter's algorithm),
        # visual_scale secondary (larger actors on top at same Y), player on top
        actors_in_viewport = gw.actor_spatial_index.get_in_bounds(
            world_left, world_top, world_right, world_bottom
        )
        sorted_actors = sorted(
            actors_in_viewport,
            key=lambda a: (
                a.y,
                getattr(a, "visual_scale", 1.0),
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
                screen_pixel_x, screen_pixel_y = graphics.console_to_screen_coords(
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

                visual_scale = getattr(actor, "visual_scale", 1.0)

                # Check for multi-character composition (character_layers)
                if actor.character_layers:
                    # Render each layer at its sub-tile offset
                    self._render_character_layers(
                        actor.character_layers,
                        float(root_x),
                        float(root_y),
                        graphics,
                        light_rgb,
                        alpha,
                        visual_scale,
                    )
                else:
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

        assert self.controller.coordinate_converter is not None

        px_x: PixelCoord = fm.cursor_manager.mouse_pixel_x
        px_y: PixelCoord = fm.cursor_manager.mouse_pixel_y
        root_tile_pos: RootConsoleTilePos = (
            self.controller.coordinate_converter.pixel_to_tile(px_x, px_y)
        )
        world_tile_pos = fm.get_world_coords_from_root_tile_coords(root_tile_pos)
        self.controller.gw.mouse_tile_location_on_map = world_tile_pos

    @record_time_live_variable("cpu.render.light_overlay_ms")
    def _render_light_overlay(self, graphics: GraphicsContext) -> Any | None:
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

        # Create a GlyphBuffer for the light overlay.
        # It's already transparent by default from its own .clear() method.
        light_glyph_buffer = GlyphBuffer(self.width, self.height)

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
        explored_mask_slice = gw.game_map.explored[world_slice]

        if np.any(explored_mask_slice):
            # Get pure light appearance for overlay
            light_app_slice = gw.game_map.light_appearance_map[world_slice]

            # Create lighting intensity array for explored tiles
            # Start with zero intensity everywhere
            explored_light_intensities = np.zeros(
                (*explored_mask_slice.shape, 3), dtype=np.float32
            )
            # Set full intensity only for visible areas (FOV)
            if np.any(visible_mask_slice):
                explored_light_intensities[visible_mask_slice] = (
                    self.current_light_intensity[visible_mask_slice]
                )

            # Add spillover effects to non-visible but explored areas
            # This allows spillover to be seen outside FOV while maintaining FOV
            # darkening
            non_visible_mask = explored_mask_slice & ~visible_mask_slice
            if np.any(non_visible_mask):
                # Sample spillover intensity for non-visible explored tiles
                nvx, nvy = np.nonzero(non_visible_mask)
                for _i, (rel_x, rel_y) in enumerate(zip(nvx, nvy, strict=True)):
                    if (
                        rel_x < self.current_light_intensity.shape[0]
                        and rel_y < self.current_light_intensity.shape[1]
                    ):
                        # Get the computed spillover intensity
                        spillover_intensity = self.current_light_intensity[rel_x, rel_y]
                        # Apply a fraction of spillover to non-visible areas (dimmed
                        # but visible)
                        explored_light_intensities[rel_x, rel_y] = (
                            spillover_intensity * 0.3
                        )

            # Map world coordinates to viewport coordinates for all explored tiles
            exp_x, exp_y = np.nonzero(explored_mask_slice)
            viewport_x = vs.offset_x + exp_x
            viewport_y = vs.offset_y + exp_y

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

                # Get pure light appearance data for all explored tiles
                light_chars = light_app_slice["ch"][
                    exp_x[valid_mask], exp_y[valid_mask]
                ]

                # light_app_slice['fg'] is (N, 3) RGB. Apply lighting, convert to RGBA.
                light_fg_rgb = light_app_slice["fg"][
                    exp_x[valid_mask], exp_y[valid_mask]
                ]
                dark_fg_rgb = gw.game_map.dark_appearance_map[world_slice]["fg"][
                    exp_x[valid_mask], exp_y[valid_mask]
                ]

                light_bg_rgb = light_app_slice["bg"][
                    exp_x[valid_mask], exp_y[valid_mask]
                ]
                dark_bg_rgb = gw.game_map.dark_appearance_map[world_slice]["bg"][
                    exp_x[valid_mask], exp_y[valid_mask]
                ]

                # Apply animated colors for tiles that have animation enabled.
                # This modifies light_fg_rgb, light_bg_rgb, and light_chars in place
                # for animated tiles, using the current animation state.
                animation_params = gw.game_map.animation_params[world_slice]
                animation_state = gw.game_map.animation_state[world_slice]
                valid_exp_x = exp_x[valid_mask]
                valid_exp_y = exp_y[valid_mask]

                for i in range(len(valid_exp_x)):
                    rel_x, rel_y = valid_exp_x[i], valid_exp_y[i]
                    params = animation_params[rel_x, rel_y]

                    if params["animates"]:
                        state = animation_state[rel_x, rel_y]
                        fg_var = params["fg_variation"]
                        bg_var = params["bg_variation"]
                        fg_vals = state["fg_values"]
                        bg_vals = state["bg_values"]

                        # Compute animated foreground color
                        # Cast to int to avoid overflow in numpy small int types
                        base_fg = light_fg_rgb[i]
                        anim_fg = np.array(
                            [
                                max(
                                    0,
                                    min(
                                        255,
                                        int(base_fg[c])
                                        + int(fg_var[c]) * int(fg_vals[c]) // 1000
                                        - int(fg_var[c]) // 2,
                                    ),
                                )
                                for c in range(3)
                            ],
                            dtype=np.uint8,
                        )
                        light_fg_rgb[i] = anim_fg

                        # Compute animated background color
                        base_bg = light_bg_rgb[i]
                        anim_bg = np.array(
                            [
                                max(
                                    0,
                                    min(
                                        255,
                                        int(base_bg[c])
                                        + int(bg_var[c]) * int(bg_vals[c]) // 1000
                                        - int(bg_var[c]) // 2,
                                    ),
                                )
                                for c in range(3)
                            ],
                            dtype=np.uint8,
                        )
                        light_bg_rgb[i] = anim_bg

                # Per-channel scaling: warm colours stay warm
                light_intensity_valid = explored_light_intensities[
                    exp_x[valid_mask], exp_y[valid_mask]
                ]  # shape (N,3)

                # Apply the same lighting blend to foreground as we do to background
                scaled_fg_rgb = (
                    light_fg_rgb.astype(np.float32) * light_intensity_valid
                    + dark_fg_rgb.astype(np.float32) * (1.0 - light_intensity_valid)
                ).astype(np.uint8)

                alpha_channel = np.full((len(scaled_fg_rgb), 1), 255, dtype=np.uint8)
                light_fg_rgba = np.hstack((scaled_fg_rgb, alpha_channel))

                # --- exact CPU blend to match pre-refactor appearance ---
                # light_intensity_valid shape: (N, 3) already carries warm torch colours
                scaled_bg_rgb = (
                    light_bg_rgb.astype(np.float32) * light_intensity_valid
                    + dark_bg_rgb.astype(np.float32) * (1.0 - light_intensity_valid)
                ).astype(np.uint8)

                # Combine scaled BG with a full alpha channel to get RGBA
                bg_rgba = np.hstack((scaled_bg_rgb, alpha_channel))

                # Now, do the vectorized assignment to the GlyphBuffer.
                # The mask is for (width, height) indexing with XY coordinates.
                final_mask_T = np.zeros((self.width, self.height), dtype=bool)
                final_mask_T[final_vp_x, final_vp_y] = True

                light_glyph_buffer.data["ch"][final_mask_T] = light_chars
                light_glyph_buffer.data["fg"][final_mask_T] = light_fg_rgba
                light_glyph_buffer.data["bg"][final_mask_T] = bg_rgba

        # Return the populated GlyphBuffer. The caller is responsible for rendering it.
        return light_glyph_buffer

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
