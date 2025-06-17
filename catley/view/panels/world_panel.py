from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from tcod.console import Console

from catley import colors, config
from catley.config import (
    LUMINANCE_THRESHOLD,
    PULSATION_MAX_BLEND_ALPHA,
    PULSATION_PERIOD,
    SELECTION_HIGHLIGHT_ALPHA,
)
from catley.util.coordinates import (
    PixelCoord,
    Rect,
    RootConsoleTilePos,
)
from catley.view.render.effects.effects import EffectLibrary
from catley.view.render.effects.environmental import EnvironmentalEffectSystem
from catley.view.render.effects.particles import (
    ParticleLayer,
    SubTileParticleSystem,
)
from catley.view.render.effects.screen_shake import ScreenShake
from catley.view.render.viewport import ViewportSystem

from .panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller, FrameManager
    from catley.game.actors import Actor
    from catley.view.render.renderer import Renderer


class WorldPanel(Panel):
    """Panel responsible for rendering the game world (map, actors, effects)."""

    def __init__(
        self,
        controller: Controller,
        screen_shake: ScreenShake,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.screen_shake = screen_shake
        # Initialize a viewport system sized using configuration defaults.
        # These defaults are replaced once resize() sets the real panel bounds.
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

    def resize(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override resize to update viewport and console dimensions."""
        super().resize(x1, y1, x2, y2)
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
        self, actor: Actor, color: colors.Color, effect: str = "solid"
    ) -> None:
        """Highlight an actor if it is visible."""
        if self.controller.gw.game_map.visible[actor.x, actor.y]:
            self.highlight_tile(actor.x, actor.y, color, effect)

    def highlight_tile(
        self, x: int, y: int, color: colors.Color, effect: str = "solid"
    ) -> None:
        """Highlight a tile with an optional effect using world coordinates."""
        vs = self.viewport_system
        if not vs.is_visible(x, y):
            return
        vp_x, vp_y = vs.world_to_screen(x, y)
        if not (
            0 <= vp_x < self.game_map_console.width
            and 0 <= vp_y < self.game_map_console.height
        ):
            return
        if effect == "pulse":
            color = self._apply_pulsating_effect(color, color)
        self._apply_replacement_highlight(vp_x, vp_y, color)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, renderer: Renderer) -> None:
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

        # Render everything to the off-screen console.
        self._render_game_world(delta_time)

        # Restore the original camera position so subsequent frames start clean.
        self.viewport_system.camera.set_position(original_cam_x, original_cam_y)

        # Blit the viewport-sized console to the root console at our panel's
        # location.
        renderer.blit_console(
            source=self.game_map_console,
            dest=renderer.root_console,
            dest_x=self.x,
            dest_y=self.y,
            width=self.width,
            height=self.height,
        )

    def present(self, renderer: Renderer) -> None:
        """Composite final frame layers in proper order."""
        super().present(renderer)

        if not self.visible:
            return

        viewport_bounds = Rect.from_bounds(0, 0, self.width - 1, self.height - 1)
        panel_offset = (self.x, self.y)

        self.particle_system.render_particles(
            renderer,
            ParticleLayer.UNDER_ACTORS,
            viewport_bounds,
            panel_offset,
        )

        if config.SMOOTH_ACTOR_RENDERING_ENABLED:
            self._render_actors_smooth(renderer)
        else:
            self._render_actors()

        self.particle_system.render_particles(
            renderer,
            ParticleLayer.OVER_ACTORS,
            viewport_bounds,
            panel_offset,
        )

        if config.ENVIRONMENTAL_EFFECTS_ENABLED:
            self.environmental_system.render_effects(
                renderer,
                viewport_bounds,
                panel_offset,
            )

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------
    def _render_game_world(self, delta_time: float) -> None:
        renderer = self.controller.renderer
        renderer.clear_console(self.game_map_console)

        # Update render positions for all visible actors
        for actor in self.controller.gw.actors:
            actor.update_render_position(delta_time)

        self._render_map()
        self._render_actors()

        if self.controller.active_mode:
            self.controller.active_mode.render_world()
        else:
            self._render_selected_actor_highlight()

        self.particle_system.update(delta_time)
        self.environmental_system.update(delta_time)

    def _render_map(self) -> None:
        gw = self.controller.gw
        vs = self.viewport_system
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
        dest_width = world_right - world_left + 1
        dest_height = world_bottom - world_top + 1
        # Offset to center smaller maps within the viewport
        dest_x_start = vs.offset_x
        dest_y_start = vs.offset_y
        # Determine the offset within the viewport-sized console where the world
        # slice will be drawn. This keeps the map centered even when smaller
        # than the viewport.
        dark_app_slice = gw.game_map.dark_appearance_map[world_slice]
        explored_mask_slice = gw.game_map.explored[world_slice]
        # Clear the viewport with an empty shroud before drawing.
        self.game_map_console.rgb[:] = (ord(" "), (0, 0, 0), (0, 0, 0))
        ex_x, ex_y = np.nonzero(explored_mask_slice)
        self.game_map_console.rgb[dest_x_start + ex_x, dest_y_start + ex_y] = (
            dark_app_slice[ex_x, ex_y]
        )
        visible_mask_slice = gw.game_map.visible[world_slice]
        if not np.any(visible_mask_slice):
            return
        # Only consider actors currently inside the viewport when calculating
        # lighting. Use the spatial index to avoid scanning the full actor list.
        relevant_actors = gw.actor_spatial_index.get_in_bounds(
            world_left, world_top, world_right, world_bottom
        )
        self.current_light_intensity = gw.lighting.compute_lighting_with_shadows(
            dest_width,
            dest_height,
            relevant_actors,
            viewport_offset=(world_left, world_top),
        )
        light_app_slice = gw.game_map.light_appearance_map[world_slice]
        dark_tiles_to_light = dark_app_slice[visible_mask_slice]
        light_tiles_to_light = light_app_slice[visible_mask_slice]
        light_intensities_to_apply = self.current_light_intensity[visible_mask_slice]
        blended_tiles = np.empty_like(dark_tiles_to_light)
        blended_tiles["ch"] = light_tiles_to_light["ch"]
        blended_tiles["fg"] = light_tiles_to_light["fg"]
        # Blend each RGB channel separately based on computed lighting intensity.
        for i in range(3):
            light_intensity_channel = light_intensities_to_apply[..., i]
            blended_tiles["bg"][..., i] = light_tiles_to_light["bg"][
                ..., i
            ] * light_intensity_channel + dark_tiles_to_light["bg"][..., i] * (
                1.0 - light_intensity_channel
            )
        vis_x, vis_y = np.nonzero(visible_mask_slice)
        self.game_map_console.rgb[dest_x_start + vis_x, dest_y_start + vis_y] = (
            blended_tiles
        )

    def _render_actors(self) -> None:
        if not config.SMOOTH_ACTOR_RENDERING_ENABLED:
            self._render_actors_traditional()
            return

        # When smooth rendering is enabled, skip console rendering
        # Actors will be drawn in present() phase
        pass

    def _render_actors_smooth(self, renderer: Renderer) -> None:
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
                self._render_single_actor_smooth(actor, renderer, bounds, vs)

    def _render_single_actor_smooth(
        self, actor: Actor, renderer: Renderer, bounds: Rect, vs: ViewportSystem
    ) -> None:
        """Render a single actor with smooth positioning and lighting."""
        # Get lighting intensity (reuse existing lighting logic)
        light_rgb = self._get_actor_lighting_intensity(actor, bounds)

        # Convert actor's render position to viewport coordinates
        vp_x, vp_y = vs.world_to_screen_float(actor.render_x, actor.render_y)

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
            actor.ch, final_color, screen_pixel_x, screen_pixel_y, light_rgb
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

    def _render_actors_traditional(self) -> None:
        """Original actor rendering method (tile-aligned)."""
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
                self._render_actor(actor)

    def _render_actor(self, a: Actor) -> None:
        if self.current_light_intensity is None:
            return
        # Use render_x and render_y for calculating screen position
        vp_x, vp_y = self.viewport_system.world_to_screen(
            a.render_x,  # pyright: ignore[reportArgumentType]
            a.render_y,  # pyright: ignore[reportArgumentType]
        )

        # We need to round to the nearest tile for console rendering
        vp_x_int, vp_y_int = round(vp_x), round(vp_y)

        if not (
            0 <= vp_x_int < self.game_map_console.width
            and 0 <= vp_y_int < self.game_map_console.height
        ):
            return

        self.game_map_console.rgb["ch"][vp_x_int, vp_y_int] = ord(a.ch)
        base_actor_color: colors.Color = a.color
        # Some simple actors in tests may not define visual_effects.
        visual_effects = getattr(a, "visual_effects", None)
        if visual_effects is not None:
            visual_effects.update()
            flash_color = visual_effects.get_flash_color()
            if flash_color:
                base_actor_color = flash_color
        # Map the actor's position into the lighting array so we can look up
        # the intensity for this tile.
        bounds = self.viewport_system.get_visible_bounds()
        world_left, world_top = bounds.x1, bounds.y1
        light_x, light_y = a.x - world_left, a.y - world_top
        if not (
            0 <= light_x < self.current_light_intensity.shape[0]
            and 0 <= light_y < self.current_light_intensity.shape[1]
        ):
            return
        light_rgb = self.current_light_intensity[light_x, light_y]
        normally_lit_fg_components: colors.Color = (
            max(0, min(255, int(base_actor_color[0] * light_rgb[0]))),
            max(0, min(255, int(base_actor_color[1] * light_rgb[1]))),
            max(0, min(255, int(base_actor_color[2] * light_rgb[2]))),
        )
        final_fg_color = normally_lit_fg_components
        if (
            self.controller.gw.selected_actor == a
            and self.controller.gw.game_map.visible[a.x, a.y]
            and not self.controller.is_targeting_mode()
        ):
            # Give the selected actor a subtle pulse so it's easy to spot.
            final_fg_color = self._apply_pulsating_effect(
                normally_lit_fg_components, base_actor_color
            )
        self.game_map_console.rgb["fg"][vp_x_int, vp_y_int] = final_fg_color

    def _render_selected_actor_highlight(self) -> None:
        if self.controller.is_targeting_mode():
            return
        actor = self.controller.gw.selected_actor
        if actor and self.controller.gw.game_map.visible[actor.x, actor.y]:
            vp_x, vp_y = self.viewport_system.world_to_screen(actor.x, actor.y)
            if 0 <= vp_x < self.width and 0 <= vp_y < self.height:
                self._apply_blended_highlight(
                    vp_x, vp_y, colors.SELECTED_HIGHLIGHT, SELECTION_HIGHLIGHT_ALPHA
                )

    def _apply_blended_highlight(
        self, x: int, y: int, target_color: colors.Color, alpha: float
    ) -> None:
        current_bg = self.game_map_console.rgb["bg"][x, y]
        blended_color = [
            int(target_color[i] * alpha + current_bg[i] * (1.0 - alpha))
            for i in range(3)
        ]
        self.game_map_console.rgb["bg"][x, y] = [
            max(0, min(255, c)) for c in blended_color
        ]

    def _apply_replacement_highlight(self, x: int, y: int, color: colors.Color) -> None:
        self.game_map_console.rgb["bg"][x, y] = color

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
