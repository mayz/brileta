from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from brileta import colors, config
from brileta.environment import tile_types
from brileta.types import InterpolationAlpha, Opacity
from brileta.util import rng
from brileta.util.caching import ResourceCache
from brileta.util.coordinates import (
    PixelCoord,
    Rect,
    RootConsoleTilePos,
)
from brileta.util.glyph_buffer import GlyphBuffer
from brileta.util.live_vars import record_time_live_variable
from brileta.view.render.actor_renderer import ActorRenderer
from brileta.view.render.effects.decals import DecalSystem
from brileta.view.render.effects.effects import EffectLibrary
from brileta.view.render.effects.environmental import EnvironmentalEffectSystem
from brileta.view.render.effects.floating_text import FloatingTextManager
from brileta.view.render.effects.particles import (
    ParticleLayer,
    SubTileParticleSystem,
)
from brileta.view.render.effects.screen_shake import ScreenShake
from brileta.view.render.graphics import GraphicsContext
from brileta.view.render.lighting.base import LightingSystem
from brileta.view.render.shadow_renderer import ShadowRenderer
from brileta.view.render.viewport import ViewportSystem

from .base import View

if TYPE_CHECKING:
    from brileta.controller import Controller, FrameManager
    from brileta.game.actors import Actor
    from brileta.game.lights import DirectionalLight

_rng = rng.get("effects.animation")


# Viewport defaults used when initializing views before they are resized.
DEFAULT_VIEWPORT_WIDTH = config.SCREEN_WIDTH
DEFAULT_VIEWPORT_HEIGHT = 40  # Initial height before layout adjustments


class WorldView(View):
    """View responsible for rendering the game world (map, actors, effects)."""

    # Extra tiles rendered around the viewport edges for smooth sub-tile scrolling.
    # When the camera moves between tiles, we offset the rendered texture by the
    # fractional amount. The padding ensures there's always content to show at edges.
    _SCROLL_PADDING: int = 1

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
            DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT
        )
        # Initialize dimension-dependent resources (glyph buffers, particle system,
        # etc.) via set_bounds. This avoids duplicating the creation logic.
        self.set_bounds(0, 0, DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT)
        self.effect_library = EffectLibrary()
        self.floating_text_manager = FloatingTextManager()
        self._gpu_actor_lightmap_texture: Any | None = None
        self._gpu_actor_lightmap_viewport_origin: tuple[int, int] | None = None
        # Note: No on_evict callback here because _active_background_texture keeps
        # an external reference to cached textures. Releasing on eviction would
        # invalidate that reference. Textures are cleaned up by GC instead.
        self._texture_cache = ResourceCache[tuple, Any](
            name="WorldViewCache",
            max_size=5,
        )
        self._active_background_texture: Any | None = None
        self._light_overlay_texture: Any | None = None
        # Screen shake offset in tiles for sub-tile rendering
        self._shake_offset: tuple[float, float] = (0.0, 0.0)
        # Camera fractional offset for smooth scrolling (set each frame in present())
        self.camera_frac_offset: tuple[float, float] = (0.0, 0.0)
        self.shadow_renderer = ShadowRenderer(
            game_map=controller.gw.game_map,
            viewport_system=self.viewport_system,
            graphics=self.graphics,
        )
        self.actor_renderer = ActorRenderer(
            viewport_system=self.viewport_system,
            graphics=self.graphics,
            shadow_renderer=self.shadow_renderer,
        )
        # Cumulative game time for decal age tracking
        self._game_time: float = 0.0
        from brileta.view.render.effects.atmospheric import (
            AtmosphericConfig,
            AtmosphericLayerSystem,
        )

        self.atmospheric_system = AtmosphericLayerSystem(
            AtmosphericConfig.create_default()
        )

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override set_bounds to update viewport and console dimensions."""
        # Only perform resize logic if the dimensions have actually changed.
        if self.width != (x2 - x1) or self.height != (y2 - y1):
            super().set_bounds(x1, y1, x2, y2)
            # Update the existing viewport's size instead of replacing it.
            self.viewport_system.viewport.resize(self.width, self.height)
            # Glyph buffer includes padding for smooth scrolling.
            pad = self._SCROLL_PADDING
            self.map_glyph_buffer = GlyphBuffer(
                self.width + 2 * pad, self.height + 2 * pad
            )
            # Source colors for the GPU compose path (light appearance + animation,
            # before blending with dark colors and light intensity).
            self.light_source_glyph_buffer = GlyphBuffer(
                self.width + 2 * pad, self.height + 2 * pad
            )
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
            game_time = self.controller.clock.last_time
            final_color = self.actor_renderer.apply_pulsating_effect(
                color, color, game_time
            )
        # Apply camera fractional offset for smooth scrolling alignment
        cam_frac_x, cam_frac_y = self.camera_frac_offset
        root_x = self.x + vp_x - cam_frac_x
        root_y = self.y + vp_y - cam_frac_y
        self.graphics.draw_tile_highlight(root_x, root_y, final_color, alpha)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _get_background_cache_key(self) -> tuple:
        """Generate a hashable key representing the state of the static background.

        Uses the actual visible bounds computed by get_visible_bounds() to ensure
        the cache key matches exactly what _render_map_unlit will render. The
        fractional camera offset is handled at presentation time for smooth scrolling.

        IMPORTANT: The bounds must be integer tile positions, not sub-tile floats.
        get_visible_bounds() uses round(camera.world_x/y) internally, so the bounds
        only change when the camera crosses an integer tile boundary. This ensures
        the cache hits during smooth sub-tile scrolling.
        """
        gw = self.controller.gw
        vs = self.viewport_system

        # Use actual visible bounds as the key - this is what determines
        # which tiles are rendered in _render_map_unlit via screen_to_world.
        # Explicit int() casts document the intent and guard against any future
        # changes that might introduce floats.
        bounds = vs.get_visible_bounds()
        bounds_key = (int(bounds.x1), int(bounds.y1), int(bounds.x2), int(bounds.y2))

        map_key = gw.game_map.structural_revision

        # The background cache must be invalidated whenever the player moves,
        # because movement updates the `explored` map, which changes the appearance
        # of the unlit background. The new `exploration_revision` is the direct
        # source of truth for this.
        exploration_key = gw.game_map.exploration_revision

        return (bounds_key, map_key, exploration_key)

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
        # Background and light overlay use different cache_key_suffix values to
        # ensure they get separate GPU render targets (they have the same dimensions).
        if config.DEBUG_DISABLE_BACKGROUND_CACHE:
            # Debug mode: always re-render (bypasses cache entirely)
            self._render_map_unlit()
        else:
            cache_key = self._get_background_cache_key()
            cached = self._texture_cache.get(cache_key)

            if cached is None:
                # CACHE MISS: Re-render the static background GlyphBuffer
                self._render_map_unlit()
                # Store a marker (not the texture) to indicate this key was rendered
                self._texture_cache.store(cache_key, True)

        # Set noise parameters once per frame so the shader produces stable
        # sub-tile brightness patterns that match the map's decoration seed.
        # The tile offset converts buffer-space tile indices to world coordinates
        # so the noise pattern stays anchored to world tiles during scrolling.
        graphics.set_noise_seed(gw.game_map.decoration_seed)
        bounds = vs.get_visible_bounds()
        pad = self._SCROLL_PADDING
        graphics.set_noise_tile_offset(
            bounds.x1 - vs.offset_x - pad,
            bounds.y1 - vs.offset_y - pad,
        )

        # Render the GlyphBuffer to a texture. The TextureRenderer's change
        # detection skips re-rendering if the buffer hasn't changed.
        with record_time_live_variable("time.render.bg_texture_upload_ms"):
            texture = graphics.render_glyph_buffer_to_texture(
                self.map_glyph_buffer, cache_key_suffix="bg"
            )

        self._active_background_texture = texture

        # Generate the light overlay DATA (GlyphBuffer)
        if config.DEBUG_DISABLE_LIGHT_OVERLAY:
            self._light_overlay_texture = None
            self._gpu_actor_lightmap_texture = None
            self._gpu_actor_lightmap_viewport_origin = None
        else:
            self._light_overlay_texture = self._render_light_overlay_gpu_compose(
                graphics,
                texture,
            )

        # Update dynamic systems that need to process every frame.
        # actor.update_render_position is a legacy call from the old rendering
        # system and is no longer needed with fixed-timestep interpolation.

        # Visual effects like particles and environmental effects should update based
        # on the frame's delta_time for maximum smoothness, not the fixed
        # logic timestep or the interpolation alpha.
        self.particle_system.update(delta_time)
        self.environmental_system.update(delta_time)
        self.floating_text_manager.update(delta_time)
        self.atmospheric_system.update(delta_time)
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

        vs = self.viewport_system
        pad = self._SCROLL_PADDING

        # Calculate pixel offsets for smooth scrolling.
        # The background texture is larger than the viewport (includes padding),
        # so we need to offset it to align the visible portion correctly.
        base_px_x, base_px_y = graphics.console_to_screen_coords(0.0, 0.0)

        # 1. Screen shake offset (existing behavior)
        shake_tile_x, shake_tile_y = self._shake_offset
        shake_px_x, shake_px_y = graphics.console_to_screen_coords(
            shake_tile_x, shake_tile_y
        )
        shake_offset_x = shake_px_x - base_px_x
        shake_offset_y = shake_px_y - base_px_y

        # 2. Camera fractional offset for smooth sub-tile scrolling.
        # The camera position is rounded when selecting which tiles to render.
        # The fractional part tells us how far between tiles the camera actually is.
        # We offset the texture in the opposite direction to compensate.
        cam_frac_x, cam_frac_y = vs.get_camera_fractional_offset()
        # Store for use by actor/particle rendering methods
        self.camera_frac_offset = (cam_frac_x, cam_frac_y)
        cam_px_x, cam_px_y = graphics.console_to_screen_coords(-cam_frac_x, -cam_frac_y)
        cam_offset_x = cam_px_x - base_px_x
        cam_offset_y = cam_px_y - base_px_y

        # 3. Padding offset: the texture starts 1 tile before the viewport origin,
        # so we shift it back by the padding amount.
        pad_px_x, pad_px_y = graphics.console_to_screen_coords(-pad, -pad)
        pad_offset_x = pad_px_x - base_px_x
        pad_offset_y = pad_px_y - base_px_y

        # Combined offset for background rendering
        offset_x_pixels = shake_offset_x + cam_offset_x + pad_offset_x
        offset_y_pixels = shake_offset_y + cam_offset_y + pad_offset_y

        # The padded texture dimensions
        tex_width = self.width + 2 * pad
        tex_height = self.height + 2 * pad

        # 1. Present the cached unlit background with combined offset
        if self._active_background_texture:
            with record_time_live_variable("time.render.present_background_ms"):
                graphics.draw_background(
                    self._active_background_texture,
                    self.x,
                    self.y,
                    tex_width,
                    tex_height,
                    offset_x_pixels,
                    offset_y_pixels,
                )

        # 2. Present the dynamic light overlay on top of the background
        if self._light_overlay_texture:
            with record_time_live_variable("time.render.present_light_overlay_ms"):
                graphics.draw_background(
                    self._light_overlay_texture,
                    self.x,
                    self.y,
                    tex_width,
                    tex_height,
                    offset_x_pixels,
                    offset_y_pixels,
                )

        # Compute visible bounds for atmospheric effects BEFORE applying shake.
        # This ensures atmospheric effects align with the background/light overlay,
        # which were rendered with the original (unshaken) camera position in draw().
        vs = self.viewport_system
        visible_bounds = vs.get_visible_bounds()
        viewport_offset = (visible_bounds.x1, visible_bounds.y1)

        # 3. Apply shake to camera for actor/particle rendering
        original_cam_x = vs.camera.world_x
        original_cam_y = vs.camera.world_y
        vs.camera.world_x += shake_tile_x
        vs.camera.world_y += shake_tile_y

        viewport_bounds = Rect.from_bounds(0, 0, self.width - 1, self.height - 1)
        # Include camera fractional offset in view_offset for particles/decals/etc.
        # This ensures they shift with the background during smooth scrolling.
        view_offset = (self.x - cam_frac_x, self.y - cam_frac_y)
        viewport_size = (self.width, self.height)
        map_size = (
            self.controller.gw.game_map.width,
            self.controller.gw.game_map.height,
        )
        px_left, px_top = graphics.console_to_screen_coords(self.x, self.y)
        px_right, px_bottom = graphics.console_to_screen_coords(
            self.x + self.width, self.y + self.height
        )
        px_left += offset_x_pixels
        px_right += offset_x_pixels
        px_top += offset_y_pixels
        px_bottom += offset_y_pixels

        if config.ATMOSPHERIC_EFFECTS_ENABLED:
            with record_time_live_variable("time.render.atmospheric_ms"):
                active_layers = self.atmospheric_system.get_active_layers()
                # Render mist first, then shadows, to keep shadows readable on top.
                active_layers.sort(
                    key=lambda layer_state: (
                        0 if layer_state[0].blend_mode == "lighten" else 1
                    )
                )
                sky_exposure_texture = None
                explored_texture = None
                visible_texture = None
                if self.lighting_system is not None:
                    get_sky_exposure_texture = getattr(
                        self.lighting_system, "get_sky_exposure_texture", None
                    )
                    if callable(get_sky_exposure_texture):
                        sky_exposure_texture = get_sky_exposure_texture()
                    get_explored_texture = getattr(
                        self.lighting_system, "get_explored_texture", None
                    )
                    if callable(get_explored_texture):
                        explored_texture = get_explored_texture()
                    get_visible_texture = getattr(
                        self.lighting_system, "get_visible_texture", None
                    )
                    if callable(get_visible_texture):
                        visible_texture = get_visible_texture()

                for layer, state in active_layers:
                    effective_strength = layer.strength
                    if layer.disable_when_overcast:
                        coverage = self.atmospheric_system.config.cloud_coverage
                        # Keep a baseline shadow presence while still responding to
                        # coverage.
                        coverage_scale = 0.35 + 0.65 * (1.0 - coverage)
                        effective_strength *= max(0.0, min(1.0, coverage_scale))

                    graphics.set_atmospheric_layer(
                        viewport_offset,
                        viewport_size,
                        map_size,
                        layer.sky_exposure_threshold,
                        sky_exposure_texture,
                        explored_texture,
                        visible_texture,
                        layer.noise_scale,
                        layer.noise_threshold_low,
                        layer.noise_threshold_high,
                        effective_strength,
                        layer.tint_color,
                        (state.drift_offset_x, state.drift_offset_y),
                        state.turbulence_offset,
                        layer.turbulence_strength,
                        layer.turbulence_scale,
                        layer.blend_mode,
                        (
                            round(px_left),
                            round(px_top),
                            round(px_right),
                            round(px_bottom),
                        ),
                    )

        # Render persistent decals (blood splatters, etc.) on the floor
        with record_time_live_variable("time.render.decals_ms"):
            graphics.render_decals(
                self.decal_system,
                viewport_bounds,
                view_offset,
                self.viewport_system,
                self._game_time,
            )

        with record_time_live_variable("time.render.particles_under_actors_ms"):
            graphics.render_particles(
                self.particle_system,
                ParticleLayer.UNDER_ACTORS,
                viewport_bounds,
                view_offset,
                self.viewport_system,
            )

        visible_actors_for_frame: list[Actor] | None = None
        if config.SHADOWS_ENABLED or config.SMOOTH_ACTOR_RENDERING_ENABLED:
            actor_bounds = vs.get_visible_bounds()
            visible_actors_for_frame = [
                actor
                for actor in self.actor_renderer.get_sorted_visible_actors(
                    actor_bounds, self.controller.gw
                )
                if self.controller.gw.game_map.visible[actor.x, actor.y]
            ]

        set_gpu_actor_lighting_context = getattr(
            graphics, "set_actor_lighting_gpu_context", None
        )
        if callable(set_gpu_actor_lighting_context):
            if (
                self._gpu_actor_lightmap_texture is not None
                and self._gpu_actor_lightmap_viewport_origin is not None
            ):
                set_gpu_actor_lighting_context(
                    self._gpu_actor_lightmap_texture,
                    self._gpu_actor_lightmap_viewport_origin,
                )
            else:
                set_gpu_actor_lighting_context(None, None)

        with (
            record_time_live_variable("time.render.actor_shadows_ms"),
            graphics.shadow_pass(),
        ):
            directional_light = self._get_directional_light()
            self.shadow_renderer.game_map = self.controller.gw.game_map
            self.shadow_renderer.viewport_system = self.viewport_system
            self.shadow_renderer.graphics = graphics
            self.shadow_renderer.render_actor_shadows(
                alpha,
                visible_actors=visible_actors_for_frame,
                directional_light=directional_light,
                lights=self.controller.gw.lights,
                view_origin=(float(self.x), float(self.y)),
                camera_frac_offset=self.camera_frac_offset,
            )

        self.actor_renderer.render_actors(
            alpha,
            game_world=self.controller.gw,
            camera_frac_offset=self.camera_frac_offset,
            view_origin=(float(self.x), float(self.y)),
            visible_actors=visible_actors_for_frame,
            viewport_bounds=actor_bounds
            if visible_actors_for_frame is not None
            else None,
            smooth=config.SMOOTH_ACTOR_RENDERING_ENABLED,
            game_time=self.controller.clock.last_time,
            is_combat=self.controller.is_combat_mode(),
        )

        # Render highlights and mode-specific UI on top of actors
        # Render all modes in the stack (bottom-to-top) so higher modes draw on top
        with record_time_live_variable("time.render.active_mode_world_ms"):
            for mode in self.controller.mode_stack:
                mode.render_world()

        with record_time_live_variable("time.render.particles_over_actors_ms"):
            graphics.render_particles(
                self.particle_system,
                ParticleLayer.OVER_ACTORS,
                viewport_bounds,
                view_offset,
                self.viewport_system,
            )

        with record_time_live_variable("time.render.floating_text_ms"):
            self.floating_text_manager.render(
                graphics,
                self.viewport_system,
                view_offset,
                self.controller.gw,
            )

        if config.ENVIRONMENTAL_EFFECTS_ENABLED:
            with record_time_live_variable("time.render.environmental_effects_ms"):
                self.environmental_system.render_effects(
                    graphics,
                    viewport_bounds,
                    view_offset,
                )

        if config.DEBUG_SHOW_TILE_GRID:
            graphics.draw_debug_tile_grid(
                (self.x, self.y),
                (self.width, self.height),
                (offset_x_pixels, offset_y_pixels),
            )

        # Restore camera position after rendering
        vs.camera.set_position(original_cam_x, original_cam_y)

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    @record_time_live_variable("time.render.map_unlit_ms")
    def _render_map_unlit(self) -> None:
        """Renders the static, unlit background of the game world.

        The glyph buffer is larger than the viewport by _SCROLL_PADDING tiles on
        each edge. This allows smooth sub-tile scrolling: when the camera moves
        between tiles, we offset the rendered texture by the fractional amount,
        and the padding ensures there's always content at the edges.

        Uses vectorized numpy operations for performance.
        """
        gw = self.controller.gw
        vs = self.viewport_system
        pad = self._SCROLL_PADDING
        game_map = gw.game_map

        # Clear the console for this view to a default black background.
        self.map_glyph_buffer.clear()

        # Get world bounds for coordinate conversion.
        # viewport_to_world formula: world = vp - offset + bounds_origin
        # And vp = buf - pad, so: world = buf - pad - offset + bounds_origin
        bounds = vs.viewport.get_world_bounds(vs.camera)
        world_origin_x = bounds.x1 - vs.offset_x - pad
        world_origin_y = bounds.y1 - vs.offset_y - pad

        buf_width = self.map_glyph_buffer.width
        buf_height = self.map_glyph_buffer.height

        # Create coordinate arrays for all buffer positions
        buf_x_coords = np.arange(buf_width)
        buf_y_coords = np.arange(buf_height)
        buf_x_grid, buf_y_grid = np.meshgrid(buf_x_coords, buf_y_coords, indexing="ij")

        # Convert buffer coords to world coords
        world_x_grid = buf_x_grid + world_origin_x
        world_y_grid = buf_y_grid + world_origin_y

        # Mask for tiles within map bounds
        in_bounds_mask = (
            (world_x_grid >= 0)
            & (world_x_grid < game_map.width)
            & (world_y_grid >= 0)
            & (world_y_grid < game_map.height)
        )

        # Get the valid world coordinates
        valid_world_x = world_x_grid[in_bounds_mask]
        valid_world_y = world_y_grid[in_bounds_mask]

        # Mask for explored tiles (subset of in-bounds)
        explored_mask = game_map.explored[valid_world_x, valid_world_y]

        if not np.any(explored_mask):
            return

        # Get final coordinates for explored tiles
        final_world_x = valid_world_x[explored_mask]
        final_world_y = valid_world_y[explored_mask]
        final_buf_x = buf_x_grid[in_bounds_mask][explored_mask]
        final_buf_y = buf_y_grid[in_bounds_mask][explored_mask]

        # Get dark appearance data for all explored tiles at once
        dark_app = game_map.dark_appearance_map[final_world_x, final_world_y]

        # Extract character codes and colors
        chars = dark_app["ch"]
        fg_rgb = dark_app["fg"]  # Shape: (N, 3)
        bg_rgb = dark_app["bg"]  # Shape: (N, 3)

        # Apply per-tile glyph and color decoration for terrain variety
        unlit_tile_ids = game_map.tiles[final_world_x, final_world_y]
        tile_types.apply_terrain_decoration(
            chars,
            fg_rgb,
            bg_rgb,
            unlit_tile_ids,
            final_world_x,
            final_world_y,
            game_map.decoration_seed,
        )

        # Add alpha channel (255) to make RGBA
        alpha = np.full((len(chars), 1), 255, dtype=np.uint8)
        fg_rgba = np.hstack((fg_rgb, alpha))
        bg_rgba = np.hstack((bg_rgb, alpha))

        # Assign to glyph buffer using coordinate indexing
        self.map_glyph_buffer.data["ch"][final_buf_x, final_buf_y] = chars
        self.map_glyph_buffer.data["fg"][final_buf_x, final_buf_y] = fg_rgba
        self.map_glyph_buffer.data["bg"][final_buf_x, final_buf_y] = bg_rgba

        # Write sub-tile jitter amplitude so the fragment shader can apply
        # per-pixel brightness variation within each tile cell.
        self.map_glyph_buffer.data["noise"][final_buf_x, final_buf_y] = (
            tile_types.get_sub_tile_jitter_map(unlit_tile_ids)
        )

    def _get_directional_light(self) -> DirectionalLight | None:
        """Return the first directional/global sun light active in the world."""
        from brileta.game.lights import DirectionalLight

        gw = self.controller.gw
        return next(
            (
                light
                for light in gw.get_global_lights()
                if isinstance(light, DirectionalLight)
            ),
            None,
        )

    def _update_actor_particles(self) -> None:
        """Emit particles from actors with particle emitters."""
        gw = self.controller.gw

        # Get all actors and check for particle emitters
        for actor in gw.actors:
            visual_effects = actor.visual_effects
            if visual_effects is not None and visual_effects.has_continuous_effects():
                # Create an effect context for this actor
                from brileta.view.render.effects.effects import EffectContext

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
        tiles_to_update = _rng.sample(
            animated_tiles, min(num_to_update, len(animated_tiles))
        )

        # Update each selected tile using random walk
        # Use small step size for subtle, organic color drift
        step_size = 80

        for world_x, world_y in tiles_to_update:
            state = animation_state[world_x, world_y]

            # Random walk for foreground RGB values
            for i in range(3):
                offset = _rng.randint(-step_size, step_size)
                new_val = int(state["fg_values"][i]) + offset
                state["fg_values"][i] = max(0, min(1000, new_val))

            # Random walk for background RGB values
            for i in range(3):
                offset = _rng.randint(-step_size, step_size)
                new_val = int(state["bg_values"][i]) + offset
                state["bg_values"][i] = max(0, min(1000, new_val))

            # Glyph is always visible - the "flicker" effect comes naturally
            # when fg and bg colors drift close together, making the glyph
            # blend into the background.

            # Write back the updated state
            animation_state[world_x, world_y] = state

    def _update_mouse_tile_location(self) -> None:
        """Update the stored world-space mouse tile based on the current camera."""
        fm: FrameManager = self.controller.frame_manager

        # Use graphics.pixel_to_tile which correctly handles letterboxing offsets,
        # rather than coordinate_converter.pixel_to_tile which assumes no offset.
        px_x: PixelCoord = fm.cursor_manager.mouse_pixel_x
        px_y: PixelCoord = fm.cursor_manager.mouse_pixel_y
        root_tile_pos: RootConsoleTilePos = self.controller.graphics.pixel_to_tile(
            px_x, px_y
        )
        world_tile_pos = fm.get_world_coords_from_root_tile_coords(root_tile_pos)

        self.controller.gw.mouse_tile_location_on_map = world_tile_pos

    def _apply_tile_light_animations(
        self,
        light_fg_rgb: np.ndarray,
        light_bg_rgb: np.ndarray,
        animation_params: np.ndarray,
        animation_state: np.ndarray,
        valid_exp_x: np.ndarray,
        valid_exp_y: np.ndarray,
    ) -> None:
        """Apply per-tile color modulation for animated terrain tiles."""
        animates_mask = animation_params["animates"][valid_exp_x, valid_exp_y]
        if not np.any(animates_mask):
            return

        anim_indices = np.nonzero(animates_mask)[0]
        anim_rel_x = valid_exp_x[animates_mask]
        anim_rel_y = valid_exp_y[animates_mask]

        fg_var = animation_params["fg_variation"][anim_rel_x, anim_rel_y].astype(
            np.int32
        )
        bg_var = animation_params["bg_variation"][anim_rel_x, anim_rel_y].astype(
            np.int32
        )
        fg_vals = animation_state["fg_values"][anim_rel_x, anim_rel_y].astype(np.int32)
        bg_vals = animation_state["bg_values"][anim_rel_x, anim_rel_y].astype(np.int32)

        base_fg = light_fg_rgb[anim_indices].astype(np.int32)
        anim_fg = base_fg + fg_var * fg_vals // 1000 - fg_var // 2
        light_fg_rgb[anim_indices] = np.clip(anim_fg, 0, 255).astype(np.uint8)

        base_bg = light_bg_rgb[anim_indices].astype(np.int32)
        anim_bg = base_bg + bg_var * bg_vals // 1000 - bg_var // 2
        light_bg_rgb[anim_indices] = np.clip(anim_bg, 0, 255).astype(np.uint8)

    @record_time_live_variable("time.render.light_overlay_ms")
    def _render_light_overlay_gpu_compose(
        self,
        graphics: GraphicsContext,
        dark_texture: Any,
    ) -> Any:
        """Render the light overlay via GPU compose pass (no full lightmap readback)."""
        if self.lighting_system is None:
            raise RuntimeError(
                "Light overlay is enabled but no lighting system is configured."
            )

        compute_texture_fn = getattr(
            self.lighting_system, "compute_lightmap_texture", None
        )
        if not callable(compute_texture_fn):
            raise RuntimeError(
                "Lighting system is missing compute_lightmap_texture(), required for "
                "GPU light overlay composition."
            )

        compose_fn = getattr(graphics, "compose_light_overlay_gpu", None)
        if not callable(compose_fn):
            raise RuntimeError(
                "Graphics context is missing compose_light_overlay_gpu(), required "
                "for GPU light overlay composition."
            )

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
        if dest_width <= 0 or dest_height <= 0:
            raise RuntimeError(
                "Computed invalid overlay viewport dimensions: "
                f"{dest_width}x{dest_height}."
            )

        viewport_bounds = Rect(world_left, world_top, dest_width, dest_height)
        lightmap_texture = compute_texture_fn(viewport_bounds)
        if lightmap_texture is None:
            raise RuntimeError(
                "Lighting system returned no lightmap texture for GPU overlay "
                "composition."
            )
        # Keep GPU actor-lighting context in sync with the latest lightmap frame.
        self._gpu_actor_lightmap_texture = lightmap_texture
        self._gpu_actor_lightmap_viewport_origin = (
            viewport_bounds.x1,
            viewport_bounds.y1,
        )

        pad = self._SCROLL_PADDING
        buf_width = self.width + 2 * pad
        buf_height = self.height + 2 * pad
        light_source_buffer = self.light_source_glyph_buffer
        light_source_buffer.clear()
        # Buffer-space visible mask used by the GPU compose shader. Keeping this
        # in the same coordinate space as the glyph buffers avoids map/viewport
        # axis mismatches when maps are centered in larger viewports.
        visible_mask_buffer = np.zeros((buf_width, buf_height), dtype=np.bool_)

        world_slice = (
            slice(world_left, world_right + 1),
            slice(world_top, world_bottom + 1),
        )
        explored_mask_slice = gw.game_map.explored[world_slice]
        visible_mask_slice = gw.game_map.visible[world_slice]
        if np.any(explored_mask_slice):
            light_app_slice = gw.game_map.light_appearance_map[world_slice]
            exp_x, exp_y = np.nonzero(explored_mask_slice)
            buffer_x = vs.offset_x + exp_x + pad
            buffer_y = vs.offset_y + exp_y + pad
            valid_mask = (
                (buffer_x >= 0)
                & (buffer_x < buf_width)
                & (buffer_y >= 0)
                & (buffer_y < buf_height)
            )
            if np.any(valid_mask):
                final_buf_x = buffer_x[valid_mask]
                final_buf_y = buffer_y[valid_mask]
                valid_exp_x = exp_x[valid_mask]
                valid_exp_y = exp_y[valid_mask]
                valid_visible = visible_mask_slice[valid_exp_x, valid_exp_y]

                light_chars = light_app_slice["ch"][valid_exp_x, valid_exp_y]
                light_fg_rgb = light_app_slice["fg"][valid_exp_x, valid_exp_y]
                light_bg_rgb = light_app_slice["bg"][valid_exp_x, valid_exp_y]

                # Apply per-tile glyph and color decoration for terrain variety.
                # valid_exp_x/y are offsets within the world slice, so add back
                # world_left/world_top to get true world coordinates for the hash.
                world_tile_ids = gw.game_map.tiles[world_slice][
                    valid_exp_x, valid_exp_y
                ]
                tile_types.apply_terrain_decoration(
                    light_chars,
                    light_fg_rgb,
                    light_bg_rgb,
                    world_tile_ids,
                    valid_exp_x + world_left,
                    valid_exp_y + world_top,
                    gw.game_map.decoration_seed,
                )

                animation_params = gw.game_map.animation_params[world_slice]
                animation_state = gw.game_map.animation_state[world_slice]
                self._apply_tile_light_animations(
                    light_fg_rgb,
                    light_bg_rgb,
                    animation_params,
                    animation_state,
                    valid_exp_x,
                    valid_exp_y,
                )

                alpha_channel = np.full((len(light_fg_rgb), 1), 255, dtype=np.uint8)
                light_fg_rgba = np.hstack((light_fg_rgb, alpha_channel))
                light_bg_rgba = np.hstack((light_bg_rgb, alpha_channel))
                light_source_buffer.data["ch"][final_buf_x, final_buf_y] = light_chars
                light_source_buffer.data["fg"][final_buf_x, final_buf_y] = light_fg_rgba
                light_source_buffer.data["bg"][final_buf_x, final_buf_y] = light_bg_rgba

                # Write sub-tile jitter amplitude for the light source buffer too,
                # so lit tiles also get per-pixel brightness variation.
                light_source_buffer.data["noise"][final_buf_x, final_buf_y] = (
                    tile_types.get_sub_tile_jitter_map(world_tile_ids)
                )

                if np.any(valid_visible):
                    visible_mask_buffer[
                        final_buf_x[valid_visible], final_buf_y[valid_visible]
                    ] = True

        with record_time_live_variable("time.render.light_texture_upload_ms"):
            light_source_texture = graphics.render_glyph_buffer_to_texture(
                light_source_buffer,
                cache_key_suffix="light_source",
            )

        composed_texture = compose_fn(
            dark_texture=dark_texture,
            light_texture=light_source_texture,
            lightmap_texture=lightmap_texture,
            visible_mask_buffer=visible_mask_buffer,
            viewport_bounds=viewport_bounds,
            viewport_offset=(vs.offset_x, vs.offset_y),
            pad_tiles=pad,
        )
        if composed_texture is None:
            raise RuntimeError(
                "GPU light overlay composition produced no texture while lighting "
                "is enabled."
            )
        return composed_texture
