from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from catley import colors, config
from catley.environment.tile_types import TileTypeID, get_shadow_height_map
from catley.types import InterpolationAlpha, Opacity
from catley.util import rng
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
from catley.view.render.effects.floating_text import FloatingTextManager
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
    from catley.game.lights import DirectionalLight

_rng = rng.get("effects.animation")


# Viewport defaults used when initializing views before they are resized.
DEFAULT_VIEWPORT_WIDTH = config.SCREEN_WIDTH
DEFAULT_VIEWPORT_HEIGHT = 40  # Initial height before layout adjustments

# Rendering effects
PULSATION_PERIOD = 2.0  # Seconds for full pulsation cycle (selected actor)
PULSATION_MAX_BLEND_ALPHA: Opacity = Opacity(0.5)  # Maximum alpha for pulsation
LUMINANCE_THRESHOLD = 127.5  # For determining light vs dark colors


class _SunShadowParams(NamedTuple):
    """Pre-computed directional shadow geometry shared by terrain and actor passes."""

    dir_x: float
    dir_y: float
    length_scale: float


# Combat outline shimmer effect (shimmering glyph outlines on targetable enemies)
COMBAT_OUTLINE_SHIMMER_PERIOD = 2.4  # Seconds for full shimmer cycle
COMBAT_OUTLINE_MIN_ALPHA: Opacity = Opacity(0.4)  # Minimum alpha during shimmer
COMBAT_OUTLINE_MAX_ALPHA: Opacity = Opacity(0.85)  # Maximum alpha during shimmer

# Contextual target outline (exploration mode)
CONTEXTUAL_OUTLINE_ALPHA: Opacity = Opacity(0.70)  # Solid outline opacity


class WorldView(View):
    """View responsible for rendering the game world (map, actors, effects)."""

    # Extra tiles rendered around the viewport edges for smooth sub-tile scrolling.
    # When the camera moves between tiles, we offset the rendered texture by the
    # fractional amount. The padding ensures there's always content to show at edges.
    _SCROLL_PADDING: int = 1
    _SUN_SHADOW_OUTDOOR_REGION_TYPES: frozenset[str] = frozenset(
        {"outdoor", "exterior", "test_outdoor"}
    )
    _SUN_SHADOW_OUTDOOR_TILE_IDS: frozenset[int] = frozenset(
        {
            int(TileTypeID.OUTDOOR_FLOOR),
            int(TileTypeID.GRASS),
            int(TileTypeID.DIRT_PATH),
            int(TileTypeID.GRAVEL),
        }
    )

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
        # Per-frame multiplicative light scales for actors shadowed by other actors.
        # Keyed by id(actor) because Actor objects are mutable and not hash-stable.
        self._actor_shadow_receive_light_scale: dict[int, float] = {}
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
        self._camera_frac_offset: tuple[float, float] = (0.0, 0.0)
        # Cumulative game time for decal age tracking
        self._game_time: float = 0.0
        from catley.view.render.effects.atmospheric import (
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
            final_color = self._apply_pulsating_effect(color, color)
        # Apply camera fractional offset for smooth scrolling alignment
        cam_frac_x, cam_frac_y = self._camera_frac_offset
        root_x = self.x + vp_x - cam_frac_x
        root_y = self.y + vp_y - cam_frac_y
        self.graphics.draw_tile_highlight(root_x, root_y, final_color, alpha)

    def render_actor_outline(
        self, actor: Actor, color: colors.Color, alpha: float
    ) -> None:
        """Render an outlined glyph for an actor at its current position.

        Used for combat targeting to show a shimmering outline around the
        enemy's glyph shape. The actor must be visible.

        Args:
            actor: The actor to render an outline for
            color: RGB color for the outline
            alpha: Opacity of the outline (0.0-1.0)
        """
        if not self.controller.gw.game_map.visible[actor.x, actor.y]:
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
        cam_frac_x, cam_frac_y = self._camera_frac_offset
        vp_x -= cam_frac_x
        vp_y -= cam_frac_y

        root_x = self.x + vp_x
        root_y = self.y + vp_y
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

    def get_shimmer_alpha(self, period: float = COMBAT_OUTLINE_SHIMMER_PERIOD) -> float:
        """Calculate oscillating alpha for shimmer effect.

        Returns an alpha value that smoothly oscillates between configured
        min and max values over the specified period, creating a breathing
        or pulsing visual effect.

        Args:
            period: Duration in seconds for one complete oscillation cycle

        Returns:
            Alpha value between COMBAT_OUTLINE_MIN_ALPHA and COMBAT_OUTLINE_MAX_ALPHA
        """
        game_time = self.controller.clock.last_time
        t = (game_time % period) / period
        # Sinusoidal oscillation from min to max alpha
        normalized = (math.sin(t * 2 * math.pi) + 1) / 2
        return (
            COMBAT_OUTLINE_MIN_ALPHA
            + (COMBAT_OUTLINE_MAX_ALPHA - COMBAT_OUTLINE_MIN_ALPHA) * normalized
        )

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
        self._camera_frac_offset = (cam_frac_x, cam_frac_y)
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
                for actor in self._get_sorted_visible_actors(actor_bounds)
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
            self._render_actor_shadows(
                graphics,
                alpha,
                visible_actors=visible_actors_for_frame,
            )

        if config.SMOOTH_ACTOR_RENDERING_ENABLED:
            self._render_actors_smooth(
                graphics,
                alpha,
                visible_actors=visible_actors_for_frame,
                viewport_bounds=actor_bounds
                if visible_actors_for_frame is not None
                else None,
            )
        else:
            self._render_actors_traditional(graphics, alpha)

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

        # Add alpha channel (255) to make RGBA
        alpha = np.full((len(chars), 1), 255, dtype=np.uint8)
        fg_rgba = np.hstack((fg_rgb, alpha))
        bg_rgba = np.hstack((bg_rgb, alpha))

        # Assign to glyph buffer using coordinate indexing
        self.map_glyph_buffer.data["ch"][final_buf_x, final_buf_y] = chars
        self.map_glyph_buffer.data["fg"][final_buf_x, final_buf_y] = fg_rgba
        self.map_glyph_buffer.data["bg"][final_buf_x, final_buf_y] = bg_rgba

    def _render_actors(self) -> None:
        # Traditional actors are now baked into the cache during the draw phase.
        # This method is now only for presenting dynamic elements on top of the cache.
        # Therefore, we do nothing if smooth rendering is disabled.
        if config.SMOOTH_ACTOR_RENDERING_ENABLED:
            # The present() method calls _render_actors_smooth() directly,
            # so this method can simply be a no-op or pass.
            pass

    def _get_sorted_visible_actors(self, bounds: Rect) -> list[Actor]:
        """Return actors in the viewport sorted for painter-style rendering."""
        gw = self.controller.gw
        actors_in_viewport = gw.actor_spatial_index.get_in_bounds(
            bounds.x1, bounds.y1, bounds.x2, bounds.y2
        )

        return sorted(
            actors_in_viewport,
            key=lambda actor: (
                actor.y,
                getattr(actor, "visual_scale", 1.0),
                actor == gw.player,
            ),
        )

    def _get_directional_light(self) -> DirectionalLight | None:
        """Return the first directional/global sun light active in the world."""
        from catley.game.lights import DirectionalLight

        gw = self.controller.gw
        return next(
            (
                light
                for light in gw.get_global_lights()
                if isinstance(light, DirectionalLight)
            ),
            None,
        )

    @staticmethod
    def _compute_sun_shadow_params(
        directional_light: DirectionalLight,
    ) -> _SunShadowParams | None:
        """Derive shadow direction and length scale from a directional light.

        Returns ``None`` when the light produces no usable shadow (e.g. zero
        direction vector or sun directly overhead).
        """
        raw_dx = -directional_light.direction.x
        raw_dy = -directional_light.direction.y
        length = math.hypot(raw_dx, raw_dy)
        if length <= 1e-6:
            return None

        dir_x = raw_dx / length
        dir_y = raw_dy / length

        elevation = max(0.0, min(90.0, directional_light.elevation_degrees))
        if elevation >= 90.0:
            return None

        tan_elev = math.tan(math.radians(elevation))
        length_scale = 8.0 if tan_elev <= 1e-6 else min(1.0 / tan_elev, 8.0)

        return _SunShadowParams(dir_x, dir_y, length_scale)

    def _get_actor_screen_position(
        self,
        actor: Actor,
        graphics: GraphicsContext,
        vs: ViewportSystem,
        interpolation_alpha: InterpolationAlpha,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute interpolated world/root/pixel coordinates for an actor."""
        alpha_value = float(interpolation_alpha)
        if getattr(actor, "_animation_controlled", False):
            interpolated_x = actor.render_x
            interpolated_y = actor.render_y
        else:
            interpolated_x = actor.prev_x * (1.0 - alpha_value) + actor.x * alpha_value
            interpolated_y = actor.prev_y * (1.0 - alpha_value) + actor.y * alpha_value

        # Apply idle drift so shadows and glyphs remain locked together.
        if (
            actor.visual_effects is not None
            and actor.health is not None
            and actor.health.is_alive()
        ):
            drift_x, drift_y = actor.visual_effects.get_idle_drift_offset()
            interpolated_x += drift_x
            interpolated_y += drift_y

        vp_x, vp_y = vs.world_to_screen_float(interpolated_x, interpolated_y)
        cam_frac_x, cam_frac_y = self._camera_frac_offset
        vp_x -= cam_frac_x
        vp_y -= cam_frac_y

        root_x = self.x + vp_x
        root_y = self.y + vp_y
        screen_pixel_x, screen_pixel_y = graphics.console_to_screen_coords(
            root_x, root_y
        )

        return (
            interpolated_x,
            interpolated_y,
            root_x,
            root_y,
            screen_pixel_x,
            screen_pixel_y,
        )

    def _render_actor_shadows(
        self,
        graphics: GraphicsContext,
        interpolation_alpha: InterpolationAlpha,
        visible_actors: list[Actor] | None = None,
    ) -> None:
        """Render projected glyph shadows for terrain objects and visible actors."""
        self._actor_shadow_receive_light_scale = {}
        if not config.SHADOWS_ENABLED:
            return

        vs = self.viewport_system
        tile_height = float(graphics.tile_dimensions[1])

        # Compute sun shadow params once for both terrain and actor passes.
        directional_light = self._get_directional_light()
        sun_params: _SunShadowParams | None = None
        if directional_light is not None:
            sun_params = self._compute_sun_shadow_params(directional_light)

        # Terrain glyph shadows (boulders, etc.) - independent of actor positions
        self._render_terrain_glyph_shadows(
            graphics,
            tile_height,
            sun_params=sun_params,
        )

        if visible_actors is None:
            gw = self.controller.gw
            bounds = vs.get_visible_bounds()
            visible_actors = [
                actor
                for actor in self._get_sorted_visible_actors(bounds)
                if gw.game_map.visible[actor.x, actor.y]
            ]
        if not visible_actors:
            return

        shadow_casters = [
            actor for actor in visible_actors if getattr(actor, "shadow_height", 0) > 0
        ]
        if not shadow_casters:
            return

        self._render_sun_actor_shadows(
            graphics,
            vs,
            shadow_casters,
            interpolation_alpha,
            tile_height,
            receivers=visible_actors,
            sun_params=sun_params,
        )
        self._render_point_light_actor_shadows(
            graphics,
            vs,
            shadow_casters,
            interpolation_alpha,
            tile_height,
            receivers=visible_actors,
        )

    def _render_sun_actor_shadows(
        self,
        graphics: GraphicsContext,
        vs: ViewportSystem,
        actors: list[Actor],
        interpolation_alpha: InterpolationAlpha,
        tile_height: float,
        receivers: list[Actor] | None = None,
        sun_params: _SunShadowParams | None = None,
    ) -> None:
        """Render actor shadows cast by the directional light."""
        if sun_params is None:
            dl = self._get_directional_light()
            if dl is not None:
                sun_params = self._compute_sun_shadow_params(dl)
        if sun_params is None:
            return

        shadow_receivers = actors if receivers is None else receivers
        shadow_dir_x, shadow_dir_y, shadow_length_scale = sun_params

        for actor in actors:
            # Sun shadows should only appear on tiles that are truly outdoor.
            # Some dungeon regions can carry elevated sky_exposure metadata while
            # still being interior rooms; this guard keeps directional shadows
            # from leaking into indoor spaces.
            if not self._can_render_sun_shadow_at_tile(actor.x, actor.y):
                continue

            shadow_height = float(getattr(actor, "shadow_height", 0))
            shadow_length_tiles = shadow_height * shadow_length_scale
            if shadow_length_tiles <= 0.0:
                continue

            clipped_length_tiles = self._clip_shadow_length_by_walls(
                actor.x, actor.y, shadow_dir_x, shadow_dir_y, shadow_length_tiles
            )
            if clipped_length_tiles <= 0.0:
                continue

            self._accumulate_actor_shadow_receiver_dimming(
                caster=actor,
                receivers=shadow_receivers,
                shadow_dir_x=shadow_dir_x,
                shadow_dir_y=shadow_dir_y,
                shadow_length_tiles=clipped_length_tiles,
                shadow_alpha=config.ACTOR_SHADOW_ALPHA,
                fade_tip=config.ACTOR_SHADOW_FADE_TIP,
            )

            _, _, root_x, root_y, screen_x, screen_y = self._get_actor_screen_position(
                actor, graphics, vs, interpolation_alpha
            )
            self._emit_actor_shadow_quads(
                actor=actor,
                graphics=graphics,
                root_x=root_x,
                root_y=root_y,
                screen_x=screen_x,
                screen_y=screen_y,
                shadow_dir_x=shadow_dir_x,
                shadow_dir_y=shadow_dir_y,
                shadow_length_pixels=clipped_length_tiles * tile_height,
                shadow_alpha=config.ACTOR_SHADOW_ALPHA,
                fade_tip=config.ACTOR_SHADOW_FADE_TIP,
            )

    def _can_render_sun_shadow_at_tile(self, x: int, y: int) -> bool:
        """Return whether a tile should receive directional sun-projected shadows."""
        game_map = self.controller.gw.game_map
        region = game_map.get_region_at((x, y))
        if region is None or region.sky_exposure <= 0.1:
            return False

        if region.region_type in self._SUN_SHADOW_OUTDOOR_REGION_TYPES:
            return True

        tile_id = int(game_map.tiles[x, y])
        return tile_id in self._SUN_SHADOW_OUTDOOR_TILE_IDS

    def _render_terrain_glyph_shadows(
        self,
        graphics: GraphicsContext,
        tile_height: float,
        sun_params: _SunShadowParams | None = None,
    ) -> None:
        """Render projected glyph shadows for small terrain objects (boulders, etc.).

        Tiles whose shadow_height is in (0, _GLYPH_SHADOW_MAX_HEIGHT] get a
        CPU-projected glyph shadow using the same draw_actor_shadow() path as
        actors.  Taller tiles (walls, doors) keep their shader-only tile shadows.
        """
        if sun_params is None:
            dl = self._get_directional_light()
            if dl is not None:
                sun_params = self._compute_sun_shadow_params(dl)
        if sun_params is None:
            return

        shadow_dir_x, shadow_dir_y, shadow_length_scale = sun_params

        gw = self.controller.gw
        vs = self.viewport_system
        game_map = gw.game_map

        # Get viewport bounds
        bounds = vs.get_visible_bounds()
        world_left = max(0, bounds.x1)
        world_top = max(0, bounds.y1)
        world_right = min(game_map.width - 1, bounds.x2)
        world_bottom = min(game_map.height - 1, bounds.y2)

        # Vectorized lookup: find visible tiles with shadow_height 1-2
        # (height 0 = no shadow, 1-2 = glyph shadow, 3+ = staircase shader shadow)
        viewport_tiles = game_map.tiles[
            world_left : world_right + 1, world_top : world_bottom + 1
        ]
        heights = get_shadow_height_map(viewport_tiles)
        visible_slice = game_map.visible[
            world_left : world_right + 1, world_top : world_bottom + 1
        ]
        candidates = np.argwhere((heights > 0) & (heights <= 2) & visible_slice)

        if len(candidates) == 0:
            return

        cam_frac_x, cam_frac_y = self._camera_frac_offset

        for rel_x, rel_y in candidates:
            world_x = world_left + int(rel_x)
            world_y = world_top + int(rel_y)

            # Only render in outdoor areas exposed to sunlight
            if not self._can_render_sun_shadow_at_tile(world_x, world_y):
                continue

            glyph_shadow_height = float(heights[rel_x, rel_y])
            shadow_length_tiles = glyph_shadow_height * shadow_length_scale
            if shadow_length_tiles <= 0.0:
                continue

            # Clip shadow by nearby walls
            clipped_length_tiles = self._clip_shadow_length_by_walls(
                world_x, world_y, shadow_dir_x, shadow_dir_y, shadow_length_tiles
            )
            if clipped_length_tiles <= 0.0:
                continue

            # Get the tile's glyph character for the shadow shape
            ch_code = int(game_map.light_appearance_map[world_x, world_y]["ch"])
            char = chr(ch_code)

            # Convert tile position to screen coordinates
            vp_x, vp_y = vs.world_to_screen(world_x, world_y)
            root_x = self.x + vp_x - cam_frac_x
            root_y = self.y + vp_y - cam_frac_y
            screen_x, screen_y = graphics.console_to_screen_coords(root_x, root_y)

            graphics.draw_actor_shadow(
                char=char,
                screen_x=screen_x,
                screen_y=screen_y,
                shadow_dir_x=shadow_dir_x,
                shadow_dir_y=shadow_dir_y,
                shadow_length_pixels=clipped_length_tiles * tile_height,
                shadow_alpha=config.TERRAIN_GLYPH_SHADOW_ALPHA,
                scale_x=1.0,
                scale_y=1.0,
                fade_tip=config.ACTOR_SHADOW_FADE_TIP,
            )

    def _render_point_light_actor_shadows(
        self,
        graphics: GraphicsContext,
        vs: ViewportSystem,
        actors: list[Actor],
        interpolation_alpha: InterpolationAlpha,
        tile_height: float,
        receivers: list[Actor] | None = None,
    ) -> None:
        """Render actor shadows cast by nearby point lights."""
        from catley.game.lights import DirectionalLight

        gw = self.controller.gw
        shadow_receivers = actors if receivers is None else receivers
        point_lights = [
            light for light in gw.lights if not isinstance(light, DirectionalLight)
        ]
        if not point_lights:
            return

        for actor in actors:
            shadow_height = float(getattr(actor, "shadow_height", 0))
            if shadow_height <= 0.0:
                continue

            (
                _actor_world_x,
                _actor_world_y,
                root_x,
                root_y,
                screen_x,
                screen_y,
            ) = self._get_actor_screen_position(
                actor, graphics, vs, interpolation_alpha
            )

            for light in point_lights:
                radius = float(light.radius)
                if radius <= 0.0:
                    continue

                # Actors should not cast directional shadows from their own lights.
                # A carried torch mainly creates local self-occlusion, not a stable
                # ground-projected self shadow.
                if getattr(light, "owner", None) is actor:
                    continue

                light_x, light_y = light.position
                # Guard against unstable direction when actor and light occupy
                # the same tile (including sub-tile drift jitter).
                if actor.x == int(light_x) and actor.y == int(light_y):
                    continue

                # Use tile-space positions for shadow direction. Idle drift should
                # move the rendered glyph, but not rotate a cardinal shadow into
                # a diagonal one when actor and light are horizontally aligned.
                dir_x = float(actor.x) - float(light_x)
                dir_y = float(actor.y) - float(light_y)
                distance = math.hypot(dir_x, dir_y)

                # Avoid undefined direction when actor and light share the same tile.
                if distance <= 1e-6 or distance > radius:
                    continue

                attenuation = max(0.0, 1.0 - distance / radius)
                light_intensity = float(getattr(light, "intensity", 1.0))
                shadow_alpha = config.ACTOR_SHADOW_ALPHA * attenuation * light_intensity
                if shadow_alpha <= 0.0:
                    continue

                dir_x /= distance
                dir_y /= distance

                clipped_length_tiles = self._clip_shadow_length_by_walls(
                    actor.x, actor.y, dir_x, dir_y, shadow_height
                )
                if clipped_length_tiles <= 0.0:
                    continue

                self._accumulate_actor_shadow_receiver_dimming(
                    caster=actor,
                    receivers=shadow_receivers,
                    shadow_dir_x=dir_x,
                    shadow_dir_y=dir_y,
                    shadow_length_tiles=clipped_length_tiles,
                    shadow_alpha=shadow_alpha,
                    fade_tip=config.ACTOR_SHADOW_FADE_TIP,
                )

                self._emit_actor_shadow_quads(
                    actor=actor,
                    graphics=graphics,
                    root_x=root_x,
                    root_y=root_y,
                    screen_x=screen_x,
                    screen_y=screen_y,
                    shadow_dir_x=dir_x,
                    shadow_dir_y=dir_y,
                    shadow_length_pixels=clipped_length_tiles * tile_height,
                    shadow_alpha=shadow_alpha,
                    fade_tip=config.ACTOR_SHADOW_FADE_TIP,
                )

    def _emit_actor_shadow_quads(
        self,
        actor: Actor,
        graphics: GraphicsContext,
        root_x: float,
        root_y: float,
        screen_x: float,
        screen_y: float,
        shadow_dir_x: float,
        shadow_dir_y: float,
        shadow_length_pixels: float,
        shadow_alpha: float,
        fade_tip: bool,
    ) -> None:
        """Emit one projected shadow quad per visual glyph layer."""
        visual_scale = getattr(actor, "visual_scale", 1.0)
        if actor.character_layers:
            for layer in actor.character_layers:
                layer_root_x = root_x + layer.offset_x
                layer_root_y = root_y + layer.offset_y
                layer_screen_x, layer_screen_y = graphics.console_to_screen_coords(
                    layer_root_x, layer_root_y
                )
                graphics.draw_actor_shadow(
                    char=layer.char,
                    screen_x=layer_screen_x,
                    screen_y=layer_screen_y,
                    shadow_dir_x=shadow_dir_x,
                    shadow_dir_y=shadow_dir_y,
                    shadow_length_pixels=shadow_length_pixels,
                    shadow_alpha=shadow_alpha,
                    scale_x=visual_scale * layer.scale_x,
                    scale_y=visual_scale * layer.scale_y,
                    fade_tip=fade_tip,
                )
            return

        graphics.draw_actor_shadow(
            char=actor.ch,
            screen_x=screen_x,
            screen_y=screen_y,
            shadow_dir_x=shadow_dir_x,
            shadow_dir_y=shadow_dir_y,
            shadow_length_pixels=shadow_length_pixels,
            shadow_alpha=shadow_alpha,
            scale_x=visual_scale,
            scale_y=visual_scale,
            fade_tip=fade_tip,
        )

    def _clip_shadow_length_by_walls(
        self,
        actor_x: int,
        actor_y: int,
        shadow_dir_x: float,
        shadow_dir_y: float,
        shadow_length_tiles: float,
    ) -> float:
        """Clamp shadow length when terrain blockers are encountered."""
        if shadow_length_tiles <= 0.0:
            return 0.0

        if abs(shadow_dir_x) <= 1e-6 and abs(shadow_dir_y) <= 1e-6:
            return 0.0

        game_map = self.controller.gw.game_map
        max_steps = min(8, math.ceil(shadow_length_tiles))
        origin_x = float(actor_x) + 0.5
        origin_y = float(actor_y) + 0.5

        for step in range(1, max_steps + 1):
            sample_x = math.floor(origin_x + shadow_dir_x * float(step))
            sample_y = math.floor(origin_y + shadow_dir_y * float(step))

            if not (0 <= sample_x < game_map.width and 0 <= sample_y < game_map.height):
                return max(0.0, min(shadow_length_tiles, float(step) - 0.5))

            # Projected glyph shadows for low-profile terrain (height 1-2) should
            # not occlude actor/terrain projected shadows; only tall blockers clip.
            if game_map.shadow_heights[sample_x, sample_y] > 2:
                return max(0.0, min(shadow_length_tiles, float(step) - 0.5))

        return shadow_length_tiles

    def _accumulate_actor_shadow_receiver_dimming(
        self,
        caster: Actor,
        receivers: list[Actor],
        shadow_dir_x: float,
        shadow_dir_y: float,
        shadow_length_tiles: float,
        shadow_alpha: float,
        fade_tip: bool,
    ) -> None:
        """Accumulate per-actor light attenuation from projected actor shadows."""
        if shadow_length_tiles <= 0.0 or shadow_alpha <= 0.0:
            return

        direction_length_sq = shadow_dir_x * shadow_dir_x + shadow_dir_y * shadow_dir_y
        if direction_length_sq <= 1e-12:
            return

        # Callers pass normalized vectors; avoid per-receiver renormalization work.
        dir_x = shadow_dir_x
        dir_y = shadow_dir_y
        caster_center_x = float(caster.x) + 0.5
        caster_center_y = float(caster.y) + 0.5

        # Taller actors should darken receivers more than shorter actors.
        caster_shadow_height = max(0.0, float(getattr(caster, "shadow_height", 0.0)))
        height_factor = min(1.0, caster_shadow_height / 4.0)
        if height_factor <= 0.0:
            return

        caster_scale = max(0.5, float(getattr(caster, "visual_scale", 1.0)))
        shadow_half_width = 0.18 + 0.22 * caster_scale

        for receiver in receivers:
            if receiver is caster:
                continue

            receiver_scale = max(0.5, float(getattr(receiver, "visual_scale", 1.0)))
            receiver_radius = 0.2 + 0.18 * receiver_scale
            receiver_center_x = float(receiver.x) + 0.5
            receiver_center_y = float(receiver.y) + 0.5

            rel_x = receiver_center_x - caster_center_x
            rel_y = receiver_center_y - caster_center_y

            # Signed distance along the projected shadow axis.
            distance_along_shadow = rel_x * dir_x + rel_y * dir_y
            if (
                distance_along_shadow <= 0.0
                or distance_along_shadow >= shadow_length_tiles
            ):
                continue

            # Perpendicular distance to the shadow axis.
            distance_from_axis = abs(rel_x * dir_y - rel_y * dir_x)
            lateral_limit = shadow_half_width + receiver_radius
            if distance_from_axis >= lateral_limit:
                continue

            lateral_factor = 1.0 - distance_from_axis / lateral_limit
            tip_factor = (
                1.0 - distance_along_shadow / shadow_length_tiles if fade_tip else 1.0
            )
            attenuation = shadow_alpha * height_factor * lateral_factor * tip_factor
            if attenuation <= 0.0:
                continue

            receiver_id = id(receiver)
            current_scale = self._actor_shadow_receive_light_scale.get(receiver_id, 1.0)
            next_scale = current_scale * (1.0 - min(0.95, attenuation))
            self._actor_shadow_receive_light_scale[receiver_id] = max(0.05, next_scale)

    @record_time_live_variable("time.render.actors_smooth_ms")
    def _render_actors_smooth(
        self,
        graphics: GraphicsContext,
        alpha: InterpolationAlpha,
        visible_actors: list[Actor] | None = None,
        viewport_bounds: Rect | None = None,
    ) -> None:
        """Render all actors with smooth sub-pixel positioning."""
        vs = self.viewport_system
        if viewport_bounds is None:
            viewport_bounds = vs.get_visible_bounds()

        if visible_actors is None:
            gw = self.controller.gw
            visible_actors = [
                actor
                for actor in self._get_sorted_visible_actors(viewport_bounds)
                if gw.game_map.visible[actor.x, actor.y]
            ]

        if not visible_actors:
            return

        for actor in visible_actors:
            self._render_single_actor_smooth(
                actor, graphics, viewport_bounds, vs, alpha
            )

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
        _, _, root_x, root_y, screen_pixel_x, screen_pixel_y = (
            self._get_actor_screen_position(actor, graphics, vs, interpolation_alpha)
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
                actor_world_pos=(actor.x, actor.y),
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
                world_pos=(actor.x, actor.y),
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
        actor_world_pos: tuple[int, int],
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
                world_pos=actor_world_pos,
            )

    def _get_actor_lighting_intensity(self, _actor: Actor, _bounds: Rect) -> tuple:
        """Get actor lighting multiplier tuple for the screen shader path."""
        receive_scale = self._actor_shadow_receive_light_scale.get(id(_actor), 1.0)
        return (receive_scale, receive_scale, receive_scale)

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

    @record_time_live_variable("time.render.actors_traditional_ms")
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
                    and not self.controller.is_combat_mode()
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
                        actor_world_pos=(actor.x, actor.y),
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
                        world_pos=(actor.x, actor.y),
                    )

    def _render_selection_and_hover_outlines(self) -> None:
        """Render outlines for selected and hovered actors.

        Outlines are rendered in priority order:
        1. selected_target (golden) - sticky click-to-select
        2. hovered_actor (subtle grey) - visual feedback only
        """
        if self.controller.is_combat_mode():
            return

        # Priority 1: Render selected target outline (golden)
        selected = self.controller.selected_target
        if (
            selected is not None
            and selected in self.controller.gw.actors
            and self.controller.gw.game_map.visible[selected.x, selected.y]
        ):
            self._draw_actor_outline(
                selected, colors.SELECTION_OUTLINE, float(CONTEXTUAL_OUTLINE_ALPHA)
            )
            return  # Don't also render hover outline for same actor

        # Priority 2: Render hover outline (white)
        hovered = self.controller.hovered_actor
        if hovered is None or hovered not in self.controller.gw.actors:
            return
        if not self.controller.gw.game_map.visible[hovered.x, hovered.y]:
            return
        self._draw_actor_outline(hovered, colors.HOVER_OUTLINE, 0.50)

    def _draw_actor_outline(
        self, actor: Actor, color: colors.Color, alpha: float
    ) -> None:
        """Draw an outline around an actor, handling content layers properly.

        If the actor has content layers (multi-character composition like bookcase),
        outlines the entire tile. Otherwise, outlines the glyph shape.

        Args:
            actor: The actor to outline
            color: RGB color for the outline
            alpha: Opacity of the outline (0.0-1.0)
        """
        if actor.character_layers or actor.has_complex_visuals:
            self._render_layered_tile_outline(actor, color, alpha)
        else:
            self.render_actor_outline(actor, color, alpha)

    def _render_layered_tile_outline(
        self, actor: Actor, color: colors.Color, alpha: float
    ) -> None:
        vs = self.viewport_system
        if not vs.is_visible(actor.x, actor.y):
            return

        vp_x, vp_y = vs.world_to_screen(actor.x, actor.y)
        root_x = self.x + vp_x
        root_y = self.y + vp_y
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
