"""Mini-map view rendered into the left sidebar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from brileta import colors, config
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.environment import tile_types
from brileta.game.actors import Character
from brileta.game.actors.boulder import Boulder
from brileta.game.actors.trees import Tree
from brileta.types import InterpolationAlpha
from brileta.util.caching import ResourceCache
from brileta.util.live_vars import record_time_live_variable
from brileta.view.render.graphics import GraphicsContext

from .base import TextView, View

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.view.render.viewport import ViewportSystem


class MiniMapView(TextView):
    """Render a compact overview of explored and visible map tiles."""

    _MAP_BORDER_PX = 2
    _EXPLORED_DIM_FACTOR = 0.4
    _TERRAIN_SATURATION = 0.7
    _TERRAIN_BRIGHTNESS = 0.9
    _VIEWPORT_RECT_COLOR: colors.Color = (198, 184, 150)

    # Minimap colors for static feature actors, keyed by actor class.
    # Trees are a dark forest green, distinctly darker than tuned grass (~69,94,56).
    # Boulders are a muted warm grey, distinct from both terrain and tree colors.
    # To add a new feature type, add an entry here - no other changes needed.
    _FEATURE_COLORS: ClassVar[dict[type, colors.Color]] = {
        Tree: (30, 70, 20),
        Boulder: (95, 93, 90),
    }

    def __init__(self, controller: Controller, viewport_system: ViewportSystem) -> None:
        super().__init__(create_texture_cache=False)
        self.controller = controller
        self.viewport_system = viewport_system
        # Terrain is rendered directly via NumPy (no canvas needed).
        # The overlay (markers, viewport rect) uses a transparent Pillow canvas.
        self.canvas = PillowImageCanvas(
            controller.graphics,
            font_path=config.UI_FONT_PATH,
            transparent=True,
        )
        self.view_width_px = 0
        self.view_height_px = 0
        self._cached_terrain_texture: Any | None = None

        self._terrain_texture_cache = ResourceCache[tuple, Any](
            name=f"{self.__class__.__name__}TerrainRender",
            max_size=1,
            on_evict=lambda tex: self.controller.graphics.release_texture(tex),
        )
        self._texture_cache = ResourceCache[tuple, Any](
            name=f"{self.__class__.__name__}OverlayRender",
            max_size=1,
            on_evict=lambda tex: self.controller.graphics.release_texture(tex),
        )

        self._visible_colors = self._build_visible_color_lut()
        self._explored_colors = tuple(
            self._dim_color(color, self._EXPLORED_DIM_FACTOR)
            for color in self._visible_colors
        )
        # NumPy lookup tables for vectorized terrain rendering, indexed by TileTypeID.
        self._visible_colors_np = np.array(self._visible_colors, dtype=np.uint8)
        self._explored_colors_np = np.array(self._explored_colors, dtype=np.uint8)

        # Precomputed feature layers, built lazily on first terrain render
        # and invalidated when structural_revision changes.
        self._feature_layer_rev: int = -1
        self._feature_mask: np.ndarray = np.empty(0, dtype=bool)
        self._feature_vis_rgb: np.ndarray = np.empty(0, dtype=np.uint8)
        self._feature_exp_rgb: np.ndarray = np.empty(0, dtype=np.uint8)

    @staticmethod
    def _dim_color(color: colors.Color, factor: float) -> colors.Color:
        """Return a dimmed RGB color."""
        return (
            int(color[0] * factor),
            int(color[1] * factor),
            int(color[2] * factor),
        )

    @classmethod
    def _tune_terrain_color(cls, color: colors.Color) -> colors.Color:
        """Reduce terrain color noise so the mini-map reads as shape first."""
        r, g, b = color
        # Desaturate toward luminance, then trim brightness slightly so markers
        # and overlays remain the strongest signals in the mini-map.
        luma = int((r * 30 + g * 59 + b * 11) / 100)
        saturation = cls._TERRAIN_SATURATION
        brightness = cls._TERRAIN_BRIGHTNESS
        tuned_r = int((luma + (r - luma) * saturation) * brightness)
        tuned_g = int((luma + (g - luma) * saturation) * brightness)
        tuned_b = int((luma + (b - luma) * saturation) * brightness)
        return (
            max(0, min(255, tuned_r)),
            max(0, min(255, tuned_g)),
            max(0, min(255, tuned_b)),
        )

    @classmethod
    def _build_visible_color_lut(cls) -> tuple[colors.Color, ...]:
        """Build a color lookup table indexed by ``TileTypeID``."""
        appearances = tile_types._tile_type_properties_light_appearance
        lut: list[colors.Color] = []
        for appearance in appearances:
            bg = appearance["bg"]
            lut.append(cls._tune_terrain_color((int(bg[0]), int(bg[1]), int(bg[2]))))
        return tuple(lut)

    def _get_feature_layers(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mask, vis_rgb, exp_rgb) arrays for static feature actors.

        Built once from the actor list, then cached until
        ``structural_revision`` changes (which covers map regeneration).
        """
        game_map = self.controller.gw.game_map
        rev = game_map.structural_revision
        if self._feature_layer_rev == rev:
            return self._feature_mask, self._feature_vis_rgb, self._feature_exp_rgb

        h, w = game_map.height, game_map.width
        mask = np.zeros((h, w), dtype=bool)
        vis_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        exp_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        dim = self._EXPLORED_DIM_FACTOR

        for actor in self.controller.gw.actors:
            color = self._FEATURE_COLORS.get(type(actor))
            if color is None:
                continue
            ax, ay = actor.x, actor.y
            if not (0 <= ax < w and 0 <= ay < h):
                continue
            mask[ay, ax] = True
            vis_rgb[ay, ax] = color
            exp_rgb[ay, ax] = self._dim_color(color, dim)

        self._feature_layer_rev = rev
        self._feature_mask = mask
        self._feature_vis_rgb = vis_rgb
        self._feature_exp_rgb = exp_rgb
        return mask, vis_rgb, exp_rgb

    def _get_pixels_per_tile(self) -> int:
        """Return the largest integer tile scale that fits the current bounds."""
        game_map = self.controller.gw.game_map
        usable_width_px = max(0, self.view_width_px - self._MAP_BORDER_PX)
        usable_height_px = max(0, self.view_height_px - self._MAP_BORDER_PX)
        max_by_width = usable_width_px // max(1, game_map.width)
        max_by_height = usable_height_px // max(1, game_map.height)
        return max(1, min(max_by_width, max_by_height))

    def _iter_visible_character_markers(
        self,
    ) -> frozenset[tuple[int, int, bool]]:
        """Return visible non-player character positions and hostility flags."""
        gw = self.controller.gw
        game_map = gw.game_map
        player = gw.player
        markers: set[tuple[int, int, bool]] = set()

        for actor in gw.actors:
            if actor is player or not isinstance(actor, Character):
                continue
            actor_health = getattr(actor, "health", None)
            if actor_health is not None and not actor_health.is_alive():
                continue
            if not (0 <= actor.x < game_map.width and 0 <= actor.y < game_map.height):
                continue
            if not game_map.visible[actor.x, actor.y]:
                continue

            is_hostile = False
            actor_ai = getattr(actor, "ai", None)
            if player is not None and actor_ai is not None:
                is_hostile = bool(actor_ai.is_hostile_toward(player))
            markers.add((actor.x, actor.y, is_hostile))

        return frozenset(markers)

    def _get_terrain_cache_key(self) -> tuple:
        """Return the cache key for the terrain/fog mini-map layer.

        Player position is included even though terrain colors don't depend on
        it directly.  The reason: the FOV (visible/explored masks) is recomputed
        from the player's position, but the masks themselves aren't hashable.
        Including ``(player_x, player_y)`` is a cheap proxy that ensures any
        FOV change triggers a terrain re-render.
        """
        gw = self.controller.gw
        game_map = gw.game_map
        player = gw.player
        player_x = player.x if player is not None else -1
        player_y = player.y if player is not None else -1

        return (
            game_map.exploration_revision,
            game_map.structural_revision,
            player_x,
            player_y,
            self.controller.graphics.tile_dimensions,
            self.view_width_px,
            self.view_height_px,
        )

    def _get_overlay_cache_key(
        self,
        markers: frozenset[tuple[int, int, bool]] | None = None,
    ) -> tuple:
        """Return the cache key for viewport and actor markers.

        Args:
            markers: Pre-computed marker set. Avoids re-iterating actors when
                the caller already has the result (e.g. to reuse in drawing).
        """
        gw = self.controller.gw
        player = gw.player
        bounds = self.viewport_system.get_visible_bounds()
        player_x = player.x if player is not None else -1
        player_y = player.y if player is not None else -1
        if markers is None:
            markers = self._iter_visible_character_markers()

        return (
            player_x,
            player_y,
            bounds.x1,
            bounds.y1,
            bounds.x2,
            bounds.y2,
            markers,
            self.controller.graphics.tile_dimensions,
            self.view_width_px,
            self.view_height_px,
        )

    def get_cache_key(self) -> tuple:
        """Return the combined mini-map cache state (terrain + overlay)."""
        return (self._get_terrain_cache_key(), self._get_overlay_cache_key())

    def _sync_view_metrics(self, graphics: GraphicsContext) -> None:
        """Sync tile and pixel dimensions from the active graphics context."""
        self.tile_dimensions = graphics.tile_dimensions
        self.view_width_px = self.width * self.tile_dimensions[0]
        self.view_height_px = self.height * self.tile_dimensions[1]

    def _configure_overlay_canvas(self) -> None:
        """Ensure the overlay canvas matches the current view size and scaling."""
        self.canvas.configure_dimensions(self.view_width_px, self.view_height_px)
        self.canvas.configure_scaling(self.tile_dimensions[1])
        self.canvas.configure_drawing_offset(self.x, self.y)

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Update bounds and cached pixel dimensions.

        Calls ``View.set_bounds`` directly instead of ``super().set_bounds()``
        to bypass ``TextView.set_bounds``, which would reconfigure the *overlay*
        canvas from the base-class side.  MiniMapView manages its own overlay
        canvas via ``_configure_overlay_canvas`` so that terrain and overlay
        pixel dimensions stay in sync.
        """
        View.set_bounds(self, x1, y1, x2, y2)
        self.view_width_px = self.width * self.tile_dimensions[0]
        self.view_height_px = self.height * self.tile_dimensions[1]
        if self.tile_dimensions != (0, 0):
            self._configure_overlay_canvas()

    def _render_terrain_pixels(self) -> np.ndarray:
        """Render the terrain/fog layer as a small RGBA pixel buffer via NumPy.

        Each map tile is exactly 1 pixel. The GPU's nearest-neighbor sampler
        handles upscaling to screen resolution, which reduces the buffer from
        ~3 MB (full resolution) to ~70 KB and eliminates the costly np.repeat
        and tobytes overhead.  The 1px border is drawn by the overlay layer at
        full resolution instead.
        """
        game_map = self.controller.gw.game_map
        px_per_tile, _ma_x, _ma_y, tile_origin_x, tile_origin_y = self._map_layout()

        # Small texture: 1 pixel per map tile, with margins for centering.
        # The GPU stretches this back to the full view area on present.
        small_w = max(1, -(-self.view_width_px // px_per_tile))
        small_h = max(1, -(-self.view_height_px // px_per_tile))
        origin_x = round(tile_origin_x / px_per_tile)
        origin_y = round(tile_origin_y / px_per_tile)

        # Transpose so axes become (row=y, col=x) for image layout.
        tile_ids = game_map.tiles.T.astype(np.intp)
        visible_t = game_map.visible.T
        explored_t = game_map.explored.T

        # Build (H, W, 3) RGB via lookup table indexing. Unexplored tiles stay black.
        map_rgb = np.zeros((game_map.height, game_map.width, 3), dtype=np.uint8)
        map_rgb[visible_t] = self._visible_colors_np[tile_ids[visible_t]]
        explored_only = explored_t & ~visible_t
        map_rgb[explored_only] = self._explored_colors_np[tile_ids[explored_only]]

        # Overlay static feature actors (trees, boulders) with a vectorized
        # mask operation.  The feature layers are built once from the actor
        # list and cached until the map structure changes.
        feat_mask, feat_vis, feat_exp = self._get_feature_layers()
        vis_features = feat_mask & visible_t
        map_rgb[vis_features] = feat_vis[vis_features]
        exp_features = feat_mask & explored_only
        map_rgb[exp_features] = feat_exp[exp_features]

        # Composite 1:1 map content into the small opaque-black buffer.
        buf = np.zeros((small_h, small_w, 4), dtype=np.uint8)
        buf[:, :, 3] = 255
        clip_h = min(game_map.height, small_h - origin_y)
        clip_w = min(game_map.width, small_w - origin_x)
        if clip_h > 0 and clip_w > 0:
            buf[origin_y : origin_y + clip_h, origin_x : origin_x + clip_w, :3] = (
                map_rgb[:clip_h, :clip_w]
            )

        return np.ascontiguousarray(buf)

    def _map_layout(self) -> tuple[int, int, int, int, int]:
        """Return (px_per_tile, map_area_x, map_area_y, tile_origin_x, tile_origin_y).

        Shared geometry used by both terrain and overlay layers.
        """
        game_map = self.controller.gw.game_map
        px_per_tile = self._get_pixels_per_tile()
        map_content_w = game_map.width * px_per_tile
        map_content_h = game_map.height * px_per_tile
        map_area_w = map_content_w + self._MAP_BORDER_PX
        map_area_h = map_content_h + self._MAP_BORDER_PX
        map_area_x = max(0, (self.view_width_px - map_area_w) // 2)
        map_area_y = max(0, (self.view_height_px - map_area_h) // 2)
        return px_per_tile, map_area_x, map_area_y, map_area_x + 1, map_area_y + 1

    def draw_content(
        self,
        graphics: GraphicsContext,
        alpha: InterpolationAlpha,
        *,
        markers: frozenset[tuple[int, int, bool]] | None = None,
    ) -> None:
        """Render the overlay layer (border, viewport rect, actor markers).

        Called from ``draw()`` after view metrics and the overlay canvas have
        already been synced for this frame.

        Args:
            graphics: The active graphics context.
            alpha: Interpolation factor (unused).
            markers: Pre-computed character markers from the cache-key step.
                Avoids re-iterating actors when the caller already built the set.
        """
        _ = alpha

        gw = self.controller.gw
        game_map = gw.game_map
        player = gw.player
        px_per_tile, map_area_x, map_area_y, tile_origin_x, tile_origin_y = (
            self._map_layout()
        )

        map_content_w = game_map.width * px_per_tile
        map_content_h = game_map.height * px_per_tile
        map_area_w = map_content_w + self._MAP_BORDER_PX
        map_area_h = map_content_h + self._MAP_BORDER_PX

        # 1px border drawn here at full resolution (the terrain layer renders
        # at reduced resolution so the border would scale incorrectly there).
        self.canvas.draw_rect(
            map_area_x,
            map_area_y,
            map_area_w,
            map_area_h,
            colors.DARK_GREY,
            fill=False,
        )

        viewport_bounds = self.viewport_system.get_visible_bounds()
        vp_x1 = max(0, min(game_map.width - 1, viewport_bounds.x1))
        vp_y1 = max(0, min(game_map.height - 1, viewport_bounds.y1))
        vp_x2 = max(0, min(game_map.width - 1, viewport_bounds.x2))
        vp_y2 = max(0, min(game_map.height - 1, viewport_bounds.y2))
        if vp_x2 >= vp_x1 and vp_y2 >= vp_y1:
            self.canvas.draw_rect(
                tile_origin_x + vp_x1 * px_per_tile,
                tile_origin_y + vp_y1 * px_per_tile,
                (vp_x2 - vp_x1 + 1) * px_per_tile,
                (vp_y2 - vp_y1 + 1) * px_per_tile,
                self._VIEWPORT_RECT_COLOR,
                fill=False,
            )

        if player is not None:
            if markers is None:
                markers = self._iter_visible_character_markers()
            for x, y, is_hostile in markers:
                self.canvas.draw_rect(
                    tile_origin_x + x * px_per_tile,
                    tile_origin_y + y * px_per_tile,
                    px_per_tile,
                    px_per_tile,
                    colors.RED if is_hostile else colors.GREY,
                    fill=True,
                )

            if 0 <= player.x < game_map.width and 0 <= player.y < game_map.height:
                self.canvas.draw_rect(
                    tile_origin_x + player.x * px_per_tile,
                    tile_origin_y + player.y * px_per_tile,
                    px_per_tile,
                    px_per_tile,
                    colors.WHITE,
                    fill=True,
                )

    def draw(self, graphics: GraphicsContext, alpha: InterpolationAlpha) -> None:
        """Render mini-map terrain and overlay with separate caches."""
        if not self.visible:
            return
        if self.width <= 0 or self.height <= 0:
            self._cached_terrain_texture = None
            self._cached_texture = None
            return

        with record_time_live_variable("time.render.minimap_ms"):
            self._sync_view_metrics(graphics)
            self._configure_overlay_canvas()

            # Terrain layer - rendered via NumPy, bypassing the canvas pipeline.
            terrain_key = self._get_terrain_cache_key()
            cached_terrain = self._terrain_texture_cache.get(terrain_key)
            if cached_terrain is None:
                terrain_pixels = self._render_terrain_pixels()
                terrain_texture = graphics.texture_from_numpy(
                    terrain_pixels, transparent=False
                )
                if terrain_texture:
                    self._terrain_texture_cache.store(terrain_key, terrain_texture)
                    self._cached_terrain_texture = terrain_texture
            else:
                self._cached_terrain_texture = cached_terrain

            # Overlay layer - viewport rect and actor markers via Pillow canvas.
            # Compute markers once and reuse for both the cache key and drawing.
            markers = self._iter_visible_character_markers()
            overlay_key = self._get_overlay_cache_key(markers=markers)
            cached_overlay = self._texture_cache.get(overlay_key)
            if cached_overlay is None:
                self.canvas.begin_frame()
                self.draw_content(graphics, alpha, markers=markers)
                overlay_artifact = self.canvas.end_frame()
                if overlay_artifact is not None:
                    overlay_texture = self.canvas.create_texture(
                        graphics, overlay_artifact
                    )
                    if overlay_texture:
                        self._texture_cache.store(overlay_key, overlay_texture)
                        self._cached_texture = overlay_texture
            else:
                self._cached_texture = cached_overlay

    def present(self, graphics: GraphicsContext, alpha: InterpolationAlpha) -> None:
        """Present terrain layer first, then the overlay markers."""
        if not self.visible:
            return
        if self._cached_terrain_texture is not None:
            graphics.present_texture(
                self._cached_terrain_texture, self.x, self.y, self.width, self.height
            )
        super().present(graphics, alpha)
