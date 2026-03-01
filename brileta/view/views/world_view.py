from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np

from brileta import colors, config
from brileta.environment import tile_types
from brileta.types import (
    Direction,
    InterpolationAlpha,
    Opacity,
    PixelCoord,
    ViewOffset,
    WorldTilePos,
    saturate,
)
from brileta.util import rng
from brileta.util.caching import ResourceCache
from brileta.util.coordinates import Rect
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
from brileta.view.render.effects.rain import RainAnimationState, RainConfig
from brileta.view.render.effects.screen_shake import ScreenShake
from brileta.view.render.graphics import GraphicsContext
from brileta.view.render.lighting.base import LightingSystem
from brileta.view.render.shadow_renderer import ShadowRenderer
from brileta.view.render.viewport import ViewportSystem

from .base import View

if TYPE_CHECKING:
    from brileta.controller import Controller, FrameManager
    from brileta.environment.generators.buildings.building import Building
    from brileta.game.actors import Actor
    from brileta.game.lights import DirectionalLight

_rng = rng.get("effects.animation")


# Viewport defaults used when initializing views before they are resized.
DEFAULT_VIEWPORT_WIDTH = 80
DEFAULT_VIEWPORT_HEIGHT = 40  # Initial height before layout adjustments

# Cardinal boundary directions only (W, N, S, E). Organic edge blending uses
# corner rounding in the shader, so diagonals are not transported separately.
_EDGE_BLEND_CARDINAL_DIRECTIONS: tuple[Direction, ...] = (
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
)

_ROOF_TILE_IDS: tuple[int, ...] = (
    int(tile_types.TileTypeID.ROOF_THATCH),
    int(tile_types.TileTypeID.ROOF_SHINGLE),
    int(tile_types.TileTypeID.ROOF_TIN),
)
_EDGE_BLEND_HARD_EDGE_NEIGHBOR_TILE_IDS: tuple[int, ...] = (
    int(tile_types.TileTypeID.WALL),
    *_ROOF_TILE_IDS,
)


def _adjust_color_brightness(
    colors: np.ndarray, offset: int | np.ndarray
) -> np.ndarray:
    """Apply a signed brightness offset to uint8 RGB colors with clamping.

    Widens to int16 internally to prevent uint8 wraparound, then clamps to
    [0, 255].  Works with scalar offsets, per-channel arrays, or per-pixel
    arrays that broadcast against the input shape.
    """
    return np.clip(colors.astype(np.int16) + offset, 0, 255).astype(np.uint8)


# Wear material IDs packed into wear_pack bits 0-7.
_WEAR_MAT_THATCH: int = 1
_WEAR_MAT_SHINGLE: int = 2
_WEAR_MAT_TIN: int = 3
_WEAR_MATERIAL_MAP: dict[str, int] = {
    "thatch": _WEAR_MAT_THATCH,
    "shingle": _WEAR_MAT_SHINGLE,
    "tin": _WEAR_MAT_TIN,
}
# How many tiles inward from the roof perimeter the edge proximity fades to
# zero.  Controls how far moss, lichen, and edge-concentrated wear extend.
_WEAR_EDGE_FADE_TILES: float = 3.0


def _apply_noise_pattern_overrides(
    tile_ids: np.ndarray, roof_result: _RoofSubstitutionResult
) -> np.ndarray:
    """Return sub-tile pattern IDs, applying per-building roof overrides.

    Roof materials like tin and shingle need orientation-dependent patterns
    (e.g., corrugation running down-slope).  The roof substitution pass stores
    these overrides in *roof_result*; this helper merges them onto the
    default tile-type pattern map.
    """
    pattern = tile_types.get_sub_tile_pattern_map(tile_ids)
    if (
        roof_result.noise_pattern is not None
        and roof_result.noise_pattern_mask is not None
        and np.any(roof_result.noise_pattern_mask)
    ):
        mask = roof_result.noise_pattern_mask
        pattern = pattern.copy()
        pattern[mask] = roof_result.noise_pattern[mask]
    return pattern


_LIGHT_OVERLAY_MASK_HIDDEN = np.uint8(0)
_LIGHT_OVERLAY_MASK_ROOF_OPAQUE = np.uint8(128)
_LIGHT_OVERLAY_MASK_ROOF_SUNLIT = np.uint8(192)
_LIGHT_OVERLAY_MASK_VISIBLE = np.uint8(255)

# Maximum northward perspective offset in tile rows. Used to expand viewport
# building detection so shifted roofs at the edge of the screen are not missed.
# Derived from ceil(MAX_PERSPECTIVE_FLOORS * WALL_HEIGHT_PER_FLOOR) = 6,
# plus 1 tile of margin for safety.
_MAX_PERSPECTIVE_SHIFT_TILES = 7


class _RoofSubstitutionResult(NamedTuple):
    """Result of roof substitution including split data for perspective offset."""

    effective_tile_ids: np.ndarray
    noise_pattern: np.ndarray | None = None  # (N,) uint8 override values
    noise_pattern_mask: np.ndarray | None = None  # (N,) bool
    split_y: np.ndarray | None = None
    split_bg: np.ndarray | None = None  # (N, 4) uint8 RGBA
    split_fg: np.ndarray | None = None  # (N, 4) uint8 RGBA
    split_noise: np.ndarray | None = None  # float32
    split_noise_pattern: np.ndarray | None = None  # uint8
    wear_pack: np.ndarray | None = None  # (N,) uint32


@dataclass(slots=True)
class _RoofStamp:
    """Cached roof visuals for one building in extended local coordinates.

    The stamp covers the building footprint plus perspective overhang north:
    world rows [fp.y1 - north_overhang, fp.y2) and columns [fp.x1, fp.x2).
    Local coordinates: lx = world_x - fp.x1,
                       ly = world_y - (fp.y1 - north_overhang).
    """

    chars: np.ndarray  # (w, ext_h) int32
    fg_rgb: np.ndarray  # (w, ext_h, 3) uint8
    bg_rgb: np.ndarray  # (w, ext_h, 3) uint8
    tile_ids: np.ndarray  # (w, ext_h) int32
    draw_mask: np.ndarray  # (w, ext_h) bool
    north_overhang: int  # Rows extending above fp.y1 for perspective shift
    noise_pattern: np.ndarray  # (w, ext_h) uint8
    noise_pattern_mask: np.ndarray  # (w, ext_h) bool
    # Sub-tile split data for perspective boundary tiles.
    split_y: np.ndarray  # (w, ext_h) float32
    split_bg: np.ndarray  # (w, ext_h, 4) uint8 RGBA
    split_fg: np.ndarray  # (w, ext_h, 4) uint8 RGBA
    split_noise: np.ndarray  # (w, ext_h) float32
    split_noise_pattern: np.ndarray  # (w, ext_h) uint8
    # Packed weathering data for the fragment shader (material|condition|edge).
    wear_pack: np.ndarray  # (w, ext_h) uint32


@dataclass(slots=True)
class _LightBufferCache:
    """Cached light-source glyph buffer tile data and visibility metadata.

    Groups all per-viewport light overlay cache fields so they can be
    invalidated atomically by replacing the entire object.
    """

    cache_key: tuple[object, ...] | None = None
    # Pre-animation colors and index metadata for animated tiles.
    anim_base_fg: np.ndarray | None = None
    anim_base_bg: np.ndarray | None = None
    anim_buf_indices: np.ndarray | None = None
    anim_exp_x: np.ndarray | None = None
    anim_exp_y: np.ndarray | None = None
    # Buffer coordinates for all explored tiles and visible subset.
    buf_x: np.ndarray | None = None
    buf_y: np.ndarray | None = None
    exp_x: np.ndarray | None = None
    exp_y: np.ndarray | None = None
    roof_covered_mask: np.ndarray | None = None
    vis_buf_x: np.ndarray | None = None
    vis_buf_y: np.ndarray | None = None
    roof_opaque_buf_x: np.ndarray | None = None
    roof_opaque_buf_y: np.ndarray | None = None
    # Viewport-local explored snapshot for incremental invalidation.
    exploration_revision: int = -1
    explored_mask: np.ndarray | None = None


def _shift_boundary_valid_mask(shape: tuple[int, int], dx: int, dy: int) -> np.ndarray:
    """Build a boolean mask marking cells whose ``(dx, dy)`` neighbor is in bounds.

    After ``np.roll(arr, shift=(-dx, -dy), axis=(0, 1))``, the rolled array
    wraps around at the edges.  This mask is ``False`` for cells whose shifted
    neighbor originated from the opposite edge (i.e. is not a true neighbor).
    """
    mask = np.ones(shape, dtype=np.bool_)
    if dx < 0:
        mask[:-dx, :] = False
    elif dx > 0:
        mask[-dx:, :] = False
    if dy < 0:
        mask[:, :-dy] = False
    elif dy > 0:
        mask[:, -dy:] = False
    return mask


def _compute_tile_edge_transition_metadata(
    tile_id_window: np.ndarray,
    edge_blend_window: np.ndarray,
    drawn_mask_window: np.ndarray,
    bg_rgb_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute neighbor masks/colors for organic tile edge transitions.

    The inputs are buffer-aligned windows in glyph-buffer tile coordinates:
    `tile_id_window[x, y]`, `edge_blend_window[x, y]`, `drawn_mask_window[x, y]`,
    and `bg_rgb_window[x, y, rgb]`.
    The returned mask uses `_EDGE_BLEND_CARDINAL_DIRECTIONS` bit ordering.
    """
    edge_neighbor_mask = np.zeros(tile_id_window.shape, dtype=np.uint8)
    edge_neighbor_bg = np.zeros(
        (*tile_id_window.shape, len(_EDGE_BLEND_CARDINAL_DIRECTIONS), 3), dtype=np.uint8
    )

    for bit_index, (dx, dy) in enumerate(_EDGE_BLEND_CARDINAL_DIRECTIONS):
        shifted_ids = np.roll(tile_id_window, shift=(-dx, -dy), axis=(0, 1))
        shifted_edge_blend = np.roll(edge_blend_window, shift=(-dx, -dy), axis=(0, 1))
        shifted_drawn = np.roll(drawn_mask_window, shift=(-dx, -dy), axis=(0, 1))
        shifted_bg = np.roll(bg_rgb_window, shift=(-dx, -dy), axis=(0, 1))

        valid_neighbor = _shift_boundary_valid_mask(tile_id_window.shape, dx, dy)

        # One-sided ownership prevents both tiles from cutting into each other.
        # The tile with the higher edge_blend value owns the boundary so organic
        # tiles feather into rigid ones regardless of enum ID ordering. When the
        # blend values tie, the lower tile type ID wins for determinism.
        owns_boundary = (edge_blend_window > shifted_edge_blend) | (
            (edge_blend_window == shifted_edge_blend) & (tile_id_window < shifted_ids)
        )
        different_neighbor_mask = (
            drawn_mask_window
            & shifted_drawn
            & valid_neighbor
            & (shifted_ids != tile_id_window)
            & owns_boundary
        )
        if not np.any(different_neighbor_mask):
            continue

        edge_neighbor_mask[different_neighbor_mask] |= np.uint8(1 << bit_index)
        color_slice = edge_neighbor_bg[:, :, bit_index, :]
        color_slice[different_neighbor_mask] = shifted_bg[different_neighbor_mask]

    return edge_neighbor_mask, edge_neighbor_bg


def _suppress_edge_blend_toward_hard_edges(
    edge_neighbor_mask: np.ndarray, tile_id_window: np.ndarray
) -> None:
    """Clear edge-blend mask bits that point at architectural hard-edge tiles.

    Organic terrain should feather into other natural terrain, but not into
    building surfaces like walls/roofs. Those boundaries should stay crisp.
    """
    if not np.any(edge_neighbor_mask):
        return

    hard_edge_tiles = np.isin(tile_id_window, _EDGE_BLEND_HARD_EDGE_NEIGHBOR_TILE_IDS)
    if not np.any(hard_edge_tiles):
        return
    source_is_hard_edge = hard_edge_tiles

    for bit_index, (dx, dy) in enumerate(_EDGE_BLEND_CARDINAL_DIRECTIONS):
        shifted_hard_edges = np.roll(hard_edge_tiles, shift=(-dx, -dy), axis=(0, 1))

        valid_neighbor = _shift_boundary_valid_mask(tile_id_window.shape, dx, dy)

        direction_bit = np.uint8(1 << bit_index)
        mask_points_to_hard_edge = (
            valid_neighbor
            & ~source_is_hard_edge
            & shifted_hard_edges
            & ((edge_neighbor_mask & direction_bit) != 0)
        )
        if not np.any(mask_points_to_hard_edge):
            continue

        edge_neighbor_mask[mask_points_to_hard_edge] &= np.uint8(0xFF ^ direction_bit)


def _override_edge_neighbor_bg_with_self_darken(
    edge_neighbor_bg: np.ndarray,
    tile_id_window: np.ndarray,
    bg_rgb_window: np.ndarray,
) -> None:
    """Replace edge blend target colors with a darkened self color when configured."""
    edge_self_darken = tile_types.get_edge_self_darken_map(tile_id_window)
    self_darken_mask = edge_self_darken > 0
    if not np.any(self_darken_mask):
        return

    # Signed cast so negation produces a proper negative offset.
    darken_amount = edge_self_darken[self_darken_mask].astype(np.int16)[:, np.newaxis]
    darkened_bg = _adjust_color_brightness(
        bg_rgb_window[self_darken_mask], -darken_amount
    )
    # Write every cardinal slot so the shader always darkens toward self-color
    # for these materials, regardless of actual neighbor type.
    edge_neighbor_bg[self_darken_mask] = darkened_bg[:, np.newaxis, :]


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
        self._zoom_index: int = config.DEFAULT_ZOOM_INDEX
        # World viewport zoom only. HUD/root-console zoom stays fixed.
        self._viewport_zoom: float = config.ZOOM_STOPS[self._zoom_index]
        # Initialize a viewport system with bootstrap defaults.
        # These defaults are replaced once resize() sets the real view bounds.
        self.viewport_system = ViewportSystem(
            DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT
        )
        # Initialize dimension-dependent resources (glyph buffers, particle system,
        # etc.) via set_bounds. This avoids duplicating the creation logic.
        self.set_bounds(0, 0, DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT)
        # Light source buffer cache: skip full rebuild when viewport and exploration
        # haven't changed since the last frame.
        self._map_unlit_buffer_cache_key: tuple[object, ...] | None = None
        self._light_cache = _LightBufferCache()
        # Persistent visible mask buffer to avoid per-frame allocation.
        self._visible_mask_buffer: np.ndarray | None = None
        self._roof_state_cache_key: tuple[object, ...] | None = None
        self._roof_state_cache_value: tuple[int | None, list[Building]] | None = None
        self._roof_stamp_cache: dict[
            tuple[int, bool], tuple[tuple[object, ...], _RoofStamp]
        ] = {}
        self.effect_library = EffectLibrary()
        self.floating_text_manager = FloatingTextManager()
        self._gpu_actor_lightmap_texture: Any | None = None
        self._gpu_actor_lightmap_viewport_origin: WorldTilePos | None = None
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
        self._shake_offset: ViewOffset = (0.0, 0.0)
        # Camera fractional offset for smooth scrolling (set each frame in present())
        self.camera_frac_offset: ViewOffset = (0.0, 0.0)
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
        # Numpy RNG for vectorized tile animation random walks.
        self._tile_anim_rng = np.random.default_rng()
        # Lazily rebuilt when GameWorld actor membership changes.
        self._particle_emitter_actors: set[Actor] = set()
        self._particle_emitter_actors_revision: int = -1
        from brileta.view.render.effects.atmospheric import (
            AtmosphericConfig,
            AtmosphericLayerSystem,
        )

        self.atmospheric_system = AtmosphericLayerSystem(
            AtmosphericConfig.create_default()
        )
        self.rain_config = RainConfig.from_config()
        self.rain_animation = RainAnimationState()

    def _compute_zoomed_viewport_size(self) -> tuple[int, int]:
        """Return visible world-tile dimensions for the fixed view rect."""
        zoom = max(config.ZOOM_STOPS[0], float(self._viewport_zoom))
        visible_width = max(1, round(self.width / zoom))
        visible_height = max(1, round(self.height / zoom))
        return (visible_width, visible_height)

    def _rebuild_viewport_dependent_resources(self) -> None:
        """Resize viewport and rebuild buffers that depend on visible tile counts."""
        if self.width <= 0 or self.height <= 0:
            return

        visible_width, visible_height = self._compute_zoomed_viewport_size()
        current_visible = (
            self.viewport_system.viewport.width_tiles,
            self.viewport_system.viewport.height_tiles,
        )
        self.viewport_system.set_display_size(self.width, self.height)
        if current_visible == (visible_width, visible_height):
            return

        self.viewport_system.viewport.resize(visible_width, visible_height)
        pad = self._SCROLL_PADDING
        self.map_glyph_buffer = GlyphBuffer(
            visible_width + 2 * pad, visible_height + 2 * pad
        )
        self.light_source_glyph_buffer = GlyphBuffer(
            visible_width + 2 * pad, visible_height + 2 * pad
        )
        # Force a full light-source rebuild after buffer resize.
        self._map_unlit_buffer_cache_key = None
        self._light_cache = _LightBufferCache()
        self._visible_mask_buffer = None
        self._active_background_texture = None
        self._light_overlay_texture = None
        # Keep dynamic effects sized to the visible viewport tile grid.
        self.particle_system = SubTileParticleSystem(visible_width, visible_height)
        self.environmental_system = EnvironmentalEffectSystem()
        self.decal_system = DecalSystem()

    def step_zoom(self, direction: int) -> bool:
        """Step world viewport zoom by one configured stop."""
        if direction == 0:
            return False
        max_index = len(config.ZOOM_STOPS) - 1
        new_index = max(
            0, min(max_index, self._zoom_index + (1 if direction > 0 else -1))
        )
        if new_index == self._zoom_index:
            return False
        self._zoom_index = new_index
        self._viewport_zoom = config.ZOOM_STOPS[self._zoom_index]
        self._rebuild_viewport_dependent_resources()
        return True

    def reset_zoom(self) -> bool:
        """Reset world viewport zoom to the configured default stop."""
        if self._zoom_index == config.DEFAULT_ZOOM_INDEX:
            return False
        self._zoom_index = config.DEFAULT_ZOOM_INDEX
        self._viewport_zoom = config.ZOOM_STOPS[self._zoom_index]
        self._rebuild_viewport_dependent_resources()
        return True

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override set_bounds to update viewport and console dimensions."""
        # Only perform resize logic if the dimensions have actually changed.
        if self.width != (x2 - x1) or self.height != (y2 - y1):
            super().set_bounds(x1, y1, x2, y2)
            self._rebuild_viewport_dependent_resources()

    @property
    def viewport_zoom(self) -> float:
        """Current world-view zoom multiplier (HUD/root-console remains fixed)."""
        return float(self._viewport_zoom)

    @property
    def is_default_viewport_zoom(self) -> bool:
        """Whether the world view is currently at the configured default zoom stop."""
        return self._zoom_index == config.DEFAULT_ZOOM_INDEX

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def highlight_actor(
        self,
        actor: Actor,
        color: colors.Color,
        effect: Literal["solid", "pulse"] = "solid",
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
        effect: Literal["solid", "pulse"] = "solid",
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
        scale_x, scale_y = vs.get_display_scale_factors()
        self.graphics.draw_tile_highlight(
            root_x,
            root_y,
            final_color,
            alpha,
            scale_x=scale_x,
            scale_y=scale_y,
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

        player = getattr(gw, "player", None)
        player_pos_key = (
            getattr(player, "x", None),
            getattr(player, "y", None),
        )

        # Roof ridge shading depends on the sun direction, so the background
        # must be invalidated when the directional light changes.
        sun_key = self._get_sun_direction_cache_key()

        return (bounds_key, map_key, exploration_key, player_pos_key, sun_key)

    @staticmethod
    def _building_identity_key(building: Building) -> tuple[object, ...]:
        """Return a hashable key capturing a building's structural identity.

        Used by multiple cache systems to detect when a building's visual
        representation should be considered changed.
        """
        return (
            int(building.id),
            int(building.footprint.x1),
            int(building.footprint.y1),
            int(building.footprint.x2),
            int(building.footprint.y2),
            str(building.roof_style),
            str(building.roof_profile),
            round(float(building.flat_section_ratio), 3),
            int(building.floor_count),
            tuple((int(x), int(y)) for x, y in building.door_positions),
            (
                None
                if building.chimney_offset is None
                else (
                    int(building.chimney_offset[0]),
                    int(building.chimney_offset[1]),
                )
            ),
            round(building.chimney_projected_height, 3),
            round(float(building.condition), 3),
        )

    @staticmethod
    def _hash_array_view(array: np.ndarray) -> bytes:
        """Return a stable digest for a NumPy array view used in render cache keys."""
        if array.size == 0:
            return b""

        # Use the array's logical layout (`order="A"`) so equivalent slices hash
        # the same regardless of contiguous/strided backing storage.
        return hashlib.blake2b(array.tobytes(order="A"), digest_size=16).digest()

    def _get_map_unlit_buffer_cache_key(self) -> tuple[object, ...]:
        """Return a cache key for `_render_map_unlit()` GlyphBuffer contents."""
        gw = self.controller.gw
        game_map = gw.game_map
        vs = self.viewport_system
        pad = self._SCROLL_PADDING

        bounds = vs.viewport.get_world_bounds(vs.camera)
        world_origin_x = bounds.x1 - vs.offset_x - pad
        world_origin_y = bounds.y1 - vs.offset_y - pad
        buf_width = self.map_glyph_buffer.width
        buf_height = self.map_glyph_buffer.height

        world_x1 = max(0, world_origin_x)
        world_y1 = max(0, world_origin_y)
        world_x2 = min(game_map.width, world_origin_x + buf_width)
        world_y2 = min(game_map.height, world_origin_y + buf_height)
        if world_x1 < world_x2 and world_y1 < world_y2:
            explored_window = game_map.explored[world_x1:world_x2, world_y1:world_y2]
            explored_key: tuple[object, ...] = (
                world_x1,
                world_y1,
                world_x2,
                world_y2,
                self._hash_array_view(explored_window),
            )
        else:
            explored_key = (world_x1, world_y1, world_x2, world_y2, b"")

        # Roof substitution changes the unlit terrain pass based on player position,
        # visible buildings, and directional light (ridge shading).
        player_building_id, viewport_buildings = self._compute_roof_state()
        roof_buildings_key = tuple(
            self._building_identity_key(building) for building in viewport_buildings
        )

        camera_pos_key = (
            round(float(vs.camera.world_x), 6),
            round(float(vs.camera.world_y), 6),
        )

        # Include LOD state so crossing the detail threshold invalidates the
        # cached buffer (tiles render differently with/without decoration).
        lod_detail = self._viewport_zoom >= config.LOD_DETAIL_ZOOM_THRESHOLD

        return (
            id(self.map_glyph_buffer),
            buf_width,
            buf_height,
            int(world_origin_x),
            int(world_origin_y),
            int(game_map.width),
            int(game_map.height),
            int(vs.offset_x),
            int(vs.offset_y),
            camera_pos_key,
            int(getattr(game_map, "structural_revision", 0)),
            int(game_map.decoration_seed),
            id(game_map.dark_appearance_map),
            explored_key,
            getattr(getattr(gw, "player", None), "x", None),
            getattr(getattr(gw, "player", None), "y", None),
            player_building_id,
            roof_buildings_key,
            self._get_sun_direction_cache_key(),
            lod_detail,
        )

    def draw(self, graphics: GraphicsContext, alpha: InterpolationAlpha) -> None:
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
        with record_time_live_variable("time.render.draw_updates_ms"):
            self.particle_system.update(delta_time)
            self.environmental_system.update(delta_time)
            self.floating_text_manager.update(delta_time)
            self.atmospheric_system.update(delta_time)
            self.rain_animation.update(delta_time, self.rain_config)
            # Update decals for age-based cleanup
            self._game_time += delta_time
            self.decal_system.update(delta_time, self._game_time)

            # Update tile animations (color oscillation, glyph flicker)
            with record_time_live_variable("time.render.draw_updates.tile_anim_ms"):
                self._update_tile_animations()

            # Emit particles from actors with particle emitters
            self._update_actor_particles()

    def present(self, graphics: GraphicsContext, alpha: InterpolationAlpha) -> None:
        """Composite final frame layers in proper order."""
        if not self.visible:
            return

        vs = self.viewport_system
        pad = self._SCROLL_PADDING
        scale_x, scale_y = vs.get_display_scale_factors()

        # Calculate pixel offsets for smooth scrolling.
        # The background texture is larger than the viewport (includes padding),
        # so we need to offset it to align the visible portion correctly.
        base_px_x, base_px_y = graphics.console_to_screen_coords(0.0, 0.0)

        # 1. Screen shake offset (existing behavior)
        shake_tile_x, shake_tile_y = self._shake_offset
        shake_px_x, shake_px_y = graphics.console_to_screen_coords(
            shake_tile_x * scale_x, shake_tile_y * scale_y
        )
        shake_offset_x = shake_px_x - base_px_x
        shake_offset_y = shake_px_y - base_px_y

        # 2. Camera fractional offset for smooth sub-tile scrolling.
        # The camera position is rounded when selecting which tiles to render.
        # The fractional part tells us how far between tiles the camera actually is.
        # We offset the texture in the opposite direction to compensate.
        cam_frac_x, cam_frac_y = vs.get_display_camera_fractional_offset()
        # Store for use by actor/particle rendering methods
        self.camera_frac_offset = (cam_frac_x, cam_frac_y)
        cam_px_x, cam_px_y = graphics.console_to_screen_coords(-cam_frac_x, -cam_frac_y)
        cam_offset_x = cam_px_x - base_px_x
        cam_offset_y = cam_px_y - base_px_y

        # Combined offset for background rendering
        offset_x_pixels = shake_offset_x + cam_offset_x
        offset_y_pixels = shake_offset_y + cam_offset_y

        # Present the padded world texture into the fixed world-view rect by
        # scaling visible world tiles into root-console tile space.
        visible_w = self.viewport_system.viewport.width_tiles
        visible_h = self.viewport_system.viewport.height_tiles
        tex_x = self.x - (pad * scale_x)
        tex_y = self.y - (pad * scale_y)
        tex_width = (visible_w + 2 * pad) * scale_x
        tex_height = (visible_h + 2 * pad) * scale_y

        # 1. Present the cached unlit background with combined offset
        if self._active_background_texture:
            with record_time_live_variable("time.render.present_background_ms"):
                graphics.draw_background(
                    self._active_background_texture,
                    tex_x,
                    tex_y,
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
                    tex_x,
                    tex_y,
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
        # Atmospheric uniforms expect visible viewport size in world tiles,
        # not the fixed on-screen world-view rect size in root-console tiles.
        viewport_size = (
            self.viewport_system.viewport.width_tiles,
            self.viewport_system.viewport.height_tiles,
        )
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
        dynamic_receivers_for_frame: list[Actor] | None = None
        roof_occluded: frozenset[Actor] = frozenset()
        actor_bounds = vs.get_visible_bounds()
        with record_time_live_variable("time.render.actor_filter_ms"):
            if config.SHADOWS_ENABLED:
                visible_actors_for_frame = [
                    actor
                    for actor in self.actor_renderer.get_sorted_visible_actors(
                        actor_bounds, self.controller.gw
                    )
                    if self.controller.gw.game_map.visible[actor.x, actor.y]
                ]
                visible_actors_for_frame, roof_occluded = (
                    self._filter_roof_occluded_actors(
                        visible_actors_for_frame,
                        actor_bounds,
                    )
                )
                # Only dynamic actors (those with an energy component) need shadow
                # receiver dimming. Exclude roof-occluded actors since they should
                # not cast or receive shadows through opaque roofs.
                dynamic_receivers_for_frame = [
                    a
                    for a in visible_actors_for_frame
                    if a.energy is not None and a not in roof_occluded
                ]
            else:
                visible_actors_for_frame, roof_occluded = (
                    self._filter_roof_occluded_actors(
                        self.actor_renderer.get_sorted_visible_actors(
                            actor_bounds, self.controller.gw
                        ),
                        actor_bounds,
                    )
                )

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

        directional_light = self._get_directional_light()
        with (
            record_time_live_variable("time.render.actor_shadows_ms"),
            graphics.shadow_pass(),
        ):
            self.shadow_renderer.game_map = self.controller.gw.game_map
            self.shadow_renderer.viewport_system = self.viewport_system
            self.shadow_renderer.graphics = graphics
            self.shadow_renderer.viewport_zoom = self._viewport_zoom
            # Exclude roof-occluded actors from shadow rendering since their
            # shadows should not appear through opaque roofs.
            shadow_actors = (
                [a for a in visible_actors_for_frame if a not in roof_occluded]
                if roof_occluded
                else visible_actors_for_frame
            )
            self.shadow_renderer.render_actor_shadows(
                alpha,
                visible_actors=shadow_actors,
                dynamic_receivers=dynamic_receivers_for_frame,
                directional_light=directional_light,
                lights=self.controller.gw.lights,
                view_origin=(float(self.x), float(self.y)),
                camera_frac_offset=self.camera_frac_offset,
            )
            # Chimney shadows use the same GPU parallelogram pipeline as
            # actor shadows, emitted inside the same shadow_pass() bracket.
            self._render_chimney_shadows(graphics, directional_light)

        # Redraw chimney cap tiles in the post-shadow layer so they cover
        # the shadow parallelogram overlap, just as actor sprites cover
        # their shadow at the base.  Only needed when shadows are enabled
        # since there is nothing to cover otherwise, and the overlay uses
        # fixed sunlit colours that could mismatch the glyph buffer at night.
        if config.SHADOWS_ENABLED:
            self._render_chimney_overlays(graphics)

        self._apply_sun_direction_to_graphics(graphics, directional_light)

        # Render under-actor highlights (e.g. hover outlines) so actor sprites
        # paint on top, producing a lasso-style framing effect.
        for mode in self.controller.mode_stack:
            mode.render_world_under_actors()

        with record_time_live_variable("time.render.render_actors_ms"):
            self.actor_renderer.render_actors(
                alpha,
                game_world=self.controller.gw,
                camera_frac_offset=self.camera_frac_offset,
                view_origin=(float(self.x), float(self.y)),
                visible_actors=visible_actors_for_frame,
                viewport_bounds=actor_bounds,
                roof_occluded_actors=roof_occluded or None,
            )

        if config.ATMOSPHERIC_EFFECTS_ENABLED:
            with record_time_live_variable("time.render.atmospheric_ms"):
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

                active_layers = self.atmospheric_system.get_active_layers(
                    is_raining=self.rain_config.enabled
                )
                # Render mist first, then shadows, to keep shadows readable on top.
                active_layers.sort(
                    key=lambda layer_state: (
                        0 if layer_state[0].blend_mode == "lighten" else 1
                    )
                )
                roof_surface_mask_buffer = self._build_atmospheric_roof_surface_mask(
                    viewport_offset, viewport_size
                )

                for layer, state in active_layers:
                    effective_strength = layer.strength
                    if layer.disable_when_overcast:
                        coverage = self.atmospheric_system.config.cloud_coverage
                        # Keep a baseline shadow presence while still responding to
                        # coverage.
                        coverage_scale = 0.35 + 0.65 * (1.0 - coverage)
                        effective_strength *= saturate(coverage_scale)

                    # This queues atmospheric uniforms for the GPU atmospheric
                    # renderer; compositing happens in finalize_present() over the
                    # current framebuffer, which already includes actors.
                    graphics.set_atmospheric_layer(
                        viewport_offset,
                        viewport_size,
                        map_size,
                        layer.sky_exposure_threshold,
                        sky_exposure_texture,
                        explored_texture,
                        visible_texture,
                        roof_surface_mask_buffer,
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
                        affects_foreground=layer.affects_foreground,
                    )

        if self.rain_config.enabled:
            rain_exclusion_mask = self._build_rain_exclusion_mask(
                viewport_offset,
                viewport_size,
            )
            rain_angle = self.rain_animation.render_angle
            # Keep perceived rain shape/density/speed stable across viewport zoom.
            # Rain tuning vars are authored at 1.0x zoom, so compensate tile-space
            # values before sending them to the shader.
            zoom_scale = max(float(config.ZOOM_STOPS[0]), float(self._viewport_zoom))
            zoom_compensation = 1.0 / zoom_scale
            graphics.set_rain_effect(
                viewport_offset=viewport_offset,
                viewport_size=viewport_size,
                tile_dimensions=graphics.tile_dimensions,
                intensity=self.rain_config.intensity,
                angle=rain_angle,
                drop_length=self.rain_config.drop_length * zoom_compensation,
                drop_speed=self.rain_config.drop_speed * zoom_compensation,
                drop_spacing=self.rain_config.drop_spacing * zoom_compensation,
                stream_spacing=self.rain_config.stream_spacing * zoom_compensation,
                rain_color=self.rain_config.color,
                time=self.rain_animation.time,
                rain_exclusion_mask_buffer=rain_exclusion_mask,
                pixel_bounds=(
                    round(px_left),
                    round(px_top),
                    round(px_right),
                    round(px_bottom),
                ),
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
                    viewport_system=self.viewport_system,
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

    def _get_roof_entrance_clear_positions(
        self, building: Building
    ) -> set[tuple[int, int]]:
        """Return roof mask carve-outs for a building's exterior doorway/approach."""
        entrance_clear_positions: set[tuple[int, int]] = set()
        for door_x, door_y in building.door_positions:
            entrance_clear_positions.add((door_x, door_y))

            outward_dx = 0
            outward_dy = 0
            if door_x == building.footprint.x1:
                outward_dx = -1
            elif door_x == building.footprint.x2 - 1:
                outward_dx = 1
            elif door_y == building.footprint.y1:
                outward_dy = -1
            elif door_y == building.footprint.y2 - 1:
                outward_dy = 1

            if outward_dx != 0 or outward_dy != 0:
                approach_x = door_x + outward_dx
                approach_y = door_y + outward_dy
                entrance_clear_positions.add((approach_x, approach_y))

                if outward_dx != 0:
                    entrance_clear_positions.add((approach_x, approach_y - 1))
                    entrance_clear_positions.add((approach_x, approach_y + 1))
                else:
                    entrance_clear_positions.add((approach_x - 1, approach_y))
                    entrance_clear_positions.add((approach_x + 1, approach_y))

        return entrance_clear_positions

    def _build_atmospheric_roof_surface_mask(
        self,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
    ) -> np.ndarray | None:
        """Build a viewport-space mask of roof tiles visible this frame.

        The atmospheric pass operates on world-derived sky exposure, but roofs are
        a view-time substitution layered over indoor tiles. This mask marks the
        roof surfaces that are actually drawn so cloud shadows can apply to them
        without changing global sky-exposure semantics.
        """
        viewport_w, viewport_h = viewport_size
        if viewport_w <= 0 or viewport_h <= 0:
            return None

        player_building_id, viewport_buildings = self._compute_roof_state()
        roof_visible_buildings = [
            b for b in viewport_buildings if b.id != player_building_id
        ]
        if not roof_visible_buildings:
            return None

        view_x, view_y = viewport_offset
        view_x2 = view_x + viewport_w
        view_y2 = view_y + viewport_h
        roof_mask = np.zeros((viewport_w, viewport_h), dtype=bool)

        for building in roof_visible_buildings:
            fp = building.footprint
            # Visual roof bounds: the roof's north edge starts at the first
            # full overhang row (no north split tile), and the south end
            # recedes by floor_offset wall face rows.
            N = building.perspective_ceil_offset
            floor_offset = building.perspective_floor_offset
            has_frac = building.perspective_has_frac

            # First full roof row north of footprint.
            visual_y1 = fp.y1 - N + (1 if has_frac else 0)
            visual_y2_excl = fp.y2 - floor_offset  # exclusive upper bound

            clip_x1 = max(view_x, fp.x1)
            clip_y1 = max(view_y, visual_y1)
            clip_x2 = min(view_x2, fp.x2)
            clip_y2 = min(view_y2, visual_y2_excl)
            if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
                continue

            roof_mask[
                (clip_x1 - view_x) : (clip_x2 - view_x),
                (clip_y1 - view_y) : (clip_y2 - view_y),
            ] = True

            entrance_clear_positions = self._get_roof_entrance_clear_positions(building)
            if entrance_clear_positions:
                for clear_x, clear_y in entrance_clear_positions:
                    if view_x <= clear_x < view_x2 and view_y <= clear_y < view_y2:
                        roof_mask[clear_x - view_x, clear_y - view_y] = False

        if not np.any(roof_mask):
            return None
        return roof_mask

    def _build_rain_exclusion_mask(
        self,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
    ) -> np.ndarray | None:
        """Build a viewport-space mask for the current player building interior.

        Rain is a full-screen stochastic overlay, so we only carve out the
        interior footprint of the building the player is currently inside.
        """
        viewport_w, viewport_h = viewport_size
        if viewport_w <= 0 or viewport_h <= 0:
            return None

        player_building_id, viewport_buildings = self._compute_roof_state()
        if player_building_id is None:
            return None

        player_building = next(
            (b for b in viewport_buildings if b.id == player_building_id),
            None,
        )
        if player_building is None:
            return None

        view_x, view_y = viewport_offset
        view_x2 = view_x + viewport_w
        view_y2 = view_y + viewport_h

        footprint = player_building.footprint
        clip_x1 = max(view_x, footprint.x1)
        clip_y1 = max(view_y, footprint.y1)
        clip_x2 = min(view_x2, footprint.x2)
        clip_y2 = min(view_y2, footprint.y2)
        if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
            return None

        mask = np.zeros((viewport_w, viewport_h), dtype=np.bool_)
        mask[
            (clip_x1 - view_x) : (clip_x2 - view_x),
            (clip_y1 - view_y) : (clip_y2 - view_y),
        ] = True
        return mask

    def _filter_roof_occluded_actors(
        self,
        actors: list[Actor],
        viewport_bounds_world: Rect,
    ) -> tuple[list[Actor], frozenset[Actor]]:
        """Tag actors under opaque roofs so they render as white outlines.

        Returns the full actor list (all actors kept, in original order) and a
        frozenset of actors that are under opaque roofs. The caller renders
        occluded actors as outline-only glyphs and excludes them from shadows.
        """
        _empty: frozenset[Actor] = frozenset()
        if not actors:
            return actors, _empty

        player_building_id, viewport_buildings = self._compute_roof_state()

        if not viewport_buildings:
            return actors, _empty

        roof_visible_buildings = [
            b for b in viewport_buildings if b.id != player_building_id
        ]
        if not roof_visible_buildings:
            return actors, _empty

        occluded_set: set[Actor] = set()
        for actor in actors:
            ax = actor.x
            ay = actor.y
            # Skip unnecessary map lookups for actors outside the viewport slice.
            if not (
                viewport_bounds_world.x1 <= ax <= viewport_bounds_world.x2
                and viewport_bounds_world.y1 <= ay <= viewport_bounds_world.y2
            ):
                continue

            for building in roof_visible_buildings:
                # Occlude actors under the shifted visual roof bounds.
                # The roof covers the footprint (minus the wall face strip at the
                # south end) plus an overhang extending N tiles north.
                fp = building.footprint
                N = building.perspective_ceil_offset
                floor_offset = building.perspective_floor_offset
                has_frac = building.perspective_has_frac

                # Visual roof X range is the footprint width.
                if not (fp.x1 <= ax < fp.x2):
                    continue

                # Visual roof Y range: first full roof row to y2-1-floor_offset
                # (where floor_offset full rows become wall face, and the south
                # boundary row is partially roof). Wall face tiles are visible.
                visual_roof_y_min = fp.y1 - N + (1 if has_frac else 0)
                visual_roof_y_max = fp.y2 - 1 - floor_offset
                if not (visual_roof_y_min <= ay <= visual_roof_y_max):
                    continue

                if (ax, ay) not in self._get_roof_entrance_clear_positions(building):
                    occluded_set.add(actor)
                    break

        return actors, frozenset(occluded_set)

    def _compute_roof_state(self) -> tuple[int | None, list[Building]]:
        """Cache per-frame roof visibility inputs shared by both render passes."""
        gw = self.controller.gw
        game_map = gw.game_map
        bounds = self.viewport_system.get_visible_bounds()
        raw_buildings = getattr(gw, "buildings", [])
        buildings = raw_buildings if isinstance(raw_buildings, list) else []

        player = getattr(gw, "player", None)
        player_x = getattr(player, "x", None)
        player_y = getattr(player, "y", None)
        cache_key: tuple[object, ...] = (
            bounds.x1,
            bounds.y1,
            bounds.x2,
            bounds.y2,
            player_x,
            player_y,
            getattr(game_map, "structural_revision", 0),
            len(buildings),
        )
        # getattr is deliberate: some test doubles bypass __init__, so
        # these attributes may not exist on partially-constructed instances.
        cached_roof_state_key = getattr(self, "_roof_state_cache_key", None)
        cached_roof_state_value = getattr(self, "_roof_state_cache_value", None)
        if cached_roof_state_key == cache_key and cached_roof_state_value is not None:
            return cached_roof_state_value

        player_building_id: int | None = None
        get_region_at = getattr(game_map, "get_region_at", None)
        if (
            callable(get_region_at)
            and isinstance(player_x, int)
            and isinstance(player_y, int)
        ):
            region = get_region_at((player_x, player_y))
            if region is not None and region.sky_exposure <= 0.1:
                player_region_id = region.id
                for building in buildings:
                    if any(
                        room.region_id == player_region_id for room in building.rooms
                    ):
                        player_building_id = building.id
                        break

            # Some interior archways/wall tiles may have no region assignment or be in
            # an indoor region not represented in building.rooms. The footprint is the
            # reliable semantic boundary for "inside the house" roof cutaway behavior.
            if player_building_id is None:
                for building in buildings:
                    if building.contains_point(player_x, player_y):
                        player_building_id = building.id
                        break

        # Include scroll padding so roof tiles stay available during smooth camera
        # motion when the unlit pass renders one extra tile around the viewport.
        pad = self._SCROLL_PADDING
        view_left = bounds.x1 - pad
        view_top = bounds.y1 - pad
        view_right = bounds.x2 + pad
        view_bottom = bounds.y2 + pad

        # Expand the vertical check to account for perspective offset: a building's
        # shifted roof extends up to _MAX_PERSPECTIVE_SHIFT_TILES north of its
        # footprint, so buildings just south of the viewport may still be visible.
        viewport_buildings: list[Building] = []
        for building in buildings:
            fp = building.footprint
            if (
                fp.x1 <= view_right
                and fp.x2 - 1 >= view_left
                and fp.y1 - _MAX_PERSPECTIVE_SHIFT_TILES <= view_bottom
                and fp.y2 - 1 >= view_top
            ):
                viewport_buildings.append(building)

        roof_state = (player_building_id, viewport_buildings)
        self._roof_state_cache_key = cache_key
        self._roof_state_cache_value = roof_state
        return roof_state

    @staticmethod
    def _building_roof_color_offset(
        building: Building, decoration_seed: int
    ) -> tuple[int, int, int]:
        """Compute a subtle deterministic per-building roof RGB tint offset."""
        footprint = building.footprint
        h = (
            (int(building.id) * 0x9E3779B1)
            ^ (int(footprint.x1) * 0x85EBCA6B)
            ^ (int(footprint.y1) * 0xC2B2AE35)
            ^ int(decoration_seed)
        ) & 0xFFFFFFFF
        # Two rounds of avalanche mixing improve byte independence while staying
        # deterministic across Python versions/platforms.
        h = ((h ^ (h >> 16)) * 0x045D9F3B) & 0xFFFFFFFF
        h = ((h ^ (h >> 16)) * 0x045D9F3B) & 0xFFFFFFFF
        h = (h ^ (h >> 16)) & 0xFFFFFFFF

        r_offset = (h & 0xFF) % 17 - 8
        g_offset = ((h >> 8) & 0xFF) % 17 - 8
        b_offset = ((h >> 16) & 0xFF) % 17 - 8
        return (r_offset, g_offset, b_offset)

    def _build_roof_stamp(
        self,
        building: Building,
        *,
        is_light: bool,
        decoration_seed: int,
        sun_dx: float,
        sun_dy: float,
    ) -> _RoofStamp:
        """Build cached perspective-aware roof visuals for a building.

        The stamp covers the building footprint extended north by the
        perspective offset (roof overhang region). It precomputes all
        zone visuals - roof surface, wall face, split boundary tiles,
        chimney - so the per-frame blit in _apply_roof_substitution is
        just an array copy.
        """
        fp = building.footprint
        width = int(fp.width)
        height = int(fp.height)
        N = building.perspective_ceil_offset
        floor_offset = building.perspective_floor_offset
        frac = building.perspective_frac
        has_frac = building.perspective_has_frac

        ext_h = height + N  # Extended height including overhang
        stamp_world_y_start = int(fp.y1) - N

        def _empty_stamp() -> _RoofStamp:
            z2 = np.zeros((width, ext_h), dtype=np.int32)
            return _RoofStamp(
                chars=z2.copy(),
                fg_rgb=np.zeros((width, ext_h, 3), dtype=np.uint8),
                bg_rgb=np.zeros((width, ext_h, 3), dtype=np.uint8),
                tile_ids=z2.copy(),
                draw_mask=np.zeros((width, ext_h), dtype=np.bool_),
                north_overhang=N,
                noise_pattern=np.zeros((width, ext_h), dtype=np.uint8),
                noise_pattern_mask=np.zeros((width, ext_h), dtype=np.bool_),
                split_y=np.zeros((width, ext_h), dtype=np.float32),
                split_bg=np.zeros((width, ext_h, 4), dtype=np.uint8),
                split_fg=np.zeros((width, ext_h, 4), dtype=np.uint8),
                split_noise=np.zeros((width, ext_h), dtype=np.float32),
                split_noise_pattern=np.zeros((width, ext_h), dtype=np.uint8),
                wear_pack=np.zeros((width, ext_h), dtype=np.uint32),
            )

        if width <= 0 or ext_h <= 0:
            return _empty_stamp()

        # Allocate stamp arrays.
        chars = np.zeros((width, ext_h), dtype=np.int32)
        fg_rgb = np.zeros((width, ext_h, 3), dtype=np.uint8)
        bg_rgb = np.zeros((width, ext_h, 3), dtype=np.uint8)
        tile_ids = np.zeros((width, ext_h), dtype=np.int32)
        noise_pattern_arr = np.zeros((width, ext_h), dtype=np.uint8)
        noise_pattern_mask_arr = np.zeros((width, ext_h), dtype=np.bool_)
        split_y_arr = np.zeros((width, ext_h), dtype=np.float32)
        split_bg_arr = np.zeros((width, ext_h, 4), dtype=np.uint8)
        split_fg_arr = np.zeros((width, ext_h, 4), dtype=np.uint8)
        split_noise_arr = np.zeros((width, ext_h), dtype=np.float32)
        split_noise_pattern_arr = np.zeros((width, ext_h), dtype=np.uint8)

        # World coordinate grids for the extended stamp region.
        x_coords = np.arange(fp.x1, fp.x2, dtype=np.int32)
        y_coords = np.arange(stamp_world_y_start, int(fp.y2), dtype=np.int32)
        world_x_grid, world_y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")

        # --- Zone boundaries in world coordinates ---
        split_y_value = (1.0 - frac) if has_frac else 0.0
        if has_frac:
            south_split_row = int(fp.y2) - 1 - floor_offset
            roof_inside_end = south_split_row  # exclusive
        else:
            south_split_row = -1  # sentinel: no split
            roof_inside_end = int(fp.y2) - N

        roof_overhang_start = int(fp.y1) - N + (1 if has_frac else 0)
        wall_y_start = int(fp.y2) - floor_offset if floor_offset > 0 else int(fp.y2)

        # --- Entrance exclusion ---
        entrance_clear = np.zeros((width, ext_h), dtype=np.bool_)
        for clear_x, clear_y in self._get_roof_entrance_clear_positions(building):
            lx = clear_x - int(fp.x1)
            ly = clear_y - stamp_world_y_start
            if 0 <= lx < width and 0 <= ly < ext_h:
                entrance_clear[lx, ly] = True

        # --- Chimney at shifted position (moves north by N tiles) ---
        chimney_mask = np.zeros((width, ext_h), dtype=np.bool_)
        chimney_pos = building.chimney_world_pos
        if chimney_pos is not None:
            cx, cy = chimney_pos
            shifted_cy = cy - N
            lx = cx - int(fp.x1)
            ly = shifted_cy - stamp_world_y_start
            if 0 <= lx < width and 0 <= ly < ext_h:
                chimney_mask[lx, ly] = True

        # --- Zone masks via world coordinate comparisons on the grid ---
        in_roof_overhang = (world_y_grid >= roof_overhang_start) & (
            world_y_grid < int(fp.y1)
        )
        in_roof_inside = (world_y_grid >= int(fp.y1)) & (world_y_grid < roof_inside_end)
        full_roof = (
            (in_roof_overhang | in_roof_inside) & ~entrance_clear & ~chimney_mask
        )

        south_split_mask = np.zeros((width, ext_h), dtype=np.bool_)
        if has_frac:
            south_split_mask = (
                (world_y_grid == south_split_row) & ~entrance_clear & ~chimney_mask
            )

        wall_mask = np.zeros((width, ext_h), dtype=np.bool_)
        if floor_offset > 0:
            wall_mask = (
                (world_y_grid >= wall_y_start)
                & (world_y_grid < int(fp.y2))
                & ~entrance_clear
            )

        draw_mask = full_roof | south_split_mask | wall_mask | chimney_mask
        if not np.any(draw_mask):
            return _empty_stamp()

        # --- Roof appearance (covers full roof + south split primary) ---
        roof_tile_id = tile_types.ROOF_STYLE_TILE_TYPES.get(
            building.roof_style, tile_types.TileTypeID.ROOF_THATCH
        )
        roof_tile_id_int = int(roof_tile_id)
        roof_noise_pattern = tile_types.get_roof_noise_pattern(
            roof_tile_id, building.ridge_axis
        )

        # Roof and split tiles both use the roof tile_id for lighting lookups.
        roof_and_split = full_roof | south_split_mask | chimney_mask
        tile_ids[roof_and_split] = roof_tile_id_int
        noise_pattern_arr[roof_and_split] = np.uint8(roof_noise_pattern)
        noise_pattern_mask_arr[roof_and_split] = True

        all_roof = full_roof | south_split_mask
        if np.any(all_roof):
            roof_data = tile_types.get_tile_type_data_by_id(roof_tile_id_int)
            roof_appearance = roof_data["light" if is_light else "dark"]

            roof_count = int(np.count_nonzero(all_roof))
            roof_chars = np.full(
                roof_count, int(roof_appearance["ch"]), dtype=chars.dtype
            )
            roof_fg = np.tile(
                np.asarray(roof_appearance["fg"], dtype=np.uint8), (roof_count, 1)
            )
            roof_bg = np.tile(
                np.asarray(roof_appearance["bg"], dtype=np.uint8), (roof_count, 1)
            )
            roof_tile_ids_flat = np.full(roof_count, roof_tile_id_int, dtype=np.int32)

            roof_world_x = world_x_grid[all_roof]
            roof_world_y = world_y_grid[all_roof]

            tile_types.apply_terrain_decoration(
                roof_chars,
                roof_fg,
                roof_bg,
                roof_tile_ids_flat,
                roof_world_x,
                roof_world_y,
                decoration_seed,
            )

            roof_color_offset = np.asarray(
                self._building_roof_color_offset(building, decoration_seed),
                dtype=np.int16,
            )
            roof_bg = _adjust_color_brightness(roof_bg, roof_color_offset)

            chars[all_roof] = roof_chars
            fg_rgb[all_roof] = roof_fg
            bg_rgb[all_roof] = roof_bg

        # --- Ridge shading on all roof surfaces ---
        # Ridge center is shifted north by N tiles to follow the visual roof.
        if np.any(all_roof):
            rx = world_x_grid[all_roof]
            ry = world_y_grid[all_roof]

            if building.ridge_axis == "horizontal":
                shifted_center = (fp.y1 + fp.y2) / 2.0 - N
                ridge_axis_offset = ry + 0.5 - shifted_center
                sun_component = sun_dy
            else:
                center = (fp.x1 + fp.x2) / 2.0
                ridge_axis_offset = rx + 0.5 - center
                sun_component = sun_dx

            abs_ridge_offset = np.abs(ridge_axis_offset)
            is_ridge = abs_ridge_offset < 0.6
            is_sun_side = (ridge_axis_offset * sun_component) > 0
            intensity = abs(sun_component)
            roof_profile = building.roof_profile

            # Corrugated tin still has a pitched profile, but its smoother,
            # manufactured surface reads better with slightly gentler ridge
            # contrast than thatch/shingle.
            if roof_tile_id_int == int(tile_types.TileTypeID.ROOF_TIN):
                ridge_peak = 4
                slope_peak = 12
            else:
                ridge_peak = 6
                slope_peak = 18

            if roof_profile == "flat":
                ridge_brightness = np.full(
                    rx.shape, np.int16(round(2 * intensity)), dtype=np.int16
                )
            elif roof_profile == "low_slope":
                short_axis_span = (
                    int(fp.height)
                    if building.ridge_axis == "horizontal"
                    else int(fp.width)
                )
                flat_half_span = max(
                    0.6, short_axis_span * float(building.flat_section_ratio) * 0.5
                )
                in_flat_section = abs_ridge_offset < flat_half_span
                # round() returns int in Python 3; peaks stay integer-valued for
                # stable per-tile brightness offsets.
                low_slope_peak = max(4, round(slope_peak * 0.6))
                flat_peak = max(1, round(ridge_peak * 0.5))
                ridge_brightness = np.where(
                    in_flat_section,
                    np.int16(round(flat_peak * intensity)),
                    np.where(
                        is_sun_side,
                        np.int16(round(low_slope_peak * intensity)),
                        np.int16(round(-low_slope_peak * intensity)),
                    ),
                )
            else:
                ridge_brightness = np.where(
                    is_ridge,
                    np.int16(round(ridge_peak * intensity)),
                    np.where(
                        is_sun_side,
                        np.int16(round(slope_peak * intensity)),
                        np.int16(round(-slope_peak * intensity)),
                    ),
                )

            ridge_brightness_offset = ridge_brightness[:, np.newaxis]
            bg_rgb[all_roof] = _adjust_color_brightness(
                bg_rgb[all_roof], ridge_brightness_offset
            )
            fg_rgb[all_roof] = _adjust_color_brightness(
                fg_rgb[all_roof], ridge_brightness_offset
            )

        # --- Eave darkening on shifted roof perimeter ---
        # The visual roof spans from the first full roof row to the south
        # split/wall boundary.
        visual_roof = full_roof | south_split_mask
        visual_roof_y_max = south_split_row if has_frac else int(fp.y2) - N - 1
        visual_roof_y_min = roof_overhang_start

        is_west_edge = world_x_grid == fp.x1
        is_east_edge = world_x_grid == fp.x2 - 1
        is_north_edge = world_y_grid == visual_roof_y_min
        is_south_edge = world_y_grid == visual_roof_y_max
        perimeter_mask = visual_roof & (
            is_west_edge | is_east_edge | is_north_edge | is_south_edge
        )
        if np.any(perimeter_mask):
            bg_rgb[perimeter_mask] = _adjust_color_brightness(
                bg_rgb[perimeter_mask], -6
            )

        # Strong darkening on south edge where roof meets wall face.
        south_roof_edge = visual_roof & is_south_edge
        if np.any(south_roof_edge):
            bg_rgb[south_roof_edge] = _adjust_color_brightness(
                bg_rgb[south_roof_edge], -10
            )

        corner_mask = visual_roof & (
            (is_west_edge | is_east_edge) & (is_north_edge | is_south_edge)
        )
        if np.any(corner_mask):
            bg_rgb[corner_mask] = _adjust_color_brightness(bg_rgb[corner_mask], -4)

        # --- Eave shadow color (shared by wall face and south split) ---
        eave_bg = np.asarray(
            colors.WALL_EAVE_SHADOW_LIGHT if is_light else colors.WALL_EAVE_SHADOW_DARK,
            dtype=np.uint8,
        )

        # --- Wall face appearance ---
        # Wall tiles keep default tile_ids so lighting treats them as visible
        # exterior surfaces, not opaque roof.  Uses WALL_FACE colors instead
        # of deriving from eave shadow so the wall reads as a lit surface
        # with the eave shadow as a distinct dark band above.
        if np.any(wall_mask):
            wall_face = np.asarray(
                colors.WALL_FACE_LIGHT if is_light else colors.WALL_FACE_DARK,
                dtype=np.uint8,
            )

            chars[wall_mask] = ord(" ")

            # Per-row darkening: each successive row loses brightness to sell
            # the receding wall face depth.
            wall_wy = world_y_grid[wall_mask]
            row_offset = (wall_wy - wall_y_start).astype(np.int16)
            wall_base = _adjust_color_brightness(
                wall_face, -(row_offset[:, np.newaxis] * 6)
            )
            fg_rgb[wall_mask] = wall_base
            bg_rgb[wall_mask] = wall_base

            # Edge darkening: west/east wall tiles suggest the wall wrapping
            # around to a side face.
            wall_and_edge = wall_mask & (is_west_edge | is_east_edge)
            if np.any(wall_and_edge):
                darkened = _adjust_color_brightness(bg_rgb[wall_and_edge], -10)
                bg_rgb[wall_and_edge] = darkened
                fg_rgb[wall_and_edge] = darkened

        # --- South split: primary=roof (above threshold), split=eave shadow ---
        # The wall portion right under the roof is the darkest part (eave
        # shadow), selling the roof-wall depth separation.
        if np.any(south_split_mask):
            eave_rgba = np.append(eave_bg, np.uint8(255))
            split_y_arr[south_split_mask] = split_y_value
            split_bg_arr[south_split_mask] = eave_rgba
            split_fg_arr[south_split_mask] = eave_rgba
            split_noise_arr[south_split_mask] = 0.012
            split_noise_pattern_arr[south_split_mask] = 0

        # --- Chimney top, projected body, and shadow ---
        if chimney_pos is not None:
            cx, cy = chimney_pos
            shifted_cy = cy - N
            lx = cx - int(fp.x1)
            ly = shifted_cy - stamp_world_y_start

            if 0 <= lx < width and 0 <= ly < ext_h:
                # Top of chimney: plan-view cap with flue opening.
                tile_ids[lx, ly] = roof_tile_id_int
                stone_color = (
                    colors.CHIMNEY_STONE_LIGHT
                    if is_light
                    else colors.CHIMNEY_STONE_DARK
                )
                flue_color = (
                    colors.CHIMNEY_FLUE_LIGHT if is_light else colors.CHIMNEY_FLUE_DARK
                )
                chars[lx, ly] = ord("\u2022")
                bg_rgb[lx, ly] = stone_color
                fg_rgb[lx, ly] = flue_color

                # Projected chimney body: the tile directly south of the top
                # gets a split at chimney_projected_height showing the south-
                # facing stone surface above and the underlying roof below.
                proj_h = building.chimney_projected_height
                body_ly = ly + 1
                body_color = (
                    colors.CHIMNEY_BODY_LIGHT if is_light else colors.CHIMNEY_BODY_DARK
                )
                body_on_roof = (
                    body_ly < ext_h
                    and full_roof[lx, body_ly]
                    and not south_split_mask[lx, body_ly]
                )
                if body_on_roof:
                    # Save existing roof colors for the lower (roof) portion.
                    saved_bg = bg_rgb[lx, body_ly].copy()
                    saved_fg = fg_rgb[lx, body_ly].copy()

                    # Primary = chimney body (above split threshold).
                    # Use matching fg/bg so the glyph is invisible - the
                    # chimney face reads as a solid stone surface.
                    bg_rgb[lx, body_ly] = body_color
                    fg_rgb[lx, body_ly] = body_color
                    tile_ids[lx, body_ly] = roof_tile_id_int
                    draw_mask[lx, body_ly] = True

                    # Split portion = the original roof appearance.
                    split_y_arr[lx, body_ly] = proj_h
                    split_bg_arr[lx, body_ly] = (*saved_bg, 255)
                    split_fg_arr[lx, body_ly] = (*saved_fg, 255)

                # Chimney shadow is rendered as a GPU parallelogram quad in
                # _render_chimney_shadows, not baked into the stamp.  This
                # produces a smooth projected shadow identical in style to
                # tree/actor shadows.

        # --- Wear pack: encode per-tile data for the fragment shader ---
        # Pack material ID, building condition, edge proximity, and a
        # per-building hash into a uint32 per roof tile.  The fragment
        # shader uses this to apply per-pixel FBM-noise wear effects
        # (stains, moss, rust, etc.) that flow across tile boundaries.
        # Bits 0-7: material, 8-15: condition, 16-23: edge, 24-31: hash.
        wear_pack_arr = np.zeros((width, ext_h), dtype=np.uint32)
        material_id = _WEAR_MATERIAL_MAP.get(building.roof_style, 0)
        if material_id > 0 and building.condition > 0.001 and np.any(all_roof):
            cond_byte = min(255, int(building.condition * 255.0))
            # Per-building hash for independent sub-effect variation.
            # Derived from footprint position so it's deterministic.
            bld_hash = (
                (int(fp.x1) * 73856093) ^ (int(fp.y1) * 19349663) ^ 0x5A3E7F1D
            ) & 0xFF

            # Edge proximity: 1.0 at roof perimeter, fading to 0.0 inward.
            roof_wx = world_x_grid[all_roof]
            roof_wy = world_y_grid[all_roof]
            dist_w = (roof_wx - fp.x1).astype(np.float32)
            dist_e = (fp.x2 - 1 - roof_wx).astype(np.float32)
            dist_n = (roof_wy - visual_roof_y_min).astype(np.float32)
            dist_s = (visual_roof_y_max - roof_wy).astype(np.float32)
            edge_dist = np.minimum(
                np.minimum(dist_w, dist_e), np.minimum(dist_n, dist_s)
            )
            edge_byte = np.minimum(
                255,
                (np.clip(1.0 - edge_dist / _WEAR_EDGE_FADE_TILES, 0.0, 1.0) * 255.0),
            ).astype(np.uint32)
            wear_pack_arr[all_roof] = (
                np.uint32(material_id)
                | (np.uint32(cond_byte) << np.uint32(8))
                | (edge_byte << np.uint32(16))
                | (np.uint32(bld_hash) << np.uint32(24))
            )

        return _RoofStamp(
            chars,
            fg_rgb,
            bg_rgb,
            tile_ids,
            draw_mask,
            N,
            noise_pattern_arr,
            noise_pattern_mask_arr,
            split_y_arr,
            split_bg_arr,
            split_fg_arr,
            split_noise_arr,
            split_noise_pattern_arr,
            wear_pack_arr,
        )

    def _render_chimney_shadows(
        self,
        graphics: GraphicsContext,
        directional_light: DirectionalLight | None,
    ) -> None:
        """Emit GPU parallelogram shadow quads for chimneys on visible roofs.

        Uses the same draw_actor_shadow pipeline as tree/actor shadows,
        producing smooth connected projected shapes instead of grid-aligned
        tile darkening.  Called inside the shadow_pass() context so the
        geometry is batched with other shadow quads.
        """
        if directional_light is None or not config.SHADOWS_ENABLED:
            return

        # Shadow direction and length scale - same math as ShadowRenderer.
        raw_dx = -directional_light.direction.x
        raw_dy = -directional_light.direction.y
        dir_len = math.hypot(raw_dx, raw_dy)
        if dir_len <= 1e-6:
            return
        shadow_dir_x = raw_dx / dir_len
        shadow_dir_y = raw_dy / dir_len

        elevation = max(0.0, min(90.0, directional_light.elevation_degrees))
        if elevation >= 90.0:
            return
        tan_elev = math.tan(math.radians(elevation))
        length_scale = 8.0 if tan_elev <= 1e-6 else min(1.0 / tan_elev, 8.0)

        player_building_id, viewport_buildings = self._compute_roof_state()
        if not viewport_buildings:
            return

        viewport_scale_x, viewport_scale_y = (
            self.viewport_system.get_display_scale_factors()
        )

        # Zoom-corrected position offset matching the actor renderer.
        # When viewport_scale != 1 the scaled glyph needs an origin shift
        # of (scale - 1) * 0.5 console-tiles to stay centred on the tile.
        zoom_dx = (viewport_scale_x - 1.0) * 0.5
        zoom_dy = (viewport_scale_y - 1.0) * 0.5

        # Screen-pixel height of one viewport tile (zoom-aware).
        tile_h_raw = float(graphics.tile_dimensions[1])

        cam_frac_x, cam_frac_y = self.camera_frac_offset
        view_ox = float(self.x)
        view_oy = float(self.y)

        for building in viewport_buildings:
            if building.id == player_building_id:
                continue
            chimney_pos = building.chimney_world_pos
            if chimney_pos is None:
                continue

            cx, cy = chimney_pos
            # Chimney visual position is shifted north by the perspective
            # offset (same as the roof it sits on).
            N = building.perspective_ceil_offset
            shifted_cy = cy - N

            # The shadow must originate from the chimney BASE (where the
            # chimney stone meets the roof surface).  The body tile is one
            # tile south of the cap, and the chimney base sits at fraction
            # chimney_projected_height from its top (the split_y boundary).
            #
            # We derive the shadow origin from the body tile's INTEGER world
            # position (not a fractional one) because the zoom correction
            # (viewport_scale - 1) * 0.5 is calibrated for integer tile
            # positions.  A fractional world offset (proj_h) would be mis-
            # scaled at non-1x zoom.
            proj_h = building.chimney_projected_height
            body_cy = shifted_cy + 1

            vp_x, vp_y = self.viewport_system.world_to_screen_float(
                float(cx), float(body_cy)
            )
            root_x = view_ox + vp_x - cam_frac_x + zoom_dx
            root_y = view_oy + vp_y - cam_frac_y + zoom_dy
            body_sx, body_sy = graphics.console_to_screen_coords(root_x, root_y)

            # draw_actor_shadow places the parallelogram base edge at:
            #   base_edge_y = screen_y + tile_h * (1 + scale_y) / 2
            #
            # The chimney base pixel within the body tile is at:
            #   chimney_base_y = body_sy + tile_h*(1-vs)/2 + proj_h*tile_h*vs
            #
            # Solving for screen_y so base_edge_y = chimney_base_y gives:
            #   screen_y = body_sy - tile_h * vs * (1 - proj_h)
            shadow_screen_y = float(body_sy) - tile_h_raw * viewport_scale_y * (
                1.0 - proj_h
            )

            shadow_length_tiles = building.chimney_shadow_height * length_scale
            shadow_length_px = shadow_length_tiles * tile_h_raw * viewport_scale_y

            graphics.draw_actor_shadow(
                char="\u2588",  # Full block - shadow matches chimney tile width.
                screen_x=float(body_sx),
                screen_y=shadow_screen_y,
                shadow_dir_x=shadow_dir_x,
                shadow_dir_y=shadow_dir_y,
                shadow_length_pixels=shadow_length_px,
                shadow_alpha=float(config.TERRAIN_GLYPH_SHADOW_ALPHA),
                scale_x=viewport_scale_x,
                scale_y=viewport_scale_y,
                fade_tip=True,
            )

    def _render_chimney_overlays(
        self,
        graphics: GraphicsContext,
    ) -> None:
        """Redraw chimney tiles in the post-shadow actor layer.

        Chimney tiles live in the glyph buffer (tile layer) which renders
        before shadows.  The shadow parallelogram therefore darkens the
        chimney itself.  This method redraws the cap and body tiles on top
        of the shadow using draw_actor (which emits post-shadow quads),
        matching how actor sprites cover their shadow overlap at their base.
        """
        player_building_id, viewport_buildings = self._compute_roof_state()
        if not viewport_buildings:
            return

        viewport_scale_x, viewport_scale_y = (
            self.viewport_system.get_display_scale_factors()
        )

        # Zoom correction matching the actor renderer (see render_actors).
        zoom_dx = (viewport_scale_x - 1.0) * 0.5
        zoom_dy = (viewport_scale_y - 1.0) * 0.5

        # Shadows only appear in sunlight, so use the lit (daylight) chimney
        # colours.  With world_pos set, GPU actor lighting applies the
        # lightmap, so the overlay receives environment-appropriate shading.
        cap_stone = colors.CHIMNEY_STONE_LIGHT
        flue_color = colors.CHIMNEY_FLUE_LIGHT
        body_stone = colors.CHIMNEY_BODY_LIGHT

        tile_h_raw = float(graphics.tile_dimensions[1])

        cam_frac_x, cam_frac_y = self.camera_frac_offset
        view_ox = float(self.x)
        view_oy = float(self.y)

        for building in viewport_buildings:
            if building.id == player_building_id:
                continue
            chimney_pos = building.chimney_world_pos
            if chimney_pos is None:
                continue

            cx, cy = chimney_pos
            N = building.perspective_ceil_offset
            shifted_cy = cy - N

            # --- Cap tile overlay (top of chimney) ---
            vp_x, vp_y = self.viewport_system.world_to_screen_float(
                float(cx), float(shifted_cy)
            )
            root_x = view_ox + vp_x - cam_frac_x + zoom_dx
            root_y = view_oy + vp_y - cam_frac_y + zoom_dy
            screen_x, screen_y = graphics.console_to_screen_coords(root_x, root_y)

            cap_wpos: WorldTilePos = (cx, shifted_cy)

            # Opaque stone background covering the shadow at the cap.
            graphics.draw_actor(
                char="\u2588",
                color=cap_stone,
                screen_x=float(screen_x),
                screen_y=float(screen_y),
                scale_x=viewport_scale_x,
                scale_y=viewport_scale_y,
                world_pos=cap_wpos,
                tile_bg=cap_stone,
            )
            # Flue opening glyph on top.
            graphics.draw_actor(
                char="\u2022",
                color=flue_color,
                screen_x=float(screen_x),
                screen_y=float(screen_y),
                scale_x=viewport_scale_x,
                scale_y=viewport_scale_y,
                world_pos=cap_wpos,
            )

            # --- Body tile overlay (south-facing chimney wall) ---
            # Only the upper chimney_projected_height fraction of the body
            # tile is chimney stone; the rest is roof.  Overlay just the
            # stone portion using a reduced scale_y and an adjusted screen_y
            # that counters draw_actor's centering so the shortened glyph
            # sits flush with the top of the body tile.
            proj_h = building.chimney_projected_height
            if proj_h > 0.01:
                body_cy = shifted_cy + 1
                vp_bx, vp_by = self.viewport_system.world_to_screen_float(
                    float(cx), float(body_cy)
                )
                body_root_x = view_ox + vp_bx - cam_frac_x + zoom_dx
                body_root_y = view_oy + vp_by - cam_frac_y + zoom_dy
                body_sx, body_sy = graphics.console_to_screen_coords(
                    body_root_x, body_root_y
                )

                body_wpos: WorldTilePos = (cx, body_cy)

                # draw_actor centres the scaled glyph within the raw tile
                # height.  To top-align the shortened body overlay with the
                # full-tile overlay's top edge we need to compensate for the
                # difference in centering between scale_y=vsy (full tile) and
                # scale_y=proj_h*vsy (body portion).  The correct offset that
                # works at all zoom levels is:
                #   tile_h * vsy * (1 - proj_h) / 2
                body_scale_y = proj_h * viewport_scale_y
                centering_offset = tile_h_raw * viewport_scale_y * (1.0 - proj_h) / 2.0

                graphics.draw_actor(
                    char="\u2588",
                    color=body_stone,
                    screen_x=float(body_sx),
                    screen_y=float(body_sy) - centering_offset,
                    scale_x=viewport_scale_x,
                    scale_y=body_scale_y,
                    world_pos=body_wpos,
                    tile_bg=body_stone,
                )

    def _get_cached_roof_stamp(
        self,
        building: Building,
        *,
        is_light: bool,
        decoration_seed: int,
        sun_dx: float,
        sun_dy: float,
        sun_direction_key: tuple[float, float] | None,
    ) -> _RoofStamp:
        """Return a cached roof stamp, rebuilding when inputs changed."""
        cache_key = (
            *self._building_identity_key(building),
            bool(is_light),
            int(decoration_seed),
            sun_direction_key,
        )

        cache_slot = (int(building.id), bool(is_light))
        cached = self._roof_stamp_cache.get(cache_slot)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        stamp = self._build_roof_stamp(
            building,
            is_light=is_light,
            decoration_seed=decoration_seed,
            sun_dx=sun_dx,
            sun_dy=sun_dy,
        )
        self._roof_stamp_cache[cache_slot] = (cache_key, stamp)
        return stamp

    def _apply_roof_substitution(
        self,
        chars: np.ndarray,
        fg_rgb: np.ndarray,
        bg_rgb: np.ndarray,
        tile_ids: np.ndarray,
        world_x: np.ndarray,
        world_y: np.ndarray,
        *,
        is_light: bool,
        decoration_seed: int,
        buf_x: np.ndarray | None = None,
        buf_y: np.ndarray | None = None,
        buf_width: int | None = None,
        buf_height: int | None = None,
        world_origin_x: int | None = None,
        world_origin_y: int | None = None,
    ) -> _RoofSubstitutionResult:
        """Overlay virtual roof and wall face visuals for pseudo-3D perspective.

        Roofs are shifted north by each building's perspective_north_offset,
        exposing a south-facing wall strip. Boundary tiles use split_y data
        so the fragment shader can render a sub-tile roof/wall boundary.

        Returns effective tile IDs and optional per-tile split data arrays.
        """
        n = len(tile_ids)
        no_change = _RoofSubstitutionResult(tile_ids, None, None)
        if n == 0:
            return no_change

        player_building_id, viewport_buildings = self._compute_roof_state()
        if not viewport_buildings:
            return no_change

        # Get sun direction for ridge shading across all buildings.
        directional_light = self._get_directional_light()
        if directional_light is not None:
            sun_dx = directional_light.direction.x
            sun_dy = directional_light.direction.y
            sun_direction_key = (
                round(float(sun_dx), 3),
                round(float(sun_dy), 3),
            )
        else:
            sun_dx, sun_dy = -0.7, -0.7
            sun_direction_key = None

        direct_indexing_available = (
            buf_x is not None
            and buf_y is not None
            and buf_width is not None
            and buf_height is not None
            and world_origin_x is not None
            and world_origin_y is not None
            and len(buf_x) == len(tile_ids)
            and len(buf_y) == len(tile_ids)
        )
        buffer_index_lookup: np.ndarray | None = None
        if direct_indexing_available:
            assert buf_x is not None
            assert buf_y is not None
            assert buf_width is not None
            assert buf_height is not None
            buffer_index_lookup = np.full(
                (int(buf_width), int(buf_height)),
                -1,
                dtype=np.int32,
            )
            buffer_index_lookup[buf_x, buf_y] = np.arange(len(tile_ids), dtype=np.int32)

        # Evict cached stamps for buildings no longer in the viewport so the
        # cache doesn't grow unboundedly over a long session.
        viewport_building_ids = {b.id for b in viewport_buildings}
        stale_keys = [
            k for k in self._roof_stamp_cache if k[0] not in viewport_building_ids
        ]
        for k in stale_keys:
            del self._roof_stamp_cache[k]

        effective_tile_ids: np.ndarray | None = None
        any_roof_applied = False
        # Split arrays allocated on first building with fractional offset.
        split_y_arr: np.ndarray | None = None
        split_bg_arr: np.ndarray | None = None
        split_fg_arr: np.ndarray | None = None
        split_noise_arr: np.ndarray | None = None
        split_noise_pattern_arr: np.ndarray | None = None
        noise_pattern_arr: np.ndarray | None = None
        noise_pattern_mask_arr: np.ndarray | None = None
        wear_pack_arr: np.ndarray | None = None

        for building in viewport_buildings:
            if building.id == player_building_id:
                continue

            stamp = self._get_cached_roof_stamp(
                building,
                is_light=is_light,
                decoration_seed=decoration_seed,
                sun_dx=float(sun_dx),
                sun_dy=float(sun_dy),
                sun_direction_key=sun_direction_key,
            )

            fp = building.footprint
            stamp_world_y_start = int(fp.y1) - stamp.north_overhang

            # --- Locate stamp cells in the render arrays ---
            if direct_indexing_available and buffer_index_lookup is not None:
                assert buf_width is not None
                assert buf_height is not None
                assert world_origin_x is not None
                assert world_origin_y is not None

                # Map extended stamp world coords to buffer coords.
                stamp_buf_x1 = int(fp.x1) - int(world_origin_x)
                stamp_buf_y1 = stamp_world_y_start - int(world_origin_y)
                stamp_buf_x2 = int(fp.x2) - int(world_origin_x)
                stamp_buf_y2 = int(fp.y2) - int(world_origin_y)

                # Clip to buffer bounds.
                clip_x1 = max(0, stamp_buf_x1)
                clip_y1 = max(0, stamp_buf_y1)
                clip_x2 = min(int(buf_width), stamp_buf_x2)
                clip_y2 = min(int(buf_height), stamp_buf_y2)
                if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
                    continue

                # Stamp-local coords for the clipped region.
                local_x1 = clip_x1 - stamp_buf_x1
                local_y1 = clip_y1 - stamp_buf_y1
                local_x2 = local_x1 + (clip_x2 - clip_x1)
                local_y2 = local_y1 + (clip_y2 - clip_y1)

                stamp_mask_view = stamp.draw_mask[local_x1:local_x2, local_y1:local_y2]
                if not np.any(stamp_mask_view):
                    continue

                lookup_view = buffer_index_lookup[clip_x1:clip_x2, clip_y1:clip_y2]
                visible_stamp_mask = stamp_mask_view & (lookup_view >= 0)
                if not np.any(visible_stamp_mask):
                    continue

                hit_local_x, hit_local_y = np.nonzero(visible_stamp_mask)
                target_indices = lookup_view[hit_local_x, hit_local_y]
                stamp_x = hit_local_x + local_x1
                stamp_y = hit_local_y + local_y1
            else:
                # Fallback: per-element mask matching against extended bounds.
                in_stamp = (
                    (world_x >= int(fp.x1))
                    & (world_x < int(fp.x2))
                    & (world_y >= stamp_world_y_start)
                    & (world_y < int(fp.y2))
                )
                if not np.any(in_stamp):
                    continue

                stamp_indices = np.nonzero(in_stamp)[0]
                local_x = world_x[stamp_indices] - int(fp.x1)
                local_y = world_y[stamp_indices] - stamp_world_y_start
                drawn = stamp.draw_mask[local_x, local_y]
                if not np.any(drawn):
                    continue

                target_indices = stamp_indices[drawn]
                stamp_x = local_x[drawn]
                stamp_y = local_y[drawn]

            if len(target_indices) == 0:
                continue

            any_roof_applied = True
            if effective_tile_ids is None:
                effective_tile_ids = tile_ids.copy()

            # Blit precomputed stamp visuals into the render arrays.
            effective_tile_ids[target_indices] = stamp.tile_ids[stamp_x, stamp_y]
            chars[target_indices] = stamp.chars[stamp_x, stamp_y]
            fg_rgb[target_indices] = stamp.fg_rgb[stamp_x, stamp_y]
            bg_rgb[target_indices] = stamp.bg_rgb[stamp_x, stamp_y]

            # Override sub-tile pattern IDs for roofs that require per-building
            # orientation (e.g., tin corrugation following roof slope direction).
            has_pattern_override = stamp.noise_pattern_mask[stamp_x, stamp_y]
            if np.any(has_pattern_override):
                if noise_pattern_arr is None:
                    noise_pattern_arr = np.zeros(n, dtype=np.uint8)
                    noise_pattern_mask_arr = np.zeros(n, dtype=np.bool_)
                assert noise_pattern_mask_arr is not None
                override_idx = target_indices[has_pattern_override]
                ox = stamp_x[has_pattern_override]
                oy = stamp_y[has_pattern_override]
                noise_pattern_arr[override_idx] = stamp.noise_pattern[ox, oy]
                noise_pattern_mask_arr[override_idx] = True

            # Copy split data from stamp for perspective boundary tiles.
            has_split = stamp.split_y[stamp_x, stamp_y] > 0
            if np.any(has_split):
                if split_y_arr is None:
                    split_y_arr = np.zeros(n, dtype=np.float32)
                    split_bg_arr = np.zeros((n, 4), dtype=np.uint8)
                    split_fg_arr = np.zeros((n, 4), dtype=np.uint8)
                    split_noise_arr = np.zeros(n, dtype=np.float32)
                    split_noise_pattern_arr = np.zeros(n, dtype=np.uint8)

                assert split_bg_arr is not None
                assert split_fg_arr is not None
                assert split_noise_arr is not None
                assert split_noise_pattern_arr is not None
                split_idx = target_indices[has_split]
                sx = stamp_x[has_split]
                sy = stamp_y[has_split]
                split_y_arr[split_idx] = stamp.split_y[sx, sy]
                split_bg_arr[split_idx] = stamp.split_bg[sx, sy]
                split_fg_arr[split_idx] = stamp.split_fg[sx, sy]
                split_noise_arr[split_idx] = stamp.split_noise[sx, sy]
                split_noise_pattern_arr[split_idx] = stamp.split_noise_pattern[sx, sy]

            # Copy wear_pack from stamp for shader-based weathering.
            has_weather = stamp.wear_pack[stamp_x, stamp_y] > 0
            if np.any(has_weather):
                if wear_pack_arr is None:
                    wear_pack_arr = np.zeros(n, dtype=np.uint32)
                w_idx = target_indices[has_weather]
                wx = stamp_x[has_weather]
                wy = stamp_y[has_weather]
                wear_pack_arr[w_idx] = stamp.wear_pack[wx, wy]

        if not any_roof_applied or effective_tile_ids is None:
            return no_change
        return _RoofSubstitutionResult(
            effective_tile_ids,
            noise_pattern_arr,
            noise_pattern_mask_arr,
            split_y_arr,
            split_bg_arr,
            split_fg_arr,
            split_noise_arr,
            split_noise_pattern_arr,
            wear_pack_arr,
        )

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

        cache_key = self._get_map_unlit_buffer_cache_key()
        if cache_key == self._map_unlit_buffer_cache_key:
            return

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
            self._map_unlit_buffer_cache_key = cache_key
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

        # Apply per-tile glyph and color decoration for terrain variety.
        # At low zoom (below LOD threshold), tiles are too small for glyph
        # variation and color jitter to be visible - skip the work.
        unlit_tile_ids = game_map.tiles[final_world_x, final_world_y]
        if self._viewport_zoom >= config.LOD_DETAIL_ZOOM_THRESHOLD:
            tile_types.apply_terrain_decoration(
                chars,
                fg_rgb,
                bg_rgb,
                unlit_tile_ids,
                final_world_x,
                final_world_y,
                game_map.decoration_seed,
            )

        roof_result = self._apply_roof_substitution(
            chars,
            fg_rgb,
            bg_rgb,
            unlit_tile_ids,
            final_world_x,
            final_world_y,
            is_light=False,
            decoration_seed=game_map.decoration_seed,
            buf_x=final_buf_x,
            buf_y=final_buf_y,
            buf_width=buf_width,
            buf_height=buf_height,
            world_origin_x=world_origin_x,
            world_origin_y=world_origin_y,
        )
        effective_unlit_tile_ids = roof_result.effective_tile_ids

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
            tile_types.get_sub_tile_jitter_map(effective_unlit_tile_ids)
        )
        self.map_glyph_buffer.data["noise_pattern"][final_buf_x, final_buf_y] = (
            _apply_noise_pattern_overrides(effective_unlit_tile_ids, roof_result)
        )
        # Edge transitions create organic feathering between terrain types.
        # At low zoom, tiles are too small for the blending to be visible.
        if self._viewport_zoom >= config.LOD_DETAIL_ZOOM_THRESHOLD:
            self._apply_tile_edge_transition_data(
                glyph_buffer=self.map_glyph_buffer,
                final_buf_x=final_buf_x,
                final_buf_y=final_buf_y,
                tile_ids=effective_unlit_tile_ids,
                decorated_bg_rgb=bg_rgb,
            )
        self._map_unlit_buffer_cache_key = cache_key

        # Write perspective offset split data for boundary tiles.
        if roof_result.split_y is not None:
            buf = self.map_glyph_buffer.data
            buf["split_y"][final_buf_x, final_buf_y] = roof_result.split_y
            buf["split_bg"][final_buf_x, final_buf_y] = roof_result.split_bg
            buf["split_fg"][final_buf_x, final_buf_y] = roof_result.split_fg
            buf["split_noise"][final_buf_x, final_buf_y] = roof_result.split_noise
            buf["split_noise_pattern"][final_buf_x, final_buf_y] = (
                roof_result.split_noise_pattern
            )

        # Write packed weathering data for per-pixel shader effects.
        if roof_result.wear_pack is not None:
            self.map_glyph_buffer.data["wear_pack"][final_buf_x, final_buf_y] = (
                roof_result.wear_pack
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

    def _get_sun_direction_cache_key(self) -> tuple[float, float] | None:
        """Return a hashable sun direction for cache invalidation.

        Rounded to 3 decimal places to avoid cache thrashing from floating
        point noise while still invalidating when the direction meaningfully
        changes (e.g. time-of-day azimuth rotation).
        """
        try:
            directional_light = self._get_directional_light()
        except (AttributeError, TypeError):
            return None
        if directional_light is None:
            return None
        return (
            round(directional_light.direction.x, 3),
            round(directional_light.direction.y, 3),
        )

    def _apply_sun_direction_to_graphics(
        self,
        graphics: GraphicsContext,
        directional_light: DirectionalLight | None,
    ) -> None:
        """Push per-frame sun direction to every active graphics context reference.

        WorldView owns renderers initialized with ``self.graphics`` and also receives
        a ``graphics`` argument in ``present()``. These are expected to be the same
        object, but updating both (when distinct) keeps sprite highlight uniforms
        consistent across wrapped/proxy contexts.
        """
        if directional_light is None:
            sun_dx, sun_dy = 0.0, 0.0
        else:
            sun_dx = directional_light.direction.x
            sun_dy = directional_light.direction.y

        target_graphics: list[GraphicsContext] = [self.graphics]
        if graphics is not self.graphics:
            target_graphics.append(graphics)

        for target in target_graphics:
            set_sun_direction = getattr(target, "set_sun_direction", None)
            if callable(set_sun_direction):
                set_sun_direction(sun_dx, sun_dy)

    def _apply_tile_edge_transition_data(
        self,
        glyph_buffer: GlyphBuffer,
        final_buf_x: np.ndarray,
        final_buf_y: np.ndarray,
        tile_ids: np.ndarray,
        decorated_bg_rgb: np.ndarray,
    ) -> None:
        """Populate per-tile organic edge transition metadata for the glyph shader."""
        if len(tile_ids) == 0:
            return

        edge_blend = tile_types.get_edge_blend_map(tile_ids)
        glyph_buffer.data["edge_blend"][final_buf_x, final_buf_y] = edge_blend
        if not np.any(edge_blend > 0.0):
            return

        tile_id_window = np.zeros(
            (glyph_buffer.width, glyph_buffer.height), dtype=np.int32
        )
        drawn_mask_window = np.zeros(
            (glyph_buffer.width, glyph_buffer.height), dtype=np.bool_
        )
        bg_rgb_window = np.zeros(
            (glyph_buffer.width, glyph_buffer.height, 3), dtype=np.uint8
        )
        edge_blend_window = np.zeros(
            (glyph_buffer.width, glyph_buffer.height), dtype=np.float32
        )

        tile_id_window[final_buf_x, final_buf_y] = tile_ids.astype(np.int32, copy=False)
        edge_blend_window[final_buf_x, final_buf_y] = edge_blend
        drawn_mask_window[final_buf_x, final_buf_y] = True
        bg_rgb_window[final_buf_x, final_buf_y] = decorated_bg_rgb

        edge_neighbor_mask, edge_neighbor_bg = _compute_tile_edge_transition_metadata(
            tile_id_window=tile_id_window,
            edge_blend_window=edge_blend_window,
            drawn_mask_window=drawn_mask_window,
            bg_rgb_window=bg_rgb_window,
        )
        _suppress_edge_blend_toward_hard_edges(
            edge_neighbor_mask=edge_neighbor_mask,
            tile_id_window=tile_id_window,
        )
        _override_edge_neighbor_bg_with_self_darken(
            edge_neighbor_bg=edge_neighbor_bg,
            tile_id_window=tile_id_window,
            bg_rgb_window=bg_rgb_window,
        )
        glyph_buffer.data["edge_neighbor_mask"][final_buf_x, final_buf_y] = (
            edge_neighbor_mask[final_buf_x, final_buf_y]
        )
        glyph_buffer.data["edge_neighbor_bg"][final_buf_x, final_buf_y] = (
            edge_neighbor_bg[final_buf_x, final_buf_y]
        )

    def _update_actor_particles(self) -> None:
        """Emit particles from actors with particle emitters."""
        gw = self.controller.gw

        # Some test doubles do not implement `actors_revision`; rebuild each call
        # in that case and keep the optimized revision path for real GameWorld.
        actors_revision = getattr(gw, "actors_revision", None)
        if (
            actors_revision is None
            or self._particle_emitter_actors_revision != actors_revision
        ):
            self._particle_emitter_actors = {
                actor
                for actor in gw.actors
                if (
                    actor.visual_effects is not None
                    and actor.visual_effects.has_continuous_effects()
                )
            }
            if actors_revision is not None:
                self._particle_emitter_actors_revision = actors_revision

        with record_time_live_variable("time.render.actor_particles_ms"):
            for actor in self._particle_emitter_actors:
                visual_effects = actor.visual_effects
                if (
                    visual_effects is not None
                    and visual_effects.has_continuous_effects()
                ):
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

        # Vectorized: slice viewport region and find visible animated tiles
        # using numpy boolean operations instead of a Python loop.
        vp_slice = (
            slice(world_left, world_right + 1),
            slice(world_top, world_bottom + 1),
        )
        visible_slice = game_map.visible[vp_slice]
        animates_slice = game_map.animation_params["animates"][vp_slice]
        animated_mask = visible_slice & animates_slice

        # Get local (within-slice) coordinates of animated tiles
        local_x, local_y = np.nonzero(animated_mask)
        num_animated = len(local_x)
        if num_animated == 0:
            return

        # Select a random subset of tiles to update (percent_of_cells %)
        num_to_update = max(1, num_animated * percent_of_cells // 100)
        if num_to_update < num_animated:
            chosen = self._tile_anim_rng.choice(
                num_animated, size=num_to_update, replace=False
            )
            local_x = local_x[chosen]
            local_y = local_y[chosen]
            num_to_update = len(local_x)

        # Convert to world coordinates for the animation_state write
        world_x = local_x + world_left
        world_y = local_y + world_top

        # Vectorized random walk: generate all offsets in one batch
        # (num_to_update, 3) for fg and bg independently
        step_size = 80
        fg_offsets = self._tile_anim_rng.integers(
            -step_size, step_size + 1, size=(num_to_update, 3), dtype=np.int16
        )
        bg_offsets = self._tile_anim_rng.integers(
            -step_size, step_size + 1, size=(num_to_update, 3), dtype=np.int16
        )

        # Read current state, apply offsets, clamp, write back
        state = game_map.animation_state
        fg_vals = state["fg_values"][world_x, world_y].astype(np.int16)
        bg_vals = state["bg_values"][world_x, world_y].astype(np.int16)
        fg_vals += fg_offsets
        bg_vals += bg_offsets
        np.clip(fg_vals, 0, 1000, out=fg_vals)
        np.clip(bg_vals, 0, 1000, out=bg_vals)
        state["fg_values"][world_x, world_y] = fg_vals
        state["bg_values"][world_x, world_y] = bg_vals

    def _update_mouse_tile_location(self) -> None:
        """Update the stored world-space mouse tile based on the current camera."""
        fm: FrameManager = self.controller.frame_manager

        # Use pixel_to_world_tile which compensates for the camera's fractional
        # scroll offset, so the detected tile matches the visually rendered grid.
        px_x: PixelCoord = fm.cursor_manager.mouse_pixel_x
        px_y: PixelCoord = fm.cursor_manager.mouse_pixel_y
        self.controller.gw.mouse_tile_location_on_map = fm.pixel_to_world_tile(
            px_x, px_y
        )

    def _apply_tile_light_animations(
        self,
        light_fg_rgb: np.ndarray,
        light_bg_rgb: np.ndarray,
        animation_params: np.ndarray,
        animation_state: np.ndarray,
        valid_exp_x: np.ndarray,
        valid_exp_y: np.ndarray,
        exclude_mask: np.ndarray | None = None,
    ) -> None:
        """Apply per-tile color modulation for animated terrain tiles.

        ``exclude_mask`` can be used to suppress animation on cells that were
        visually substituted this frame (e.g., opaque roof overlay tiles).
        """
        animates_mask = animation_params["animates"][valid_exp_x, valid_exp_y]
        if exclude_mask is not None:
            animates_mask = animates_mask & ~exclude_mask
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

    def _can_reuse_light_source_buffer_cache(
        self,
        cache_key: tuple[object, ...],
        explored_mask_slice: np.ndarray,
        exploration_revision: int,
    ) -> bool:
        """Return whether cached light-source tile data is still valid.

        Global exploration revision changes often, but the expensive rebuild is only
        needed when the explored state inside the current viewport changes.
        """
        if cache_key != self._light_cache.cache_key:
            return False
        if self._light_cache.exploration_revision == exploration_revision:
            return True
        cached_explored_mask = self._light_cache.explored_mask
        if cached_explored_mask is None:
            return False
        if cached_explored_mask.shape != explored_mask_slice.shape:
            return False
        if not np.array_equal(cached_explored_mask, explored_mask_slice):
            return False

        # Exploration changed elsewhere on the map, but not inside this viewport.
        self._light_cache.exploration_revision = exploration_revision
        return True

    def _populate_light_overlay_visible_mask_from_cache(
        self,
        visible_mask_buffer: np.ndarray,
        visible_mask_slice: np.ndarray,
    ) -> None:
        """Update the compose-visible mask from cached tile coordinates.

        Tile appearance data stays cached, but visibility/FOV can change every frame
        as the player moves. Recomputing only this mask avoids a full tile rebuild.
        """
        if (
            self._light_cache.roof_opaque_buf_x is not None
            and self._light_cache.roof_opaque_buf_y is not None
        ):
            visible_mask_buffer[
                self._light_cache.roof_opaque_buf_x,
                self._light_cache.roof_opaque_buf_y,
            ] = _LIGHT_OVERLAY_MASK_ROOF_SUNLIT

        cached_buf_x = self._light_cache.buf_x
        cached_buf_y = self._light_cache.buf_y
        cached_exp_x = self._light_cache.exp_x
        cached_exp_y = self._light_cache.exp_y
        if (
            cached_buf_x is None
            or cached_buf_y is None
            or cached_exp_x is None
            or cached_exp_y is None
        ):
            self._light_cache.vis_buf_x = None
            self._light_cache.vis_buf_y = None
            return

        visible_now = visible_mask_slice[cached_exp_x, cached_exp_y]
        roof_covered_mask = self._light_cache.roof_covered_mask
        if roof_covered_mask is not None:
            visible_now = visible_now & ~roof_covered_mask

        if not np.any(visible_now):
            self._light_cache.vis_buf_x = None
            self._light_cache.vis_buf_y = None
            return

        vis_buf_x = cached_buf_x[visible_now]
        vis_buf_y = cached_buf_y[visible_now]
        visible_mask_buffer[vis_buf_x, vis_buf_y] = _LIGHT_OVERLAY_MASK_VISIBLE
        self._light_cache.vis_buf_x = vis_buf_x
        self._light_cache.vis_buf_y = vis_buf_y

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
        # Include player_building_id so entering/exiting a building invalidates
        # the cached roof overlay. At low zoom the viewport can cover the entire
        # map, making the viewport bounds constant and insufficient for invalidation.
        player_building_id, _viewport_buildings = self._compute_roof_state()
        cache_key = (
            world_left,
            world_top,
            world_right,
            world_bottom,
            getattr(gw.game_map, "structural_revision", 0),
            self._get_sun_direction_cache_key(),
            self._viewport_zoom >= config.LOD_DETAIL_ZOOM_THRESHOLD,
            player_building_id,
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
        light_source_buffer = self.light_source_glyph_buffer
        buf_width = light_source_buffer.width
        buf_height = light_source_buffer.height
        # Buffer-space visible mask used by the GPU compose shader. Keeping this
        # in the same coordinate space as the glyph buffers avoids map/viewport
        # axis mismatches when maps are centered in larger viewports.
        if self._visible_mask_buffer is None or self._visible_mask_buffer.shape != (
            buf_width,
            buf_height,
        ):
            self._visible_mask_buffer = np.zeros(
                (buf_width, buf_height), dtype=np.uint8
            )
        else:
            self._visible_mask_buffer.fill(_LIGHT_OVERLAY_MASK_HIDDEN)
        visible_mask_buffer = self._visible_mask_buffer

        world_slice = (
            slice(world_left, world_right + 1),
            slice(world_top, world_bottom + 1),
        )
        exploration_revision = int(gw.game_map.exploration_revision)
        explored_mask_slice = gw.game_map.explored[world_slice]
        visible_mask_slice = gw.game_map.visible[world_slice]
        if self._can_reuse_light_source_buffer_cache(
            cache_key,
            explored_mask_slice,
            exploration_revision,
        ):
            self._populate_light_overlay_visible_mask_from_cache(
                visible_mask_buffer,
                visible_mask_slice,
            )

            if self._light_cache.anim_buf_indices is not None:
                anim_base_fg = self._light_cache.anim_base_fg
                anim_base_bg = self._light_cache.anim_base_bg
                anim_exp_x = self._light_cache.anim_exp_x
                anim_exp_y = self._light_cache.anim_exp_y
                cached_buf_x = self._light_cache.buf_x
                cached_buf_y = self._light_cache.buf_y
                assert anim_base_fg is not None
                assert anim_base_bg is not None
                assert anim_exp_x is not None
                assert anim_exp_y is not None
                assert cached_buf_x is not None
                assert cached_buf_y is not None

                anim_fg_rgb = anim_base_fg.copy()
                anim_bg_rgb = anim_base_bg.copy()
                self._apply_tile_light_animations(
                    anim_fg_rgb,
                    anim_bg_rgb,
                    gw.game_map.animation_params[world_slice],
                    gw.game_map.animation_state[world_slice],
                    anim_exp_x,
                    anim_exp_y,
                )

                alpha_channel = np.full((len(anim_fg_rgb), 1), 255, dtype=np.uint8)
                anim_buf_x = cached_buf_x[self._light_cache.anim_buf_indices]
                anim_buf_y = cached_buf_y[self._light_cache.anim_buf_indices]
                light_source_buffer.data["fg"][anim_buf_x, anim_buf_y] = np.hstack(
                    (anim_fg_rgb, alpha_channel)
                )
                light_source_buffer.data["bg"][anim_buf_x, anim_buf_y] = np.hstack(
                    (anim_bg_rgb, alpha_channel)
                )
        else:
            light_source_buffer.clear()
            self._light_cache = _LightBufferCache(
                exploration_revision=exploration_revision,
                explored_mask=explored_mask_slice.copy(),
            )

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
                    # At low zoom, tiles are too small for decoration to be visible.
                    world_tile_ids = gw.game_map.tiles[world_slice][
                        valid_exp_x, valid_exp_y
                    ]
                    if self._viewport_zoom >= config.LOD_DETAIL_ZOOM_THRESHOLD:
                        tile_types.apply_terrain_decoration(
                            light_chars,
                            light_fg_rgb,
                            light_bg_rgb,
                            world_tile_ids,
                            valid_exp_x + world_left,
                            valid_exp_y + world_top,
                            gw.game_map.decoration_seed,
                        )
                    lit_roof_result = self._apply_roof_substitution(
                        light_chars,
                        light_fg_rgb,
                        light_bg_rgb,
                        world_tile_ids,
                        valid_exp_x + world_left,
                        valid_exp_y + world_top,
                        is_light=True,
                        decoration_seed=gw.game_map.decoration_seed,
                        buf_x=final_buf_x,
                        buf_y=final_buf_y,
                        buf_width=buf_width,
                        buf_height=buf_height,
                        world_origin_x=world_left - vs.offset_x - pad,
                        world_origin_y=world_top - vs.offset_y - pad,
                    )
                    effective_world_tile_ids = lit_roof_result.effective_tile_ids
                    roof_covered_mask = np.isin(
                        effective_world_tile_ids, _ROOF_TILE_IDS
                    )
                    if np.any(roof_covered_mask):
                        # Roofs must be fully opaque from outside: hide roof-covered
                        # cells from the GPU light/FOV compose mask so interior light
                        # patterns do not project through the roof surface.
                        valid_visible = valid_visible & ~roof_covered_mask

                    animation_params = gw.game_map.animation_params[world_slice]
                    animation_state = gw.game_map.animation_state[world_slice]
                    animates_mask = animation_params["animates"][
                        valid_exp_x, valid_exp_y
                    ]
                    if np.any(roof_covered_mask):
                        animates_mask = animates_mask & ~roof_covered_mask
                    if np.any(animates_mask):
                        anim_indices = np.nonzero(animates_mask)[0]
                        self._light_cache.anim_base_fg = light_fg_rgb[
                            anim_indices
                        ].copy()
                        self._light_cache.anim_base_bg = light_bg_rgb[
                            anim_indices
                        ].copy()
                        self._light_cache.anim_buf_indices = anim_indices
                        self._light_cache.anim_exp_x = valid_exp_x[animates_mask]
                        self._light_cache.anim_exp_y = valid_exp_y[animates_mask]

                    self._apply_tile_light_animations(
                        light_fg_rgb,
                        light_bg_rgb,
                        animation_params,
                        animation_state,
                        valid_exp_x,
                        valid_exp_y,
                        exclude_mask=roof_covered_mask,
                    )

                    alpha_channel = np.full((len(light_fg_rgb), 1), 255, dtype=np.uint8)
                    light_fg_rgba = np.hstack((light_fg_rgb, alpha_channel))
                    light_bg_rgba = np.hstack((light_bg_rgb, alpha_channel))
                    light_source_buffer.data["ch"][final_buf_x, final_buf_y] = (
                        light_chars
                    )
                    light_source_buffer.data["fg"][final_buf_x, final_buf_y] = (
                        light_fg_rgba
                    )
                    light_source_buffer.data["bg"][final_buf_x, final_buf_y] = (
                        light_bg_rgba
                    )

                    # Write sub-tile jitter amplitude for the light source buffer too,
                    # so lit tiles also get per-pixel brightness variation.
                    light_source_buffer.data["noise"][final_buf_x, final_buf_y] = (
                        tile_types.get_sub_tile_jitter_map(effective_world_tile_ids)
                    )
                    light_source_buffer.data["noise_pattern"][
                        final_buf_x, final_buf_y
                    ] = _apply_noise_pattern_overrides(
                        effective_world_tile_ids, lit_roof_result
                    )
                    if self._viewport_zoom >= config.LOD_DETAIL_ZOOM_THRESHOLD:
                        self._apply_tile_edge_transition_data(
                            glyph_buffer=light_source_buffer,
                            final_buf_x=final_buf_x,
                            final_buf_y=final_buf_y,
                            tile_ids=effective_world_tile_ids,
                            decorated_bg_rgb=light_bg_rgb,
                        )

                    # Write perspective offset split data for boundary tiles.
                    if lit_roof_result.split_y is not None:
                        lbuf = light_source_buffer.data
                        lbuf["split_y"][final_buf_x, final_buf_y] = (
                            lit_roof_result.split_y
                        )
                        lbuf["split_bg"][final_buf_x, final_buf_y] = (
                            lit_roof_result.split_bg
                        )
                        lbuf["split_fg"][final_buf_x, final_buf_y] = (
                            lit_roof_result.split_fg
                        )
                        lbuf["split_noise"][final_buf_x, final_buf_y] = (
                            lit_roof_result.split_noise
                        )
                        lbuf["split_noise_pattern"][final_buf_x, final_buf_y] = (
                            lit_roof_result.split_noise_pattern
                        )

                    # Write packed weathering data for per-pixel shader effects.
                    if lit_roof_result.wear_pack is not None:
                        light_source_buffer.data["wear_pack"][
                            final_buf_x, final_buf_y
                        ] = lit_roof_result.wear_pack

                    self._light_cache.buf_x = final_buf_x
                    self._light_cache.buf_y = final_buf_y
                    self._light_cache.exp_x = valid_exp_x
                    self._light_cache.exp_y = valid_exp_y
                    self._light_cache.roof_covered_mask = roof_covered_mask
                    if np.any(roof_covered_mask):
                        roof_buf_x = final_buf_x[roof_covered_mask]
                        roof_buf_y = final_buf_y[roof_covered_mask]
                        visible_mask_buffer[roof_buf_x, roof_buf_y] = (
                            _LIGHT_OVERLAY_MASK_ROOF_SUNLIT
                        )
                        self._light_cache.roof_opaque_buf_x = roof_buf_x
                        self._light_cache.roof_opaque_buf_y = roof_buf_y
                    if np.any(valid_visible):
                        vis_buf_x = final_buf_x[valid_visible]
                        vis_buf_y = final_buf_y[valid_visible]
                        visible_mask_buffer[vis_buf_x, vis_buf_y] = (
                            _LIGHT_OVERLAY_MASK_VISIBLE
                        )
                        self._light_cache.vis_buf_x = vis_buf_x
                        self._light_cache.vis_buf_y = vis_buf_y

            self._light_cache.cache_key = cache_key

        with record_time_live_variable("time.render.light_texture_upload_ms"):
            light_source_texture = graphics.render_glyph_buffer_to_texture(
                light_source_buffer,
                cache_key_suffix="light_source",
            )

        composed_texture = graphics.compose_light_overlay_gpu(
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
