from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from brileta import config
from brileta.environment.tile_types import TileTypeID, get_shadow_height_map
from brileta.types import InterpolationAlpha

from .graphics import GraphicsContext
from .viewport import ViewportSystem

if TYPE_CHECKING:
    from brileta.game.actors import Actor
    from brileta.game.lights import DirectionalLight


def compute_actor_screen_position(
    actor: Actor,
    graphics: GraphicsContext,
    viewport_system: ViewportSystem,
    interpolation_alpha: InterpolationAlpha,
    camera_frac_offset: tuple[float, float],
    view_origin: tuple[float, float],
) -> tuple[float, float, float, float, float, float]:
    """Compute interpolated world/root/pixel coordinates for an actor."""
    alpha_value = float(interpolation_alpha)
    if actor._animation_controlled:
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

    vp_x, vp_y = viewport_system.world_to_screen_float(interpolated_x, interpolated_y)
    cam_frac_x, cam_frac_y = camera_frac_offset
    vp_x -= cam_frac_x
    vp_y -= cam_frac_y

    root_x = view_origin[0] + vp_x
    root_y = view_origin[1] + vp_y
    screen_pixel_x, screen_pixel_y = graphics.console_to_screen_coords(root_x, root_y)

    return (
        interpolated_x,
        interpolated_y,
        root_x,
        root_y,
        screen_pixel_x,
        screen_pixel_y,
    )


class _SunShadowParams(NamedTuple):
    """Pre-computed directional shadow geometry shared by terrain and actor passes."""

    dir_x: float
    dir_y: float
    length_scale: float


class _ActorBatchData(NamedTuple):
    """Pre-extracted actor data shared by sun and point-light shadow passes.

    Eliminates duplicate per-actor attribute access and numpy array construction
    that was happening independently in both shadow passes. Intermediate arrays
    (interpolation sources, drift, animation flags) are computed inside
    _extract_actor_batch_data but kept as locals - only the final consumed
    values are stored here.
    """

    # Integer tile positions.
    actor_x: np.ndarray  # int32
    actor_y: np.ndarray  # int32
    shadow_heights: np.ndarray  # float64
    visual_scale: np.ndarray  # float64
    # Sprite data.
    sprite_anchor_y: np.ndarray  # float64
    sprite_uvs: np.ndarray  # float32 (N, 4)
    sprite_batchable: np.ndarray  # bool
    # Pre-computed screen positions (root console and pixel coords).
    root_x: np.ndarray  # float64
    root_y: np.ndarray  # float64
    screen_x: np.ndarray  # float64
    screen_y: np.ndarray  # float64


class ShadowRenderer:
    """Render projected terrain and actor shadows for a world view."""

    _MIN_RENDERED_ACTOR_SHADOW_LENGTH_PX: float = 2.5
    _SUN_SHADOW_OUTDOOR_REGION_TYPES: frozenset[str] = frozenset(
        {"outdoor", "exterior", "test_outdoor"}
    )
    _SUN_SHADOW_OUTDOOR_TILE_IDS: frozenset[int] = frozenset(
        {
            int(TileTypeID.COBBLESTONE),
            int(TileTypeID.GRASS),
            int(TileTypeID.DIRT),
            int(TileTypeID.GRAVEL),
        }
    )

    def __init__(
        self,
        game_map: Any,
        viewport_system: ViewportSystem,
        graphics: GraphicsContext,
    ) -> None:
        self.game_map = game_map
        self.viewport_system = viewport_system
        self.graphics = graphics
        # Current world-view zoom multiplier, set by the owning view each frame.
        # Used for LOD gating of expensive shadow detail at low zoom levels.
        self.viewport_zoom: float = 1.0
        self._view_origin: tuple[float, float] = (0.0, 0.0)
        self._camera_frac_offset: tuple[float, float] = (0.0, 0.0)
        self._frame_directional_light: DirectionalLight | None = None
        self._frame_lights: Sequence[Any] = ()
        # Per-frame multiplicative light scales for actors shadowed by other actors.
        self._actor_shadow_receive_light_scale: dict[Actor, float] = {}

    @property
    def actor_shadow_receive_light_scale(self) -> dict[Actor, float]:
        """Per-frame actor receive-dimming scale populated during shadow rendering."""
        return self._actor_shadow_receive_light_scale

    def set_view_transform(
        self,
        *,
        view_origin: tuple[float, float],
        camera_frac_offset: tuple[float, float],
    ) -> None:
        """Set frame-specific world-view transform used by shadow projection."""
        self._view_origin = view_origin
        self._camera_frac_offset = camera_frac_offset

    def set_frame_lighting(
        self,
        *,
        directional_light: DirectionalLight | None,
        lights: Sequence[Any],
    ) -> None:
        """Cache frame lighting inputs for helper methods and tests."""
        self._frame_directional_light = directional_light
        self._frame_lights = lights

    def _get_viewport_display_scale_factors(self) -> tuple[float, float]:
        """Return viewport display scale factors."""
        return self.viewport_system.get_display_scale_factors()

    def _zoomed_tile_draw_origin(
        self,
        root_x: float,
        root_y: float,
        *,
        ground_anchor_y: float | None = None,
    ) -> tuple[float, float]:
        """Shift root origin so scaled shadow draws remain tile-aligned."""
        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale_factors()
        corrected_x = root_x + (viewport_scale_x - 1.0) * 0.5
        if ground_anchor_y is None:
            corrected_y = root_y + (viewport_scale_y - 1.0) * 0.5
        else:
            corrected_y = root_y + (viewport_scale_y - 1.0) * float(ground_anchor_y)
        return (corrected_x, corrected_y)

    def render_actor_shadows(
        self,
        interpolation_alpha: InterpolationAlpha,
        *,
        visible_actors: list[Actor] | None,
        dynamic_receivers: list[Actor] | None = None,
        directional_light: DirectionalLight | None,
        lights: Sequence[Any],
        view_origin: tuple[float, float],
        camera_frac_offset: tuple[float, float],
    ) -> None:
        """Render projected glyph shadows for terrain objects and visible actors.

        Args:
            dynamic_receivers: Optional restricted list of actors that should
                receive shadow dimming (typically only dynamic actors like the
                player and NPCs). When provided, the O(N*N) receiver-dimming
                inner loop becomes O(N*D) where D is len(dynamic_receivers).
                Shadow quads are still drawn for ALL casters regardless.
        """
        self.set_view_transform(
            view_origin=view_origin,
            camera_frac_offset=camera_frac_offset,
        )
        self.set_frame_lighting(directional_light=directional_light, lights=lights)
        self._actor_shadow_receive_light_scale = {}
        if not config.SHADOWS_ENABLED:
            return

        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale_factors()
        tile_height = float(self.graphics.tile_dimensions[1]) * viewport_scale_y

        # Compute sun shadow params once for both terrain and actor passes.
        sun_params: _SunShadowParams | None = None
        if directional_light is not None:
            sun_params = self._compute_sun_shadow_params(directional_light)

        # Terrain glyph shadows (boulders, etc.) - independent of actor positions.
        self._render_terrain_glyph_shadows(
            tile_height,
            viewport_scale=(viewport_scale_x, viewport_scale_y),
            sun_params=sun_params,
        )

        if not visible_actors:
            return

        shadow_casters = [actor for actor in visible_actors if actor.shadow_height > 0]
        if not shadow_casters:
            return

        # Extract actor data into numpy arrays once and share between both
        # shadow passes. Previously each pass independently iterated actors
        # and built its own arrays, duplicating ~50 per-actor attribute reads.
        batch_data = self._extract_actor_batch_data(shadow_casters, interpolation_alpha)

        self._render_sun_actor_shadows(
            shadow_casters,
            interpolation_alpha,
            tile_height,
            receivers=visible_actors,
            dynamic_receivers=dynamic_receivers,
            sun_params=sun_params,
            directional_light=directional_light,
            batch_data=batch_data,
        )
        self._render_point_light_actor_shadows(
            shadow_casters,
            interpolation_alpha,
            tile_height,
            receivers=visible_actors,
            dynamic_receivers=dynamic_receivers,
            lights=lights,
            batch_data=batch_data,
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

    def _get_min_rendered_actor_shadow_length_tiles(self) -> float:
        """Return the minimum shadow length worth rendering at the current zoom."""
        _viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale_factors()
        tile_height_pixels = float(self.graphics.tile_dimensions[1]) * viewport_scale_y
        if tile_height_pixels <= 1e-6:
            return math.inf
        return self._MIN_RENDERED_ACTOR_SHADOW_LENGTH_PX / tile_height_pixels

    def _get_actor_screen_position(
        self,
        actor: Actor,
        interpolation_alpha: InterpolationAlpha,
    ) -> tuple[float, float, float, float, float, float]:
        return compute_actor_screen_position(
            actor=actor,
            graphics=self.graphics,
            viewport_system=self.viewport_system,
            interpolation_alpha=interpolation_alpha,
            camera_frac_offset=self._camera_frac_offset,
            view_origin=self._view_origin,
        )

    def _world_to_screen_float_batch(
        self,
        world_x: np.ndarray,
        world_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Vectorized ``ViewportSystem.world_to_screen_float`` for real viewports."""
        if not isinstance(self.viewport_system, ViewportSystem):
            return None

        bounds = self.viewport_system.viewport.get_world_bounds(
            self.viewport_system.camera
        )
        scale_x, scale_y = self.viewport_system.get_display_scale_factors()
        left = float(bounds.x1)
        top = float(bounds.y1)
        offset_x = float(self.viewport_system.viewport.offset_x)
        offset_y = float(self.viewport_system.viewport.offset_y)
        return (
            (world_x - left + offset_x) * scale_x,
            (world_y - top + offset_y) * scale_y,
        )

    def _console_to_screen_coords_batch(
        self,
        console_x: np.ndarray,
        console_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized ``GraphicsContext.console_to_screen_coords``."""
        # letterbox_geometry is not on the GraphicsContext ABC, so we use
        # getattr once.  The remaining attributes (tile_dimensions,
        # console_width_tiles, console_height_tiles) are guaranteed by the ABC.
        letterbox_geometry = getattr(self.graphics, "letterbox_geometry", None)
        if not isinstance(letterbox_geometry, tuple):
            tile_w, tile_h = self.graphics.tile_dimensions
            return (
                np.trunc(console_x * float(tile_w)),
                np.trunc(console_y * float(tile_h)),
            )

        offset_x, offset_y, scaled_w, scaled_h = letterbox_geometry
        console_w = self.graphics.console_width_tiles
        console_h = self.graphics.console_height_tiles
        return (
            float(offset_x) + console_x * (float(scaled_w) / float(console_w)),
            float(offset_y) + console_y * (float(scaled_h) / float(console_h)),
        )

    def _extract_actor_batch_data(
        self,
        actors: list[Actor],
        interpolation_alpha: InterpolationAlpha,
    ) -> _ActorBatchData:
        """Extract actor attributes into numpy arrays and pre-compute positions.

        Collects per-actor data (positions, drift, visual scale, sprite info)
        into contiguous arrays and computes interpolated world positions and
        screen coordinates in one vectorized pass. Called once by
        render_actor_shadows and consumed by both the sun and point-light
        shadow passes, eliminating the duplicate extraction.
        """
        n = len(actors)
        actor_x = np.empty(n, dtype=np.int32)
        actor_y = np.empty(n, dtype=np.int32)
        shadow_heights = np.empty(n, dtype=np.float64)
        prev_x = np.empty(n, dtype=np.float64)
        prev_y = np.empty(n, dtype=np.float64)
        curr_x = np.empty(n, dtype=np.float64)
        curr_y = np.empty(n, dtype=np.float64)
        render_x = np.empty(n, dtype=np.float64)
        render_y = np.empty(n, dtype=np.float64)
        animation_controlled = np.zeros(n, dtype=bool)
        drift_x = np.zeros(n, dtype=np.float64)
        drift_y = np.zeros(n, dtype=np.float64)
        visual_scale = np.ones(n, dtype=np.float64)
        sprite_anchor_y = np.ones(n, dtype=np.float64)
        sprite_uvs = np.zeros((n, 4), dtype=np.float32)
        sprite_batchable = np.zeros(n, dtype=bool)

        for idx, actor in enumerate(actors):
            actor_x[idx] = int(actor.x)
            actor_y[idx] = int(actor.y)
            shadow_heights[idx] = float(actor.shadow_height)
            prev_x[idx] = float(actor.prev_x)
            prev_y[idx] = float(actor.prev_y)
            curr_x[idx] = float(actor.x)
            curr_y[idx] = float(actor.y)
            render_x[idx] = float(actor.render_x)
            render_y[idx] = float(actor.render_y)
            animation_controlled[idx] = actor._animation_controlled
            visual_scale[idx] = float(actor.visual_scale)

            if (
                actor.visual_effects is not None
                and actor.health is not None
                and actor.health.is_alive()
            ):
                dx, dy = actor.visual_effects.get_idle_drift_offset()
                drift_x[idx] = float(dx)
                drift_y[idx] = float(dy)

            sprite_uv = actor.sprite_uv
            if sprite_uv is not None:
                sprite_anchor_y[idx] = float(actor.sprite_ground_anchor_y)
                sprite_uvs[idx, 0] = float(sprite_uv.u1)
                sprite_uvs[idx, 1] = float(sprite_uv.v1)
                sprite_uvs[idx, 2] = float(sprite_uv.u2)
                sprite_uvs[idx, 3] = float(sprite_uv.v2)
                sprite_batchable[idx] = True

        # Vectorized interpolation.
        alpha_value = float(interpolation_alpha)
        interpolated_x = np.where(
            animation_controlled,
            render_x,
            prev_x * (1.0 - alpha_value) + curr_x * alpha_value,
        )
        interpolated_y = np.where(
            animation_controlled,
            render_y,
            prev_y * (1.0 - alpha_value) + curr_y * alpha_value,
        )
        interpolated_x += drift_x
        interpolated_y += drift_y

        # Vectorized screen position computation.
        viewport_pos = self._world_to_screen_float_batch(interpolated_x, interpolated_y)
        if viewport_pos is not None:
            vp_x, vp_y = viewport_pos
            cam_frac_x, cam_frac_y = self._camera_frac_offset
            root_x_arr = self._view_origin[0] + (vp_x - float(cam_frac_x))
            root_y_arr = self._view_origin[1] + (vp_y - float(cam_frac_y))
            screen_x_arr, screen_y_arr = self._console_to_screen_coords_batch(
                root_x_arr, root_y_arr
            )
        else:
            # Per-actor scalar fallback for test doubles.
            root_x_arr = np.empty(n, dtype=np.float64)
            root_y_arr = np.empty(n, dtype=np.float64)
            screen_x_arr = np.empty(n, dtype=np.float64)
            screen_y_arr = np.empty(n, dtype=np.float64)
            for idx in range(n):
                (
                    _,
                    _,
                    root_x_arr[idx],
                    root_y_arr[idx],
                    screen_x_arr[idx],
                    screen_y_arr[idx],
                ) = self._get_actor_screen_position(actors[idx], interpolation_alpha)

        return _ActorBatchData(
            actor_x=actor_x,
            actor_y=actor_y,
            shadow_heights=shadow_heights,
            visual_scale=visual_scale,
            sprite_anchor_y=sprite_anchor_y,
            sprite_uvs=sprite_uvs,
            sprite_batchable=sprite_batchable,
            root_x=root_x_arr,
            root_y=root_y_arr,
            screen_x=screen_x_arr,
            screen_y=screen_y_arr,
        )

    def _render_sun_actor_shadows(
        self,
        actors: list[Actor],
        interpolation_alpha: InterpolationAlpha,
        tile_height: float,
        receivers: list[Actor] | None = None,
        dynamic_receivers: list[Actor] | None = None,
        sun_params: _SunShadowParams | None = None,
        directional_light: DirectionalLight | None = None,
        batch_data: _ActorBatchData | None = None,
    ) -> None:
        """Render actor shadows cast by the directional light.

        Vectorized implementation: uses pre-extracted actor batch data,
        filters by outdoor eligibility, optionally clips by walls and
        computes receiver dimming (when LOD detail is enabled), then emits
        shadow geometry. Sprite actors use the batch sprite-shadow API
        when available, otherwise fall through to per-actor emission.
        """
        if sun_params is None:
            dl = directional_light or self._frame_directional_light
            if dl is not None:
                sun_params = self._compute_sun_shadow_params(dl)
        if sun_params is None:
            return

        # Use the restricted dynamic_receivers for the dimming loop when
        # provided, falling back to the full receivers list for backwards
        # compatibility (tests pass receivers directly).
        shadow_receivers = (
            dynamic_receivers
            if dynamic_receivers is not None
            else (actors if receivers is None else receivers)
        )
        shadow_dir_x, shadow_dir_y, shadow_length_scale = sun_params
        min_shadow_length_tiles = self._get_min_rendered_actor_shadow_length_tiles()

        # LOD flag: at low zoom, wall-clipping and receiver-dimming are
        # imperceptible and skipped.
        lod_detail: bool = self.viewport_zoom >= config.LOD_DETAIL_ZOOM_THRESHOLD

        # ------------------------------------------------------------------
        # 1. Use pre-extracted batch data (or extract on demand for tests).
        # ------------------------------------------------------------------
        if batch_data is None:
            batch_data = self._extract_actor_batch_data(actors, interpolation_alpha)

        actor_count = len(actors)
        actor_x = batch_data.actor_x
        actor_y = batch_data.actor_y
        shadow_heights = batch_data.shadow_heights
        visual_scale = batch_data.visual_scale
        sprite_anchor_y = batch_data.sprite_anchor_y
        sprite_uvs = batch_data.sprite_uvs
        sprite_batchable = batch_data.sprite_batchable

        # ------------------------------------------------------------------
        # 2. Shadow length filtering.
        # ------------------------------------------------------------------
        shadow_length_tiles = shadow_heights * float(shadow_length_scale)
        eligible_mask = (shadow_length_tiles > 0.0) & (
            shadow_length_tiles >= float(min_shadow_length_tiles)
        )
        if not np.any(eligible_mask):
            return

        # ------------------------------------------------------------------
        # 3. Outdoor eligibility filtering.
        # ------------------------------------------------------------------
        in_bounds = (
            (actor_x >= 0)
            & (actor_x < int(self.game_map.width))
            & (actor_y >= 0)
            & (actor_y < int(self.game_map.height))
        )

        # Use cached eligibility grid when available (vectorized),
        # otherwise fall back to per-actor tile check.
        eligible_grid = self._get_sun_shadow_eligibility_grid()
        outdoor_mask = np.zeros(actor_count, dtype=bool)
        outdoor_indices = np.flatnonzero(eligible_mask & in_bounds)
        if len(outdoor_indices) > 0:
            if eligible_grid is not None:
                outdoor_mask[outdoor_indices] = np.asarray(
                    eligible_grid[actor_x[outdoor_indices], actor_y[outdoor_indices]],
                    dtype=bool,
                )
            else:
                for oi in outdoor_indices:
                    outdoor_mask[oi] = self._can_render_sun_shadow_at_tile(
                        int(actor_x[oi]), int(actor_y[oi])
                    )
        eligible_mask &= outdoor_mask
        if not np.any(eligible_mask):
            return

        # ------------------------------------------------------------------
        # 4. Wall clipping (LOD detail only).
        # ------------------------------------------------------------------
        if lod_detail:
            eligible_indices_for_clip = np.flatnonzero(eligible_mask)
            if len(eligible_indices_for_clip) > 0:
                elig_x = actor_x[eligible_indices_for_clip]
                elig_y = actor_y[eligible_indices_for_clip]
                elig_lengths = shadow_length_tiles[eligible_indices_for_clip]

                max_steps = 8
                origin_x_f = elig_x.astype(np.float64) + 0.5
                origin_y_f = elig_y.astype(np.float64) + 0.5
                steps = np.arange(1, max_steps + 1, dtype=np.float64)

                # Sample positions for all actors x all steps: [N_elig, 8]
                sample_sx = np.floor(
                    origin_x_f[:, np.newaxis] + shadow_dir_x * steps[np.newaxis, :]
                ).astype(np.int32)
                sample_sy = np.floor(
                    origin_y_f[:, np.newaxis] + shadow_dir_y * steps[np.newaxis, :]
                ).astype(np.int32)

                map_w = int(self.game_map.width)
                map_h = int(self.game_map.height)
                step_in_bounds = (
                    (sample_sx >= 0)
                    & (sample_sx < map_w)
                    & (sample_sy >= 0)
                    & (sample_sy < map_h)
                )

                safe_sx = np.clip(sample_sx, 0, map_w - 1)
                safe_sy = np.clip(sample_sy, 0, map_h - 1)
                sampled_h = self.game_map.shadow_heights[safe_sx, safe_sy]

                # A step blocks if out of bounds or hits tall terrain.
                blocks = ~step_in_bounds | (sampled_h > 2)
                # Only count steps within the actor's shadow range.
                step_in_range = steps[np.newaxis, :] <= np.ceil(
                    elig_lengths[:, np.newaxis]
                )
                blocks &= step_in_range

                has_blocker = np.any(blocks, axis=1)
                first_blocker_idx = np.argmax(blocks, axis=1)
                clip_at_step = steps[first_blocker_idx]

                clipped_lengths = np.where(
                    has_blocker,
                    np.maximum(0.0, np.minimum(elig_lengths, clip_at_step - 0.5)),
                    elig_lengths,
                )

                shadow_length_tiles[eligible_indices_for_clip] = clipped_lengths
                eligible_mask &= shadow_length_tiles >= float(min_shadow_length_tiles)
                eligible_mask &= shadow_length_tiles > 0.0
                if not np.any(eligible_mask):
                    return

        # ------------------------------------------------------------------
        # 5. Receiver dimming (LOD detail only).
        # ------------------------------------------------------------------
        if lod_detail and shadow_receivers:
            eligible_indices_for_dimming = np.flatnonzero(eligible_mask)
            n_casters = len(eligible_indices_for_dimming)
            n_receivers = len(shadow_receivers)

            if n_casters > 0 and n_receivers > 0:
                c_center_x = (
                    actor_x[eligible_indices_for_dimming].astype(np.float64) + 0.5
                )
                c_center_y = (
                    actor_y[eligible_indices_for_dimming].astype(np.float64) + 0.5
                )
                c_shadow_h = shadow_heights[eligible_indices_for_dimming]
                c_vis_scale = visual_scale[eligible_indices_for_dimming]
                c_shadow_len = shadow_length_tiles[eligible_indices_for_dimming]

                r_center_x = np.array(
                    [float(r.x) + 0.5 for r in shadow_receivers],
                    dtype=np.float64,
                )
                r_center_y = np.array(
                    [float(r.y) + 0.5 for r in shadow_receivers],
                    dtype=np.float64,
                )
                r_vis_scale = np.array(
                    [float(r.visual_scale) for r in shadow_receivers],
                    dtype=np.float64,
                )

                # Build identity mask to skip self-shadowing.
                caster_ids = [id(actors[ci]) for ci in eligible_indices_for_dimming]
                receiver_id_map = {id(r): ri for ri, r in enumerate(shadow_receivers)}
                identity_mask = np.zeros((n_casters, n_receivers), dtype=bool)
                for ci, cid in enumerate(caster_ids):
                    ri = receiver_id_map.get(cid)
                    if ri is not None:
                        identity_mask[ci, ri] = True

                # Pairwise geometry [N_casters, D_receivers].
                rel_x = r_center_x[np.newaxis, :] - c_center_x[:, np.newaxis]
                rel_y = r_center_y[np.newaxis, :] - c_center_y[:, np.newaxis]

                dist_along = rel_x * shadow_dir_x + rel_y * shadow_dir_y
                dist_perp = np.abs(rel_x * shadow_dir_y - rel_y * shadow_dir_x)

                height_factor = np.minimum(1.0, c_shadow_h / 4.0)
                shadow_half_width = 0.18 + 0.22 * np.maximum(0.5, c_vis_scale)
                receiver_radius = 0.2 + 0.18 * np.maximum(0.5, r_vis_scale)
                lateral_limit = (
                    shadow_half_width[:, np.newaxis] + receiver_radius[np.newaxis, :]
                )

                valid = (
                    ~identity_mask
                    & (dist_along > 0.0)
                    & (dist_along < c_shadow_len[:, np.newaxis])
                    & (dist_perp < lateral_limit)
                )

                shadow_alpha_dim = float(config.ACTOR_SHADOW_ALPHA)
                fade_tip_dim = bool(config.ACTOR_SHADOW_FADE_TIP)

                lateral_factor = np.where(valid, 1.0 - dist_perp / lateral_limit, 0.0)
                tip_factor = np.where(
                    valid & fade_tip_dim,
                    1.0 - dist_along / c_shadow_len[:, np.newaxis],
                    np.where(valid, 1.0, 0.0),
                )
                attenuation = (
                    shadow_alpha_dim
                    * height_factor[:, np.newaxis]
                    * lateral_factor
                    * tip_factor
                )
                attenuation = np.where(valid, attenuation, 0.0)

                per_pair_factor = np.where(
                    attenuation > 0.0,
                    1.0 - np.minimum(0.95, attenuation),
                    1.0,
                )
                per_receiver_scale = np.prod(per_pair_factor, axis=0)
                per_receiver_scale = np.maximum(0.05, per_receiver_scale)

                for ri, receiver in enumerate(shadow_receivers):
                    if per_receiver_scale[ri] < 1.0 - 1e-9:
                        self._actor_shadow_receive_light_scale[receiver] = float(
                            per_receiver_scale[ri]
                        )

        # ------------------------------------------------------------------
        # 6. Use pre-computed screen positions from batch data.
        # ------------------------------------------------------------------
        root_x_arr = batch_data.root_x
        root_y_arr = batch_data.root_y
        screen_x_arr = batch_data.screen_x
        screen_y_arr = batch_data.screen_y

        # ------------------------------------------------------------------
        # 7. Emit shadow geometry.
        # ------------------------------------------------------------------
        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale_factors()
        shadow_length_pixels = shadow_length_tiles * float(tile_height)
        shadow_alpha = float(config.ACTOR_SHADOW_ALPHA)
        fade_tip = bool(config.ACTOR_SHADOW_FADE_TIP)
        eligible_indices = np.flatnonzero(eligible_mask)

        # Batch sprite-shadow API (WGPU). Resolved once per frame, not per-actor.
        draw_sprite_shadow_batch = getattr(
            self.graphics, "draw_sprite_shadow_batch", None
        )

        def flush_sprite_batch(batch_indices: list[int]) -> None:
            """Emit accumulated sprite shadow quads as a single batch."""
            if not batch_indices:
                return
            batch_idx = np.asarray(batch_indices, dtype=np.int32)

            if draw_sprite_shadow_batch is not None:
                draw_root_x = root_x_arr[batch_idx] + (
                    (float(viewport_scale_x) - 1.0) * 0.5
                )
                draw_root_y = root_y_arr[batch_idx] + (
                    (float(viewport_scale_y) - 1.0) * sprite_anchor_y[batch_idx]
                )
                sprite_screen_x, sprite_screen_y = self._console_to_screen_coords_batch(
                    draw_root_x, draw_root_y
                )
                draw_sprite_shadow_batch(
                    sprite_uvs=sprite_uvs[batch_idx],
                    screen_x=sprite_screen_x,
                    screen_y=sprite_screen_y,
                    shadow_dir_x=shadow_dir_x,
                    shadow_dir_y=shadow_dir_y,
                    shadow_length_pixels=shadow_length_pixels[batch_idx],
                    shadow_alpha=shadow_alpha,
                    scale_x=(visual_scale[batch_idx] * float(viewport_scale_x)),
                    scale_y=(visual_scale[batch_idx] * float(viewport_scale_y)),
                    ground_anchor_y=sprite_anchor_y[batch_idx],
                    fade_tip=fade_tip,
                )
                return

            # Per-actor fallback when batch sprite API is unavailable.
            for scalar_idx in batch_indices:
                self._emit_actor_shadow_quads(
                    actor=actors[scalar_idx],
                    root_x=float(root_x_arr[scalar_idx]),
                    root_y=float(root_y_arr[scalar_idx]),
                    screen_x=float(screen_x_arr[scalar_idx]),
                    screen_y=float(screen_y_arr[scalar_idx]),
                    shadow_dir_x=shadow_dir_x,
                    shadow_dir_y=shadow_dir_y,
                    shadow_length_pixels=float(shadow_length_pixels[scalar_idx]),
                    shadow_alpha=shadow_alpha,
                    fade_tip=fade_tip,
                )

        # Walk eligible actors in order, batching consecutive sprite actors
        # and flushing non-sprite actors individually.
        sprite_run: list[int] = []
        for idx_value in eligible_indices.tolist():
            if sprite_batchable[idx_value]:
                sprite_run.append(idx_value)
                continue

            flush_sprite_batch(sprite_run)
            sprite_run = []

            self._emit_actor_shadow_quads(
                actor=actors[idx_value],
                root_x=float(root_x_arr[idx_value]),
                root_y=float(root_y_arr[idx_value]),
                screen_x=float(screen_x_arr[idx_value]),
                screen_y=float(screen_y_arr[idx_value]),
                shadow_dir_x=shadow_dir_x,
                shadow_dir_y=shadow_dir_y,
                shadow_length_pixels=float(shadow_length_pixels[idx_value]),
                shadow_alpha=shadow_alpha,
                fade_tip=fade_tip,
            )

        flush_sprite_batch(sprite_run)

    def _get_sun_shadow_eligibility_grid(self) -> np.ndarray | None:
        """Return the cached eligibility grid, or None if the map doesn't support it."""
        get_grid = getattr(self.game_map, "get_sun_shadow_eligibility_grid", None)
        if not callable(get_grid):
            return None
        return get_grid(
            outdoor_region_types=self._SUN_SHADOW_OUTDOOR_REGION_TYPES,
            outdoor_tile_ids=self._SUN_SHADOW_OUTDOOR_TILE_IDS,
        )

    def _can_render_sun_shadow_at_tile(self, x: int, y: int) -> bool:
        """Return whether a tile should receive directional sun-projected shadows."""
        if not (0 <= x < self.game_map.width and 0 <= y < self.game_map.height):
            return False

        eligible_grid = self._get_sun_shadow_eligibility_grid()
        if eligible_grid is not None:
            return bool(eligible_grid[x, y])

        region = self.game_map.get_region_at((x, y))
        if region is None or region.sky_exposure <= 0.1:
            return False

        if region.region_type in self._SUN_SHADOW_OUTDOOR_REGION_TYPES:
            return True

        tile_id = int(self.game_map.tiles[x, y])
        return tile_id in self._SUN_SHADOW_OUTDOOR_TILE_IDS

    def _render_terrain_glyph_shadows(
        self,
        tile_height: float,
        *,
        viewport_scale: tuple[float, float] = (1.0, 1.0),
        sun_params: _SunShadowParams | None = None,
        directional_light: DirectionalLight | None = None,
    ) -> None:
        """Render projected glyph shadows for small terrain objects (boulders, etc.).

        Tiles whose shadow_height is in (0, 2] get a CPU-projected glyph shadow
        using the same draw_actor_shadow() path as actors. Taller tiles (walls,
        doors) keep their shader-only tile shadows.
        """
        # At low zoom, terrain glyph shadows (boulders etc.) are rendered at
        # ~10px per tile - the small projected shadow shapes are imperceptible.
        # Skip the entire terrain shadow pass to avoid scanning 16k+ tiles.
        if self.viewport_zoom < config.LOD_DETAIL_ZOOM_THRESHOLD:
            return

        if sun_params is None:
            dl = directional_light or self._frame_directional_light
            if dl is not None:
                sun_params = self._compute_sun_shadow_params(dl)
        if sun_params is None:
            return

        shadow_dir_x, shadow_dir_y, shadow_length_scale = sun_params
        viewport_scale_x, viewport_scale_y = viewport_scale

        # Get viewport bounds.
        bounds = self.viewport_system.get_visible_bounds()
        world_left = max(0, bounds.x1)
        world_top = max(0, bounds.y1)
        world_right = min(self.game_map.width - 1, bounds.x2)
        world_bottom = min(self.game_map.height - 1, bounds.y2)

        # Vectorized lookup: find visible tiles with shadow_height 1-2.
        # (height 0 = no shadow, 1-2 = glyph shadow, 3+ = staircase shader shadow)
        viewport_tiles = self.game_map.tiles[
            world_left : world_right + 1, world_top : world_bottom + 1
        ]
        heights = get_shadow_height_map(viewport_tiles)
        visible_slice = self.game_map.visible[
            world_left : world_right + 1, world_top : world_bottom + 1
        ]
        candidates = np.argwhere((heights > 0) & (heights <= 2) & visible_slice)

        if len(candidates) == 0:
            return

        cam_frac_x, cam_frac_y = self._camera_frac_offset

        for rel_x, rel_y in candidates:
            world_x = world_left + int(rel_x)
            world_y = world_top + int(rel_y)

            # Only render in outdoor areas exposed to sunlight.
            if not self._can_render_sun_shadow_at_tile(world_x, world_y):
                continue

            glyph_shadow_height = float(heights[rel_x, rel_y])
            shadow_length_tiles = glyph_shadow_height * shadow_length_scale
            if shadow_length_tiles <= 0.0:
                continue

            # Clip shadow by nearby walls.
            clipped_length_tiles = self._clip_shadow_length_by_walls(
                world_x, world_y, shadow_dir_x, shadow_dir_y, shadow_length_tiles
            )
            if clipped_length_tiles <= 0.0:
                continue

            # Get the tile's glyph character for the shadow shape.
            ch_code = int(self.game_map.light_appearance_map[world_x, world_y]["ch"])
            char = chr(ch_code)

            # Convert tile position to screen coordinates.
            vp_x, vp_y = self.viewport_system.world_to_screen(world_x, world_y)
            root_x = self._view_origin[0] + vp_x - cam_frac_x
            root_y = self._view_origin[1] + vp_y - cam_frac_y
            draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(root_x, root_y)
            screen_x, screen_y = self.graphics.console_to_screen_coords(
                draw_root_x, draw_root_y
            )

            self.graphics.draw_actor_shadow(
                char=char,
                screen_x=screen_x,
                screen_y=screen_y,
                shadow_dir_x=shadow_dir_x,
                shadow_dir_y=shadow_dir_y,
                shadow_length_pixels=clipped_length_tiles * tile_height,
                shadow_alpha=config.TERRAIN_GLYPH_SHADOW_ALPHA,
                scale_x=viewport_scale_x,
                scale_y=viewport_scale_y,
                fade_tip=config.ACTOR_SHADOW_FADE_TIP,
            )

    def _render_point_light_actor_shadows(
        self,
        actors: list[Actor],
        interpolation_alpha: InterpolationAlpha,
        tile_height: float,
        receivers: list[Actor] | None = None,
        dynamic_receivers: list[Actor] | None = None,
        lights: Sequence[Any] | None = None,
        batch_data: _ActorBatchData | None = None,
    ) -> None:
        """Render actor shadows cast by nearby point lights.

        Uses pre-computed batch data for screen positions, eliminating
        per-actor compute_actor_screen_position calls.
        """
        from brileta.game.lights import DirectionalLight

        # Use restricted dynamic_receivers for dimming when available.
        shadow_receivers = (
            dynamic_receivers
            if dynamic_receivers is not None
            else (actors if receivers is None else receivers)
        )
        active_lights = self._frame_lights if lights is None else lights
        point_lights = [
            light for light in active_lights if not isinstance(light, DirectionalLight)
        ]
        if not point_lights:
            return

        min_shadow_length_tiles = self._get_min_rendered_actor_shadow_length_tiles()
        lod_detail = self.viewport_zoom >= config.LOD_DETAIL_ZOOM_THRESHOLD

        # Use pre-extracted batch data (or extract on demand for tests).
        if batch_data is None:
            batch_data = self._extract_actor_batch_data(actors, interpolation_alpha)

        shadow_heights_arr = batch_data.shadow_heights
        root_x_arr = batch_data.root_x
        root_y_arr = batch_data.root_y
        screen_x_arr = batch_data.screen_x
        screen_y_arr = batch_data.screen_y
        actor_x_arr = batch_data.actor_x
        actor_y_arr = batch_data.actor_y

        # Pre-filter actors by shadow height to avoid per-light checks.
        min_sh = float(min_shadow_length_tiles)
        eligible_mask = (shadow_heights_arr > 0.0) & (shadow_heights_arr >= min_sh)
        eligible_indices = np.flatnonzero(eligible_mask)
        if len(eligible_indices) == 0:
            return

        # Cache config values outside the loop.
        actor_shadow_alpha_cfg = float(config.ACTOR_SHADOW_ALPHA)
        fade_tip = bool(config.ACTOR_SHADOW_FADE_TIP)

        for idx_val in eligible_indices.tolist():
            actor = actors[idx_val]
            shadow_height = float(shadow_heights_arr[idx_val])
            root_x = float(root_x_arr[idx_val])
            root_y = float(root_y_arr[idx_val])
            screen_x = float(screen_x_arr[idx_val])
            screen_y = float(screen_y_arr[idx_val])
            ax = int(actor_x_arr[idx_val])
            ay = int(actor_y_arr[idx_val])

            for light in point_lights:
                radius = float(light.radius)
                if radius <= 0.0:
                    continue

                # Actors should not cast directional shadows from their own lights.
                if light.owner is actor:
                    continue

                light_x, light_y = light.position
                # Guard against unstable direction when actor and light occupy
                # the same tile (including sub-tile drift jitter).
                if ax == int(light_x) and ay == int(light_y):
                    continue

                # Use tile-space positions for shadow direction. Idle drift should
                # move the rendered glyph, but not rotate a cardinal shadow into
                # a diagonal one when actor and light are horizontally aligned.
                dir_x = float(ax) - float(light_x)
                dir_y = float(ay) - float(light_y)
                distance = math.hypot(dir_x, dir_y)

                # Avoid undefined direction or out-of-range actors.
                if distance <= 1e-6 or distance > radius:
                    continue

                attenuation = max(0.0, 1.0 - distance / radius)
                light_intensity = float(light.intensity)
                shadow_alpha = actor_shadow_alpha_cfg * attenuation * light_intensity
                if shadow_alpha <= 0.0:
                    continue

                dir_x /= distance
                dir_y /= distance

                if lod_detail:
                    clipped_length_tiles = self._clip_shadow_length_by_walls(
                        ax, ay, dir_x, dir_y, shadow_height
                    )
                    if clipped_length_tiles <= 0.0 or clipped_length_tiles < min_sh:
                        continue
                else:
                    clipped_length_tiles = shadow_height

                if lod_detail:
                    self._accumulate_actor_shadow_receiver_dimming(
                        caster=actor,
                        receivers=shadow_receivers,
                        shadow_dir_x=dir_x,
                        shadow_dir_y=dir_y,
                        shadow_length_tiles=clipped_length_tiles,
                        shadow_alpha=shadow_alpha,
                        fade_tip=fade_tip,
                    )

                self._emit_actor_shadow_quads(
                    actor=actor,
                    root_x=root_x,
                    root_y=root_y,
                    screen_x=screen_x,
                    screen_y=screen_y,
                    shadow_dir_x=dir_x,
                    shadow_dir_y=dir_y,
                    shadow_length_pixels=clipped_length_tiles * tile_height,
                    shadow_alpha=shadow_alpha,
                    fade_tip=fade_tip,
                )

    # CP437 solid block (█) used as a fallback when sprite-silhouette shadows
    # are unavailable on the current graphics backend.
    _SOLID_BLOCK_CHAR = chr(219)

    def _emit_actor_shadow_quads(
        self,
        actor: Actor,
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
        """Emit projected shadow quads for an actor.

        Sprite actors use sprite-atlas silhouettes when supported by the
        graphics backend, with a solid-glyph fallback for compatibility.
        Character-layer and single-glyph actors use their existing per-layer
        shadow path.
        """
        visual_scale = actor.visual_scale
        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale_factors()

        # Sprite actors: prefer sprite-silhouette shadow when supported.
        # draw_sprite_shadow is defined on GraphicsContext with a no-op default,
        # so it's always callable.
        sprite_uv = actor.sprite_uv
        if sprite_uv is not None:
            sprite_ground_anchor_y = float(actor.sprite_ground_anchor_y)
            draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(
                root_x,
                root_y,
                ground_anchor_y=sprite_ground_anchor_y,
            )
            screen_x, screen_y = self.graphics.console_to_screen_coords(
                draw_root_x, draw_root_y
            )
            self.graphics.draw_sprite_shadow(
                sprite_uv=sprite_uv,
                screen_x=screen_x,
                screen_y=screen_y,
                shadow_dir_x=shadow_dir_x,
                shadow_dir_y=shadow_dir_y,
                shadow_length_pixels=shadow_length_pixels,
                shadow_alpha=shadow_alpha,
                scale_x=visual_scale * viewport_scale_x,
                scale_y=visual_scale * viewport_scale_y,
                ground_anchor_y=sprite_ground_anchor_y,
                fade_tip=fade_tip,
            )
            return

        # Character-layer actors: one shadow per glyph layer.
        if actor.character_layers:
            for layer in actor.character_layers:
                layer_root_x = root_x + (layer.offset_x * viewport_scale_x)
                layer_root_y = root_y + (layer.offset_y * viewport_scale_y)
                layer_root_x, layer_root_y = self._zoomed_tile_draw_origin(
                    layer_root_x, layer_root_y
                )
                layer_screen_x, layer_screen_y = self.graphics.console_to_screen_coords(
                    layer_root_x, layer_root_y
                )
                self.graphics.draw_actor_shadow(
                    char=layer.char,
                    screen_x=layer_screen_x,
                    screen_y=layer_screen_y,
                    shadow_dir_x=shadow_dir_x,
                    shadow_dir_y=shadow_dir_y,
                    shadow_length_pixels=shadow_length_pixels,
                    shadow_alpha=shadow_alpha,
                    scale_x=visual_scale * layer.scale_x * viewport_scale_x,
                    scale_y=visual_scale * layer.scale_y * viewport_scale_y,
                    fade_tip=fade_tip,
                )
            return

        # Single-glyph fallback.
        draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(root_x, root_y)
        screen_x, screen_y = self.graphics.console_to_screen_coords(
            draw_root_x, draw_root_y
        )
        self.graphics.draw_actor_shadow(
            char=actor.ch,
            screen_x=screen_x,
            screen_y=screen_y,
            shadow_dir_x=shadow_dir_x,
            shadow_dir_y=shadow_dir_y,
            shadow_length_pixels=shadow_length_pixels,
            shadow_alpha=shadow_alpha,
            scale_x=visual_scale * viewport_scale_x,
            scale_y=visual_scale * viewport_scale_y,
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

        max_steps = min(8, math.ceil(shadow_length_tiles))
        origin_x = float(actor_x) + 0.5
        origin_y = float(actor_y) + 0.5

        for step in range(1, max_steps + 1):
            sample_x = math.floor(origin_x + shadow_dir_x * float(step))
            sample_y = math.floor(origin_y + shadow_dir_y * float(step))

            if not (
                0 <= sample_x < self.game_map.width
                and 0 <= sample_y < self.game_map.height
            ):
                return max(0.0, min(shadow_length_tiles, float(step) - 0.5))

            # Projected glyph shadows for low-profile terrain (height 1-2) should
            # not occlude actor/terrain projected shadows; only tall blockers clip.
            if self.game_map.shadow_heights[sample_x, sample_y] > 2:
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
        caster_shadow_height = max(0.0, float(caster.shadow_height))
        height_factor = min(1.0, caster_shadow_height / 4.0)
        if height_factor <= 0.0:
            return

        caster_scale = max(0.5, float(caster.visual_scale))
        shadow_half_width = 0.18 + 0.22 * caster_scale

        for receiver in receivers:
            if receiver is caster:
                continue

            receiver_scale = max(0.5, float(receiver.visual_scale))
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

            current_scale = self._actor_shadow_receive_light_scale.get(receiver, 1.0)
            next_scale = current_scale * (1.0 - min(0.95, attenuation))
            self._actor_shadow_receive_light_scale[receiver] = max(0.05, next_scale)
