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


class ShadowRenderer:
    """Render projected terrain and actor shadows for a world view."""

    _SUN_SHADOW_OUTDOOR_REGION_TYPES: frozenset[str] = frozenset(
        {"outdoor", "exterior", "test_outdoor"}
    )
    _SUN_SHADOW_OUTDOOR_TILE_IDS: frozenset[int] = frozenset(
        {
            int(TileTypeID.COBBLESTONE),
            int(TileTypeID.GRASS),
            int(TileTypeID.DIRT_PATH),
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

    def render_actor_shadows(
        self,
        interpolation_alpha: InterpolationAlpha,
        *,
        visible_actors: list[Actor] | None,
        directional_light: DirectionalLight | None,
        lights: Sequence[Any],
        view_origin: tuple[float, float],
        camera_frac_offset: tuple[float, float],
    ) -> None:
        """Render projected glyph shadows for terrain objects and visible actors."""
        self.set_view_transform(
            view_origin=view_origin,
            camera_frac_offset=camera_frac_offset,
        )
        self.set_frame_lighting(directional_light=directional_light, lights=lights)
        self._actor_shadow_receive_light_scale = {}
        if not config.SHADOWS_ENABLED:
            return

        tile_height = float(self.graphics.tile_dimensions[1])

        # Compute sun shadow params once for both terrain and actor passes.
        sun_params: _SunShadowParams | None = None
        if directional_light is not None:
            sun_params = self._compute_sun_shadow_params(directional_light)

        # Terrain glyph shadows (boulders, etc.) - independent of actor positions.
        self._render_terrain_glyph_shadows(
            tile_height,
            sun_params=sun_params,
        )

        if not visible_actors:
            return

        shadow_casters = [
            actor for actor in visible_actors if getattr(actor, "shadow_height", 0) > 0
        ]
        if not shadow_casters:
            return

        self._render_sun_actor_shadows(
            shadow_casters,
            interpolation_alpha,
            tile_height,
            receivers=visible_actors,
            sun_params=sun_params,
            directional_light=directional_light,
        )
        self._render_point_light_actor_shadows(
            shadow_casters,
            interpolation_alpha,
            tile_height,
            receivers=visible_actors,
            lights=lights,
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

    def _render_sun_actor_shadows(
        self,
        actors: list[Actor],
        interpolation_alpha: InterpolationAlpha,
        tile_height: float,
        receivers: list[Actor] | None = None,
        sun_params: _SunShadowParams | None = None,
        directional_light: DirectionalLight | None = None,
    ) -> None:
        """Render actor shadows cast by the directional light."""
        if sun_params is None:
            dl = directional_light or self._frame_directional_light
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
                actor, interpolation_alpha
            )
            self._emit_actor_shadow_quads(
                actor=actor,
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
        sun_params: _SunShadowParams | None = None,
        directional_light: DirectionalLight | None = None,
    ) -> None:
        """Render projected glyph shadows for small terrain objects (boulders, etc.).

        Tiles whose shadow_height is in (0, 2] get a CPU-projected glyph shadow
        using the same draw_actor_shadow() path as actors. Taller tiles (walls,
        doors) keep their shader-only tile shadows.
        """
        if sun_params is None:
            dl = directional_light or self._frame_directional_light
            if dl is not None:
                sun_params = self._compute_sun_shadow_params(dl)
        if sun_params is None:
            return

        shadow_dir_x, shadow_dir_y, shadow_length_scale = sun_params

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
            screen_x, screen_y = self.graphics.console_to_screen_coords(root_x, root_y)

            self.graphics.draw_actor_shadow(
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
        actors: list[Actor],
        interpolation_alpha: InterpolationAlpha,
        tile_height: float,
        receivers: list[Actor] | None = None,
        lights: Sequence[Any] | None = None,
    ) -> None:
        """Render actor shadows cast by nearby point lights."""
        from brileta.game.lights import DirectionalLight

        shadow_receivers = actors if receivers is None else receivers
        active_lights = self._frame_lights if lights is None else lights
        point_lights = [
            light for light in active_lights if not isinstance(light, DirectionalLight)
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
            ) = self._get_actor_screen_position(actor, interpolation_alpha)

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
                    scale_x=visual_scale * layer.scale_x,
                    scale_y=visual_scale * layer.scale_y,
                    fade_tip=fade_tip,
                )
            return

        self.graphics.draw_actor_shadow(
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

            current_scale = self._actor_shadow_receive_light_scale.get(receiver, 1.0)
            next_scale = current_scale * (1.0 - min(0.95, attenuation))
            self._actor_shadow_receive_light_scale[receiver] = max(0.05, next_scale)
