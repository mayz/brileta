"""Actor rendering and outline drawing for the world view.

Handles smooth sub-pixel actor positioning, traditional tile-aligned rendering,
character layer composition, and contextual/combat outline drawing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

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

# Roof occlusion outline (actors behind opaque roofs)
_ROOF_OCCLUDED_OUTLINE_ALPHA: Opacity = Opacity(0.45)


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
        roof_occluded_actors: frozenset[Actor] | None = None,
    ) -> None:
        """Render all visible actors with smooth sub-pixel positioning.

        Sprite actors are batched into a single vectorized draw call to avoid
        per-actor Python overhead. Non-sprite actors (character layers, single
        glyphs) are rendered individually between sprite batches to maintain
        correct painter's algorithm ordering.

        Actors in ``roof_occluded_actors`` are rendered as white outline-only
        glyphs instead of their normal appearance, indicating they are behind
        an opaque building roof.

        Args:
            interpolation_alpha: Interpolation factor for smooth movement.
            game_world: The current game world state.
            camera_frac_offset: Fractional camera offset for smooth scrolling.
            view_origin: Root console origin of the viewport (x, y).
            visible_actors: Pre-filtered list of visible actors, or None to compute.
            viewport_bounds: Viewport bounds for culling, or None to compute.
            roof_occluded_actors: Actors to render as outline-only (behind roofs).
        """
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
        n = len(visible_actors)

        # Classify actors and collect sprite data in one pass.
        sprite_indices: list[int] = []
        non_sprite_indices: list[int] = []

        # Pre-allocate arrays for sprite actor data.
        prev_x = np.empty(n, dtype=np.float64)
        prev_y = np.empty(n, dtype=np.float64)
        curr_x = np.empty(n, dtype=np.float64)
        curr_y = np.empty(n, dtype=np.float64)
        render_x = np.empty(n, dtype=np.float64)
        render_y = np.empty(n, dtype=np.float64)
        anim_controlled = np.zeros(n, dtype=bool)
        drift_x = np.zeros(n, dtype=np.float64)
        drift_y = np.zeros(n, dtype=np.float64)
        actor_colors = np.empty((n, 3), dtype=np.uint8)
        sprite_uvs = np.empty((n, 4), dtype=np.float32)
        visual_scales = np.ones(n, dtype=np.float64)
        anchor_y = np.ones(n, dtype=np.float64)
        world_pos_arr = np.empty((n, 2), dtype=np.int32)
        tile_bg_arr = np.zeros((n, 3), dtype=np.uint8)
        light_scale_arr = np.ones(n, dtype=np.float64)

        # Single pass: classify and extract sprite data.
        # Roof-occluded actors are forced into non_sprite_indices so they get
        # individual outline rendering instead of being batched with sprites.
        shadow_receive = self.shadow_renderer.actor_shadow_receive_light_scale
        for idx, actor in enumerate(visible_actors):
            if roof_occluded_actors is not None and actor in roof_occluded_actors:
                non_sprite_indices.append(idx)
                continue
            sprite_uv = actor.sprite_uv
            if sprite_uv is not None and not actor.character_layers:
                sprite_indices.append(idx)
                prev_x[idx] = float(actor.prev_x)
                prev_y[idx] = float(actor.prev_y)
                curr_x[idx] = float(actor.x)
                curr_y[idx] = float(actor.y)
                render_x[idx] = float(actor.render_x)
                render_y[idx] = float(actor.render_y)
                anim_controlled[idx] = actor._animation_controlled
                visual_scales[idx] = float(actor.visual_scale)
                world_pos_arr[idx, 0] = actor.x
                world_pos_arr[idx, 1] = actor.y

                # Idle drift.
                if (
                    actor.visual_effects is not None
                    and actor.health is not None
                    and actor.health.is_alive()
                ):
                    dx, dy = actor.visual_effects.get_idle_drift_offset()
                    drift_x[idx] = float(dx)
                    drift_y[idx] = float(dy)

                # Color with flash override.
                color = actor.color
                ve = actor.visual_effects
                if ve is not None:
                    ve.update()
                    flash = ve.get_flash_color()
                    if flash:
                        color = flash
                actor_colors[idx] = color

                # Sprite UV.
                sprite_uvs[idx, 0] = float(sprite_uv.u1)
                sprite_uvs[idx, 1] = float(sprite_uv.v1)
                sprite_uvs[idx, 2] = float(sprite_uv.u2)
                sprite_uvs[idx, 3] = float(sprite_uv.v2)

                anchor_y[idx] = float(actor.sprite_ground_anchor_y)

                # Tile background for shader contrast.
                bg = game_map.light_appearance_map[actor.x, actor.y]["bg"]
                tile_bg_arr[idx, 0] = int(bg[0])
                tile_bg_arr[idx, 1] = int(bg[1])
                tile_bg_arr[idx, 2] = int(bg[2])

                # Shadow receive dimming.
                light_scale_arr[idx] = shadow_receive.get(actor, 1.0)
            else:
                non_sprite_indices.append(idx)

        if not sprite_indices:
            # All non-sprite: use per-actor path.
            for idx in non_sprite_indices:
                actor = visible_actors[idx]
                if roof_occluded_actors is not None and actor in roof_occluded_actors:
                    self._render_roof_occluded_outline(
                        actor,
                        interpolation_alpha,
                        camera_frac_offset=camera_frac_offset,
                        view_origin=view_origin,
                    )
                else:
                    self._render_single_actor(
                        actor,
                        viewport_bounds,
                        vs,
                        interpolation_alpha,
                        game_map=game_map,
                        camera_frac_offset=camera_frac_offset,
                        view_origin=view_origin,
                    )
            return

        # Vectorized screen position computation for sprite actors.
        si = np.array(sprite_indices, dtype=np.int32)
        alpha_val = float(interpolation_alpha)
        interp_x = np.where(
            anim_controlled[si],
            render_x[si],
            prev_x[si] * (1.0 - alpha_val) + curr_x[si] * alpha_val,
        )
        interp_y = np.where(
            anim_controlled[si],
            render_y[si],
            prev_y[si] * (1.0 - alpha_val) + curr_y[si] * alpha_val,
        )
        interp_x += drift_x[si]
        interp_y += drift_y[si]

        # Vectorized world-to-screen coordinate transform.
        viewport_scale_x, viewport_scale_y = vs.get_display_scale_factors()
        bounds = vs.viewport.get_world_bounds(vs.camera)
        left = float(bounds.x1)
        top = float(bounds.y1)
        offset_x = float(vs.viewport.offset_x)
        offset_y = float(vs.viewport.offset_y)
        vp_x = (interp_x - left + offset_x) * viewport_scale_x
        vp_y = (interp_y - top + offset_y) * viewport_scale_y

        cam_frac_x, cam_frac_y = camera_frac_offset
        root_x = float(view_origin[0]) + (vp_x - float(cam_frac_x))
        root_y = float(view_origin[1]) + (vp_y - float(cam_frac_y))

        # Apply zoomed tile draw origin correction for sprites.
        draw_root_x = root_x + (viewport_scale_x - 1.0) * 0.5
        draw_root_y = root_y + (viewport_scale_y - 1.0) * anchor_y[si]

        # Vectorized console-to-screen-coords.
        letterbox_geometry = self.graphics.letterbox_geometry
        if isinstance(letterbox_geometry, tuple):
            lx, ly, lw, lh = letterbox_geometry
            console_w = self.graphics.console_width_tiles
            console_h = self.graphics.console_height_tiles
            screen_px_x = float(lx) + draw_root_x * (float(lw) / float(console_w))
            screen_px_y = float(ly) + draw_root_y * (float(lh) / float(console_h))
        else:
            tile_dims = self.graphics.tile_dimensions
            screen_px_x = np.trunc(draw_root_x * float(tile_dims[0]))
            screen_px_y = np.trunc(draw_root_y * float(tile_dims[1]))

        # Build light intensity array.
        ls = light_scale_arr[si]
        light_intensity = np.column_stack([ls, ls, ls]).astype(np.float32)

        # Maintain painter's algorithm by processing in draw order, batching
        # contiguous runs of sprite actors and flushing when interrupted by
        # non-sprite actors.  Both index lists are already sorted, so we
        # merge-iterate them rather than using a set lookup per actor.
        sprite_batch_pos = {idx: pos for pos, idx in enumerate(sprite_indices)}

        def flush_sprite_run(run: list[int]) -> None:
            """Emit a contiguous batch of sprite actors."""
            if not run:
                return
            batch_pos = np.array(
                [sprite_batch_pos[idx] for idx in run],
                dtype=np.int32,
            )
            actor_idx = si[batch_pos]
            self.graphics.draw_sprite_smooth_batch(
                sprite_uvs=sprite_uvs[actor_idx],
                actor_colors=actor_colors[actor_idx],
                screen_x=screen_px_x[batch_pos],
                screen_y=screen_px_y[batch_pos],
                light_intensity=light_intensity[batch_pos],
                scale_x=(visual_scales[actor_idx] * viewport_scale_x).astype(
                    np.float32,
                ),
                scale_y=(visual_scales[actor_idx] * viewport_scale_y).astype(
                    np.float32,
                ),
                ground_anchor_y=anchor_y[actor_idx].astype(np.float32),
                world_pos=world_pos_arr[actor_idx],
                tile_bg=tile_bg_arr[actor_idx],
            )

        current_run: list[int] = []
        ns_iter = iter(non_sprite_indices)
        next_ns = next(ns_iter, n)  # sentinel: n is past end
        for idx in range(n):
            if idx < next_ns:
                # This actor is a sprite (comes before the next non-sprite).
                current_run.append(idx)
            else:
                # Hit a non-sprite boundary - flush the sprite run first.
                flush_sprite_run(current_run)
                current_run = []
                actor = visible_actors[idx]
                if roof_occluded_actors is not None and actor in roof_occluded_actors:
                    self._render_roof_occluded_outline(
                        actor,
                        interpolation_alpha,
                        camera_frac_offset=camera_frac_offset,
                        view_origin=view_origin,
                    )
                else:
                    self._render_single_actor(
                        actor,
                        viewport_bounds,
                        vs,
                        interpolation_alpha,
                        game_map=game_map,
                        camera_frac_offset=camera_frac_offset,
                        view_origin=view_origin,
                    )
                next_ns = next(ns_iter, n)
        flush_sprite_run(current_run)

    def _get_viewport_display_scale(self) -> tuple[float, float]:
        """Return world-tile-to-root-console scaling for the current viewport."""
        return self.viewport_system.get_display_scale_factors()

    def _zoomed_tile_draw_origin(
        self,
        root_x: float,
        root_y: float,
        *,
        ground_anchor_y: float | None = None,
    ) -> tuple[float, float]:
        """Shift root-console origin so scaled draws stay anchored to zoomed tiles.

        Backend draw methods scale around a single root-console tile cell. During
        world-only zoom, a world tile spans multiple root-console tiles, so we
        pre-shift the draw origin into that larger cell before the backend applies
        its centering/anchor math.
        """
        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale()
        corrected_x = root_x + (viewport_scale_x - 1.0) * 0.5
        if ground_anchor_y is None:
            corrected_y = root_y + (viewport_scale_y - 1.0) * 0.5
        else:
            corrected_y = root_y + (viewport_scale_y - 1.0) * float(ground_anchor_y)
        return (corrected_x, corrected_y)

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
        """Render an actor-shaped outline at the actor's current position.

        Used for combat targeting to show a shimmering outline around an
        enemy. Sprite actors use sprite-contour outlines; glyph actors use
        CP437 glyph outlines. The actor must be visible.

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
        if actor._animation_controlled:
            world_x = actor.render_x
            world_y = actor.render_y
        else:
            world_x = float(actor.x)
            world_y = float(actor.y)
        vp_x, vp_y = vs.world_to_screen_float(world_x, world_y)

        # Apply camera fractional offset for smooth scrolling alignment
        cam_frac_x, cam_frac_y = camera_frac_offset
        vp_x -= cam_frac_x
        vp_y -= cam_frac_y

        root_x = view_origin[0] + vp_x
        root_y = view_origin[1] + vp_y
        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale()

        sprite_uv = actor.sprite_uv
        if sprite_uv is not None:
            sprite_ground_anchor_y = float(actor.sprite_ground_anchor_y)
            draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(
                root_x,
                root_y,
                ground_anchor_y=sprite_ground_anchor_y,
            )
            screen_x, screen_y = self.graphics.console_to_screen_coords(
                draw_root_x,
                draw_root_y,
            )
            self.graphics.draw_sprite_outline(
                sprite_uv,
                screen_x,
                screen_y,
                color,
                alpha,
                scale_x=actor.visual_scale * viewport_scale_x,
                scale_y=actor.visual_scale * viewport_scale_y,
                ground_anchor_y=sprite_ground_anchor_y,
            )
            return

        if actor.character_layers or actor.has_complex_visuals:
            self._render_layered_tile_outline(
                actor,
                color,
                alpha,
                camera_frac_offset=camera_frac_offset,
                view_origin=view_origin,
            )
            return

        draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(root_x, root_y)
        screen_x, screen_y = self.graphics.console_to_screen_coords(
            draw_root_x, draw_root_y
        )
        self.graphics.draw_actor_outline(
            actor.ch,
            screen_x,
            screen_y,
            color,
            alpha,
            scale_x=actor.visual_scale * viewport_scale_x,
            scale_y=actor.visual_scale * viewport_scale_y,
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

        Sorted by Y position (painter's algorithm), then player on top (so
        large boulders/trees at the same Y never occlude the player), then
        visual_scale (larger on top at same Y among non-player actors).
        """
        actors_in_viewport = game_world.actor_spatial_index.get_in_bounds(
            bounds.x1, bounds.y1, bounds.x2, bounds.y2
        )

        # Use `is` identity check instead of `==` equality to avoid
        # Actor.__eq__ dispatch (isinstance + actor_id comparison).
        player = game_world.player
        return sorted(
            actors_in_viewport,
            key=lambda actor: (
                actor.y,
                actor is player,
                actor.visual_scale,
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

    def _render_roof_occluded_outline(
        self,
        actor: Actor,
        interpolation_alpha: InterpolationAlpha,
        *,
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
    ) -> None:
        """Render a white outline glyph for an actor hidden behind an opaque roof.

        Uses the pre-generated outlined atlas (same as hover/selection outlines)
        to draw just the character silhouette edge in white, indicating the
        actor's position without showing the full colored glyph.

        Idle drift is deliberately excluded: the thin 1-pixel outline is very
        sensitive to sub-pixel position changes, causing a buzzing/flickering
        artifact when the drift shifts which pixels are lit each frame.
        """
        vs = self.viewport_system
        alpha_value = float(interpolation_alpha)
        if actor._animation_controlled:
            world_x = actor.render_x
            world_y = actor.render_y
        else:
            world_x = actor.prev_x * (1.0 - alpha_value) + actor.x * alpha_value
            world_y = actor.prev_y * (1.0 - alpha_value) + actor.y * alpha_value

        vp_x, vp_y = vs.world_to_screen_float(world_x, world_y)
        cam_frac_x, cam_frac_y = camera_frac_offset
        vp_x -= cam_frac_x
        vp_y -= cam_frac_y
        root_x = view_origin[0] + vp_x
        root_y = view_origin[1] + vp_y
        draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(root_x, root_y)
        screen_x, screen_y = self.graphics.console_to_screen_coords(
            draw_root_x, draw_root_y
        )
        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale()

        sprite_uv = actor.sprite_uv
        if sprite_uv is not None:
            sprite_ground_anchor_y = float(actor.sprite_ground_anchor_y)
            draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(
                root_x,
                root_y,
                ground_anchor_y=sprite_ground_anchor_y,
            )
            screen_x, screen_y = self.graphics.console_to_screen_coords(
                draw_root_x,
                draw_root_y,
            )
            self.graphics.draw_sprite_outline(
                sprite_uv,
                screen_x,
                screen_y,
                colors.WHITE,
                _ROOF_OCCLUDED_OUTLINE_ALPHA,
                scale_x=actor.visual_scale * viewport_scale_x,
                scale_y=actor.visual_scale * viewport_scale_y,
                ground_anchor_y=sprite_ground_anchor_y,
            )
            return

        if actor.character_layers or actor.has_complex_visuals:
            # Non-sprite complex actors fall back to a tile outline marker.
            screen_rect_x, screen_rect_y = self.graphics.console_to_screen_coords(
                root_x,
                root_y,
            )
            tile_w, tile_h = self.graphics.tile_dimensions
            self.graphics.draw_rect_outline(
                int(screen_rect_x),
                int(screen_rect_y),
                max(1, round(tile_w * viewport_scale_x)),
                max(1, round(tile_h * viewport_scale_y)),
                colors.WHITE,
                _ROOF_OCCLUDED_OUTLINE_ALPHA,
            )
            return

        self.graphics.draw_actor_outline(
            actor.ch,
            screen_x,
            screen_y,
            colors.WHITE,
            _ROOF_OCCLUDED_OUTLINE_ALPHA,
            scale_x=actor.visual_scale * viewport_scale_x,
            scale_y=actor.visual_scale * viewport_scale_y,
        )

    @record_time_live_variable("time.render.actors_smooth_ms")
    def _render_single_actor(
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
        visual_scale = actor.visual_scale
        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale()

        # Send tile background to GPU shader for actor-vs-tile contrast checks.
        tile_bg_np = game_map.light_appearance_map[actor.x, actor.y]["bg"]
        tile_bg = (
            int(tile_bg_np[0]),
            int(tile_bg_np[1]),
            int(tile_bg_np[2]),
        )

        # Priority 1: Sprite atlas path (procedurally generated sprites).
        sprite_uv = actor.sprite_uv
        if sprite_uv is not None:
            sprite_ground_anchor_y = float(actor.sprite_ground_anchor_y)
            draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(
                root_x,
                root_y,
                ground_anchor_y=sprite_ground_anchor_y,
            )
            screen_pixel_x, screen_pixel_y = self.graphics.console_to_screen_coords(
                draw_root_x,
                draw_root_y,
            )
            self.graphics.draw_sprite_smooth(
                sprite_uv,
                final_color,
                screen_pixel_x,
                screen_pixel_y,
                light_rgb,
                interpolation_alpha,
                scale_x=visual_scale * viewport_scale_x,
                scale_y=visual_scale * viewport_scale_y,
                ground_anchor_y=sprite_ground_anchor_y,
                world_pos=(actor.x, actor.y),
                tile_bg=tile_bg,
            )
        elif actor.character_layers:
            # Priority 2: Multi-character composition (character_layers).
            self._render_character_layers(
                actor.character_layers,
                root_x,
                root_y,
                light_rgb,
                interpolation_alpha,
                visual_scale,
                actor_world_pos=(actor.x, actor.y),
                viewport_scale=(viewport_scale_x, viewport_scale_y),
            )
        else:
            # Priority 3: Single CP437 glyph fallback.
            draw_root_x, draw_root_y = self._zoomed_tile_draw_origin(root_x, root_y)
            screen_pixel_x, screen_pixel_y = self.graphics.console_to_screen_coords(
                draw_root_x,
                draw_root_y,
            )
            self.graphics.draw_actor(
                actor.ch,
                final_color,
                screen_pixel_x,
                screen_pixel_y,
                light_rgb,
                interpolation_alpha,
                scale_x=visual_scale * viewport_scale_x,
                scale_y=visual_scale * viewport_scale_y,
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
        viewport_scale: tuple[float, float],
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
        viewport_scale_x, viewport_scale_y = viewport_scale
        for layer in layers:
            # Calculate this layer's position by adding its offset to the base position
            layer_x = root_x + (layer.offset_x * viewport_scale_x)
            layer_y = root_y + (layer.offset_y * viewport_scale_y)
            layer_x, layer_y = self._zoomed_tile_draw_origin(layer_x, layer_y)

            # Convert to screen pixel coordinates
            pixel_x, pixel_y = graphics.console_to_screen_coords(layer_x, layer_y)

            # Combine actor scale with per-layer scale (non-uniform)
            combined_scale_x = visual_scale * layer.scale_x * viewport_scale_x
            combined_scale_y = visual_scale * layer.scale_y * viewport_scale_y

            # Render this layer
            graphics.draw_actor(
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
        if (
            actor.sprite_uv is not None
            or actor.character_layers
            or actor.has_complex_visuals
        ):
            self._render_layered_tile_outline(
                actor,
                color,
                alpha,
                camera_frac_offset=camera_frac_offset,
                view_origin=view_origin,
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
        camera_frac_offset: ViewOffset,
        view_origin: ViewOffset,
    ) -> None:
        """Render a full-tile rectangle outline for actors with complex visuals."""
        vs = self.viewport_system
        if not vs.is_visible(actor.x, actor.y):
            return

        vp_x, vp_y = vs.world_to_screen_float(float(actor.x), float(actor.y))

        # Apply camera fractional offset for smooth scrolling alignment.
        # Without this, the outline snaps to integer tile boundaries while the
        # actor layers render at fractionally-offset positions, causing the
        # outline to straddle tiles or jitter during camera panning.
        cam_frac_x, cam_frac_y = camera_frac_offset
        vp_x -= cam_frac_x
        vp_y -= cam_frac_y

        root_x = view_origin[0] + vp_x
        root_y = view_origin[1] + vp_y
        screen_x, screen_y = self.graphics.console_to_screen_coords(root_x, root_y)

        tile_w, tile_h = self.graphics.tile_dimensions
        viewport_scale_x, viewport_scale_y = self._get_viewport_display_scale()
        self.graphics.draw_rect_outline(
            int(screen_x),
            int(screen_y),
            max(1, round(tile_w * viewport_scale_x)),
            max(1, round(tile_h * viewport_scale_y)),
            color,
            alpha,
        )
