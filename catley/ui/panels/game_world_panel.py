from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from tcod.console import Console

from catley import colors
from catley.config import (
    HELP_HEIGHT,
    LUMINANCE_THRESHOLD,
    MOUSE_HIGHLIGHT_ALPHA,
    PULSATION_MAX_BLEND_ALPHA,
    PULSATION_PERIOD,
    SELECTION_HIGHLIGHT_ALPHA,
)
from catley.render.effects import EffectLibrary
from catley.render.particles import SubTileParticleSystem
from catley.render.screen_shake import ScreenShake

from .panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor
    from catley.render.renderer import Renderer


class GameWorldPanel(Panel):
    """Panel responsible for rendering the game world (map, actors, effects)."""

    def __init__(self, controller: Controller, screen_shake: ScreenShake) -> None:
        super().__init__(
            x=0,
            y=HELP_HEIGHT,
            width=controller.gw.game_map.width,
            height=controller.gw.game_map.height,
        )
        self.controller = controller
        self.screen_shake = screen_shake

        self.game_map_console = Console(
            controller.gw.game_map.width, controller.gw.game_map.height, order="F"
        )
        self.particle_system = SubTileParticleSystem(
            controller.gw.game_map.width, controller.gw.game_map.height
        )
        self.effect_library = EffectLibrary()
        self.current_light_intensity: np.ndarray | None = None

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
        """Highlight a tile with an optional effect."""
        if effect == "pulse":
            color = self._apply_pulsating_effect(color, color)
        self._apply_replacement_highlight(x, y, color)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, renderer: Renderer) -> None:
        if not self.visible:
            return

        delta_time = self.controller.clock.last_delta_time
        self._render_game_world(delta_time)
        shake_x, shake_y = self.screen_shake.update(delta_time)
        renderer.blit_console(
            source=self.game_map_console,
            dest=renderer.root_console,
            dest_x=self.x + shake_x,
            dest_y=self.y + shake_y,
        )

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------
    def _render_game_world(self, delta_time: float) -> None:
        renderer = self.controller.renderer
        renderer.clear_console(self.game_map_console)
        self._render_map()
        self._render_actors()

        if self.controller.active_mode:
            self.controller.active_mode.render_world()
        else:
            self._render_selected_actor_highlight()
            self._render_mouse_cursor_highlight()

        self.particle_system.update(delta_time)
        self.particle_system.render_to_console(self.game_map_console)

    def _render_map(self) -> None:
        gw = self.controller.gw
        shroud = (ord(" "), (0, 0, 0), (0, 0, 0))
        self.game_map_console.rgb[:] = shroud
        dark_app_map = gw.game_map.dark_appearance_map
        light_app_map = gw.game_map.light_appearance_map
        explored_mask = gw.game_map.explored
        self.game_map_console.rgb[explored_mask] = dark_app_map[explored_mask]
        visible_mask = gw.game_map.visible
        visible_y, visible_x = np.where(visible_mask)
        if len(visible_y) > 0:
            self.current_light_intensity = gw.lighting.compute_lighting_with_shadows(
                gw.game_map.width, gw.game_map.height, gw.actors
            )
            dark_tiles_visible = dark_app_map[visible_y, visible_x]
            light_tiles_visible = light_app_map[visible_y, visible_x]
            cell_light = self.current_light_intensity[visible_y, visible_x]
            blended_tiles = np.empty_like(dark_tiles_visible)
            blended_tiles["ch"] = light_tiles_visible["ch"]
            blended_tiles["fg"] = light_tiles_visible["fg"]
            for i in range(3):
                light_intensity_channel = cell_light[..., i]
                blended_tiles["bg"][..., i] = light_tiles_visible["bg"][
                    ..., i
                ] * light_intensity_channel + dark_tiles_visible["bg"][..., i] * (
                    1.0 - light_intensity_channel
                )
            self.game_map_console.rgb[visible_y, visible_x] = blended_tiles

    def _render_actors(self) -> None:
        gw = self.controller.gw
        for a in gw.actors:
            if a == gw.player:
                continue
            if gw.game_map.visible[a.x, a.y]:
                self._render_actor(a)
        self._render_actor(gw.player)

    def _render_actor(self, a: Actor) -> None:
        if self.current_light_intensity is None:
            return

        self.game_map_console.rgb["ch"][a.x, a.y] = ord(a.ch)
        base_actor_color: colors.Color = a.color

        visual_effects = a.visual_effects
        if visual_effects is not None:
            visual_effects.update()
            flash_color = visual_effects.get_flash_color()
            if flash_color:
                base_actor_color = flash_color

        light_rgb = self.current_light_intensity[a.x, a.y]
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
            final_fg_color = self._apply_pulsating_effect(
                normally_lit_fg_components, base_actor_color
            )

        self.game_map_console.rgb["fg"][a.x, a.y] = final_fg_color

    def _render_selected_actor_highlight(self) -> None:
        if self.controller.is_targeting_mode():
            return
        actor = self.controller.gw.selected_actor
        if actor and self.controller.gw.game_map.visible[actor.x, actor.y]:
            self._apply_blended_highlight(
                actor.x,
                actor.y,
                colors.SELECTED_HIGHLIGHT,
                SELECTION_HIGHLIGHT_ALPHA,
            )

    def _render_mouse_cursor_highlight(self) -> None:
        if self.controller.is_targeting_mode():
            return
        if not self.controller.gw.mouse_tile_location_on_map:
            return
        mx, my = self.controller.gw.mouse_tile_location_on_map
        if not (
            0 <= mx < self.game_map_console.width
            and 0 <= my < self.game_map_console.height
        ):
            return
        target_color = (
            colors.WHITE if self.controller.gw.game_map.visible[mx, my] else colors.GREY
        )
        self._apply_blended_highlight(mx, my, target_color, MOUSE_HIGHLIGHT_ALPHA)

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
