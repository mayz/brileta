import math
from typing import TYPE_CHECKING

import numpy as np
from tcod.console import Console
from tcod.context import Context

from catley import colors
from catley.config import (
    HELP_HEIGHT,
    LUMINANCE_THRESHOLD,
    MESSAGE_LOG_HEIGHT,
    MOUSE_HIGHLIGHT_ALPHA,
    PULSATION_MAX_BLEND_ALPHA,
    PULSATION_PERIOD,
    SELECTION_HIGHLIGHT_ALPHA,
)
from catley.game.actors import Actor
from catley.ui.message_log import MessageLog
from catley.util.clock import Clock
from catley.world.game_state import GameWorld

if TYPE_CHECKING:
    from catley.ui.menu_system import MenuSystem


class FPSDisplay:
    SHOW_FPS = False

    def __init__(self, clock: Clock, update_interval: float = 0.5) -> None:
        self.clock = clock
        self.update_interval = update_interval
        self.last_update = clock.last_time
        self.display_string = "FPS: 0.0"

    def update(self) -> None:
        current_time = self.clock.last_time
        if current_time - self.last_update >= self.update_interval:
            self.display_string = f"FPS: {self.clock.last_fps:.1f}"
            self.last_update = current_time

    def render(self, renderer: "Renderer", x: int, y: int) -> None:
        self.update()
        renderer._render_text(x, y, self.display_string, fg=colors.YELLOW)


class Renderer:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        game_world: GameWorld,
        clock: Clock,
        message_log: MessageLog,
        menu_system: "MenuSystem",
        context: Context,
        root_console: Console,
    ) -> None:
        self.context = context
        self.root_console = root_console

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gw = game_world
        self.message_log = message_log

        # Define message log geometry - help at top, then message log
        self.help_height = HELP_HEIGHT
        self.message_log_height = MESSAGE_LOG_HEIGHT
        self.message_log_x = 0
        self.message_log_y = self.help_height  # Start after help text
        self.message_log_width = screen_width

        # Create game map console
        self.game_map_console: Console = Console(
            game_world.game_map.width, game_world.game_map.height, order="F"
        )

        self.fps_display = FPSDisplay(clock)
        self.menu_system = menu_system

    def render_all(self) -> None:
        self.game_map_console.clear()

        # Clear the root console first
        self.root_console.clear()

        # Render map and actors to game console
        self._render_map()
        self._render_actors()
        self._render_selected_actor_highlight()
        self._render_mouse_cursor_highlight()

        # Blit game console to root console (below help and message log)
        self.game_map_console.blit(
            dest=self.root_console,
            dest_x=0,
            dest_y=self.help_height
            + self.message_log_height,  # Start after help + message log (line 7)
            width=self.game_map_console.width,
            height=self.game_map_console.height,
        )
        # Render help text above message log
        if not self.menu_system.has_active_menus():
            self._render_help_text()

        # Render the message log below the help text
        self.message_log.render(
            console=self.root_console,
            x=self.message_log_x,
            y=self.message_log_y,
            width=self.message_log_width,
            height=self.message_log_height,
        )

        if self.fps_display.SHOW_FPS:
            self.fps_display.render(self, self.screen_width - 12, 0)

        # Render menus last (on top of everything else)
        self.menu_system.render(self.root_console)

        # Present the final console and handle vsync timing
        self.context.present(self.root_console, keep_aspect=True, integer_scaling=True)

    def _render_selected_actor_highlight(self) -> None:
        """Renders a highlight on the selected actor's tile by blending colors."""
        if self.gw.selected_actor:
            actor = self.gw.selected_actor
            # Only highlight if the actor is visible in FOV
            if (
                self.gw.game_map.visible[actor.x, actor.y]
                and
                # Ensure coordinates are within the game_map_console bounds
                0 <= actor.x < self.game_map_console.width
                and 0 <= actor.y < self.game_map_console.height
            ):
                # Define highlight properties
                target_color = colors.SELECTED_HIGHLIGHT
                alpha = SELECTION_HIGHLIGHT_ALPHA
                self._apply_blended_highlight(actor.x, actor.y, target_color, alpha)

    def _render_mouse_cursor_highlight(self) -> None:
        if not self.gw.mouse_tile_location_on_map:
            return

        mx, my = self.gw.mouse_tile_location_on_map

        # Bounds check for the game_map_console
        if not (
            0 <= mx < self.game_map_console.width
            and 0 <= my < self.game_map_console.height
        ):
            return

        target_highlight_color: colors.Color
        if self.gw.game_map.visible[mx, my]:
            # Tile is IN FOV - target a bright highlight
            target_highlight_color = colors.WHITE
        else:
            # Tile is OUTSIDE FOV - target a muted highlight
            target_highlight_color = colors.GREY
        # Alpha blending factor (0.0 = fully transparent, 1.0 = fully opaque)
        alpha = MOUSE_HIGHLIGHT_ALPHA
        self._apply_blended_highlight(mx, my, target_highlight_color, alpha)

    def _apply_blended_highlight(
        self, x: int, y: int, target_highlight_color: colors.Color, alpha: float
    ) -> None:
        """
        Blends a target highlight color with the existing background color at (x, y)
        on the game_map_console and applies it.
        """
        # Ensure coordinates are within the game_map_console bounds
        if not (
            0 <= x < self.game_map_console.width
            and 0 <= y < self.game_map_console.height
        ):
            return

        current_bg_color = self.game_map_console.rgb["bg"][x, y]

        # Perform alpha blending:
        # NewColor = TargetColor * alpha + CurrentColor * (1 - alpha)
        blended_color = [
            int(target_highlight_color[i] * alpha + current_bg_color[i] * (1.0 - alpha))
            for i in range(3)
        ]

        # Ensure colors stay within 0-255 range
        self.game_map_console.rgb["bg"][x, y] = [
            max(0, min(255, c)) for c in blended_color
        ]

    def _render_help_text(self) -> None:
        """Render helpful key bindings at the very top."""
        help_items = ["?: Help", "I: Inventory"]  # Start with always-available items

        # Conditionally add "Get items" prompt
        player_x, player_y = self.gw.player.x, self.gw.player.y
        if self.gw.has_pickable_items_at_location(player_x, player_y):
            help_items.append("G: Get items")

        help_text = " | ".join(help_items)
        help_x = 1
        help_y = 0
        self.root_console.print(help_x, help_y, help_text, fg=colors.GREY)

    def _render_map(self) -> None:
        # Start with unexplored (shroud)
        shroud = (ord(" "), (0, 0, 0), (0, 0, 0))
        self.game_map_console.rgb[:] = shroud

        # Show explored areas with dark graphics
        explored_mask = self.gw.game_map.explored
        self.game_map_console.rgb[explored_mask] = self.gw.game_map.tiles["dark"][
            explored_mask
        ]

        # Apply dynamic lighting to visible areas only
        visible_mask = self.gw.game_map.visible
        visible_y, visible_x = np.where(visible_mask)

        if len(visible_y) > 0:
            # Compute lighting for visible areas
            self.current_light_intensity = self.gw.lighting.compute_lighting(
                self.gw.game_map.width, self.gw.game_map.height
            )

            # Get the tile graphics for visible areas
            dark_tiles = self.gw.game_map.tiles["dark"][visible_y, visible_x]
            light_tiles = self.gw.game_map.tiles["light"][visible_y, visible_x]

            # Get light intensity for blending
            cell_light = self.current_light_intensity[visible_y, visible_x]

            # Create blended tiles
            blended_tiles = np.empty_like(dark_tiles)
            blended_tiles["ch"] = light_tiles["ch"]
            blended_tiles["fg"] = light_tiles["fg"]

            # Blend background colors based on light intensity
            for i in range(3):  # RGB channels
                light_intensity = cell_light[..., i]
                blended_tiles["bg"][..., i] = light_tiles["bg"][
                    ..., i
                ] * light_intensity + dark_tiles["bg"][..., i] * (1.0 - light_intensity)

            # Apply the dynamically lit tiles
            self.game_map_console.rgb[visible_y, visible_x] = blended_tiles

    def _render_actors(self) -> None:
        for a in self.gw.actors:
            if a == self.gw.player:
                continue

            # Only render actors that are visible
            if self.gw.game_map.visible[a.x, a.y]:
                self._render_actor(a)

        # Always draw the player last.
        self._render_actor(self.gw.player)

    def _render_actor(self, a: Actor) -> None:
        self.game_map_console.rgb["ch"][a.x, a.y] = ord(a.ch)

        # Calculate the base lit color of the actor
        base_actor_color: colors.Color = a.color

        # Apply a flash effect if appropriate.
        visual_effects = a.visual_effects
        if visual_effects:
            visual_effects.update()  # Update counter, clear if done
            flash_color = visual_effects.get_flash_color()
            if flash_color:
                base_actor_color = flash_color

        light_rgb = self.current_light_intensity[a.x, a.y]

        # Apply RGB lighting to each color channel of the base_actor_color
        # Ensure components are clamped to valid color range [0, 255]
        normally_lit_fg_components = [
            max(0, min(255, int(base_actor_color[i] * light_rgb[i]))) for i in range(3)
        ]

        final_fg_color = normally_lit_fg_components

        # If selected and in FOV, apply pulsation blending on top of the lit color
        if self.gw.selected_actor == a and self.gw.game_map.visible[a.x, a.y]:
            final_fg_color = self._apply_pulsating_effect(
                normally_lit_fg_components, base_actor_color
            )

        self.game_map_console.rgb["fg"][a.x, a.y] = final_fg_color

    def _apply_pulsating_effect(
        self, input_color: colors.Color, base_actor_color: colors.Color
    ) -> colors.Color:
        game_time = self.fps_display.clock.last_time
        # alpha_oscillation will go from 0.0 to 1.0 and back over PULSATION_PERIOD
        alpha_oscillation = (
            math.sin((game_time % PULSATION_PERIOD) / PULSATION_PERIOD * 2 * math.pi)
            + 1
        ) / 2.0

        # current_blend_alpha will oscillate from 0 to PULSATION_MAX_BLEND_ALPHA
        current_blend_alpha = alpha_oscillation * PULSATION_MAX_BLEND_ALPHA

        # Determine dynamic pulsation target color based on actor's base color
        r, g, b = base_actor_color

        # Calculate luminance using the Rec. 709 formula.
        # See: https://en.wikipedia.org/wiki/Luma_(video)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

        dynamic_pulsation_target_color: colors.Color = (
            colors.DARK_GREY if luminance > LUMINANCE_THRESHOLD else colors.LIGHT_GREY
        )

        # Blend input_color with dynamic_pulsation_target_color
        blended_r = int(
            dynamic_pulsation_target_color[0] * current_blend_alpha
            + input_color[0] * (1.0 - current_blend_alpha)
        )
        blended_g = int(
            dynamic_pulsation_target_color[1] * current_blend_alpha
            + input_color[1] * (1.0 - current_blend_alpha)
        )
        blended_b = int(
            dynamic_pulsation_target_color[2] * current_blend_alpha
            + input_color[2] * (1.0 - current_blend_alpha)
        )

        return (
            max(0, min(255, blended_r)),
            max(0, min(255, blended_g)),
            max(0, min(255, blended_b)),
        )

    def _render_text(
        self, x: int, y: int, text: str, fg: colors.Color = colors.WHITE
    ) -> None:
        """Render text at a specific position with a given color"""
        self.root_console.print(x=x, y=y, text=text, fg=fg)
