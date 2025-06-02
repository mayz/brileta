import math
from typing import TYPE_CHECKING

import numpy as np
import tcod.render
import tcod.sdl.render
from tcod.console import Console

from catley import colors
from catley.config import (
    HELP_HEIGHT,
    LUMINANCE_THRESHOLD,
    MESSAGE_LOG_HEIGHT,
    MOUSE_HIGHLIGHT_ALPHA,
    PERFORMANCE_TESTING,
    PULSATION_MAX_BLEND_ALPHA,
    PULSATION_PERIOD,
    SCREEN_SHAKE_ENABLED,
    SCREEN_SHAKE_INTENSITY_MULTIPLIER,
    SELECTION_HIGHLIGHT_ALPHA,
    SHOW_FPS,
)
from catley.game.actors import Actor
from catley.render.particles import SubTileParticleSystem
from catley.render.screen_shake import ScreenShake
from catley.ui.message_log import MessageLog
from catley.util.clock import Clock
from catley.util.coordinates import CoordinateConverter
from catley.world.game_state import GameWorld

if TYPE_CHECKING:
    from catley.ui.menu_core import MenuSystem


class FPSDisplay:
    def __init__(self, clock: Clock, update_interval: float = 0.5) -> None:
        self.clock = clock
        self.update_interval = update_interval
        self.last_update = clock.last_time
        self.display_string = "FPS: 0.0"

    def update(self) -> None:
        current_time = self.clock.last_time
        if current_time - self.last_update >= self.update_interval:
            if PERFORMANCE_TESTING:
                self.display_string = (
                    f"FPS: mean {self.clock.mean_fps:.0f}, "
                    f"last {self.clock.last_fps:.0f}"
                )
            else:
                self.display_string = f"FPS: {self.clock.mean_fps:.1f}"

            self.last_update = current_time

    def render(self, renderer: "Renderer") -> None:
        self.update()

        fps_width = len(self.display_string)
        x_position = max(0, renderer.screen_width - fps_width - 1)
        renderer._render_text(x_position, 0, self.display_string, fg=colors.YELLOW)


class Renderer:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        game_world: GameWorld,
        clock: Clock,
        message_log: MessageLog,
        menu_system: "MenuSystem",
        context: tcod.context.Context,
        root_console: Console,
        tile_dimensions: tuple[int, int],
    ) -> None:
        # Extract SDL components from context
        sdl_renderer = context.sdl_renderer
        assert sdl_renderer is not None

        sdl_atlas = context.sdl_atlas
        assert sdl_atlas is not None

        self.context = context
        self.sdl_renderer: tcod.sdl.render.Renderer = sdl_renderer
        self.console_render = tcod.render.SDLConsoleRender(sdl_atlas)
        self.root_console = root_console

        self.tile_dimensions = tile_dimensions

        # Set up coordinate conversion
        renderer_width, renderer_height = self.sdl_renderer.output_size
        self.coordinate_converter = CoordinateConverter(
            console_width=root_console.width,
            console_height=root_console.height,
            tile_width=self.tile_dimensions[0],
            tile_height=self.tile_dimensions[1],
            renderer_width=renderer_width,
            renderer_height=renderer_height,
        )

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
        self.screen_shake = ScreenShake()

        self.particle_system = SubTileParticleSystem(
            self.gw.game_map.width, self.gw.game_map.height
        )

    def render_all(self) -> None:
        """
        Main rendering pipeline that composites the final frame.

        Pipeline stages:
        1. Preparation - Clear buffers, calculate timing
        2. Game World - Render static and dynamic game elements
        3. Effects & Compositing - Apply screen effects and combine layers
        4. UI Overlays - Render interface elements on top
        5. Presentation - Display the final result
        """
        # 1. PREPARATION PHASE
        delta_time = self._prepare_frame()

        # 2. GAME WORLD RENDERING (to game_map_console)
        self._render_game_world(delta_time)

        # 3. DYNAMIC EFFECTS & COMPOSITING
        self._apply_effects_and_composite(delta_time)

        # 4. UI OVERLAY RENDERING (to root_console)
        self._render_ui_overlays()

        # 5. PRESENTATION
        self._present_frame()

    def _prepare_frame(self) -> float:
        """Clear rendering buffers and calculate frame timing."""
        # Clear both rendering targets
        self.game_map_console.clear()
        self.root_console.clear()

        # Calculate delta time for this frame (used by multiple systems)
        return getattr(self.fps_display.clock, "last_delta_time", 1 / 60)

    def _render_game_world(self, delta_time: float) -> None:
        """Render all game world elements to the game_map_console."""
        # Render the static world (map, actors, selection indicators)
        self._render_map()
        self._render_actors()
        self._render_selected_actor_highlight()
        self._render_mouse_cursor_highlight()

        # Render dynamic effects (particles) on top of the world
        self.particle_system.update(delta_time)
        self.particle_system.render_to_console(self.game_map_console)

    def _apply_effects_and_composite(self, delta_time: float) -> None:
        """Apply screen effects and composite game world onto root console.

        Compositing means combining multiple visual layers into a single image.
        Here we take the game_map_console (containing the game world) and
        combine it with the root_console, applying screen shake offset during
        the combination process.
        """
        # Calculate screen shake effect
        shake_x, shake_y = self.screen_shake.update(delta_time)

        # Composite: Blit game world onto root console with shake offset
        # This is where we combine the game world layer with the UI layer
        self.game_map_console.blit(
            dest=self.root_console,
            dest_x=shake_x,
            dest_y=self.help_height + self.message_log_height + shake_y,
            width=self.game_map_console.width,
            height=self.game_map_console.height,
        )

    def _render_ui_overlays(self) -> None:
        """Render all user interface elements on top of the game world."""
        # Only show game UI when menus aren't active
        if not self.menu_system.has_active_menus():
            self._render_help_text()
            self.render_equipment_status()

        # Always show message log (it can show combat results even with menus)
        self.message_log.render(
            console=self.root_console,
            x=self.message_log_x,
            y=self.message_log_y,
            width=self.message_log_width,
            height=self.message_log_height,
        )

        # Debug/development overlays
        if SHOW_FPS:
            self.fps_display.render(self)

        # Menus render last so they appear on top of everything
        self.menu_system.render(self.root_console)

    def _present_frame(self) -> None:
        """Present the final composited frame via SDL."""
        # Convert TCOD console to SDL texture
        console_texture = self.console_render.render(self.root_console)

        # Copy texture to screen and present
        self.sdl_renderer.clear()
        self.sdl_renderer.copy(console_texture)
        self.sdl_renderer.present()

    def trigger_screen_shake(self, intensity: float, duration: float = 0.3):
        """Trigger screen shake effect. Call this from combat actions."""
        if not SCREEN_SHAKE_ENABLED:
            return

        scaled_intensity = intensity * SCREEN_SHAKE_INTENSITY_MULTIPLIER
        self.screen_shake.trigger(scaled_intensity, duration)

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

        # Get the cached appearance maps.
        dark_app_map = self.gw.game_map.dark_appearance_map
        light_app_map = self.gw.game_map.light_appearance_map

        # Show explored areas with dark graphics
        explored_mask = self.gw.game_map.explored
        self.game_map_console.rgb[explored_mask] = dark_app_map[explored_mask]

        # Apply dynamic lighting to visible areas only
        visible_mask = self.gw.game_map.visible
        visible_y, visible_x = np.where(visible_mask)

        if len(visible_y) > 0:
            # Compute lighting for visible areas
            self.current_light_intensity = self.gw.lighting.compute_lighting(
                self.gw.game_map.width, self.gw.game_map.height
            )

            # Get the tile graphics for visible areas
            dark_tiles_visible = dark_app_map[visible_y, visible_x]
            light_tiles_visible = light_app_map[visible_y, visible_x]

            # Get light intensity for blending
            cell_light = self.current_light_intensity[visible_y, visible_x]

            # Create blended tiles
            blended_tiles = np.empty_like(dark_tiles_visible)
            blended_tiles["ch"] = light_tiles_visible["ch"]
            blended_tiles["fg"] = light_tiles_visible["fg"]

            # Blend background colors based on light intensity
            for i in range(3):  # RGB channels
                light_intensity_channel = cell_light[..., i]
                blended_tiles["bg"][..., i] = light_tiles_visible["bg"][
                    ..., i
                ] * light_intensity_channel + dark_tiles_visible["bg"][..., i] * (
                    1.0 - light_intensity_channel
                )

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
        normally_lit_fg_components: colors.Color = (
            max(0, min(255, int(base_actor_color[0] * light_rgb[0]))),
            max(0, min(255, int(base_actor_color[1] * light_rgb[1]))),
            max(0, min(255, int(base_actor_color[2] * light_rgb[2]))),
        )

        final_fg_color = normally_lit_fg_components

        # If selected and in FOV, apply pulsation blending on top of the lit color
        if self.gw.selected_actor == a and self.gw.game_map.visible[a.x, a.y]:
            final_fg_color = self._apply_pulsating_effect(
                normally_lit_fg_components, base_actor_color
            )

        self.game_map_console.rgb["fg"][a.x, a.y] = final_fg_color

    def render_equipment_status(self):
        """Render equipment status showing all attack slots"""
        y_start = self.screen_height - 4  # Make room for more slots

        player = self.gw.player
        for i, item in enumerate(player.inventory.attack_slots):
            if i >= 3:  # Only show first 3 slots to avoid screen overflow
                break

            slot_name = player.inventory.get_slot_display_name(i)

            if item:
                item_text = f"{slot_name}: {item.name}"
                if item.ranged_attack:
                    item_text += (
                        f" [{item.ranged_attack.current_ammo}/"
                        f"{item.ranged_attack.max_ammo}]"
                    )

                color = colors.WHITE
            else:
                item_text = f"{slot_name}: Empty"
                color = colors.GREY

            self.root_console.print(1, y_start + i, item_text, fg=color)

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

    def convert_mouse_coordinates(self, pixel_x: int, pixel_y: int) -> tuple[int, int]:
        return self.coordinate_converter.pixel_to_tile(pixel_x, pixel_y)
