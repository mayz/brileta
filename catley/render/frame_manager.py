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
    SHOW_FPS,
)
from catley.game.actors import Actor
from catley.render.effects import EffectContext, EffectLibrary
from catley.render.old_render import FPSDisplay
from catley.render.particles import SubTileParticleSystem
from catley.render.screen_shake import ScreenShake
from catley.ui.cursor_manager import CursorManager
from catley.ui.message_log_panel import MessageLogPanel
from catley.ui.panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.render.renderer import Renderer


class FrameManager:
    """Coordinates panels and manages the frame lifecycle, including global effects."""

    def __init__(self, controller: Controller):
        self.controller = controller
        self.renderer: Renderer = controller.renderer
        self.cursor_manager = CursorManager(self.renderer)

        self.fps_display = FPSDisplay(controller.clock)

        # Global effect systems
        self.screen_shake = ScreenShake()
        self.particle_system = SubTileParticleSystem(
            controller.gw.game_map.width, controller.gw.game_map.height
        )
        self.effect_library = EffectLibrary()

        # FrameManager owns the game map console and layout constants
        self.help_height = HELP_HEIGHT
        self.game_map_console = Console(
            controller.gw.game_map.width, controller.gw.game_map.height, order="F"
        )

        self.current_light_intensity: np.ndarray | None = None

        # UI Components
        self.panels: list[Panel] = []

        # Create and add the message log panel
        self.add_panel(
            MessageLogPanel(
                message_log=self.controller.message_log,
                x=1,
                y=self.renderer.root_console.height - 5 - 1,
                width=30,
                height=5,
            )
        )

    def add_panel(self, panel: Panel) -> None:
        """Add a UI panel to be rendered each frame."""
        self.panels.append(panel)

    def render_frame(self, delta_time: float) -> None:
        """
        Main rendering pipeline that composites the final frame.

        Pipeline stages:
        1. Preparation - Clear buffers, update screen shake
        2. Game World - Render map, actors, and effects to an off-screen console
        3. Compositing - Blit the world console to the root console with shake
        4. UI Panels - Draw all registered UI panels (e.g., health, messages)
        5. Menus - Draw any active menus on top of everything
        6. Presentation - Draw custom cursor and display the final frame
        """
        # 1. PREPARATION PHASE
        self.renderer.clear_console(self.renderer.root_console)
        shake_x, shake_y = self.screen_shake.update(delta_time)

        # 2. GAME WORLD RENDERING (to game_map_console)
        self._render_game_world(delta_time)

        # 3. COMPOSITING PHASE
        # Apply screen effects and composite game world onto root console.
        # Compositing means combining multiple visual layers into a single image.
        # Here we take the game_map_console (containing the game world) and
        # combine it with the root_console, applying screen shake offset during
        # the combination process.
        self.renderer.blit_console(
            source=self.game_map_console,
            dest=self.renderer.root_console,
            dest_x=shake_x,
            dest_y=self.help_height + shake_y,
        )

        # 4. UI PANEL RENDERING
        # Only show game UI when menus aren't active
        if not self.controller.menu_system.has_active_menus():
            self._render_help_text()
            self._render_equipment_status()

            # Let active mode render its UI (e.g., "TARGETING" text)
            if self.controller.active_mode:
                self.controller.active_mode.render_ui(self.renderer.root_console)

        # Draw all registered UI panels
        for panel in self.panels:
            if panel.visible:
                panel.draw(self.renderer)

        # Debug/development overlays
        if SHOW_FPS:
            self.fps_display.render(self)

        # 5. MENU RENDERING
        self.controller.menu_system.render(self.renderer.root_console)

        # 6. PRESENTATION
        # Copy the final console state to the backbuffer
        self.renderer.prepare_to_present()

        # Allow panels and other systems to perform low-level SDL drawing
        self.cursor_manager.draw_cursor()
        for panel in self.panels:
            if panel.visible:
                panel.present(self.renderer.sdl_renderer)

        # Flip the backbuffer to the screen
        self.renderer.finalize_present()

    def get_tile_map_coords_from_root_coords(
        self, root_tile_coords: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Converts root console tile coordinates to game map tile coordinates."""
        map_render_offset_x = 0
        map_render_offset_y = self.help_height

        map_x = root_tile_coords[0] - map_render_offset_x
        map_y = root_tile_coords[1] - map_render_offset_y

        if (
            0 <= map_x < self.game_map_console.width
            and 0 <= map_y < self.game_map_console.height
        ):
            return map_x, map_y
        return None

    def highlight_actor(
        self, actor: Actor, color: colors.Color, effect: str = "solid"
    ) -> None:
        """Public API for highlighting actors if they're visible."""
        if self.controller.gw.game_map.visible[actor.x, actor.y]:
            self.highlight_tile(actor.x, actor.y, color, effect)

    def highlight_tile(
        self, x: int, y: int, color: colors.Color, effect: str = "solid"
    ) -> None:
        """Public API for highlighting tiles with optional effects."""
        if effect == "pulse":
            color = self._apply_pulsating_effect(color, color)
        self._apply_replacement_highlight(x, y, color)

    def trigger_screen_shake(self, intensity: float, duration: float) -> None:
        """Trigger screen shake effect. Call this from combat actions."""
        from catley.config import (
            SCREEN_SHAKE_ENABLED,
            SCREEN_SHAKE_INTENSITY_MULTIPLIER,
        )

        if not SCREEN_SHAKE_ENABLED:
            return
        scaled_intensity = intensity * SCREEN_SHAKE_INTENSITY_MULTIPLIER
        self.screen_shake.trigger(scaled_intensity, duration)

    def create_effect(
        self,
        effect_name: str,
        x: int,
        y: int,
        intensity: float = 1.0,
        direction_x: float = 0.0,
        direction_y: float = 0.0,
    ) -> None:
        """Unified interface for creating effects."""
        context = EffectContext(
            particle_system=self.particle_system,
            x=x,
            y=y,
            intensity=intensity,
            direction_x=direction_x,
            direction_y=direction_y,
        )
        self.effect_library.trigger(effect_name, context)

    def _render_game_world(self, delta_time: float) -> None:
        """Render all game world elements to the game_map_console."""
        self.renderer.clear_console(self.game_map_console)
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
        """Renders the game map tiles to the game_map_console."""
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
            # Compute lighting and store it so _render_actors can use it.
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
        """Renders all actors to the game_map_console, applying lighting."""
        gw = self.controller.gw
        for a in gw.actors:
            if a == gw.player:
                continue

            # Only render actors that are visible
            if gw.game_map.visible[a.x, a.y]:
                self._render_actor(a)

        # Always draw the player last.
        self._render_actor(gw.player)

    def _render_actor(self, a: Actor) -> None:
        """Draws a single actor, applying all visual effects and lighting."""
        if self.current_light_intensity is None:
            return  # Can't render actors without a light map

        self.game_map_console.rgb["ch"][a.x, a.y] = ord(a.ch)

        # Calculate the base lit color of the actor
        base_actor_color: colors.Color = a.color

        # Apply a flash effect if appropriate.
        visual_effects = a.visual_effects
        if visual_effects is not None:
            visual_effects.update()  # Update counter, clear if done
            flash_color = visual_effects.get_flash_color()
            if flash_color:
                base_actor_color = flash_color

        light_rgb = self.current_light_intensity[a.x, a.y]

        # Apply RGB lighting to each color channel of the base_actor_color
        normally_lit_fg_components: colors.Color = (
            max(0, min(255, int(base_actor_color[0] * light_rgb[0]))),
            max(0, min(255, int(base_actor_color[1] * light_rgb[1]))),
            max(0, min(255, int(base_actor_color[2] * light_rgb[2]))),
        )

        final_fg_color = normally_lit_fg_components

        # If selected and in FOV (but not in targeting mode), apply pulsation blending
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
        """Renders a highlight on the selected actor's tile by blending colors."""
        if self.controller.is_targeting_mode():
            return
        actor = self.controller.gw.selected_actor
        if actor and self.controller.gw.game_map.visible[actor.x, actor.y]:
            self._apply_blended_highlight(
                actor.x, actor.y, colors.SELECTED_HIGHLIGHT, SELECTION_HIGHLIGHT_ALPHA
            )

    def _render_mouse_cursor_highlight(self) -> None:
        """Renders a highlight on the tile under the mouse cursor."""
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
        """
        Blends a target highlight color with the existing background color at (x, y)
        on the game_map_console and applies it.
        """
        current_bg = self.game_map_console.rgb["bg"][x, y]
        blended_color = [
            int(target_color[i] * alpha + current_bg[i] * (1.0 - alpha))
            for i in range(3)
        ]
        self.game_map_console.rgb["bg"][x, y] = [
            max(0, min(255, c)) for c in blended_color
        ]

    def _apply_replacement_highlight(self, x: int, y: int, color: colors.Color) -> None:
        """Replace the background color entirely (no blending)."""
        self.game_map_console.rgb["bg"][x, y] = color

    def _apply_pulsating_effect(
        self, input_color: colors.Color, base_actor_color: colors.Color
    ) -> colors.Color:
        """Calculates a new color by blending with a pulsating highlight."""
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

    def _render_help_text(self) -> None:
        """Render helpful key bindings at the very top."""
        help_items = ["?: Help", "I: Inventory"]

        player_x, player_y = self.controller.gw.player.x, self.controller.gw.player.y
        if self.controller.gw.has_pickable_items_at_location(player_x, player_y):
            help_items.append("G: Get items")

        help_text = " | ".join(help_items)
        self.renderer.draw_text(1, 0, help_text, fg=colors.GREY)

    def _render_equipment_status(self) -> None:
        """Render equipment status showing all attack slots with active indicator"""
        y_start = self.renderer.root_console.height - 4
        player = self.controller.gw.player

        # Add weapon switching and reload hints
        hint_text = "Weapons: [1][2] to switch"
        active_weapon = player.inventory.get_active_weapon()
        if (
            active_weapon
            and active_weapon.ranged_attack
            and active_weapon.ranged_attack.current_ammo
            < active_weapon.ranged_attack.max_ammo
        ):
            hint_text += " | [R] to reload"

        self.renderer.draw_text(1, y_start - 1, hint_text, fg=colors.GREY)

        # Keep all the existing weapon display code
        for i, item in enumerate(player.inventory.attack_slots):
            if i >= 2:  # Only show first 2 slots
                break
            # Show which slot is "active"
            active_marker = ">" if i == player.inventory.active_weapon_slot else " "
            slot_name = f"{active_marker}{i + 1}"
            if item:
                item_text = f"{slot_name}: {item.name}"
                if item.ranged_attack:
                    item_text += (
                        f" [{item.ranged_attack.current_ammo}/"
                        f"{item.ranged_attack.max_ammo}]"
                    )
                color = (
                    colors.WHITE
                    if i == player.inventory.active_weapon_slot
                    else colors.LIGHT_GREY
                )
            else:
                item_text = f"{slot_name}: Empty"
                color = colors.GREY
            self.renderer.draw_text(1, y_start + i, item_text, fg=color)
