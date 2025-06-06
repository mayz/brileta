from __future__ import annotations

from typing import TYPE_CHECKING

from catley.config import (
    HELP_HEIGHT,
    SHOW_FPS,
)
from catley.render.effects import EffectContext
from catley.render.screen_shake import ScreenShake
from catley.ui.cursor_manager import CursorManager
from catley.ui.panels.equipment_panel import EquipmentPanel
from catley.ui.panels.fps_panel import FPSPanel
from catley.ui.panels.game_world_panel import GameWorldPanel
from catley.ui.panels.health_panel import HealthPanel
from catley.ui.panels.help_text_panel import HelpTextPanel
from catley.ui.panels.message_log_panel import MessageLogPanel
from catley.ui.panels.panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.render.renderer import Renderer


class FrameManager:
    """Coordinates panels and manages the frame lifecycle, including global effects."""

    def __init__(self, controller: Controller) -> None:
        """Initialize the frame manager and supporting systems."""
        self.controller = controller
        self.renderer: Renderer = controller.renderer
        self.cursor_manager = CursorManager(self.renderer)

        # Global effect systems
        self.screen_shake = ScreenShake()
        # Layout constants
        self.help_height = HELP_HEIGHT

        self._setup_game_ui()

    def _setup_game_ui(self) -> None:
        """Configure and position panels for the main game interface."""
        # Layout calculations.

        screen_width_tiles = self.renderer.root_console.width
        screen_height_tiles = self.renderer.root_console.height

        # Calculate UI panel space requirements
        message_log_height = 7
        equipment_height = 4
        # Reserve space for bottom panels plus margins
        bottom_ui_height = message_log_height + 1

        # Game world gets dedicated space - not overlapped by bottom UI
        game_world_y = self.help_height
        game_world_height = screen_height_tiles - game_world_y - bottom_ui_height

        # Position bottom panels conservatively to ensure they fit
        # Use much larger margins to account for screen size differences
        message_log_y = screen_height_tiles - message_log_height - 5
        equipment_y = screen_height_tiles - equipment_height - 2

        equipment_width = 25  # Approximate width needed for equipment display
        equipment_x = screen_width_tiles - equipment_width - 2

        # Create and register panels.
        self.fps_panel = FPSPanel(self.controller.clock)
        self.fps_panel.visible = SHOW_FPS

        self.help_text_panel = HelpTextPanel(self.controller, y=0)

        self.game_world_panel = GameWorldPanel(
            self.controller, self.screen_shake, game_world_height=game_world_height
        )
        self.message_log_panel = MessageLogPanel(
            message_log=self.controller.message_log,
            x=1,
            y=message_log_y,
            width=30,
            height=message_log_height,
            tile_dimensions=self.renderer.tile_dimensions,
        )
        self.equipment_panel = EquipmentPanel(
            self.controller,
            x=equipment_x,
            y=equipment_y,
        )
        self.health_panel = HealthPanel(self.controller, y=0)

        self.panels: list[Panel] = [
            self.help_text_panel,
            self.game_world_panel,
            self.health_panel,
            self.equipment_panel,
            self.message_log_panel,
            self.fps_panel,
        ]

    def add_panel(self, panel: Panel) -> None:
        """Add a UI panel to be rendered each frame."""
        self.panels.append(panel)

    def render_frame(self, delta_time: float) -> None:
        """
        Main rendering pipeline that composites the final frame.

        Pipeline stages:
        1. Preparation - Clear buffers
        2. UI Panels - Each panel renders itself
        3. Menus - Draw any active menus on top of everything
        4. Presentation - Draw custom cursor and display the final frame
        """
        # 1. PREPARATION PHASE
        self.renderer.clear_console(self.renderer.root_console)

        # 2. UI PANEL RENDERING
        for panel in self.panels:
            if panel.visible:
                panel.draw(self.renderer)

        # Allow active mode to render its additional UI
        if self.controller.active_mode:
            self.controller.active_mode.render_ui(self.renderer.root_console)

        # Panels may render overlays like FPS after game UI.

        # 3. MENU RENDERING
        self.controller.menu_system.render(self.renderer.root_console)

        # 4. PRESENTATION
        # Copy the final console state to the backbuffer
        self.renderer.prepare_to_present()

        # Allow panels and other systems to perform low-level SDL drawing
        self.cursor_manager.draw_cursor()
        for panel in self.panels:
            if panel.visible:
                panel.present(self.renderer)

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
            0 <= map_x < self.game_world_panel.game_map_console.width
            and 0 <= map_y < self.game_world_panel.game_map_console.height
        ):
            return map_x, map_y
        return None

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
            particle_system=self.game_world_panel.particle_system,
            x=x,
            y=y,
            intensity=intensity,
            direction_x=direction_x,
            direction_y=direction_y,
        )
        self.game_world_panel.effect_library.trigger(effect_name, context)
