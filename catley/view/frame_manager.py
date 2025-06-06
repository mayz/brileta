from __future__ import annotations

from typing import TYPE_CHECKING

from catley.config import (
    HELP_HEIGHT,
    SHOW_FPS,
)

from .panels.equipment_panel import EquipmentPanel
from .panels.fps_panel import FPSPanel
from .panels.game_world_panel import GameWorldPanel
from .panels.health_panel import HealthPanel
from .panels.help_text_panel import HelpTextPanel
from .panels.message_log_panel import MessageLogPanel
from .panels.panel import Panel
from .render.effects import EffectContext
from .render.renderer import Renderer
from .render.screen_shake import ScreenShake
from .ui.cursor_manager import CursorManager

if TYPE_CHECKING:
    from catley.controller import Controller


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
        # Create panels (dimensions will be set via resize() calls below)
        self.fps_panel = FPSPanel(self.controller.clock)
        self.fps_panel.visible = SHOW_FPS

        self.help_text_panel = HelpTextPanel(self.controller)

        self.game_world_panel = GameWorldPanel(self.controller, self.screen_shake)
        self.message_log_panel = MessageLogPanel(
            message_log=self.controller.message_log,
            tile_dimensions=self.renderer.tile_dimensions,
        )
        self.equipment_panel = EquipmentPanel(self.controller)

        self.health_panel = HealthPanel(self.controller)

        self.panels: list[Panel] = [
            self.help_text_panel,
            self.game_world_panel,
            self.health_panel,
            self.equipment_panel,
            self.message_log_panel,
            self.fps_panel,
        ]

        # Set panel boundaries using resize()
        self._resize_panels()

    def _resize_panels(self) -> None:
        """Calculate and set panel boundaries based on current screen size."""
        screen_width_tiles = self.renderer.root_console.width
        screen_height_tiles = self.renderer.root_console.height

        # Recalculate layout
        message_log_height = 10
        equipment_height = 4  # Equipment panel needs 4 lines (2 weapons + 2 hints)
        bottom_ui_height = message_log_height + 1

        game_world_y = self.help_height
        game_world_height = screen_height_tiles - game_world_y - bottom_ui_height

        message_log_y = screen_height_tiles - message_log_height - 1
        equipment_y = screen_height_tiles - equipment_height - 2

        equipment_width = 25
        equipment_x = screen_width_tiles - equipment_width - 2

        # Resize all panels
        self.help_text_panel.resize(0, 0, screen_width_tiles, self.help_height)
        self.game_world_panel.resize(
            0, game_world_y, screen_width_tiles, game_world_y + game_world_height
        )
        self.health_panel.resize(
            screen_width_tiles - 20, 0, screen_width_tiles, self.help_height
        )
        self.equipment_panel.resize(
            equipment_x, equipment_y, screen_width_tiles, screen_height_tiles
        )
        self.message_log_panel.resize(1, message_log_y, 31, screen_height_tiles - 1)
        self.fps_panel.resize(0, 0, 15, 3)

    def on_window_resized(self) -> None:
        """Called when the game window is resized to update panel layouts."""
        self._resize_panels()

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
        self.controller.overlay_system.render(self.renderer.root_console)

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
