"""
Coordinates the main frame lifecycle, UI layout, and global effects.

The FrameManager is the top-level coordinator for the entire view system. It
instantiates and manages all the persistent `View`s and temporary `Overlay`s.

Key Responsibilities:
- Creates and owns a list of all primary `View`s (e.g., WorldView, MessageLogView).
- Manages the layout by calculating and setting the screen bounds for each `View`.
- Manages the `Overlay` stack, controlling which menus or dialogs are active.
- Orchestrates the main `render_frame` loop, telling all visible components
  to draw themselves in the correct order.
- Subscribes to and triggers global visual effects like screen shake.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from catley import config
from catley.config import (
    HELP_HEIGHT,
)
from catley.events import (
    EffectEvent,
    ScreenShakeEvent,
    subscribe_to_event,
)
from catley.util.coordinates import RootConsoleTilePos, ViewportTileCoord, WorldTilePos

from .render.effects.effects import EffectContext
from .render.effects.screen_shake import ScreenShake
from .render.renderer import Renderer
from .ui.cursor_manager import CursorManager
from .views.base import View
from .views.equipment_view import EquipmentView
from .views.health_view import HealthView
from .views.help_text_view import HelpTextView
from .views.message_log_view import MessageLogView
from .views.status_view import StatusView
from .views.world_view import WorldView

if TYPE_CHECKING:
    from catley.controller import Controller


class FrameManager:
    """Coordinates views and manages the frame lifecycle, including global effects."""

    def __init__(self, controller: Controller) -> None:
        """Initialize the frame manager and supporting systems."""
        self.controller = controller
        self.renderer: Renderer = controller.renderer
        self.cursor_manager = CursorManager(self.renderer)

        # Global effect systems
        self.screen_shake = ScreenShake()
        # Layout constants
        self.help_height = HELP_HEIGHT

        # Subscribe to visual effect events
        subscribe_to_event(EffectEvent, self._handle_effect_event)
        subscribe_to_event(ScreenShakeEvent, self._handle_screen_shake_event)

        self._setup_game_ui()

    def _setup_game_ui(self) -> None:
        """Configure and position views for the main game interface."""
        # Create views (dimensions will be set via resize() calls below)
        self.help_text_view = HelpTextView(self.controller, renderer=self.renderer)

        self.world_view = WorldView(self.controller, self.screen_shake)
        self.message_log_view = MessageLogView(
            message_log=self.controller.message_log,
            renderer=self.renderer,
        )
        self.equipment_view = EquipmentView(
            self.controller,
            renderer=self.renderer,
        )

        self.health_view = HealthView(self.controller, renderer=self.renderer)
        self.status_view = StatusView(self.controller, renderer=self.renderer)

        self.views: list[View] = [
            self.help_text_view,
            self.world_view,
            self.health_view,
            self.status_view,
            self.equipment_view,
            self.message_log_view,
        ]

        # Set view boundaries using layout
        self._layout_views()

    def _layout_views(self) -> None:
        """Calculate and set view boundaries based on current screen size."""
        screen_width_tiles = self.renderer.root_console.width
        screen_height_tiles = self.renderer.root_console.height
        tile_dimensions = self.renderer.tile_dimensions

        # Recalculate layout
        message_log_height = 10
        equipment_height = 4  # Equipment view needs 4 lines (2 weapons + 2 hints)
        bottom_ui_height = message_log_height + 1

        game_world_y = self.help_height
        game_world_height = screen_height_tiles - game_world_y - bottom_ui_height

        message_log_y = screen_height_tiles - message_log_height - 1
        equipment_y = screen_height_tiles - equipment_height - 2

        equipment_width = 25
        equipment_x = screen_width_tiles - equipment_width - 2

        # Status view positioned to the left of health view to avoid overlap
        status_view_width = 25
        status_view_x = screen_width_tiles - 20 - status_view_width - 1
        status_view_y = self.help_height + 1
        status_view_height = 10

        # Set bounds for all views
        self.help_text_view.tile_dimensions = tile_dimensions
        self.help_text_view.set_bounds(0, 0, screen_width_tiles, self.help_height)
        self.world_view.tile_dimensions = tile_dimensions
        self.world_view.set_bounds(
            0, game_world_y, screen_width_tiles, game_world_y + game_world_height
        )
        self.health_view.tile_dimensions = tile_dimensions
        self.health_view.set_bounds(
            screen_width_tiles - 20, 0, screen_width_tiles, self.help_height
        )
        self.status_view.tile_dimensions = tile_dimensions
        self.status_view.set_bounds(
            status_view_x,
            status_view_y,
            status_view_x + status_view_width,
            status_view_y + status_view_height,
        )
        self.equipment_view.tile_dimensions = tile_dimensions
        self.equipment_view.set_bounds(
            equipment_x, equipment_y, screen_width_tiles, screen_height_tiles
        )
        self.message_log_view.tile_dimensions = tile_dimensions
        self.message_log_view.set_bounds(1, message_log_y, 31, screen_height_tiles)

    def on_window_resized(self) -> None:
        """Called when the game window is resized to update view layouts."""
        self._layout_views()

    def add_view(self, view: View) -> None:
        """Add a UI view to be rendered each frame."""
        self.views.append(view)

    def render_frame(self, delta_time: float) -> None:
        """
        Main rendering pipeline that composites the final frame.

        Pipeline stages:
        1. Preparation - Clear buffers
        2. UI Views - Each view renders itself
        3. Menus - Draw any active menus on top of everything
        4. Presentation - Composite overlays then draw the mouse cursor
        """
        # 1. PREPARATION PHASE
        self.renderer.clear_console(self.renderer.root_console)
        # 2. UI VIEW RENDERING
        for view in self.views:
            if view.visible:
                view.draw(self.renderer)

        # Allow active mode to render its additional UI
        if self.controller.active_mode:
            self.controller.active_mode.render_ui(self.renderer.root_console)

        # Views may render overlays like FPS after game UI.

        # 3. OVERLAY DRAWING (texture generation phase)
        self.controller.overlay_system.draw_overlays()

        # 4. PRESENTATION
        # Copy the final console state to the backbuffer
        self.renderer.prepare_to_present()

        # Allow views and other systems to perform low-level SDL drawing
        for view in self.views:
            if view.visible:
                view.present(self.renderer)

        # 5. OVERLAY PRESENTATION (texture blitting phase)
        self.controller.overlay_system.present_overlays()

        # Draw the mouse cursor on top of all overlays
        self.cursor_manager.draw_cursor()

        # Flip the backbuffer to the screen
        self.renderer.finalize_present()

        self._maybe_show_action_processing_metrics()

    def get_world_coords_from_root_tile_coords(
        self, root_tile_pos: RootConsoleTilePos
    ) -> WorldTilePos | None:
        """Converts root console tile coordinates to game map tile coordinates."""
        root_x, root_y = root_tile_pos

        # Translate from root-console coordinates to viewport coordinates by
        # subtracting the view's screen position.
        vp_x: ViewportTileCoord = root_x - self.world_view.x
        vp_y: ViewportTileCoord = root_y - self.world_view.y

        # Ignore coordinates that fall outside the visible game world view.
        if not (
            0 <= vp_x < self.world_view.width and 0 <= vp_y < self.world_view.height
        ):
            return None

        # Use the viewport system to convert viewport coords to world map coords.
        world_x, world_y = self.world_view.viewport_system.screen_to_world(vp_x, vp_y)
        gw = self.controller.gw
        if 0 <= world_x < gw.game_map.width and 0 <= world_y < gw.game_map.height:
            return world_x, world_y
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
        """Create a visual effect if the world position is visible."""
        vs = self.world_view.viewport_system
        if not vs.is_visible(x, y):
            return
        vp_x, vp_y = vs.world_to_screen(x, y)
        context = EffectContext(
            particle_system=self.world_view.particle_system,
            environmental_system=self.world_view.environmental_system,
            x=vp_x,
            y=vp_y,
            intensity=intensity,
            direction_x=direction_x,
            direction_y=direction_y,
        )
        self.world_view.effect_library.trigger(effect_name, context)

    def _handle_effect_event(self, event: EffectEvent) -> None:
        """Handle effect events from the global event bus."""
        self.create_effect(
            event.effect_name,
            event.x,
            event.y,
            event.intensity,
            event.direction_x,
            event.direction_y,
        )

    def _handle_screen_shake_event(self, event: ScreenShakeEvent) -> None:
        """Handle screen shake events from the global event bus."""
        self.trigger_screen_shake(event.intensity, event.duration)

    def _maybe_show_action_processing_metrics(self) -> None:
        if (
            config.SHOW_ACTION_PROCESSING_METRICS
            and self.controller.last_input_time is not None
            and not self.controller.input_handler.movement_keys
            and not self.controller.turn_manager.has_pending_actions()
        ):
            action_count = self.controller.action_count_for_latency_metric
            if action_count > 0:
                total_latency = (
                    time.perf_counter() - self.controller.last_input_time
                ) * 1000
                avg_action_time = total_latency / action_count
                print(
                    f"Processed {action_count} actions in {total_latency:.2f} ms. "
                    f"Avg Action Processing Time: {avg_action_time:.2f} ms"
                )

            # Reset the state here after printing metrics
            self.controller.last_input_time = None
            self.controller.action_count_for_latency_metric = 0
