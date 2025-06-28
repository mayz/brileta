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
from typing import TYPE_CHECKING, cast

from catley import config
from catley.backends.tcod.graphics import TCODGraphicsContext
from catley.config import (
    HELP_HEIGHT,
)
from catley.events import (
    EffectEvent,
    ScreenShakeEvent,
    subscribe_to_event,
)
from catley.types import (
    DeltaTime,
    InterpolationAlpha,
    RootConsoleTilePos,
    ViewportTileCoord,
    WorldTilePos,
)
from catley.util.live_vars import live_variable_registry

from .render.effects.effects import EffectContext
from .render.effects.screen_shake import ScreenShake
from .render.graphics import GraphicsContext
from .ui.cursor_manager import CursorManager
from .ui.debug_stats_overlay import DebugStatsOverlay
from .ui.dev_console_overlay import DevConsoleOverlay
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

    def __init__(self, controller: Controller, graphics: GraphicsContext) -> None:
        """Initialize the frame manager and supporting systems."""
        self.controller = controller

        self.graphics = graphics
        self.cursor_manager = CursorManager()

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
        assert self.controller is not None
        assert self.controller.overlay_system is not None

        # Create views (dimensions will be set via resize() calls below)
        self.help_text_view = HelpTextView(self.controller)

        self.world_view = WorldView(
            self.controller,
            self.screen_shake,
            self.controller.gw.lighting_system,
        )
        self.message_log_view = MessageLogView(
            self.controller.message_log,
            self.graphics,
        )
        self.equipment_view = EquipmentView(
            self.controller,
            self.graphics,
        )

        self.health_view = HealthView(self.controller, self.graphics)
        self.status_view = StatusView(self.controller)

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

        # Create and manage overlays owned by the frame manager
        self.debug_stats_overlay = DebugStatsOverlay(self.controller)
        self.controller.overlay_system.show_overlay(self.debug_stats_overlay)
        self.dev_console_overlay = DevConsoleOverlay(self.controller)

        # Register live variables owned by the frame manager
        live_variable_registry.register(
            "dev.fps",
            getter=lambda: self.controller.clock.mean_fps,
            formatter=lambda v: f"{v:4.0f}",
            description="Current frames per second.",
        )

        if config.SHOW_FPS:
            live_variable_registry.watch("dev.fps")

    def _layout_views(self) -> None:
        """Calculate and set view boundaries based on current screen size."""
        # Use the configured screen dimensions so the entire UI maintains
        # a consistent aspect ratio regardless of window size.
        screen_width_tiles = config.SCREEN_WIDTH
        screen_height_tiles = config.SCREEN_HEIGHT
        tile_dimensions = self.graphics.tile_dimensions

        # Consolidated bottom bar dimensions
        bottom_ui_height = 10
        bottom_ui_y = screen_height_tiles - bottom_ui_height

        # World view is everything between the top bar and bottom bar
        game_world_y = self.help_height

        # Fixed widths for bottom bar components
        message_log_width = 30
        equipment_width = 25

        message_log_x1 = 1
        message_log_x2 = message_log_x1 + message_log_width

        equipment_x1 = screen_width_tiles - equipment_width - 1
        equipment_x2 = equipment_x1 + equipment_width

        status_x1 = message_log_x2 + 1
        status_x2 = equipment_x1 - 1

        self.help_text_view.tile_dimensions = tile_dimensions
        self.help_text_view.set_bounds(0, 0, screen_width_tiles, self.help_height)

        self.world_view.tile_dimensions = tile_dimensions
        self.world_view.set_bounds(0, game_world_y, screen_width_tiles, bottom_ui_y)

        self.health_view.tile_dimensions = tile_dimensions
        self.health_view.set_bounds(
            screen_width_tiles - 20, 0, screen_width_tiles, self.help_height
        )

        self.status_view.tile_dimensions = tile_dimensions
        self.status_view.set_bounds(
            status_x1, bottom_ui_y, status_x2, screen_height_tiles
        )

        self.equipment_view.tile_dimensions = tile_dimensions
        self.equipment_view.set_bounds(
            equipment_x1, bottom_ui_y, equipment_x2, screen_height_tiles
        )

        self.message_log_view.tile_dimensions = tile_dimensions
        self.message_log_view.set_bounds(
            message_log_x1, bottom_ui_y, message_log_x2, screen_height_tiles
        )

    def register_metrics(self) -> None:
        """Registers live variables specific to the App layer."""
        live_variable_registry.register_metric(
            "cpu.render_ms",
            description="CPU time for rendering",
            num_samples=500,
        )
        live_variable_registry.register_metric(
            "cpu.render.light_overlay_ms",
            description="CPU time for light overlay rendering",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.map_unlit_ms",
            description="CPU time for unlit map rendering",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.actors_smooth_ms",
            description="CPU time for smooth actor rendering",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.actors_traditional_ms",
            description="CPU time for traditional actor rendering",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.selected_actor_highlight_ms",
            description="CPU time for selected actor highlight rendering",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.present_background_ms",
            description="CPU time for presenting background texture",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.present_light_overlay_ms",
            description="CPU time for presenting light overlay texture",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.particles_under_actors_ms",
            description="CPU time for rendering particles under actors",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.particles_over_actors_ms",
            description="CPU time for rendering particles over actors",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.active_mode_world_ms",
            description="CPU time for rendering active mode world elements",
            num_samples=100,
        )
        live_variable_registry.register_metric(
            "cpu.render.environmental_effects_ms",
            description="CPU time for rendering environmental effects",
            num_samples=100,
        )

    def on_window_resized(self) -> None:
        """Called when the game window is resized to update view layouts."""
        self._layout_views()

    def add_view(self, view: View) -> None:
        """Add a UI view to be rendered each frame."""
        self.views.append(view)

    def render_frame(self, alpha: InterpolationAlpha) -> None:
        """
        Main rendering pipeline that composites the final frame.

        Pipeline stages:
        - UI Views - Each view renders itself
        - Menus - Draw any active menus on top of everything
        - Presentation - Composite overlays then draw the mouse cursor
        """
        assert self.controller.overlay_system is not None

        # UI VIEW RENDERING
        for view in self.views:
            if view.visible:
                view.draw(self.graphics, alpha)

        # Allow any active mode to render its additional UI
        if self.controller.active_mode:
            # FIXME: This needs to be adapted. It can't assume a TCOD console.
            # A better way would be for the mode to have its own canvas,
            # which gets drawn like any other overlay.
            # For now, we accept this as technical debt.
            tcod_graphics = cast(TCODGraphicsContext, self.graphics)
            self.controller.active_mode.render_ui(tcod_graphics.root_console)

        # Views may render overlays like FPS after game UI.

        # OVERLAY DRAWING (texture generation phase)
        self.controller.overlay_system.draw_overlays()

        # PRESENTATION
        # Allow views and other systems to perform low-level SDL drawing
        for view in self.views:
            if view.visible:
                view.present(self.graphics, alpha)

        # OVERLAY PRESENTATION (texture blitting phase)
        self.controller.overlay_system.present_overlays()

        # Draw view outlines if enabled
        if config.DEBUG_DRAW_VIEW_OUTLINES:
            from catley import (
                colors as debug_colors,
            )  # Import locally to avoid circular dependency issues

            for i, view in enumerate(self.views):
                if not view.visible:
                    continue

                # Convert tile-based view bounds to pixel coordinates
                px_x, px_y = self.graphics.console_to_screen_coords(view.x, view.y)
                px_x2, px_y2 = self.graphics.console_to_screen_coords(
                    view.x + view.width, view.y + view.height
                )
                px_w = px_x2 - px_x
                px_h = px_y2 - px_y

                # Pick a color from the debug palette, wrapping around if needed
                color = debug_colors.DEBUG_COLORS[i % len(debug_colors.DEBUG_COLORS)]

                self.graphics.draw_debug_rect(
                    int(px_x), int(px_y), int(px_w), int(px_h), color
                )

        # Draw the mouse cursor on top of all overlays
        self.graphics.draw_mouse_cursor(self.cursor_manager)

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

    def trigger_screen_shake(self, intensity: float, duration: DeltaTime) -> None:
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
        assert self.controller.input_handler is not None

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
