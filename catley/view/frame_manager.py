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
from catley.util.coordinates import Rect
from catley.util.live_vars import live_variable_registry

from .render.effects.effects import EffectContext
from .render.effects.screen_shake import ScreenShake
from .render.graphics import GraphicsContext
from .ui.combat_tooltip_overlay import CombatTooltipOverlay
from .ui.cursor_manager import CursorManager
from .ui.debug_stats_overlay import DebugStatsOverlay
from .ui.dev_console_overlay import DevConsoleOverlay
from .views.action_panel_view import ActionPanelView
from .views.base import View
from .views.equipment_view import EquipmentView
from .views.message_log_view import MessageLogView
from .views.player_status_view import PlayerStatusView
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

        # Subscribe to visual effect events
        subscribe_to_event(EffectEvent, self._handle_effect_event)
        subscribe_to_event(ScreenShakeEvent, self._handle_screen_shake_event)

        self._setup_game_ui()

    def _setup_game_ui(self) -> None:
        """Configure and position views for the main game interface."""
        assert self.controller is not None
        assert self.controller.overlay_system is not None

        # Create views (dimensions will be set via resize() calls below)
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

        self.player_status_view = PlayerStatusView(self.controller, self.graphics)
        self.action_panel_view = ActionPanelView(self.controller)

        self.views: list[View] = [
            self.world_view,
            self.player_status_view,
            self.equipment_view,
            self.message_log_view,
            self.action_panel_view,
        ]

        # Set view boundaries using layout
        self._layout_views()

        # Create and manage overlays owned by the frame manager
        self.debug_stats_overlay = DebugStatsOverlay(self.controller)
        self.controller.overlay_system.show_overlay(self.debug_stats_overlay)
        self.dev_console_overlay = DevConsoleOverlay(self.controller)
        self.combat_tooltip_overlay = CombatTooltipOverlay(self.controller)

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

        # Left sidebar dimensions (action panel only now)
        left_sidebar_width = 20
        left_sidebar_x = 0

        # Bottom bar dimensions
        bottom_ui_height = 10
        bottom_ui_y = screen_height_tiles - bottom_ui_height

        # Action panel fills entire left sidebar (above bottom bar)
        action_panel_y = 0

        # World view starts after left sidebar
        world_view_x = left_sidebar_width

        # Bottom bar components - dynamic equipment width based on content
        equipment_width = self.equipment_view.calculate_min_width()
        equipment_x1 = screen_width_tiles - equipment_width - 1
        equipment_x2 = equipment_x1 + equipment_width

        # Message log spans bottom bar from left to equipment
        message_log_x2 = equipment_x1 - 1

        # Set view bounds
        self.world_view.tile_dimensions = tile_dimensions
        self.world_view.set_bounds(world_view_x, 0, screen_width_tiles, bottom_ui_y)

        # Player status view in upper-right (transparent HUD overlay)
        self.player_status_view.tile_dimensions = tile_dimensions
        self.player_status_view.set_bounds(
            screen_width_tiles - 40, 0, screen_width_tiles, 20
        )

        # Action panel fills left sidebar (above bottom bar)
        self.action_panel_view.tile_dimensions = tile_dimensions
        self.action_panel_view.set_bounds(
            left_sidebar_x, action_panel_y, left_sidebar_width, bottom_ui_y
        )

        # Message log spans bottom bar (left edge to equipment)
        self.message_log_view.tile_dimensions = tile_dimensions
        self.message_log_view.set_bounds(
            0, bottom_ui_y, message_log_x2, screen_height_tiles
        )

        # Equipment view in bottom right
        self.equipment_view.tile_dimensions = tile_dimensions
        self.equipment_view.set_bounds(
            equipment_x1, bottom_ui_y, equipment_x2, screen_height_tiles
        )

    def get_visible_bounds(self) -> Rect | None:
        """Return the world-space bounds currently visible in the world view."""
        return self.world_view.viewport_system.get_visible_bounds()

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
        if self.controller.overlay_system is not None:
            self.controller.overlay_system.invalidate_all()

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

        # Check if equipment panel needs to grow (grow-only dynamic width)
        needed_width = self.equipment_view.calculate_min_width()
        if needed_width > self.equipment_view.width:
            self._layout_views()

        # UI VIEW RENDERING
        for view in self.views:
            if view.visible:
                view.draw(self.graphics, alpha)

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

                self.graphics.draw_rect_outline(
                    int(px_x), int(px_y), int(px_w), int(px_h), color, 1.0
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
        """Trigger screen shake effect.

        Args:
            intensity: Amplitude of shake in tiles (0.0-0.3 typical).
            duration: Duration of the shake in seconds.
        """
        if not config.SCREEN_SHAKE_ENABLED:
            return
        self.screen_shake.trigger(intensity, duration)

    def create_effect(
        self,
        effect_name: str,
        x: int,
        y: int,
        intensity: float = 1.0,
        direction_x: float = 0.0,
        direction_y: float = 0.0,
        ray_count: int | None = None,
    ) -> None:
        """Create a visual effect if the world position is visible."""
        vs = self.world_view.viewport_system
        if not vs.is_visible(x, y):
            return
        _vp_x, _vp_y = vs.world_to_screen(x, y)
        context = EffectContext(
            particle_system=self.world_view.particle_system,
            environmental_system=self.world_view.environmental_system,
            x=x,
            y=y,
            intensity=intensity,
            direction_x=direction_x,
            direction_y=direction_y,
            decal_system=self.world_view.decal_system,
            ray_count=ray_count,
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
            event.ray_count,
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
