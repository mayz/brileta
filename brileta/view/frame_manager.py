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

from brileta import config
from brileta.events import (
    EffectEvent,
    ScreenShakeEvent,
    subscribe_to_event,
)
from brileta.types import (
    DeltaTime,
    InterpolationAlpha,
    Opacity,
    PixelCoord,
    RootConsoleTilePos,
    ViewportTileCoord,
    WorldTilePos,
)
from brileta.util.coordinates import Rect
from brileta.util.live_vars import MetricSpec, live_variable_registry

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
from .views.mini_map_view import MiniMapView
from .views.player_status_view import PlayerStatusView
from .views.world_view import WorldView

if TYPE_CHECKING:
    from brileta.controller import Controller

# Render sub-metrics (time.render.cpu_ms parent is declared in app.py).
_RENDER_METRICS: list[MetricSpec] = [
    MetricSpec("time.render.light_overlay_ms", "Light overlay rendering"),
    MetricSpec(
        "time.render.light_overlay_gpu_compose_ms",
        "GPU light overlay composition pass",
    ),
    MetricSpec(
        "time.render.light_overlay_gpu_readback_ms",
        "GPU light overlay readback/map time",
    ),
    MetricSpec(
        "time.render.actor_light_gpu_readback_ms",
        "GPU actor-light point-sample readback/map time",
    ),
    MetricSpec("time.render.map_unlit_ms", "Unlit map rendering"),
    MetricSpec("time.render.actor_shadows_ms", "Projected actor shadow rendering"),
    MetricSpec("time.render.actors_smooth_ms", "Smooth actor rendering"),
    MetricSpec("time.render.actors_traditional_ms", "Traditional actor rendering"),
    MetricSpec("time.render.present_background_ms", "Presenting background texture"),
    MetricSpec(
        "time.render.present_light_overlay_ms", "Presenting light overlay texture"
    ),
    MetricSpec("time.render.particles_under_actors_ms", "Particles under actors"),
    MetricSpec("time.render.particles_over_actors_ms", "Particles over actors"),
    MetricSpec("time.render.active_mode_world_ms", "Active mode world elements"),
    MetricSpec("time.render.environmental_effects_ms", "Environmental effects"),
    MetricSpec("time.render.bg_texture_upload_ms", "Background texture GPU upload"),
    MetricSpec(
        "time.render.light_texture_upload_ms", "Light overlay texture GPU upload"
    ),
    MetricSpec("time.render.atmospheric_ms", "Atmospheric effects (clouds, shadows)"),
    MetricSpec("time.render.decals_ms", "Decal rendering"),
    MetricSpec("time.render.floating_text_ms", "Floating text rendering"),
    MetricSpec("time.render.minimap_ms", "Mini-map draw/caching"),
    MetricSpec("time.render.actor_particles_ms", "Actor particle emitter check loop"),
    MetricSpec(
        "time.render.gpu_visible_texture_ms",
        "GPU visible texture upload (full-map)",
    ),
    MetricSpec(
        "time.render.gpu_explored_texture_ms",
        "GPU explored texture upload (full-map)",
    ),
    # TextureRenderer breakdown
    MetricSpec("time.render.texture.fbo_bind_clear_ms", "FBO bind and clear"),
    MetricSpec("time.render.texture.vbo_update_ms", "VBO vertex encode and upload"),
    MetricSpec("time.render.texture.render_ms", "Render pass"),
]
live_variable_registry.register_metrics(_RENDER_METRICS)


class FrameManager:
    """Coordinates views and manages the frame lifecycle, including global effects."""

    world_view: WorldView
    message_log_view: MessageLogView
    equipment_view: EquipmentView
    player_status_view: PlayerStatusView
    action_panel_view: ActionPanelView
    mini_map_view: MiniMapView
    views: list[View]
    debug_stats_overlay: DebugStatsOverlay
    dev_console_overlay: DevConsoleOverlay
    combat_tooltip_overlay: CombatTooltipOverlay

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
        self.mini_map_view = MiniMapView(
            self.controller, self.world_view.viewport_system
        )

        self.views: list[View] = [
            self.world_view,
            self.player_status_view,
            self.equipment_view,
            self.message_log_view,
            self.action_panel_view,
            self.mini_map_view,
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
            metric=True,
        )

        if config.SHOW_FPS:
            live_variable_registry.watch("dev.fps")

    def _layout_views(self) -> None:
        """Calculate and set view boundaries based on current screen size."""
        # Layout follows the renderer's current dynamic console dimensions.
        screen_width_tiles = self.graphics.console_width_tiles
        screen_height_tiles = self.graphics.console_height_tiles
        tile_dimensions = self.graphics.tile_dimensions

        # Reference layout at the historical baseline console size.
        # As the console shrinks (high zoom / smaller window), panel tile counts
        # should shrink faster than linearly to prevent UI from consuming most
        # of the available play area.
        reference_width_tiles = 80
        reference_height_tiles = 50
        reference_left_sidebar_width = 20
        reference_bottom_ui_height = 10

        layout_scale = min(
            1.0,
            screen_width_tiles / reference_width_tiles,
            screen_height_tiles / reference_height_tiles,
        )
        ui_scale = self._compute_ui_scale(layout_scale)

        min_world_width_tiles = 24
        min_world_height_tiles = 14
        preferred_left_sidebar_width = self._clamp_tiles(
            round(reference_left_sidebar_width * ui_scale),
            min_value=10,
            max_value=reference_left_sidebar_width,
        )
        left_sidebar_width = min(
            preferred_left_sidebar_width,
            max(0, screen_width_tiles - min_world_width_tiles),
        )
        left_sidebar_x = 0

        # Bottom UI bar uses the same scaling model as the sidebar.
        # Also cap by pixel height so zoom/DPI increases don't balloon this
        # region and leave excessive empty black space under compact HUD content.
        native_tile_height = max(
            1,
            int(getattr(self.graphics, "native_tile_size", tile_dimensions)[1]),
        )
        display_tile_height = max(1, tile_dimensions[1])
        reference_bottom_ui_height_px = reference_bottom_ui_height * native_tile_height
        pixel_capped_bottom_ui_height = self._clamp_tiles(
            round(reference_bottom_ui_height_px / display_tile_height),
            min_value=3,
            max_value=reference_bottom_ui_height,
        )
        preferred_bottom_ui_height = self._clamp_tiles(
            round(reference_bottom_ui_height * ui_scale),
            min_value=3,
            max_value=reference_bottom_ui_height,
        )
        bottom_ui_height = min(
            preferred_bottom_ui_height,
            pixel_capped_bottom_ui_height,
            max(0, screen_height_tiles - min_world_height_tiles),
        )
        bottom_ui_y = screen_height_tiles - bottom_ui_height
        # Reserve one extra row between world rendering and bottom HUD regions.
        # Smooth camera/sub-tile actor rendering can push glyphs fractionally
        # past the last visible world row; this guard row prevents glyph bottoms
        # from being clipped by the bottom UI when zoom is high.
        world_to_hud_guard_rows = 1 if bottom_ui_height > 0 else 0
        content_bottom_y = max(0, bottom_ui_y - world_to_hud_guard_rows)

        # Left sidebar content stack (action panel + optional mini-map)
        action_panel_y = 0

        # World view starts after left sidebar
        world_view_x = left_sidebar_width

        # Bottom bar components - dynamic equipment width based on content.
        # Equipment panel extends to the right screen edge to avoid uncovered
        # gap columns where the world texture would bleed through.
        equipment_width_requested = self.equipment_view.calculate_min_width()
        min_message_log_width = 8
        max_equipment_width = max(0, screen_width_tiles - min_message_log_width)
        if max_equipment_width > 0:
            equipment_width = min(equipment_width_requested, max_equipment_width)
        else:
            equipment_width = screen_width_tiles
        equipment_x2 = screen_width_tiles
        equipment_x1 = equipment_x2 - equipment_width

        # Message log spans bottom bar from left to equipment (no gap)
        message_log_x2 = equipment_x1

        # Set view bounds
        self.world_view.tile_dimensions = tile_dimensions
        self.world_view.set_bounds(
            world_view_x, 0, screen_width_tiles, content_bottom_y
        )

        # Player status view in upper-right (transparent HUD overlay)
        player_status_width = min(
            self._clamp_tiles(
                round(screen_width_tiles * 0.50),
                min_value=18,
                max_value=40,
            ),
            screen_width_tiles,
        )
        player_status_height = min(20, max(0, content_bottom_y))
        self.player_status_view.tile_dimensions = tile_dimensions
        self.player_status_view.set_bounds(
            screen_width_tiles - player_status_width,
            0,
            screen_width_tiles,
            player_status_height,
        )

        sidebar_width_px = left_sidebar_width * max(1, tile_dimensions[0])
        map_width_tiles = self.controller.gw.game_map.width
        map_height_tiles = self.controller.gw.game_map.height
        # Reserve enough vertical space for the largest integer mini-map scale
        # that can fit the sidebar's current pixel width.
        minimap_px_per_tile = max(1, max(0, sidebar_width_px - 2) // map_width_tiles)
        minimap_required_px = (map_height_tiles * minimap_px_per_tile) + 2
        tile_height_px = max(1, tile_dimensions[1])
        minimap_height_tiles = (
            minimap_required_px + tile_height_px - 1
        ) // tile_height_px

        sidebar_content_height = max(0, content_bottom_y - action_panel_y)
        min_action_panel_height_tiles = 12
        max_minimap_height_tiles = max(
            0, sidebar_content_height - min_action_panel_height_tiles
        )
        minimap_height_tiles = min(minimap_height_tiles, max_minimap_height_tiles)

        if self.mini_map_view.visible and minimap_height_tiles > 0:
            minimap_y1 = content_bottom_y - minimap_height_tiles
            action_panel_y2 = minimap_y1
            minimap_bounds = (
                left_sidebar_x,
                minimap_y1,
                left_sidebar_width,
                content_bottom_y,
            )
        else:
            action_panel_y2 = content_bottom_y
            minimap_bounds = (0, 0, 0, 0)

        # Action panel fills left sidebar above the mini-map when visible.
        self.action_panel_view.tile_dimensions = tile_dimensions
        self.action_panel_view.set_bounds(
            left_sidebar_x, action_panel_y, left_sidebar_width, action_panel_y2
        )

        self.mini_map_view.tile_dimensions = tile_dimensions
        self.mini_map_view.set_bounds(*minimap_bounds)

        # Message log spans bottom bar (left edge to equipment)
        self.message_log_view.tile_dimensions = tile_dimensions
        self.message_log_view.set_bounds(
            0, content_bottom_y, message_log_x2, screen_height_tiles
        )

        # Equipment view in bottom right
        self.equipment_view.tile_dimensions = tile_dimensions
        self.equipment_view.set_bounds(
            equipment_x1, content_bottom_y, equipment_x2, screen_height_tiles
        )

    @staticmethod
    def _clamp_tiles(value: int, min_value: int, max_value: int) -> int:
        """Clamp an integer tile count to an inclusive range."""
        return max(min_value, min(max_value, value))

    @staticmethod
    def _compute_ui_scale(layout_scale: float) -> float:
        """Return non-linear UI scaling factor from overall layout scale.

        This shrinks HUD panels faster than linear scaling, but less aggressively
        than a strict quadratic curve, improving intermediate-size behavior.
        """
        return layout_scale * (0.5 + 0.5 * layout_scale)

    def get_visible_bounds(self) -> Rect | None:
        """Return the world-space bounds currently visible in the world view."""
        return self.world_view.viewport_system.get_visible_bounds()

    def toggle_minimap(self) -> None:
        """Toggle mini-map visibility and recompute HUD layout."""
        if self.mini_map_view.visible:
            self.mini_map_view.hide()
        else:
            self.mini_map_view.show()
        self._layout_views()

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
            from brileta import (
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
                    int(px_x), int(px_y), int(px_w), int(px_h), color, Opacity(1.0)
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

    def pixel_to_world_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> WorldTilePos | None:
        """Convert display-scaled pixel coordinates to world tile coordinates.

        Unlike the raw ``pixel_to_tile`` -> ``get_world_coords`` pipeline, this
        method compensates for the camera's fractional scroll offset so that
        click and hover detection align with the visually rendered tile grid.

        Args:
            pixel_x: X position in display-scaled pixels (after applying
                ``get_display_scale_factor``).
            pixel_y: Y position in display-scaled pixels.

        Returns:
            The world tile coordinate under the cursor, or None if the pixel
            falls outside the game map area.
        """
        graphics = self.graphics

        # Compute the camera fractional offset in pixel space.
        # Smooth scrolling shifts all visual content by -cam_frac tiles during
        # presentation. Adding the equivalent pixel shift to the click position
        # undoes this before pixel_to_tile truncates to integer tiles.
        cam_frac_x, cam_frac_y = (
            self.world_view.viewport_system.get_camera_fractional_offset()
        )
        base_px_x, base_px_y = graphics.console_to_screen_coords(0.0, 0.0)
        frac_px_x, frac_px_y = graphics.console_to_screen_coords(cam_frac_x, cam_frac_y)

        adjusted_x = pixel_x + (frac_px_x - base_px_x)
        adjusted_y = pixel_y + (frac_px_y - base_px_y)

        root_tile_pos = graphics.pixel_to_tile(adjusted_x, adjusted_y)
        return self.get_world_coords_from_root_tile_coords(root_tile_pos)

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
