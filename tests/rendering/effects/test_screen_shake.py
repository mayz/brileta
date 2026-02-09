from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np

from brileta import config
from brileta.controller import Controller
from brileta.types import DeltaTime, InterpolationAlpha
from brileta.util.spatial import SpatialHashGrid
from brileta.view.render.effects.screen_shake import ScreenShake
from brileta.view.render.graphics import GraphicsContext


class DummyActor:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.ch = "@"
        self.color = (255, 255, 255)
        self.render_x = float(x)
        self.render_y = float(y)
        self.visual_effects = None
        self.blocks_movement = True


def test_screen_shake_update_and_completion() -> None:
    shake = ScreenShake()
    shake.trigger(intensity=0.2, duration=DeltaTime(1.0))

    # First update - should return float offsets
    offset = shake.update(DeltaTime(0.5))
    assert isinstance(offset[0], float)
    assert isinstance(offset[1], float)
    # With amplitude 0.2 and some fade, offsets should be non-zero
    assert offset != (0.0, 0.0)
    assert shake.is_active()

    # After duration expires
    offset = shake.update(DeltaTime(0.6))
    assert offset == (0.0, 0.0)
    assert not shake.is_active()


def test_screen_shake_trigger_overwrite() -> None:
    shake = ScreenShake()
    shake.trigger(intensity=0.2, duration=DeltaTime(0.5))
    shake.trigger(intensity=0.8, duration=DeltaTime(1.0))
    assert shake.intensity == 0.8
    assert shake.duration == 1.0
    assert shake.time_remaining == 1.0


def test_screen_shake_amplitude_bounds() -> None:
    """Verify shake amplitude stays within expected bounds."""
    shake = ScreenShake()
    shake.trigger(intensity=0.3, duration=DeltaTime(1.0))

    # Sample multiple frames
    max_offset = 0.0
    for _ in range(100):
        x, y = shake.update(DeltaTime(0.01))
        max_offset = max(max_offset, abs(x), abs(y))

    # Offset should never exceed intensity (amplitude)
    # The sine wave sum has max value of 1.0, so offset <= intensity
    assert max_offset <= 0.3 + 0.01  # Small epsilon for float precision


class DummyGameMap:
    def __init__(self, width: int, height: int) -> None:
        from brileta.environment import tile_types
        from brileta.environment.map import TileAnimationState

        self.width = width
        self.height = height
        self.structural_revision = 0
        self.exploration_revision = 0
        self.dark_appearance_map = np.zeros(
            (width, height), dtype=tile_types.TileTypeAppearance
        )
        self.light_appearance_map = np.zeros(
            (width, height), dtype=tile_types.TileTypeAppearance
        )
        self.tiles = np.zeros((width, height), dtype=np.uint8)
        self.decoration_seed = 0
        self.explored = np.zeros((width, height), dtype=bool)
        self.visible = np.zeros((width, height), dtype=bool)
        # Animation properties for animated tile rendering
        self.animation_params = np.zeros(
            (width, height), dtype=tile_types.TileAnimationParams
        )
        self.animation_state = np.zeros((width, height), dtype=TileAnimationState)


class DummyGW:
    def __init__(self) -> None:
        self.player = DummyActor(0, 0)
        self.actors = [self.player]
        self.actor_spatial_index = SpatialHashGrid(cell_size=16)
        self.actor_spatial_index.add(self.player)
        self.game_map = DummyGameMap(10, 10)
        self.game_map.visible[0, 0] = True
        self.selected_actor = None
        self.mouse_tile_location_on_map = None
        self.lighting = MagicMock()
        self.lighting.compute_lighting_with_shadows.side_effect = (
            lambda w, h, *_args, **_kwargs: 1.0 * np.ones((w, h, 3))
        )
        self.lighting._generate_cache_key = MagicMock(return_value="test_lighting_key")
        self.lighting_system = MagicMock()
        self.lighting_system.revision = 0

    def add_actor(self, actor: DummyActor) -> None:
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)

    def get_actor_at_location(self, x: int, y: int) -> DummyActor | None:
        actors_at_point = self.actor_spatial_index.get_at_point(x, y)
        if not actors_at_point:
            return None
        for actor in actors_at_point:
            if getattr(actor, "blocks_movement", False):
                return actor
        return actors_at_point[0]


class DummyController:
    def __init__(self) -> None:
        from types import SimpleNamespace

        self.gw = DummyGW()
        self.graphics = SimpleNamespace(
            clear_console=lambda *args, **kwargs: None,
            root_console=None,
            blit_console=lambda *args, **kwargs: None,
            texture_from_console=lambda console: f"mock_texture_for_{id(console)}",
            render_glyph_buffer_to_texture=lambda gb, **kw: (
                f"mock_texture_for_{id(gb)}"
            ),
            present_texture=lambda *args, **kwargs: None,
            pixel_to_tile=lambda x, y: (int(x), int(y)),
            set_noise_seed=lambda seed: None,
            set_noise_tile_offset=lambda offset_x, offset_y: None,
        )
        self.frame_manager = SimpleNamespace(
            cursor_manager=SimpleNamespace(mouse_pixel_x=0, mouse_pixel_y=0),
            get_world_coords_from_root_tile_coords=lambda pos: pos,
        )
        self.clock = SimpleNamespace(last_delta_time=0.016)
        self.active_mode = None
        self.is_combat_mode = lambda: False


def test_world_view_applies_screen_shake_before_render(monkeypatch) -> None:
    """Test that shake offset is stored in draw() and applied in present()."""
    from brileta.view.views.world_view import WorldView

    controller = DummyController()
    shake = ScreenShake()
    # Use a float offset (0.15 tiles) to test sub-tile shake
    shake.update = lambda dt: (0.15, 0.0)
    view = WorldView(cast("Controller", controller), shake)
    view.set_bounds(0, 0, 10, 10)

    # Mock light overlay composition to isolate screen-shake behavior.
    view._render_light_overlay_gpu_compose = lambda renderer, dark_texture: (
        "mock_light_overlay_texture"
    )
    view._update_tile_animations = lambda: None

    view.draw(cast(GraphicsContext, controller.graphics), InterpolationAlpha(0.0))

    # Verify shake offset is stored after draw()
    assert view._shake_offset == (0.15, 0.0)

    # Camera should NOT be modified in draw() - shake is applied in present()
    base_cam_pos = (4.5, 4.5)  # Default camera position for 10x10 map
    assert view.viewport_system.camera.world_x == base_cam_pos[0]
    assert view.viewport_system.camera.world_y == base_cam_pos[1]


def test_world_view_screen_shake_does_not_overflow(monkeypatch) -> None:
    """Ensure screen shake offsets keep rendering within console bounds."""
    from brileta.view.views.world_view import WorldView

    controller = DummyController()
    shake = ScreenShake()
    # Use a larger float offset to test bounds checking
    shake.update = lambda dt: (0.5, 0.0)
    view = WorldView(cast("Controller", controller), shake)
    view.set_bounds(0, 0, 10, 10)

    captured: dict[str, Any] = {}

    original_render_map = view._render_map_unlit

    def wrapped_render_map() -> None:
        gw = controller.gw
        vs = view.viewport_system
        bounds = vs.get_visible_bounds()
        bounds.x1 = max(0, bounds.x1)
        bounds.y1 = max(0, bounds.y1)
        world_right = min(gw.game_map.width - 1, bounds.x2)
        world_bottom = min(gw.game_map.height - 1, bounds.y2)
        captured["dest_width"] = world_right - bounds.x1 + 1
        captured["dest_height"] = world_bottom - bounds.y1 + 1
        original_render_map()

    view._render_map_unlit = wrapped_render_map

    # Mock light overlay composition to isolate screen-shake behavior.
    view._render_light_overlay_gpu_compose = lambda renderer, dark_texture: (
        "mock_light_overlay_texture"
    )
    view._update_tile_animations = lambda: None

    view.draw(cast(GraphicsContext, controller.graphics), InterpolationAlpha(0.0))

    assert captured["dest_width"] <= view.width
    assert captured["dest_height"] <= view.height


def test_small_map_actor_alignment(monkeypatch) -> None:
    """Actors should align with the map when it is smaller than the viewport."""
    from brileta.view.views.world_view import WorldView

    controller = DummyController()
    controller.gw.game_map = DummyGameMap(5, 5)
    controller.gw.game_map.visible[:] = True
    controller.gw.game_map.explored[:] = True  # All tiles are explored

    shake = ScreenShake()
    shake.update = lambda dt: (0.0, 0.0)
    view = WorldView(cast("Controller", controller), shake)
    view.set_bounds(0, 0, 10, 8)

    # Mock light overlay composition to isolate alignment behavior.
    view._render_light_overlay_gpu_compose = lambda renderer, dark_texture: (
        "mock_light_overlay_texture"
    )
    view._update_tile_animations = lambda: None

    view.draw(cast(GraphicsContext, controller.graphics), InterpolationAlpha(0.0))

    vs = view.viewport_system
    vp_x, vp_y = vs.world_to_screen(controller.gw.player.x, controller.gw.player.y)
    # The glyph buffer is larger than the viewport by _SCROLL_PADDING on each edge.
    # Buffer coordinates = viewport coordinates + padding.
    pad = view._SCROLL_PADDING
    buf_x, buf_y = vp_x + pad, vp_y + pad
    # With smooth rendering enabled, actors are drawn via SDL, not console
    if config.SMOOTH_ACTOR_RENDERING_ENABLED:
        # Terrain decoration fills explored tiles with glyphs (space for WALL).
        assert view.map_glyph_buffer.data["ch"][buf_x, buf_y] == ord(" ")
    else:
        assert view.map_glyph_buffer.data["ch"][buf_x, buf_y] == ord("@")
    assert (vp_x, vp_y) == (
        vs.offset_x + controller.gw.player.x,
        vs.offset_y + controller.gw.player.y,
    )
