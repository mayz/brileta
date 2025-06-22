from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np

from catley import config
from catley.controller import Controller
from catley.util.spatial import SpatialHashGrid
from catley.view.render.effects.screen_shake import ScreenShake


def test_screen_shake_update_and_completion() -> None:
    shake = ScreenShake()
    shake.trigger(intensity=1.0, duration=1.0)

    with (
        patch("random.random", return_value=0.0),
        patch("random.choice", side_effect=lambda seq: seq[0]),
    ):
        offset = shake.update(0.5)
    assert offset == (-1, -1)
    assert shake.is_active()

    offset = shake.update(0.6)
    assert offset == (0, 0)
    assert not shake.is_active()


def test_screen_shake_trigger_overwrite() -> None:
    shake = ScreenShake()
    shake.trigger(intensity=0.2, duration=0.5)
    shake.trigger(intensity=0.8, duration=1.0)
    assert shake.intensity == 0.8
    assert shake.duration == 1.0
    assert shake.time_remaining == 1.0


class DummyActor:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.ch = "@"
        self.color = (255, 255, 255)
        self.render_x = float(x)
        self.render_y = float(y)

    def update_render_position(self, _dt: float) -> None:
        self.render_x = float(self.x)
        self.render_y = float(self.y)


class DummyGameMap:
    def __init__(self, width: int, height: int) -> None:
        from catley.environment import tile_types

        self.width = width
        self.height = height
        self.revision = 0
        self.dark_appearance_map = np.zeros(
            (width, height), dtype=tile_types.TileTypeAppearance
        )
        self.light_appearance_map = np.zeros(
            (width, height), dtype=tile_types.TileTypeAppearance
        )
        self.explored = np.zeros((width, height), dtype=bool)
        self.visible = np.zeros((width, height), dtype=bool)


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
        self.renderer = SimpleNamespace(
            clear_console=lambda *args, **kwargs: None,
            root_console=None,
            blit_console=lambda *args, **kwargs: None,
            texture_from_console=lambda console: f"mock_texture_for_{id(console)}",
        )
        self.clock = SimpleNamespace(last_delta_time=0.016)
        self.active_mode = None
        self.is_targeting_mode = lambda: False


def test_world_view_applies_screen_shake_before_render(monkeypatch) -> None:
    from catley.view.views.world_view import WorldView

    controller = DummyController()
    shake = ScreenShake()
    shake.update = lambda dt: (1, 0)  # type: ignore[assignment]
    view = WorldView(cast("Controller", controller), shake)
    view.set_bounds(0, 0, 10, 10)

    captured = {}
    original_render_map = view._render_map

    def wrapped_render_map() -> None:
        captured["cam_pos"] = (
            view.viewport_system.camera.world_x,
            view.viewport_system.camera.world_y,
        )
        original_render_map()

    view._render_map = wrapped_render_map  # type: ignore[assignment]

    from catley.view.render.renderer import Renderer

    view.draw(cast(Renderer, controller.renderer))

    assert captured["cam_pos"] == (5.5, 4.5)


def test_world_view_screen_shake_does_not_overflow(monkeypatch) -> None:
    """Ensure screen shake offsets keep rendering within console bounds."""
    from catley.view.views.world_view import WorldView

    controller = DummyController()
    shake = ScreenShake()
    shake.update = lambda dt: (1, 0)  # type: ignore[assignment]
    view = WorldView(cast("Controller", controller), shake)
    view.set_bounds(0, 0, 10, 10)

    captured: dict[str, Any] = {}

    original_render_map = view._render_map

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

    view._render_map = wrapped_render_map  # type: ignore[assignment]

    from catley.view.render.renderer import Renderer

    view.draw(cast(Renderer, controller.renderer))

    assert captured["dest_width"] <= view.width
    assert captured["dest_height"] <= view.height


def test_small_map_actor_alignment(monkeypatch) -> None:
    """Actors should align with the map when it is smaller than the viewport."""
    from catley.view.views.world_view import WorldView

    controller = DummyController()
    controller.gw.game_map = DummyGameMap(5, 5)
    controller.gw.game_map.visible[:] = True

    shake = ScreenShake()
    shake.update = lambda dt: (0, 0)  # type: ignore[assignment]
    view = WorldView(cast("Controller", controller), shake)
    view.set_bounds(0, 0, 10, 8)

    from catley.view.render.renderer import Renderer

    view.draw(cast(Renderer, controller.renderer))

    vs = view.viewport_system
    px, py = vs.world_to_screen(controller.gw.player.x, controller.gw.player.y)
    # With smooth rendering enabled, actors are drawn via SDL, not console
    if config.SMOOTH_ACTOR_RENDERING_ENABLED:
        assert view.game_map_console.rgb["ch"][px, py] == 0  # No character in console
    else:
        assert view.game_map_console.rgb["ch"][px, py] == ord("@")
    assert (px, py) == (
        vs.offset_x + controller.gw.player.x,
        vs.offset_y + controller.gw.player.y,
    )
