"""Tests for projected actor shadow quad rendering."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, call

import numpy as np
import pytest

from brileta import config
from brileta.backends.wgpu.graphics import WGPUGraphicsContext
from brileta.backends.wgpu.screen_renderer import VERTEX_DTYPE, WGPUScreenRenderer
from brileta.environment.tile_types import TileTypeAppearance, TileTypeID
from brileta.game.actors import Actor
from brileta.game.actors.core import CharacterLayer
from brileta.game.lights import DirectionalLight
from brileta.types import InterpolationAlpha
from brileta.util.coordinates import Rect
from brileta.view.render.actor_renderer import ActorRenderer
from brileta.view.render.graphics import GraphicsContext
from brileta.view.render.shadow_renderer import ShadowRenderer
from brileta.view.render.viewport import ViewportSystem


def _build_actor(
    x: int = 0,
    y: int = 0,
    shadow_height: int = 1,
    character_layers: list[CharacterLayer] | None = None,
) -> Actor:
    return Actor(
        x=x,
        y=y,
        ch="@",
        color=(255, 255, 255),
        shadow_height=shadow_height,
        character_layers=character_layers,
    )


def _build_shadow_renderer(lights: list[object]) -> ShadowRenderer:
    default_shadow_grid = np.zeros((16, 16), dtype=np.uint8)
    default_tiles = np.full((16, 16), int(TileTypeID.FLOOR), dtype=np.uint8)
    default_visible = np.zeros((16, 16), dtype=bool)
    default_light_appearance = np.zeros((16, 16), dtype=TileTypeAppearance)
    game_map = SimpleNamespace(
        width=16,
        height=16,
        tiles=default_tiles,
        visible=default_visible,
        light_appearance_map=default_light_appearance,
        shadow_heights=default_shadow_grid,
        get_region_at=lambda _pos: SimpleNamespace(
            sky_exposure=1.0, region_type="exterior"
        ),
    )
    viewport = SimpleNamespace(
        get_visible_bounds=lambda: SimpleNamespace(x1=0, y1=0, x2=15, y2=15),
        world_to_screen=lambda x, y: (x, y),
        world_to_screen_float=lambda x, y: (x, y),
    )
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    renderer = ShadowRenderer(
        game_map=game_map,
        viewport_system=cast(ViewportSystem, viewport),
        graphics=cast(GraphicsContext, graphics),
    )
    renderer.set_view_transform(
        view_origin=(0.0, 0.0),
        camera_frac_offset=(0.0, 0.0),
    )
    renderer.set_frame_lighting(
        directional_light=next(
            (light for light in lights if isinstance(light, DirectionalLight)),
            None,
        ),
        lights=lights,
    )
    return renderer


def test_add_parallelogram_vertices() -> None:
    renderer = object.__new__(WGPUScreenRenderer)
    renderer.cpu_vertex_buffer = np.zeros(6, dtype=VERTEX_DTYPE)
    renderer.vertex_count = 0

    renderer.add_parallelogram(
        corners=((1.0, 2.0), (5.0, 2.0), (3.0, 6.0), (7.0, 6.0)),
        uv_coords=(0.1, 0.2, 0.3, 0.4),
        base_color=(0.0, 0.0, 0.0, 0.5),
        tip_color=(0.0, 0.0, 0.0, 0.0),
    )

    assert renderer.vertex_count == 6
    vertices = renderer.cpu_vertex_buffer
    assert tuple(vertices[0]["position"]) == pytest.approx((1.0, 2.0))
    assert tuple(vertices[1]["position"]) == pytest.approx((5.0, 2.0))
    assert tuple(vertices[2]["position"]) == pytest.approx((3.0, 6.0))
    assert tuple(vertices[5]["position"]) == pytest.approx((7.0, 6.0))
    assert tuple(vertices[0]["uv"]) == pytest.approx((0.1, 0.4))
    assert tuple(vertices[2]["uv"]) == pytest.approx((0.1, 0.2))
    assert float(vertices[0]["color"][3]) == pytest.approx(0.5)
    assert float(vertices[2]["color"][3]) == pytest.approx(0.0)


def test_shadow_geometry_sun() -> None:
    graphics = object.__new__(WGPUGraphicsContext)
    graphics._tile_dimensions = (16, 24)
    graphics.screen_renderer = Mock()
    graphics.uv_map = np.zeros((256, 4), dtype=np.float32)
    graphics.uv_map[ord("@")] = (0.0, 0.0, 1.0, 1.0)

    graphics.draw_actor_shadow(
        char="@",
        screen_x=100.0,
        screen_y=200.0,
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_pixels=30.0,
        shadow_alpha=0.35,
        fade_tip=True,
    )

    add_call = graphics.screen_renderer.add_parallelogram.call_args
    corners = add_call.args[0]
    # Horizontal shadow: UV-correct rectangle from the side edge, with slight
    # overlap into the glyph and base-anchored thickness.
    # overlap=6.4, near_x=109.6, far_x=139.6, bottom=224, top=216
    assert corners[0] == pytest.approx((109.6, 224.0))  # base_left: near-bottom
    assert corners[1] == pytest.approx((139.6, 224.0))  # base_right: far-bottom
    assert corners[2] == pytest.approx((109.6, 216.0))  # tip_left: near-top
    assert corners[3] == pytest.approx((139.6, 216.0))  # tip_right: far-top

    # UV is horizontally mirrored (shadow puppet projection effect).
    uv_passed = add_call.args[1]
    assert tuple(uv_passed) == pytest.approx((1.0, 0.0, 0.0, 1.0))

    # Per-vertex colours fade near-to-far, not bottom-to-top.
    vertex_colors = add_call.kwargs.get("vertex_colors")
    assert vertex_colors is not None
    base_color = add_call.args[2]
    tip_color = add_call.args[3]
    assert vertex_colors[0] == base_color  # near-bottom: opaque
    assert vertex_colors[1] == tip_color  # far-bottom: faded
    assert vertex_colors[2] == base_color  # near-top: opaque
    assert vertex_colors[3] == tip_color  # far-top: faded


def test_shadow_geometry_horizontal_direction_preserves_area() -> None:
    graphics = object.__new__(WGPUGraphicsContext)
    graphics._tile_dimensions = (16, 24)
    graphics.screen_renderer = Mock()
    graphics.uv_map = np.zeros((256, 4), dtype=np.float32)
    graphics.uv_map[ord("@")] = (0.0, 0.0, 1.0, 1.0)

    graphics.draw_actor_shadow(
        char="@",
        screen_x=50.0,
        screen_y=60.0,
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_pixels=20.0,
        shadow_alpha=0.35,
        fade_tip=True,
    )

    corners = graphics.screen_renderer.add_parallelogram.call_args.args[0]
    base_left, base_right, tip_left, _tip_right = corners

    # Regression guard: horizontal shadows use a thinner, base-anchored band.
    assert abs(base_left[1] - tip_left[1]) == pytest.approx(8.0)
    assert base_left[1] == pytest.approx(84.0)
    # Shadow extends horizontally from the glyph's side edge.
    assert abs(base_right[0] - base_left[0]) > 0


def test_shadow_geometry_diagonal_direction_unchanged_projection() -> None:
    graphics = object.__new__(WGPUGraphicsContext)
    graphics._tile_dimensions = (20, 20)
    graphics.screen_renderer = Mock()
    graphics.uv_map = np.zeros((256, 4), dtype=np.float32)
    graphics.uv_map[ord("@")] = (0.0, 0.0, 1.0, 1.0)

    graphics.draw_actor_shadow(
        char="@",
        screen_x=100.0,
        screen_y=200.0,
        shadow_dir_x=1.0,
        shadow_dir_y=1.0,
        shadow_length_pixels=30.0,
        shadow_alpha=0.35,
        fade_tip=True,
    )

    corners = graphics.screen_renderer.add_parallelogram.call_args.args[0]
    # For non-degenerate directions, tip is base + normalized_dir * length.
    expected_dx = 30.0 / np.sqrt(2.0)
    expected_dy = 30.0 / np.sqrt(2.0)
    assert corners[2] == pytest.approx((100.0 + expected_dx, 220.0 + expected_dy))
    assert corners[3] == pytest.approx((120.0 + expected_dx, 220.0 + expected_dy))


def test_shadow_geometry_near_horizontal_uses_width_thickness() -> None:
    graphics = object.__new__(WGPUGraphicsContext)
    graphics._tile_dimensions = (16, 24)
    graphics.screen_renderer = Mock()
    graphics.uv_map = np.zeros((256, 4), dtype=np.float32)
    graphics.uv_map[ord("@")] = (0.0, 0.0, 1.0, 1.0)

    # Simulate slight drift-induced vertical component while actor is effectively
    # horizontal to the light.
    graphics.draw_actor_shadow(
        char="@",
        screen_x=100.0,
        screen_y=200.0,
        shadow_dir_x=1.0,
        shadow_dir_y=0.02,
        shadow_length_pixels=40.0,
        shadow_alpha=0.35,
        fade_tip=True,
    )

    corners = graphics.screen_renderer.add_parallelogram.call_args.args[0]
    base_left, base_right, tip_left, tip_right = corners
    # Near-horizontal directions should still use the same subtle thickness.
    assert abs(base_left[1] - tip_left[1]) == pytest.approx(8.0)
    # Base edge is horizontal (both base corners share the same y).
    assert base_left[1] == pytest.approx(base_right[1])
    # Tip edge is parallel to base edge (same y for both tip corners).
    assert tip_left[1] == pytest.approx(tip_right[1])
    # Band is anchored to glyph bottom for stronger contact.
    assert base_left[1] == pytest.approx(224.0)
    assert tip_left[1] == pytest.approx(216.0)


def test_shadow_geometry_respects_non_unit_scale() -> None:
    graphics = object.__new__(WGPUGraphicsContext)
    graphics._tile_dimensions = (20, 20)
    graphics.screen_renderer = Mock()
    graphics.uv_map = np.zeros((256, 4), dtype=np.float32)
    graphics.uv_map[ord("@")] = (0.0, 0.0, 1.0, 1.0)

    graphics.draw_actor_shadow(
        char="@",
        screen_x=100.0,
        screen_y=200.0,
        shadow_dir_x=0.0,
        shadow_dir_y=1.0,
        shadow_length_pixels=30.0,
        shadow_alpha=0.35,
        scale_x=2.0,
        scale_y=0.5,
        fade_tip=True,
    )

    corners = graphics.screen_renderer.add_parallelogram.call_args.args[0]
    assert corners[0] == pytest.approx((90.0, 215.0))
    assert corners[1] == pytest.approx((130.0, 215.0))
    assert corners[2] == pytest.approx((90.0, 245.0))
    assert corners[3] == pytest.approx((130.0, 245.0))


def test_shadow_geometry_point_light() -> None:
    actor = _build_actor(x=2, y=2, shadow_height=1)
    light = SimpleNamespace(position=(1, 2), radius=6)
    renderer = _build_shadow_renderer([light])
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_point_light_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    call = graphics.draw_actor_shadow.call_args
    assert call.kwargs["shadow_dir_x"] == pytest.approx(1.0)
    assert call.kwargs["shadow_dir_y"] == pytest.approx(0.0)


def test_point_light_shadow_direction_ignores_idle_drift() -> None:
    actor = _build_actor(x=2, y=2, shadow_height=1)
    actor.visual_effects = SimpleNamespace(get_idle_drift_offset=lambda: (0.0, 0.45))
    actor.health = SimpleNamespace(is_alive=lambda: True)
    light = SimpleNamespace(position=(4, 2), radius=6)
    renderer = _build_shadow_renderer([light])
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_point_light_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    call = graphics.draw_actor_shadow.call_args
    assert call.kwargs["shadow_dir_x"] == pytest.approx(-1.0)
    assert call.kwargs["shadow_dir_y"] == pytest.approx(0.0)


def test_shadow_fade_alpha() -> None:
    graphics = object.__new__(WGPUGraphicsContext)
    graphics._tile_dimensions = (20, 20)
    graphics.screen_renderer = Mock()
    graphics.uv_map = np.zeros((256, 4), dtype=np.float32)
    graphics.uv_map[ord("@")] = (0.0, 0.0, 1.0, 1.0)

    graphics.draw_actor_shadow(
        "@", 0.0, 0.0, 1.0, 0.0, 10.0, config.ACTOR_SHADOW_ALPHA, fade_tip=True
    )
    tip_color_faded = graphics.screen_renderer.add_parallelogram.call_args.args[3]
    assert tip_color_faded[3] == pytest.approx(0.0)

    graphics.draw_actor_shadow(
        "@", 0.0, 0.0, 1.0, 0.0, 10.0, config.ACTOR_SHADOW_ALPHA, fade_tip=False
    )
    base_color = graphics.screen_renderer.add_parallelogram.call_args.args[2]
    tip_color = graphics.screen_renderer.add_parallelogram.call_args.args[3]
    assert base_color[3] == pytest.approx(config.ACTOR_SHADOW_ALPHA)
    assert tip_color[3] == pytest.approx(config.ACTOR_SHADOW_ALPHA)


def test_point_light_shadow_attenuation() -> None:
    actor = _build_actor(x=3, y=4, shadow_height=1)
    light = SimpleNamespace(position=(0, 0), radius=10)
    renderer = _build_shadow_renderer([light])
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_point_light_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    shadow_alpha = graphics.draw_actor_shadow.call_args.kwargs["shadow_alpha"]
    assert shadow_alpha == pytest.approx(config.ACTOR_SHADOW_ALPHA * 0.5)


def test_composite_layers_shadow() -> None:
    layers = [
        CharacterLayer("A", (255, 255, 255), offset_x=-0.2, offset_y=0.1),
        CharacterLayer("B", (255, 255, 255), offset_x=0.3, offset_y=-0.1),
    ]
    actor = _build_actor(character_layers=layers)
    renderer = _build_shadow_renderer([])
    graphics = SimpleNamespace(
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    renderer.graphics = cast(GraphicsContext, graphics)

    renderer._emit_actor_shadow_quads(
        actor=actor,
        root_x=10.0,
        root_y=20.0,
        screen_x=100.0,
        screen_y=200.0,
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_pixels=12.0,
        shadow_alpha=0.2,
        fade_tip=True,
    )

    assert graphics.draw_actor_shadow.call_count == 2
    first_char = graphics.draw_actor_shadow.call_args_list[0].kwargs["char"]
    second_char = graphics.draw_actor_shadow.call_args_list[1].kwargs["char"]
    assert first_char == "A"
    assert second_char == "B"


def test_no_shadow_zero_height() -> None:
    actor = _build_actor(x=3, y=4, shadow_height=0)
    light = SimpleNamespace(position=(0, 0), radius=10)
    renderer = _build_shadow_renderer([light])
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_point_light_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    graphics.draw_actor_shadow.assert_not_called()


def test_shadow_wall_clipping() -> None:
    renderer = _build_shadow_renderer([])
    shadow_grid = np.zeros((10, 10), dtype=np.uint8)
    shadow_grid[4, 2] = 4  # Blocking wall tile two steps from actor in +X
    renderer.game_map = SimpleNamespace(width=10, height=10, shadow_heights=shadow_grid)

    clipped = renderer._clip_shadow_length_by_walls(
        actor_x=2,
        actor_y=2,
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_tiles=5.0,
    )

    assert clipped == pytest.approx(1.5)


def test_shadow_wall_clipping_ignores_low_shadow_heights() -> None:
    renderer = _build_shadow_renderer([])
    shadow_grid = np.zeros((10, 10), dtype=np.uint8)
    shadow_grid[4, 2] = 2  # Boulder/glyph height should not clip projected shadows.
    renderer.game_map = SimpleNamespace(width=10, height=10, shadow_heights=shadow_grid)

    clipped = renderer._clip_shadow_length_by_walls(
        actor_x=2,
        actor_y=2,
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_tiles=5.0,
    )

    assert clipped == pytest.approx(5.0)


def test_shadow_vertex_range_markers() -> None:
    renderer = object.__new__(WGPUScreenRenderer)
    renderer.vertex_count = 12
    renderer._shadow_start = 0
    renderer._shadow_end = 0

    renderer.mark_shadow_start()
    assert renderer._shadow_start == 12

    renderer.vertex_count = 24
    renderer.mark_shadow_end()
    assert renderer._shadow_end == 24

    renderer.begin_frame()
    assert renderer.vertex_count == 0
    assert renderer._shadow_start == 0
    assert renderer._shadow_end == 0


def test_shadow_pass_marks_end_on_exception() -> None:
    graphics = object.__new__(WGPUGraphicsContext)
    graphics.screen_renderer = Mock()

    with pytest.raises(RuntimeError, match="shadow-failure"), graphics.shadow_pass():
        raise RuntimeError("shadow-failure")

    graphics.screen_renderer.mark_shadow_start.assert_called_once_with()
    graphics.screen_renderer.mark_shadow_end.assert_called_once_with()


def _build_screen_renderer_for_split_tests(
    vertex_count: int,
    shadow_start: int,
    shadow_end: int,
) -> WGPUScreenRenderer:
    renderer = object.__new__(WGPUScreenRenderer)
    renderer.vertex_count = vertex_count
    renderer._shadow_start = shadow_start
    renderer._shadow_end = shadow_end
    renderer.cpu_vertex_buffer = np.zeros(vertex_count, dtype=VERTEX_DTYPE)
    renderer.resource_manager = SimpleNamespace(queue=Mock())
    renderer.uniform_buffer = Mock()
    renderer.vertex_buffer = Mock()
    renderer.pipeline = Mock()
    renderer._cached_bind_group = Mock(name="cached_bind_group")
    renderer._shadow_bind_group = Mock(name="shadow_bind_group")
    renderer._actor_lighting_enabled = False
    renderer._actor_light_viewport_origin = (0, 0)
    return renderer


def test_render_to_screen_splits_pre_shadow_post_ranges() -> None:
    renderer = _build_screen_renderer_for_split_tests(
        vertex_count=18,
        shadow_start=6,
        shadow_end=12,
    )
    render_pass = Mock()

    renderer.render_to_screen(render_pass=render_pass, window_size=(800, 600))

    render_pass.set_bind_group.assert_has_calls(
        [
            call(0, renderer._cached_bind_group),
            call(0, renderer._shadow_bind_group),
            call(0, renderer._cached_bind_group),
        ]
    )
    render_pass.draw.assert_has_calls(
        [
            call(6, 1, 0),
            call(6, 1, 6),
            call(6, 1, 12),
        ]
    )


def test_render_to_screen_clamps_shadow_range_to_full_shadow_pass() -> None:
    renderer = _build_screen_renderer_for_split_tests(
        vertex_count=18,
        shadow_start=-5,
        shadow_end=999,
    )
    render_pass = Mock()

    renderer.render_to_screen(render_pass=render_pass, window_size=(800, 600))

    render_pass.set_bind_group.assert_called_once_with(0, renderer._shadow_bind_group)
    render_pass.draw.assert_called_once_with(18, 1, 0)


def test_render_to_screen_clamps_shadow_range_to_empty_shadow_pass() -> None:
    renderer = _build_screen_renderer_for_split_tests(
        vertex_count=18,
        shadow_start=30,
        shadow_end=40,
    )
    render_pass = Mock()

    renderer.render_to_screen(render_pass=render_pass, window_size=(800, 600))

    render_pass.set_bind_group.assert_called_once_with(0, renderer._cached_bind_group)
    render_pass.draw.assert_called_once_with(18, 1, 0)


def test_sun_shadow_skips_indoor_regions() -> None:
    actor = _build_actor(x=2, y=2, shadow_height=1)
    sun = DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.8)
    renderer = _build_shadow_renderer([sun])
    renderer.game_map.get_region_at = lambda _pos: SimpleNamespace(
        sky_exposure=0.0, region_type="room"
    )
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_sun_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    graphics.draw_actor_shadow.assert_not_called()


def test_sun_shadow_renders_outdoor_regions() -> None:
    actor = _build_actor(x=2, y=2, shadow_height=1)
    sun = DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.8)
    renderer = _build_shadow_renderer([sun])
    renderer.game_map.get_region_at = lambda _pos: SimpleNamespace(
        sky_exposure=1.0, region_type="exterior"
    )
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_sun_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    graphics.draw_actor_shadow.assert_called_once()


def test_sun_shadow_skips_room_regions_even_with_high_exposure() -> None:
    actor = _build_actor(x=2, y=2, shadow_height=1)
    sun = DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.8)
    renderer = _build_shadow_renderer([sun])
    renderer.game_map.tiles[2, 2] = int(TileTypeID.FLOOR)
    renderer.game_map.get_region_at = lambda _pos: SimpleNamespace(
        sky_exposure=1.0, region_type="room"
    )
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_sun_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    graphics.draw_actor_shadow.assert_not_called()


def test_point_light_skips_owned_light_shadow() -> None:
    actor = _build_actor(x=3, y=4, shadow_height=1)
    owned_light = SimpleNamespace(position=(0, 0), radius=10, owner=actor)
    renderer = _build_shadow_renderer([owned_light])
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_point_light_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    graphics.draw_actor_shadow.assert_not_called()


def test_point_light_skips_same_tile_drift_jitter() -> None:
    actor = _build_actor(x=3, y=4, shadow_height=1)
    # Simulate subtle idle drift that would otherwise create near-zero direction.
    actor.visual_effects = SimpleNamespace(
        get_idle_drift_offset=lambda: (0.03, 0.0),
    )
    actor.health = SimpleNamespace(is_alive=lambda: True)
    colocated_light = SimpleNamespace(position=(3, 4), radius=10)
    renderer = _build_shadow_renderer([colocated_light])
    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (x, y),
    )
    viewport = SimpleNamespace(world_to_screen_float=lambda x, y: (x, y))

    renderer.graphics = cast(GraphicsContext, graphics)
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer._render_point_light_actor_shadows(
        [actor],
        InterpolationAlpha(1.0),
        20.0,
    )

    graphics.draw_actor_shadow.assert_not_called()


def test_horizontal_shadow_uses_base_band_with_overlap_not_full_height() -> None:
    """Regression: horizontal shadows should be subtle and connected.

    When a point light is horizontally adjacent to an actor, the shadow should
    extend from the glyph's side edge with slight overlap and without expanding
    to full glyph height.
    """
    graphics = object.__new__(WGPUGraphicsContext)
    graphics._tile_dimensions = (16, 24)
    graphics.screen_renderer = Mock()
    graphics.uv_map = np.zeros((256, 4), dtype=np.float32)
    graphics.uv_map[ord("@")] = (0.0, 0.0, 1.0, 1.0)

    # Horizontal direction (light to the left, shadow to the right)
    graphics.draw_actor_shadow(
        char="@",
        screen_x=100.0,
        screen_y=200.0,
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_pixels=30.0,
        shadow_alpha=0.35,
        fade_tip=True,
    )

    corners = graphics.screen_renderer.add_parallelogram.call_args.args[0]
    base_left, base_right, tip_left, tip_right = corners

    # The shadow is a rectangle projecting from the glyph's right edge.
    # With UV-correct layout: left/right = screen x, base/tip = bottom/top.
    # Near edge (at glyph's right side) = base_left.x and tip_left.x.
    assert base_left[0] == pytest.approx(109.6)  # near-bottom
    assert tip_left[0] == pytest.approx(109.6)  # near-top

    # Far edge at base_right.x and tip_right.x.
    assert base_right[0] == pytest.approx(139.6)  # far-bottom
    assert tip_right[0] == pytest.approx(139.6)  # far-top

    # Shadow thickness is subtle and base-anchored.
    shadow_height = abs(base_left[1] - tip_left[1])
    assert shadow_height == pytest.approx(8.0)
    assert base_left[1] == pytest.approx(224.0)

    # Shadow extends rightward by shadow_length_pixels.
    shadow_extent = base_right[0] - base_left[0]
    assert shadow_extent == pytest.approx(30.0)
    # Near edge overlaps into glyph interior for visual contact.
    assert base_left[0] < 116.0


def test_leftward_horizontal_shadow_projects_from_left_edge() -> None:
    """Leftward horizontal shadows should project from the glyph's left edge."""
    graphics = object.__new__(WGPUGraphicsContext)
    graphics._tile_dimensions = (16, 24)
    graphics.screen_renderer = Mock()
    graphics.uv_map = np.zeros((256, 4), dtype=np.float32)
    graphics.uv_map[ord("@")] = (0.0, 0.0, 1.0, 1.0)

    graphics.draw_actor_shadow(
        char="@",
        screen_x=100.0,
        screen_y=200.0,
        shadow_dir_x=-1.0,
        shadow_dir_y=0.0,
        shadow_length_pixels=30.0,
        shadow_alpha=0.35,
        fade_tip=True,
    )

    corners = graphics.screen_renderer.add_parallelogram.call_args.args[0]
    base_left, base_right, tip_left, tip_right = corners

    # UV-correct layout: left/right = screen x, base/tip = bottom/top.
    # Far edge (leftward) at base_left.x and tip_left.x.
    assert base_left[0] == pytest.approx(76.4)  # far-bottom
    assert tip_left[0] == pytest.approx(76.4)  # far-top

    # Near edge (glyph's left side) at base_right.x and tip_right.x.
    assert base_right[0] == pytest.approx(106.4)  # near-bottom
    assert tip_right[0] == pytest.approx(106.4)  # near-top

    # Subtle, base-anchored thickness.
    shadow_height = abs(base_left[1] - tip_left[1])
    assert shadow_height == pytest.approx(8.0)
    assert base_left[1] == pytest.approx(224.0)

    # Per-vertex colours: right (near char) = opaque, left (far) = faded.
    add_call = graphics.screen_renderer.add_parallelogram.call_args
    vertex_colors = add_call.kwargs.get("vertex_colors")
    assert vertex_colors is not None
    base_color = add_call.args[2]
    tip_color = add_call.args[3]
    assert vertex_colors[0] == tip_color  # far-bottom: faded
    assert vertex_colors[1] == base_color  # near-bottom: opaque
    assert vertex_colors[2] == tip_color  # far-top: faded
    assert vertex_colors[3] == base_color  # near-top: opaque


def test_actor_shadow_receiver_dimming_for_overlap() -> None:
    caster = _build_actor(x=2, y=2, shadow_height=4)
    receiver = _build_actor(x=3, y=2, shadow_height=1)
    renderer = _build_shadow_renderer([])
    renderer.actor_shadow_receive_light_scale.clear()

    renderer._accumulate_actor_shadow_receiver_dimming(
        caster=caster,
        receivers=[caster, receiver],
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_tiles=4.0,
        shadow_alpha=config.ACTOR_SHADOW_ALPHA,
        fade_tip=False,
    )

    receiver_scale = renderer.actor_shadow_receive_light_scale[receiver]
    assert receiver_scale < 1.0
    assert caster not in renderer.actor_shadow_receive_light_scale


def test_actor_shadow_receiver_dimming_scales_with_caster_height() -> None:
    receiver_short = _build_actor(x=3, y=2, shadow_height=1)
    receiver_tall = _build_actor(x=3, y=2, shadow_height=1)
    short_caster = _build_actor(x=2, y=2, shadow_height=1)
    tall_caster = _build_actor(x=2, y=2, shadow_height=4)

    renderer = _build_shadow_renderer([])
    renderer.actor_shadow_receive_light_scale.clear()
    renderer._accumulate_actor_shadow_receiver_dimming(
        caster=short_caster,
        receivers=[receiver_short],
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_tiles=4.0,
        shadow_alpha=config.ACTOR_SHADOW_ALPHA,
        fade_tip=False,
    )
    short_scale = renderer.actor_shadow_receive_light_scale[receiver_short]

    renderer.actor_shadow_receive_light_scale.clear()
    renderer._accumulate_actor_shadow_receiver_dimming(
        caster=tall_caster,
        receivers=[receiver_tall],
        shadow_dir_x=1.0,
        shadow_dir_y=0.0,
        shadow_length_tiles=4.0,
        shadow_alpha=config.ACTOR_SHADOW_ALPHA,
        fade_tip=False,
    )
    tall_scale = renderer.actor_shadow_receive_light_scale[receiver_tall]

    assert tall_scale < short_scale


def test_world_view_actor_lighting_reads_shadow_renderer_scale() -> None:
    actor = _build_actor(x=1, y=1, shadow_height=1)
    renderer = object.__new__(ActorRenderer)
    renderer.shadow_renderer = SimpleNamespace(
        actor_shadow_receive_light_scale={actor: 0.5}
    )

    light_rgb = renderer._get_actor_lighting_intensity(actor, Rect(0, 0, 4, 4))
    assert light_rgb == pytest.approx((0.5, 0.5, 0.5))


def _build_terrain_shadow_view() -> tuple[ShadowRenderer, SimpleNamespace]:
    """Build a ShadowRenderer wired for terrain glyph shadow tests.

    Returns the renderer and a mock graphics object. The 16x16 game map defaults
    to FLOOR tiles (no glyph shadows) with an outdoor region and a directional sun.
    """
    sun = DirectionalLight.create_sun(elevation_degrees=45.0, intensity=0.8)
    renderer = _build_shadow_renderer([sun])

    # Viewport system with simple identity transforms
    viewport = SimpleNamespace(
        get_visible_bounds=lambda: SimpleNamespace(x1=0, y1=0, x2=15, y2=15),
        world_to_screen=lambda x, y: (x, y),
    )

    graphics = SimpleNamespace(
        tile_dimensions=(20, 20),
        draw_actor_shadow=Mock(),
        console_to_screen_coords=lambda x, y: (float(x) * 20.0, float(y) * 20.0),
    )
    renderer.viewport_system = cast(ViewportSystem, viewport)
    renderer.graphics = cast(GraphicsContext, graphics)
    return renderer, graphics


def test_terrain_glyph_shadow_for_boulder() -> None:
    """Boulder tiles (shadow_height=2) emit glyph shadow quads."""
    renderer, graphics = _build_terrain_shadow_view()
    game_map = renderer.game_map

    # Place a boulder at (5, 5) and make it visible
    game_map.tiles[5, 5] = int(TileTypeID.BOULDER)
    game_map.visible[5, 5] = True
    game_map.light_appearance_map[5, 5]["ch"] = ord("#")

    renderer._render_terrain_glyph_shadows(tile_height=20.0)

    graphics.draw_actor_shadow.assert_called_once()
    call_kwargs = graphics.draw_actor_shadow.call_args.kwargs
    assert call_kwargs["char"] == "#"
    assert call_kwargs["shadow_alpha"] == config.TERRAIN_GLYPH_SHADOW_ALPHA


def test_no_terrain_glyph_shadow_for_walls() -> None:
    """Wall tiles (shadow_height > 2) don't emit terrain glyph shadows."""
    renderer, graphics = _build_terrain_shadow_view()
    game_map = renderer.game_map

    # Place a wall at (5, 5) - shadow_height=4, above the glyph shadow threshold
    game_map.tiles[5, 5] = int(TileTypeID.WALL)
    game_map.visible[5, 5] = True
    game_map.light_appearance_map[5, 5]["ch"] = ord("#")

    renderer._render_terrain_glyph_shadows(tile_height=20.0)

    graphics.draw_actor_shadow.assert_not_called()


def test_terrain_glyph_shadow_clipped_by_walls() -> None:
    """Terrain glyph shadows shorten when a tall blocker is directly ahead."""
    renderer, graphics = _build_terrain_shadow_view()
    game_map = renderer.game_map

    # Force deterministic right-to-left projection: sun in the east.
    sun = DirectionalLight.create_sun(
        elevation_degrees=45.0,
        azimuth_degrees=90.0,
        intensity=0.8,
    )
    renderer.set_frame_lighting(directional_light=sun, lights=[sun])

    # Boulder caster at (5,5), tall blocker directly west at (4,5).
    game_map.tiles[5, 5] = int(TileTypeID.BOULDER)
    game_map.visible[5, 5] = True
    game_map.light_appearance_map[5, 5]["ch"] = ord("#")
    game_map.shadow_heights[4, 5] = 4

    renderer._render_terrain_glyph_shadows(tile_height=20.0)

    graphics.draw_actor_shadow.assert_called_once()
    shadow_length_pixels = graphics.draw_actor_shadow.call_args.kwargs[
        "shadow_length_pixels"
    ]
    assert shadow_length_pixels == pytest.approx(10.0)
