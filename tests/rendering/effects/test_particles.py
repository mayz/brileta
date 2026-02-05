import math
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from catley.types import DeltaTime
from catley.util.coordinates import Rect
from catley.view.render.effects.particles import (
    ParticleLayer,
    SubTileParticleSystem,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager


@pytest.mark.parametrize(
    ("dir_x", "dir_y", "angle"),
    [
        (1, 0, 0.0),
        (0, 1, math.pi / 2),
        (-1, 0, math.pi),
        (0, -1, -math.pi / 2),
        (1, 1, math.pi / 4),
    ],
)
def test_emit_directional_cone_velocity(
    dir_x: int,
    dir_y: int,
    angle: float,
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    ps = SubTileParticleSystem(10, 10, subdivision=2)
    # spread=0.0, speed=5.0, lifetime=0.2
    with patch_particles_rng([0.0, 5.0, 0.2]):
        ps.emit_directional_cone(
            1,
            1,
            dir_x,
            dir_y,
            count=1,
            cone_spread=0.0,
            speed_range=(5.0, 5.0),
            lifetime_range=(0.2, 0.2),
            colors_and_chars=[((1, 2, 3), "*")],
        )
    assert ps.active_count == 1
    expected_vx = math.cos(angle) * 5.0 * ps.subdivision
    expected_vy = math.sin(angle) * 5.0 * ps.subdivision
    assert pytest.approx(expected_vx) == ps.velocities[0, 0]
    assert pytest.approx(expected_vy) == ps.velocities[0, 1]
    assert ps.lifetimes[0] == 0.2


def test_emit_directional_cone_spread(
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    ps = SubTileParticleSystem(5, 5, subdivision=1)
    # spread=0.1, speed=2.0, lifetime=0.1
    with patch_particles_rng([0.1, 2.0, 0.1]):
        ps.emit_directional_cone(
            0,
            0,
            1,
            0,
            count=1,
            cone_spread=0.5,
            speed_range=(2.0, 2.0),
            lifetime_range=(0.1, 0.1),
            colors_and_chars=[((0, 0, 0), "+")],
        )
    expected_angle = 0.0 + 0.1
    assert pytest.approx(math.cos(expected_angle) * 2.0) == ps.velocities[0, 0]
    assert pytest.approx(math.sin(expected_angle) * 2.0) == ps.velocities[0, 1]


def test_emit_directional_cone_origin_offset_zero(
    ps_subdivision_2: SubTileParticleSystem,
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    """Test that with zero offset, particles originate from the tile center."""
    ps = ps_subdivision_2
    tile_x, tile_y = 1, 1
    expected_sub_x = tile_x * ps.subdivision + ps.subdivision / 2.0
    expected_sub_y = tile_y * ps.subdivision + ps.subdivision / 2.0

    # spread=0.0, speed=5.0, lifetime=0.2
    with patch_particles_rng([0.0, 5.0, 0.2]):
        ps.emit_directional_cone(
            tile_x,
            tile_y,
            direction_x=1,
            direction_y=0,
            count=1,
            cone_spread=0.0,
            speed_range=(5.0, 5.0),
            lifetime_range=(0.2, 0.2),
            colors_and_chars=[((1, 2, 3), "*")],
            origin_offset_tiles=0.0,
        )
    assert ps.active_count == 1
    assert ps.positions[0, 0] == pytest.approx(expected_sub_x)
    assert ps.positions[0, 1] == pytest.approx(expected_sub_y)


@pytest.mark.parametrize(
    ("dir_x", "dir_y", "offset_tiles", "subdiv"),
    [
        (1, 0, 0.5, 2),
        (0, 1, 0.3, 3),
        (-1, 0, 1.0, 1),
        (0, -1, 0.7, 4),
        (1, 1, 0.2, 2),
    ],
)
def test_emit_directional_cone_origin_offset_varied(
    dir_x: int,
    dir_y: int,
    offset_tiles: float,
    subdiv: int,
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    """Test origin offset in various directions and subdivisions."""
    ps = SubTileParticleSystem(10, 10, subdivision=subdiv)
    tile_x, tile_y = 2, 3

    center_sub_x = tile_x * ps.subdivision + ps.subdivision / 2.0
    center_sub_y = tile_y * ps.subdivision + ps.subdivision / 2.0

    base_angle = math.atan2(dir_y, dir_x)
    offset_dist_sub_pixels = offset_tiles * ps.subdivision

    expected_origin_sub_x = center_sub_x + math.cos(base_angle) * offset_dist_sub_pixels
    expected_origin_sub_y = center_sub_y + math.sin(base_angle) * offset_dist_sub_pixels

    # spread=0.0, speed=1.0, lifetime=0.1
    with patch_particles_rng([0.0, 1.0, 0.1]):
        ps.emit_directional_cone(
            tile_x,
            tile_y,
            direction_x=dir_x,
            direction_y=dir_y,
            count=1,
            cone_spread=0.0,
            speed_range=(1.0, 1.0),
            lifetime_range=(0.1, 0.1),
            colors_and_chars=[((0, 0, 0), "*")],
            origin_offset_tiles=offset_tiles,
        )

    assert ps.active_count == 1
    assert ps.positions[0, 0] == pytest.approx(expected_origin_sub_x)
    assert ps.positions[0, 1] == pytest.approx(expected_origin_sub_y)


@pytest.fixture
def ps_subdivision_2() -> SubTileParticleSystem:
    return SubTileParticleSystem(10, 10, subdivision=2)


@pytest.fixture
def ps_default() -> SubTileParticleSystem:
    return SubTileParticleSystem(map_width=10, map_height=10, subdivision=3)


def test_particle_update_movement_and_lifetime(
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    ps = SubTileParticleSystem(3, 3, subdivision=1)
    # x_spread=0.0, y_spread=0.0
    with patch_particles_rng([0.0, 0.0]):
        ps.add_particle(0, 0, vel_x=20.0, vel_y=-10.0, lifetime=1.0)

    delta_time = 0.1
    ps.update(DeltaTime(delta_time))
    assert ps.positions[0, 0] == pytest.approx(0.5 + 20.0 * delta_time)
    assert ps.positions[0, 1] == pytest.approx(0.5 - 10.0 * delta_time)
    assert ps.lifetimes[0] == pytest.approx(1.0 - delta_time)
    assert ps.lifetimes[0] / ps.max_lifetimes[0] == pytest.approx(1.0 - delta_time)

    ps.lifetimes[0] = 0.05
    ps.update(DeltaTime(delta_time))
    assert ps.active_count == 0


def test_particle_update_gravity_effect(
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    ps = SubTileParticleSystem(3, 3, subdivision=1)
    # x_spread=0.0, y_spread=0.0
    with patch_particles_rng([0.0, 0.0]):
        ps.add_particle(0, 0, vel_x=0.0, vel_y=0.0, lifetime=1.0)

    ps.gravity[0] = 100.0
    delta_time = 0.1

    ps.update(DeltaTime(delta_time))
    assert ps.velocities[0, 1] == pytest.approx(10.0)
    assert ps.positions[0, 1] == pytest.approx(0.5)

    ps.update(DeltaTime(delta_time))
    assert ps.velocities[0, 1] == pytest.approx(20.0)
    assert ps.positions[0, 1] == pytest.approx(0.5 + 10.0 * delta_time)

    ps.update(DeltaTime(delta_time))
    assert ps.velocities[0, 1] == pytest.approx(30.0)
    assert ps.positions[0, 1] == pytest.approx(
        0.5 + 10.0 * delta_time + 20.0 * delta_time
    )


def test_add_particle_origin_is_centered_with_spread(
    ps_default: SubTileParticleSystem,
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    ps = ps_default
    tile_x, tile_y = 2.0, 3.0

    expected_center_sub_x = (tile_x + 0.5) * ps.subdivision
    expected_center_sub_y = (tile_y + 0.5) * ps.subdivision

    mocked_spread_x = 0.2
    mocked_spread_y = -0.3
    # x_spread, y_spread
    with patch_particles_rng([mocked_spread_x, mocked_spread_y]):
        ps.add_particle(tile_x, tile_y, vel_x=0, vel_y=0, lifetime=1)

    assert ps.active_count == 1
    assert ps.positions[0, 0] == pytest.approx(expected_center_sub_x + mocked_spread_x)
    assert ps.positions[0, 1] == pytest.approx(expected_center_sub_y + mocked_spread_y)


def test_system_update_removes_dead_particles_from_list(
    ps_default: SubTileParticleSystem,
) -> None:
    ps = ps_default
    ps.add_particle(tile_x=1, tile_y=1, vel_x=0, vel_y=0, lifetime=0.05)
    ps.add_particle(tile_x=2, tile_y=2, vel_x=0, vel_y=0, lifetime=1.0)

    assert ps.active_count == 2
    ps.update(delta_time=DeltaTime(0.1))

    assert ps.active_count == 1
    assert ps.max_lifetimes[0] == 1.0


def test_emit_radial_burst_origin_is_centered(
    ps_default: SubTileParticleSystem,
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    ps = ps_default
    tile_x, tile_y = 1.0, 1.0
    expected_sub_x = (tile_x + 0.5) * ps.subdivision
    expected_sub_y = (tile_y + 0.5) * ps.subdivision

    # angle=pi/4, speed=5.0, lifetime=0.2
    with patch_particles_rng([math.pi / 4, 5.0, 0.2]):
        ps.emit_radial_burst(
            tile_x,
            tile_y,
            count=1,
            speed_range=(5.0, 5.0),
            lifetime_range=(0.2, 0.2),
            colors_and_chars=[((1, 2, 3), "*")],
        )
    assert ps.active_count == 1
    assert ps.positions[0, 0] == pytest.approx(expected_sub_x)
    assert ps.positions[0, 1] == pytest.approx(expected_sub_y)


def test_particle_layer_assignment() -> None:
    ps = SubTileParticleSystem(3, 3, subdivision=1)
    ps.add_particle(0, 0, 0.0, 0.0, lifetime=1.0, layer=ParticleLayer.UNDER_ACTORS)
    ps.add_particle(0, 0, 0.0, 0.0, lifetime=1.0)
    assert ps.layers[0] == ParticleLayer.UNDER_ACTORS.value
    assert ps.layers[1] == ParticleLayer.OVER_ACTORS.value


def test_render_particles_filters_by_layer(
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    ps = SubTileParticleSystem(2, 2, subdivision=1)
    # x_spread=0.0, y_spread=0.0 for each particle
    with patch_particles_rng([0.0, 0.0, 0.0, 0.0]):
        ps.add_particle(
            0, 0, 0.0, 0.0, lifetime=1.0, char="a", layer=ParticleLayer.UNDER_ACTORS
        )
        ps.add_particle(
            1, 0, 0.0, 0.0, lifetime=1.0, char="b", layer=ParticleLayer.OVER_ACTORS
        )

    renderer = MagicMock()
    renderer.console_to_screen_coords.return_value = (5.0, 5.0)
    renderer._draw_particle_smooth = MagicMock()

    with patch.object(
        renderer, "_convert_particle_to_screen_coords", return_value=(5.0, 5.0)
    ):
        for i in range(ps.active_count):
            if ps.layers[i] == ParticleLayer.OVER_ACTORS.value:
                renderer._draw_particle_smooth(
                    ps.chars[i],
                    tuple(ps.colors[i]),
                    5.0,
                    5.0,
                    ps.lifetimes[i] / ps.max_lifetimes[i],
                )

    renderer._draw_particle_smooth.assert_called_once()
    called_char = renderer._draw_particle_smooth.call_args[0][0]
    assert called_char == "b"


def test_convert_particle_to_screen_coords(
    patch_particles_rng: "Callable[[list[float]], AbstractContextManager[Any]]",
) -> None:
    ps = SubTileParticleSystem(3, 3, subdivision=1)
    # x_spread=0.0, y_spread=0.0
    with patch_particles_rng([0.0, 0.0]):
        ps.add_particle(1, 1, 0.0, 0.0, lifetime=1.0, char="c")
    renderer = MagicMock()
    renderer.console_to_screen_coords.return_value = (10.0, 20.0)

    viewport = Rect.from_bounds(0, 0, 2, 2)
    result = ps._convert_particle_to_screen_coords(0, viewport, (2, 3), renderer)
    x_arg, y_arg = renderer.console_to_screen_coords.call_args[0]
    assert pytest.approx(3.5, rel=1e-2) == x_arg
    assert pytest.approx(4.5, rel=1e-2) == y_arg
    assert result == (10.0, 20.0)

    off_viewport = Rect.from_bounds(2, 2, 3, 3)
    assert (
        ps._convert_particle_to_screen_coords(0, off_viewport, (0, 0), renderer) is None
    )
