import math
from unittest.mock import patch

import pytest

from catley.render.particles import SubParticle, SubTileParticleSystem


class FixedUniform:
    def __init__(self, values: list[float]):
        self.values = values
        self.index = 0

    def __call__(self, _a: float, _b: float) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


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
def test_emit_directional_cone_velocity(dir_x: int, dir_y: int, angle: float) -> None:
    ps = SubTileParticleSystem(10, 10, subdivision=2)
    fu = FixedUniform([0.0, 5.0, 0.2])
    with patch("random.uniform", fu):
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
    assert len(ps.particles) == 1
    p = ps.particles[0]
    expected_vx = math.cos(angle) * 5.0 * ps.subdivision
    expected_vy = math.sin(angle) * 5.0 * ps.subdivision
    assert pytest.approx(expected_vx) == p.vel_x
    assert pytest.approx(expected_vy) == p.vel_y
    assert p.lifetime == 0.2


def test_emit_directional_cone_spread() -> None:
    ps = SubTileParticleSystem(5, 5, subdivision=1)
    fu = FixedUniform([0.1, 2.0, 0.1])
    with patch("random.uniform", fu):
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
    particle = ps.particles[0]
    expected_angle = 0.0 + 0.1
    assert pytest.approx(math.cos(expected_angle) * 2.0) == particle.vel_x
    assert pytest.approx(math.sin(expected_angle) * 2.0) == particle.vel_y


def test_emit_directional_cone_origin_offset_zero(
    ps_subdivision_2,
):  # Assuming you have a fixture for ps
    """Test that with zero offset, particles originate from the tile center."""
    ps = ps_subdivision_2  # Example: ps = SubTileParticleSystem(10, 10, subdivision=2)
    tile_x, tile_y = 1, 1
    # Expected center for tile (1,1) with subdivision 2:
    # sub_x = 1*2 + 2/2 = 3.0
    # sub_y = 1*2 + 2/2 = 3.0
    expected_sub_x = tile_x * ps.subdivision + ps.subdivision / 2.0
    expected_sub_y = tile_y * ps.subdivision + ps.subdivision / 2.0

    # Mock random.uniform to remove randomness from spread if it affects origin
    # (it doesn't here)
    # For this test, we only care about origin, so speed/lifetime randomness is fine.
    # Setting cone_spread=0 removes velocity angle randomness.
    with patch(
        "random.uniform", FixedUniform([0.0, 5.0, 0.2])
    ):  # spread, speed, lifetime
        ps.emit_directional_cone(
            tile_x,
            tile_y,
            direction_x=1,
            direction_y=0,  # Shoot right
            count=1,
            cone_spread=0.0,  # No angular spread for velocity
            speed_range=(5.0, 5.0),
            lifetime_range=(0.2, 0.2),
            colors_and_chars=[((1, 2, 3), "*")],
            origin_offset_tiles=0.0,  # Explicitly zero offset
        )
    assert len(ps.particles) == 1
    p = ps.particles[0]
    assert p.sub_x == pytest.approx(expected_sub_x)
    assert p.sub_y == pytest.approx(expected_sub_y)


@pytest.mark.parametrize(
    ("dir_x", "dir_y", "offset_tiles", "subdiv"),
    [
        (1, 0, 0.5, 2),  # Right, offset by 0.5 tile, subdiv 2
        (0, 1, 0.3, 3),  # Down, offset by 0.3 tile, subdiv 3
        (-1, 0, 1.0, 1),  # Left, offset by 1.0 tile, subdiv 1
        (0, -1, 0.7, 4),  # Up, offset by 0.7 tile, subdiv 4
        (1, 1, 0.2, 2),  # Down-Right (approx), offset by 0.2 tile, subdiv 2
    ],
)
def test_emit_directional_cone_origin_offset_varied(
    dir_x, dir_y, offset_tiles, subdiv
) -> None:
    """Test origin offset in various directions and subdivisions."""
    ps = SubTileParticleSystem(10, 10, subdivision=subdiv)
    tile_x, tile_y = 2, 3  # Arbitrary source tile

    # Calculate expected origin
    center_sub_x = tile_x * ps.subdivision + ps.subdivision / 2.0
    center_sub_y = tile_y * ps.subdivision + ps.subdivision / 2.0

    base_angle = math.atan2(dir_y, dir_x)
    offset_dist_sub_pixels = offset_tiles * ps.subdivision

    expected_origin_sub_x = center_sub_x + math.cos(base_angle) * offset_dist_sub_pixels
    expected_origin_sub_y = center_sub_y + math.sin(base_angle) * offset_dist_sub_pixels

    with patch(
        "random.uniform", FixedUniform([0.0, 1.0, 0.1])
    ):  # spread, speed, lifetime
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

    assert len(ps.particles) == 1
    p = ps.particles[0]
    assert p.sub_x == pytest.approx(expected_origin_sub_x)
    assert p.sub_y == pytest.approx(expected_origin_sub_y)


@pytest.fixture
def ps_subdivision_2():
    return SubTileParticleSystem(10, 10, subdivision=2)


# Add these to tests/test_particles.py


@pytest.fixture
def ps_default():
    return SubTileParticleSystem(map_width=10, map_height=10, subdivision=3)


def test_sub_particle_update_movement_and_lifetime() -> None:
    particle = SubParticle(1.0, 1.0, 20.0, -10.0, 1.0)  # vel in sub-pixels/sec
    particle.max_lifetime = 1.0  # Manually set for intensity testing if needed
    delta_time = 0.1

    is_alive = particle.update(delta_time)
    assert is_alive
    assert particle.sub_x == pytest.approx(1.0 + 20.0 * 0.1)
    assert particle.sub_y == pytest.approx(1.0 - 10.0 * 0.1)
    assert particle.lifetime == pytest.approx(1.0 - 0.1)
    assert particle.intensity == pytest.approx((1.0 - 0.1) / 1.0)

    particle.lifetime = 0.05  # Almost dead
    is_alive = particle.update(delta_time)  # This should kill it based on lifetime
    assert not is_alive
    assert particle.lifetime <= 0


def test_sub_particle_update_gravity_effect():
    # Using vel in sub-pixels/sec for SubParticle constructor
    particle = SubParticle(sub_x=0, sub_y=0, vel_x=0, vel_y=0, lifetime=1)
    particle.gravity = 100.0  # Gravity in sub-pixels/sec^2
    delta_time = 0.1

    # First update
    # Initial state: sub_y = 0, vel_y = 0
    particle.update(delta_time)
    # Step 1: sub_y = sub_y + vel_y * delta_time = 0 + 0 * 0.1 = 0
    # Step 2: vel_y = vel_y + gravity * delta_time = 0 + 100.0 * 0.1 = 10.0

    expected_vel_y_after_first_update = 10.0
    expected_sub_y_after_first_update = (
        0.0  # Because vel_y was 0 when sub_y was updated
    )

    assert particle.vel_y == pytest.approx(expected_vel_y_after_first_update)
    assert particle.sub_y == pytest.approx(expected_sub_y_after_first_update)

    # Second update
    # Initial state for this update: sub_y = 0.0, vel_y = 10.0
    particle.update(delta_time)
    # Step 1: sub_y = sub_y + vel_y * delta_time = 0.0 + 10.0 * 0.1 = 1.0
    # Step 2: vel_y = vel_y + gravity * delta_time = 10.0 + 100.0 * 0.1 = 20.0

    expected_vel_y_after_second_update = 20.0
    expected_sub_y_after_second_update = 1.0

    assert particle.vel_y == pytest.approx(expected_vel_y_after_second_update)
    assert particle.sub_y == pytest.approx(expected_sub_y_after_second_update)

    # Third update (to further confirm the pattern)
    # Initial state for this update: sub_y = 1.0, vel_y = 20.0
    particle.update(delta_time)
    # Step 1: sub_y = sub_y + vel_y * delta_time = 1.0 + 20.0 * 0.1 = 3.0
    # Step 2: vel_y = vel_y + gravity * delta_time = 20.0 + 100.0 * 0.1 = 30.0

    expected_vel_y_after_third_update = 30.0
    expected_sub_y_after_third_update = 3.0

    assert particle.vel_y == pytest.approx(expected_vel_y_after_third_update)
    assert particle.sub_y == pytest.approx(expected_sub_y_after_third_update)


def test_add_particle_origin_is_centered_with_spread(
    ps_default: SubTileParticleSystem,
) -> None:
    ps = ps_default
    tile_x, tile_y = 2.0, 3.0  # Using float to match add_particle signature

    # Expected center of tile (2,3) in sub-pixels if subdivision is 3
    # (2.0 + 0.5) * 3 = 7.5
    # (3.0 + 0.5) * 3 = 10.5
    expected_center_sub_x = (tile_x + 0.5) * ps.subdivision
    expected_center_sub_y = (tile_y + 0.5) * ps.subdivision

    # Mock random.uniform to control the spread.
    # Assuming your spread_sub_pixels = 0.5
    # First call for x_spread, second for y_spread
    mocked_spread_x = 0.2
    mocked_spread_y = -0.3
    with patch("random.uniform", FixedUniform([mocked_spread_x, mocked_spread_y])):
        ps.add_particle(tile_x, tile_y, vel_x=0, vel_y=0, lifetime=1)

    assert len(ps.particles) == 1
    p = ps.particles[0]
    # Spread is relative to the sub-pixel grid, not tile grid
    assert p.sub_x == pytest.approx(expected_center_sub_x + mocked_spread_x)
    assert p.sub_y == pytest.approx(expected_center_sub_y + mocked_spread_y)


def test_system_update_removes_dead_particles_from_list(
    ps_default: SubTileParticleSystem,
) -> None:
    ps = ps_default
    # vel in tiles/sec for add_particle
    ps.add_particle(
        tile_x=1, tile_y=1, vel_x=0, vel_y=0, lifetime=0.05
    )  # Will die soon
    ps.add_particle(tile_x=2, tile_y=2, vel_x=0, vel_y=0, lifetime=1.0)  # Will live

    assert len(ps.particles) == 2
    ps.update(delta_time=0.1)  # First particle should have its lifetime <= 0

    assert len(ps.particles) == 1
    # Check the remaining particle is the one that should be alive
    # (based on initial higher lifetime)
    remaining_particle = ps.particles[0]
    assert (
        remaining_particle.max_lifetime == 1.0
    )  # Assuming max_lifetime is set in SubParticle or add_particle sets it


def test_emit_radial_burst_origin_is_centered(
    ps_default: SubTileParticleSystem,
) -> None:
    ps = ps_default
    tile_x, tile_y = 1.0, 1.0
    expected_sub_x = (tile_x + 0.5) * ps.subdivision
    expected_sub_y = (tile_y + 0.5) * ps.subdivision

    # Mock random.uniform for angle, speed, lifetime.
    # The angle determines velocity, not origin for radial.
    with patch("random.uniform", FixedUniform([math.pi / 4, 5.0, 0.2])):
        ps.emit_radial_burst(
            tile_x,
            tile_y,
            count=1,
            speed_range=(5.0, 5.0),  # tiles/sec
            lifetime_range=(0.2, 0.2),
            colors_and_chars=[((1, 2, 3), "*")],
        )
    assert len(ps.particles) == 1
    p = ps.particles[0]
    assert p.sub_x == pytest.approx(expected_sub_x)
    assert p.sub_y == pytest.approx(expected_sub_y)
