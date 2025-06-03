import math
from unittest.mock import patch

import pytest

from catley.render.particles import SubTileParticleSystem


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
