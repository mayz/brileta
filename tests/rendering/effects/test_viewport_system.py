from typing import cast

import pytest

from brileta.game.actors import Actor
from brileta.view.render.viewport import ViewportSystem


class DummyActor:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


def test_world_to_screen_roundtrip() -> None:
    vs = ViewportSystem(8, 6)
    vs.camera.set_position(4.0, 3.0)

    vp_x, vp_y = vs.world_to_screen(4, 3)
    assert (vp_x, vp_y) == (4, 3)
    world = vs.screen_to_world(vp_x, vp_y)
    assert world == (4, 3)


def test_update_camera_clamps_to_bounds() -> None:
    vs = ViewportSystem(8, 6)
    actor = cast(Actor, DummyActor(0, 0))

    vs.update_camera(actor, 10, 10)
    assert vs.camera.world_x == pytest.approx(3.5)
    assert vs.camera.world_y == pytest.approx(2.5)


def test_is_visible() -> None:
    vs = ViewportSystem(8, 6)
    vs.camera.set_position(5.0, 5.0)

    assert vs.is_visible(5, 5)
    assert not vs.is_visible(0, 0)


def test_screen_offset_for_small_map() -> None:
    vs = ViewportSystem(10, 8)
    actor = cast(Actor, DummyActor(0, 0))

    vs.update_camera(actor, 5, 5)
    assert vs.offset_x == 2
    assert vs.offset_y == 1
    assert vs.world_to_screen(0, 0) == (2, 1)
    assert vs.screen_to_world(2, 1) == (0, 0)


def test_world_to_screen_clamps_when_shaken() -> None:
    """Screen coordinates should remain aligned when the camera is offset."""
    vs = ViewportSystem(10, 8)
    actor = cast(Actor, DummyActor(0, 0))
    vs.update_camera(actor, 50, 50)

    vs.camera.world_x -= 1.0
    vs.camera.world_y -= 1.0

    assert vs.world_to_screen(0, 0) == (0, 0)


def test_get_camera_fractional_offset_at_integer() -> None:
    """Fractional offset should be zero when camera is at integer position."""
    vs = ViewportSystem(10, 8)
    vs.camera.set_position(25.0, 30.0)

    frac_x, frac_y = vs.get_camera_fractional_offset()
    assert frac_x == pytest.approx(0.0)
    assert frac_y == pytest.approx(0.0)


def test_get_camera_fractional_offset_positive() -> None:
    """Fractional offset should be positive when camera is past tile center."""
    vs = ViewportSystem(10, 8)
    vs.camera.set_position(25.3, 30.7)

    frac_x, frac_y = vs.get_camera_fractional_offset()
    # 25.3 rounds to 25, so fractional is 0.3
    assert frac_x == pytest.approx(0.3)
    # 30.7 rounds to 31, so fractional is -0.3
    assert frac_y == pytest.approx(-0.3)


def test_get_camera_fractional_offset_negative() -> None:
    """Fractional offset should be negative when camera is before tile center."""
    vs = ViewportSystem(10, 8)
    vs.camera.set_position(25.4, 30.4)

    frac_x, frac_y = vs.get_camera_fractional_offset()
    # 25.4 rounds to 25, so fractional is 0.4
    assert frac_x == pytest.approx(0.4)
    # 30.4 rounds to 30, so fractional is 0.4
    assert frac_y == pytest.approx(0.4)


def test_get_camera_fractional_offset_at_half() -> None:
    """Fractional offset at 0.5 depends on banker's rounding."""
    vs = ViewportSystem(10, 8)
    # Python's round() uses banker's rounding: 25.5 -> 26, 26.5 -> 26
    vs.camera.set_position(25.5, 26.5)

    frac_x, frac_y = vs.get_camera_fractional_offset()
    # 25.5 rounds to 26, fractional is -0.5
    assert frac_x == pytest.approx(-0.5)
    # 26.5 rounds to 26, fractional is 0.5
    assert frac_y == pytest.approx(0.5)
