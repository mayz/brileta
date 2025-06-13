from typing import cast

import pytest

from catley.game.actors import Actor
from catley.view.render.viewport import ViewportSystem


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

    assert vs.is_visible(5, 5, 10, 10)
    assert not vs.is_visible(0, 0, 10, 10)


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
