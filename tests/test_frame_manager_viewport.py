from dataclasses import dataclass
from unittest.mock import MagicMock

from catley.view.frame_manager import FrameManager
from catley.view.viewport import ViewportSystem


@dataclass
class DummyGameWorldPanel:
    x: int
    y: int
    width: int
    height: int
    viewport_system: ViewportSystem


@dataclass
class DummyController:
    gw: MagicMock


@dataclass
class DummyFM:
    controller: DummyController
    game_world_panel: DummyGameWorldPanel


def make_dummy_fm() -> DummyFM:
    """Create a minimal FrameManager stand-in with typed attributes."""
    gw = MagicMock()
    gw.game_map.width = 20
    gw.game_map.height = 20
    controller = DummyController(gw=gw)

    vs = ViewportSystem(10, 8)
    vs.camera.set_position(4.0, 4.0)
    panel = DummyGameWorldPanel(x=5, y=3, width=10, height=8, viewport_system=vs)

    return DummyFM(controller=controller, game_world_panel=panel)


def test_root_to_world_conversion_inside_panel() -> None:
    fm = make_dummy_fm()
    root_coords = (fm.game_world_panel.x + 5, fm.game_world_panel.y + 2)

    result = FrameManager.get_tile_map_coords_from_root_coords(fm, root_coords)
    assert result == (4, 2)


def test_root_to_world_conversion_outside_panel() -> None:
    fm = make_dummy_fm()
    root_coords = (0, 0)

    result = FrameManager.get_tile_map_coords_from_root_coords(fm, root_coords)
    assert result is None
