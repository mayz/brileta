from dataclasses import dataclass
from typing import Any
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
    particle_system: Any = None
    effect_library: Any = None


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


def test_create_effect_visibility_and_coords() -> None:
    fm = make_dummy_fm()
    fm.game_world_panel.particle_system = MagicMock()
    fm.game_world_panel.effect_library = MagicMock()
    vs = fm.game_world_panel.viewport_system

    FrameManager.create_effect(fm, "spark", 4, 4)
    fm.game_world_panel.effect_library.trigger.assert_called_once()
    called_name, ctx = fm.game_world_panel.effect_library.trigger.call_args[0]
    assert called_name == "spark"
    assert (ctx.x, ctx.y) == vs.world_to_screen(4, 4)

    fm.game_world_panel.effect_library.trigger.reset_mock()
    vs.camera.set_position(12.0, 12.0)
    FrameManager.create_effect(fm, "spark", 4, 4)
    fm.game_world_panel.effect_library.trigger.assert_not_called()
