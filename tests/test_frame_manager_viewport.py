from dataclasses import dataclass
from unittest.mock import MagicMock

from catley.view.frame_manager import FrameManager
from catley.view.panels.world_panel import WorldPanel
from catley.view.render.effects.effects import EffectLibrary
from catley.view.render.viewport import ViewportSystem


@dataclass
class DummyController:
    gw: MagicMock


def make_dummy_data() -> tuple[DummyController, WorldPanel]:
    """Creates the data structures needed for the FrameManager methods."""
    gw = MagicMock()
    gw.game_map.width = 20
    gw.game_map.height = 20
    controller = DummyController(gw=gw)

    vs = ViewportSystem(10, 8)
    # The camera's center is at (4, 4) in world space.
    vs.camera.set_position(4.0, 4.0)

    # To make the test more realistic, we must call update_camera,
    # as it's called every frame in the real game. This sets the map
    # dimensions on the viewport, which enables coordinate clamping.
    mock_player = MagicMock()
    mock_player.x = 4  # Position doesn't matter much for this test
    mock_player.y = 4
    vs.update_camera(mock_player, map_width=20, map_height=20)

    # Use the real GameWorldPanel, but we don't need to give it a real controller
    # since we are only testing coordinate transforms.
    panel = WorldPanel(controller=None, screen_shake=None)  # type: ignore
    # Manually resize and set the viewport

    x1, y1 = 5, 3
    width, height = 10, 8
    x2, y2 = x1 + width, y1 + height
    panel.resize(x1=x1, y1=y1, x2=x2, y2=y2)

    panel.viewport_system = vs

    return controller, panel


def test_root_to_world_conversion_inside_panel() -> None:
    controller, panel = make_dummy_data()

    # 1. Create a mock object to stand in for the FrameManager instance.
    mock_fm_self = MagicMock(spec=FrameManager)

    # 2. Attach the attributes the real method needs, using the correct names.
    mock_fm_self.controller = controller
    mock_fm_self.world_panel = panel

    root_coords = (panel.x + 5, panel.y + 2)  # This is (10, 5)

    # 3. Call the class method, passing our mock as the 'self' argument.
    result = FrameManager.get_tile_map_coords_from_root_coords(
        mock_fm_self, root_coords
    )

    # Let's trace the math:
    # root_coords = (10, 5)
    # panel top-left = (5, 3)
    # vp_x = 10 - 5 = 5
    # vp_y = 5 - 3 = 2
    # ---
    # camera center = (4, 4), viewport size = (10, 8)
    # world bounds left = round(4) - (10/2) = 4 - 5 = -1 -> clamped to 0
    # world bounds top = round(4) - (8/2) = 4 - 4 = 0
    # ---
    # world_x = vp_x + left = 5 + 0 = 5
    # world_y = vp_y + top = 2 + 0 = 2
    assert result == (5, 2)


def test_root_to_world_conversion_outside_panel() -> None:
    controller, panel = make_dummy_data()
    mock_fm_self = MagicMock(spec=FrameManager)
    mock_fm_self.controller = controller
    mock_fm_self.world_panel = panel

    root_coords = (0, 0)  # Clearly outside the panel which starts at (5, 3)

    result = FrameManager.get_tile_map_coords_from_root_coords(
        mock_fm_self, root_coords
    )
    assert result is None


def test_create_effect_visibility_and_coords() -> None:
    controller, panel = make_dummy_data()
    mock_fm_self = MagicMock(spec=FrameManager)
    mock_fm_self.controller = controller
    mock_fm_self.world_panel = panel

    # Create a MagicMock to stand in for the EffectLibrary.
    # Using `spec=EffectLibrary` is good practice as it ensures the mock
    # can only be called with methods that the real class has.
    mock_effect_library = MagicMock(spec=EffectLibrary)

    # Replace the real effect_library on the panel with our mock.
    panel.effect_library = mock_effect_library

    vs = panel.viewport_system

    # The point (4, 4) is visible by the camera at (4, 4).
    FrameManager.create_effect(mock_fm_self, "spark", 4, 4)

    # Now we assert against our mock object, which has the assertion methods.
    mock_effect_library.trigger.assert_called_once()
    called_name, ctx = mock_effect_library.trigger.call_args[0]
    assert called_name == "spark"
    assert (ctx.x, ctx.y) == vs.world_to_screen(4, 4)

    # Move camera far away, so (4, 4) is no longer visible.
    mock_effect_library.trigger.reset_mock()
    vs.camera.set_position(100.0, 100.0)
    FrameManager.create_effect(mock_fm_self, "spark", 4, 4)
    mock_effect_library.trigger.assert_not_called()
