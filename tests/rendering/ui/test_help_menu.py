"""Tests for HelpMenu command listing and actions."""

from __future__ import annotations

from brileta.view.ui.help_menu import HelpMenu
from tests.helpers import get_controller_with_dummy_world


def test_help_menu_lists_mini_map_toggle() -> None:
    """Help menu should advertise the mini-map hotkey."""
    controller = get_controller_with_dummy_world()
    menu = HelpMenu(controller)

    menu.show()

    assert any(opt.key == "M" and opt.text == "Toggle mini-map" for opt in menu.options)


def test_help_menu_mini_map_action_toggles_visibility() -> None:
    """Help action should toggle the same mini-map view used in gameplay."""
    controller = get_controller_with_dummy_world()
    menu = HelpMenu(controller)
    fm = controller.frame_manager
    assert fm is not None

    fm.mini_map_view.visible = True
    menu._on_toggle_minimap()
    assert fm.mini_map_view.visible is False

    menu._on_toggle_minimap()
    assert fm.mini_map_view.visible is True
