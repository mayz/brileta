from types import SimpleNamespace
from typing import Any, cast

import tcod.event

from catley.controller import Controller
from catley.view.ui.context_menu import ContextMenu
from tests.test_text_backends import _make_renderer


def _make_controller() -> Controller:
    renderer = _make_renderer()
    cast(Any, renderer.root_console).width = 80
    cast(Any, renderer.root_console).height = 50
    converter = SimpleNamespace(pixel_to_tile=lambda x, y: (x, y))
    return cast(
        Controller, SimpleNamespace(renderer=renderer, coordinate_converter=converter)
    )


def test_context_menu_closes_on_click_outside() -> None:
    controller = _make_controller()
    menu = ContextMenu(controller, None, (5, 5))
    menu.show()
    menu._calculate_dimensions()

    outside = (menu.x_tiles + menu.width + 1, menu.y_tiles + menu.height + 1)
    event = tcod.event.MouseButtonDown(outside, outside, tcod.event.MouseButton.LEFT)

    consumed = menu.handle_input(event)

    assert consumed
    assert not menu.is_active


def test_context_menu_stays_open_on_click_inside() -> None:
    controller = _make_controller()
    menu = ContextMenu(controller, None, (5, 5))
    menu.show()
    menu._calculate_dimensions()

    inside = (menu.x_tiles + 1, menu.y_tiles + 1)
    event = tcod.event.MouseButtonDown(inside, inside, tcod.event.MouseButton.LEFT)

    consumed = menu.handle_input(event)

    assert not consumed
    assert menu.is_active
