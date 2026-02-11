"""
User interface commands that control application behavior.

Defines commands that handle user interface interactions and application-level
functionality, distinct from in-world game actions.

UICommand:
    Base class for all interface commands. These handle user input that
    affects the application state, display, or user experience rather than
    the game world itself.

Examples:
    - QuitUICommand: Exit the application
    - ToggleFullscreenUICommand: Change display mode
    - OpenMenuUICommand: Show/hide interface views

UI commands are immediate and don't consume game turns. They handle the
"meta" layer of interaction - how the player interacts with the game
interface rather than how their character acts within the game world.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING

from brileta.view.ui.overlays import Menu

if TYPE_CHECKING:
    from brileta.app import App
    from brileta.controller import Controller


class UICommand(abc.ABC):
    """Commands that affect the UI/application."""

    @abc.abstractmethod
    def execute(self) -> None:
        pass


class ToggleFullscreenUICommand(UICommand):
    """Command for toggling fullscreen mode."""

    def __init__(self, app: App) -> None:
        self.app = app

    def execute(self) -> None:
        self.app.toggle_fullscreen()


class QuitUICommand(UICommand):
    """Command for quitting the game."""

    def __init__(self, app: App) -> None:
        self.app = app

    def execute(self) -> None:
        self.app.quit()


class OpenMenuUICommand(UICommand):
    """Command to open a menu, like the inventory or help menu."""

    def __init__(
        self,
        controller: Controller,
        menu_class: Callable[[Controller], Menu],
    ) -> None:
        self.controller = controller
        self.menu_class = menu_class

    def execute(self) -> None:
        assert self.controller.overlay_system is not None
        menu = self.menu_class(self.controller)
        self.controller.overlay_system.show_menu(menu)


class OpenExistingMenuUICommand(UICommand):
    """Command to open an already-created menu instance."""

    def __init__(self, controller: Controller, menu: Menu) -> None:
        self.controller = controller
        self.menu = menu

    def execute(self) -> None:
        assert self.controller.overlay_system is not None
        self.controller.overlay_system.show_menu(self.menu)


def open_inventory_or_loot(controller: Controller) -> None:
    """Open inventory, or loot mode if standing on items.

    This is the shared implementation for opening the inventory menu.
    If the player is standing on pickable items, opens a two-pane loot
    interface showing both the ground items and player inventory.
    Otherwise, opens the regular inventory menu.

    Args:
        controller: The game controller.
    """
    from brileta.view.ui.inventory import DualPaneMenu, ExternalInventory

    player = controller.gw.player
    if controller.gw.has_pickable_items_at_location(player.x, player.y):
        source = ExternalInventory((player.x, player.y), "On the ground")
        menu = DualPaneMenu(controller, source=source)
        OpenExistingMenuUICommand(controller, menu).execute()
    else:
        OpenMenuUICommand(controller, DualPaneMenu).execute()
