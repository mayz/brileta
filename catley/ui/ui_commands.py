from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from .menu_system import Menu, PickupMenu

if TYPE_CHECKING:
    import tcod.context

    from catley.controller import Controller
    from catley.game.entities import Entity


class UICommand(abc.ABC):
    """Commands that affect the UI/application."""

    @abc.abstractmethod
    def execute(self) -> None:
        pass


class ToggleFullscreenUICommand(UICommand):
    """Action for toggling fullscreen mode."""

    def __init__(self, context: tcod.context.Context) -> None:
        self.context = context

    def execute(self) -> None:
        self.context.present(self.context.console, keep_aspect=True)


class QuitUICommand(UICommand):
    """Action for quitting the game."""

    def execute(self) -> None:
        raise SystemExit()


class SelectOrDeselectEntityUICommand(UICommand):
    """
    Action to select a new entity or deselect (clear) the current selection.
    """

    def __init__(self, controller: Controller, selection: Entity | None):
        """
        Args:
            controller: The game controller.
            selection: The entity to select (or None to deselect).
        """
        super().__init__()
        self.controller = controller
        self.selection = selection

    def execute(self) -> None:
        self.controller.gw.selected_entity = self.selection


class OpenMenuUICommand(UICommand):
    """Action to open a menu, like the inventory or help menu."""

    def __init__(self, controller: Controller, menu_class: type[Menu]):
        self.controller = controller
        self.menu_class = menu_class

    def execute(self) -> None:
        menu = self.menu_class(self.controller)
        self.controller.menu_system.show_menu(menu)


class OpenPickupMenuUICommand(OpenMenuUICommand):
    """Action to open the pickup menu for items at the player's location."""

    def __init__(self, controller: Controller):
        self.controller = controller
        self.player = controller.gw.player

    def execute(self) -> None:
        # Check if there are items to pick up, otherwise do nothing.
        if self.controller.gw.has_pickable_items_at_location(
            self.player.x, self.player.y
        ):
            menu = PickupMenu(self.controller, (self.player.x, self.player.y))
            self.controller.menu_system.show_menu(menu)
