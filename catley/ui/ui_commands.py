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
    - OpenMenuUICommand: Show/hide interface panels
    - SelectOrDeselectActorUICommand: Change UI selection state

UI commands are immediate and don't consume game turns. They handle the
"meta" layer of interaction - how the player interacts with the game
interface rather than how their character acts within the game world.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from catley.ui.pickup_menu import PickupMenu
from catley.ui.target_menu import QuickActionBar, TargetMenu

if TYPE_CHECKING:
    import tcod.context

    from catley.controller import Controller
    from catley.game.actors import Actor
    from catley.ui.menu_core import Menu


class UICommand(abc.ABC):
    """Commands that affect the UI/application."""

    @abc.abstractmethod
    def execute(self) -> None:
        pass


class ToggleFullscreenUICommand(UICommand):
    """Command for toggling fullscreen mode."""

    def __init__(self, context: tcod.context.Context) -> None:
        self.context = context

    def execute(self) -> None:
        self.context.present(self.context.console, keep_aspect=True)


class QuitUICommand(UICommand):
    """Command for quitting the game."""

    def execute(self) -> None:
        raise SystemExit()


class SelectOrDeselectActorUICommand(UICommand):
    """
    Command to select a new actor or deselect (clear) the current selection.
    """

    def __init__(self, controller: Controller, selection: Actor | None):
        """
        Args:
            controller: The game controller.
            selection: The actor to select (or None to deselect).
        """
        super().__init__()
        self.controller = controller
        self.selection = selection

    def execute(self) -> None:
        self.controller.gw.selected_actor = self.selection

        # Hide any existing quick action bars first
        # Only hide menus that are QuickActionBar instances
        menus_to_remove = []
        for menu in self.controller.menu_system.active_menus:
            if menu.__class__.__name__ == "QuickActionBar":
                menus_to_remove.append(menu)

        for menu in menus_to_remove:
            menu.hide()
            if menu in self.controller.menu_system.active_menus:
                self.controller.menu_system.active_menus.remove(menu)

        # If selecting a non-player actor, show quick action bar
        if (
            self.selection
            and self.selection != self.controller.gw.player
            and self.selection.health
            and self.selection.health.is_alive()
        ):
            quick_bar = QuickActionBar(self.controller, self.selection)
            self.controller.menu_system.show_menu(quick_bar)


class OpenMenuUICommand(UICommand):
    """Command to open a menu, like the inventory or help menu."""

    def __init__(self, controller: Controller, menu_class: type[Menu]):
        self.controller = controller
        self.menu_class = menu_class

    def execute(self) -> None:
        menu = self.menu_class(self.controller)
        self.controller.menu_system.show_menu(menu)


class OpenPickupMenuUICommand(OpenMenuUICommand):
    """Command to open the pickup menu for items at the player's location."""

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


class OpenTargetMenuUICommand(UICommand):
    """Command to open the target menu for a specific actor or location."""

    def __init__(
        self,
        controller: Controller,
        target_actor: Actor | None = None,
        target_location: tuple[int, int] | None = None,
    ):
        self.controller = controller
        self.target_actor = target_actor
        self.target_location = target_location

    def execute(self) -> None:
        menu = TargetMenu(self.controller, self.target_actor, self.target_location)
        self.controller.menu_system.show_menu(menu)
