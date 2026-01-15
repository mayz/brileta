"""Executor for container-related actions like searching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor

if TYPE_CHECKING:
    from catley.game.actions.environment import SearchContainerIntent


class SearchContainerExecutor(ActionExecutor):
    """Opens the loot UI for searching a container or corpse.

    This executor handles the SearchContainerIntent by opening the DualPaneMenu
    with the target actor as the item source. The menu allows bidirectional
    item transfer between the player's inventory and the container/corpse.
    """

    def execute(self, intent: SearchContainerIntent) -> GameActionResult:  # type: ignore[override]
        """Execute the search action by opening the loot UI.

        Args:
            intent: The SearchContainerIntent specifying what to search

        Returns:
            GameActionResult indicating success/failure
        """
        target = intent.target

        # Validate target has an inventory to search
        if target.inventory is None:
            return GameActionResult(succeeded=False, block_reason="Nothing to search")

        # Check if inventory is empty
        if len(target.inventory) == 0:
            return GameActionResult(succeeded=False, block_reason="Nothing to loot")

        # Import here to avoid circular imports
        from catley.view.ui.dual_pane_menu import ActorInventorySource, DualPaneMenu

        # Create the source descriptor for the target
        source = ActorInventorySource(actor=target, label=target.name)

        # Open the dual-pane loot menu
        menu = DualPaneMenu(intent.controller, source=source)
        intent.controller.overlay_system.show(menu)

        return GameActionResult(succeeded=True)
