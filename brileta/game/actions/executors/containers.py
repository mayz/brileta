"""Executor for container-related actions like searching."""

from __future__ import annotations

from brileta.game.actions.base import GameActionResult
from brileta.game.actions.environment import SearchContainerIntent
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.enums import ActionBlockReason


class SearchContainerExecutor(ActionExecutor[SearchContainerIntent]):
    """Opens the loot UI for searching a container or corpse.

    This executor handles the SearchContainerIntent by opening the DualPaneMenu
    with the target actor as the item source. The menu allows bidirectional
    item transfer between the player's inventory and the container/corpse.
    """

    def execute(self, intent: SearchContainerIntent) -> GameActionResult:
        """Execute the search action by opening the loot UI.

        Args:
            intent: The SearchContainerIntent specifying what to search

        Returns:
            GameActionResult indicating success/failure
        """
        target = intent.target

        # Validate target has an inventory to search
        if target.inventory is None:
            return GameActionResult(
                succeeded=False, block_reason=ActionBlockReason.NOTHING_TO_SEARCH
            )

        # For temporary containers (corpses, ground piles), block if empty
        # For permanent containers (bookcases, crates), allow opening to deposit items
        if len(target.inventory) == 0 and not target.blocks_movement:
            return GameActionResult(
                succeeded=False, block_reason=ActionBlockReason.NOTHING_TO_LOOT
            )

        # Import here to avoid circular imports
        from brileta.view.ui.inventory import ActorInventorySource, DualPaneMenu

        # Create the source descriptor for the target
        source = ActorInventorySource(actor=target, label=target.name)

        # Open the dual-pane loot menu
        menu = DualPaneMenu(intent.controller, source=source)
        intent.controller.overlay_system.show_overlay(menu)

        return GameActionResult(succeeded=True)
