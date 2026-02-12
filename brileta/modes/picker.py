"""PickerMode - Modal target/tile selection for any game action.

PickerMode provides a standalone selection interface that any mode can use.
It handles cursor changes, mouse input, and blocking of other input while
a selection is being made. The calling mode configures callbacks for
selection and cancellation.

Example use cases:
- CombatMode: "pick a target to attack"
- Future consumable use: "pick where to throw this item"
- Future abilities: "pick a tile for area effect"
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brileta import input_events
from brileta.game.actors import Actor, Character
from brileta.modes.base import Mode
from brileta.types import WorldTilePos

if TYPE_CHECKING:
    from brileta.controller import Controller


@dataclass
class PickerResult:
    """Result of a picker session.

    Attributes:
        actor: The actor at the selected tile, if any.
        tile: The world coordinates (x, y) of the selected tile.
    """

    actor: Actor | None
    tile: WorldTilePos


class PickerMode(Mode):
    """Modal mode for selecting an actor or tile in the world.

    Blocks all input until user clicks a selection or cancels with Escape.
    The only visual change is the cursor (crosshair). Pops itself from the
    mode stack when selection is made or cancelled.

    Typical usage:
        def on_select(result: PickerResult) -> None:
            if result.actor:
                print(f"Selected {result.actor}")

        picker_mode.start(
            on_select=on_select,
            on_cancel=lambda: print("Cancelled"),
            valid_filter=lambda x, y: is_valid_target(x, y),
        )
    """

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)

        # Callbacks configured by start()
        self._on_select: Callable[[PickerResult], None] | None = None
        self._on_cancel: Callable[[], None] | None = None
        self._valid_filter: Callable[[int, int], bool] | None = None
        self._render_underneath: Callable[[], None] | None = None

        # Cache cursor manager reference (may be None in tests)
        self._cursor_manager = None
        if controller.frame_manager is not None and hasattr(
            controller.frame_manager, "cursor_manager"
        ):
            self._cursor_manager = controller.frame_manager.cursor_manager

    def start(
        self,
        on_select: Callable[[PickerResult], None],
        on_cancel: Callable[[], None] | None = None,
        valid_filter: Callable[[int, int], bool] | None = None,
        render_underneath: Callable[[], None] | None = None,
    ) -> None:
        """Configure and push this mode onto the stack.

        Args:
            on_select: Called with result when a valid selection is made.
            on_cancel: Called when user cancels (Escape). Optional.
            valid_filter: Optional filter function (x, y) -> bool.
                          If provided, only tiles where this returns True
                          are valid selection targets.
            render_underneath: Optional callback to render the calling mode's
                               visuals (e.g., combat highlights) while picking.
        """
        self._on_select = on_select
        self._on_cancel = on_cancel
        self._valid_filter = valid_filter
        self._render_underneath = render_underneath
        self.controller.push_mode(self)

    def enter(self) -> None:
        """Activate picker mode and show crosshair cursor."""
        super().enter()
        if self._cursor_manager is not None:
            self._cursor_manager.set_active_cursor_type("crosshair")

    def _exit(self) -> None:
        """Clean up picker state and restore arrow cursor."""
        if self._cursor_manager is not None:
            self._cursor_manager.set_active_cursor_type("arrow")

        # Clear callbacks to avoid holding references
        self._on_select = None
        self._on_cancel = None
        self._valid_filter = None
        self._render_underneath = None

        super()._exit()

    def handle_input(self, event: input_events.InputEvent) -> bool:
        """Handle picker input.

        Escape or T cancels and pops. Left click selects if valid.
        Other input falls through to modes below (inventory, weapon switching, etc.).
        """
        if not self.active:
            return False

        match event:
            case input_events.KeyDown(sym=input_events.KeySym.ESCAPE):
                # Escape cancels picking
                callback = self._on_cancel
                self.controller.pop_mode()
                if callback:
                    callback()
                return True

            case input_events.MouseButtonDown(button=input_events.MouseButton.LEFT):
                tile = self._get_tile_at_mouse(event)
                if tile and self._is_valid_selection(tile):
                    actor = self._get_actor_at_tile(tile)
                    result = PickerResult(actor=actor, tile=tile)
                    callback = self._on_select
                    self.controller.pop_mode()
                    if callback:
                        callback(result)
                    return True
                # Only consume clicks on the game map (invalid targets).
                # Clicks outside the game map (on UI) fall through to modes below.
                return tile is not None

        # Let other input fall through to modes below (inventory, weapon slots, etc.)
        return False

    def render_world(self) -> None:
        """Render picking-specific visuals.

        Optionally renders the calling mode's visuals (e.g., combat highlights)
        if render_underneath was provided.
        """
        if not self.active:
            return

        if self._render_underneath:
            self._render_underneath()

    def _get_tile_at_mouse(
        self, event: input_events.MouseButtonDown
    ) -> WorldTilePos | None:
        """Convert mouse click position to world tile coordinates."""
        if self.controller.frame_manager is None:
            return None

        graphics = self.controller.graphics
        scale_x, scale_y = graphics.get_display_scale_factor()
        scaled_px_x = event.position.x * scale_x
        scaled_px_y = event.position.y * scale_y
        root_tile_x, root_tile_y = graphics.pixel_to_tile(scaled_px_x, scaled_px_y)

        root_tile_pos = (int(root_tile_x), int(root_tile_y))
        world_tile_pos = (
            self.controller.frame_manager.get_world_coords_from_root_tile_coords(
                root_tile_pos
            )
        )

        if world_tile_pos is None:
            return None

        world_x, world_y = world_tile_pos
        gw = self.controller.gw

        # Check bounds
        if not (0 <= world_x < gw.game_map.width and 0 <= world_y < gw.game_map.height):
            return None

        return (world_x, world_y)

    def _is_valid_selection(self, tile: WorldTilePos) -> bool:
        """Check if a tile is a valid selection target."""
        if self._valid_filter is None:
            return True
        return self._valid_filter(tile[0], tile[1])

    def _get_actor_at_tile(self, tile: WorldTilePos) -> Actor | None:
        """Get the actor at a tile position, if any."""
        world_x, world_y = tile
        gw = self.controller.gw

        # Check visibility
        if not gw.game_map.visible[world_x, world_y]:
            return None

        actor = gw.get_actor_at_location(world_x, world_y)
        if actor is None:
            return None

        # Only return living characters
        if isinstance(actor, Character) and not actor.health.is_alive():
            return None

        return actor
