"""MoveIntentGenerator - Converts held movement keys into MoveIntent objects.

This utility handles the timing logic for held-key movement, providing a smooth
"tap vs. hold" experience similar to text editors or OS key repeat.

Key Repeat Behavior:
- First Move: Immediate response when movement key is first pressed
- Delay Phase: Brief pause (KEY_REPEAT_DELAY) before repeat begins
- Repeat Phase: Continuous movement at faster intervals (KEY_REPEAT_INTERVAL)

Integration:
This class is owned by modes (e.g., ExploreMode) that support movement.
The mode tracks which keys are held and calls generate_intent() each frame
in its update() method. Modes that don't support movement (e.g., LockpickingMode)
simply don't use this class.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from catley import config, input_events
from catley.game.actions.movement import MoveIntent
from catley.input_events import Keys

if TYPE_CHECKING:
    from catley.controller import Controller


class MoveIntentGenerator:
    """Generates MoveIntent objects from held movement keys with key repeat timing.

    This class handles the timing logic that makes held keys feel smooth:
    - Tap: Immediate single move
    - Hold: Delay, then continuous movement at repeat interval

    The class does NOT track which keys are held - that's the mode's responsibility.
    It only handles the timing logic for generating appropriately-spaced intents.
    """

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.next_move_time: float = 0.0
        self.is_first_move_of_burst: bool = True

    def generate_intent(
        self, movement_keys: set[input_events.KeySym]
    ) -> MoveIntent | None:
        """Generate a MoveIntent if timing conditions are met.

        Args:
            movement_keys: Set of currently-held movement keys (tracked by mode)

        Returns:
            MoveIntent if a move should happen this frame, None otherwise.
        """
        # Don't generate movement if player is dead
        if not self.controller.gw.player.health.is_alive():
            return None

        dx, dy = 0, 0
        if input_events.KeySym.UP in movement_keys or Keys.KEY_K in movement_keys:
            dy -= 1
        if input_events.KeySym.DOWN in movement_keys or Keys.KEY_J in movement_keys:
            dy += 1
        if input_events.KeySym.LEFT in movement_keys or Keys.KEY_H in movement_keys:
            dx -= 1
        if input_events.KeySym.RIGHT in movement_keys or Keys.KEY_L in movement_keys:
            dx += 1

        if dx == 0 and dy == 0:
            self.is_first_move_of_burst = True
            return None

        current_time = time.perf_counter()

        if self.is_first_move_of_burst:
            self.is_first_move_of_burst = False
            self.next_move_time = current_time + config.MOVEMENT_KEY_REPEAT_DELAY
            return MoveIntent(
                self.controller,
                self.controller.gw.player,
                dx,
                dy,
                duration_ms=config.HELD_KEY_MOVE_DURATION_MS,
            )

        if current_time >= self.next_move_time:
            self.next_move_time = current_time + config.MOVEMENT_KEY_REPEAT_INTERVAL
            return MoveIntent(
                self.controller,
                self.controller.gw.player,
                dx,
                dy,
                duration_ms=config.HELD_KEY_MOVE_DURATION_MS,
            )

        return None
