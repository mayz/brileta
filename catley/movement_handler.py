from __future__ import annotations

import time
from typing import TYPE_CHECKING

import tcod.event

from catley import config
from catley.game.actions.movement import MoveIntent

if TYPE_CHECKING:
    from catley.controller import Controller

class MovementInputHandler:
    """Handles smooth movement input with key repeat timing for turn-based gameplay.

    This specialized input handler manages the timing and state for directional movement
    keys, providing a smooth movement experience in a turn-based game by implementing
    a "tap vs. hold" system similar to text editors or operating system key repeat.

    Turn-based games typically require discrete actions, but movement benefits from
    feeling fluid when players hold directional keys. This handler bridges that gap
    by converting held movement keys into appropriately-timed movement intents while
    maintaining the turn-based nature of the game.

    Without this system, players would need to press and release movement keys for
    every single tile movement, which feels jerky. With it, they
    can tap for single moves or hold for continuous movement.

    Key Repeat Behavior:
    -------------------
    - **First Move**: Immediate response when movement key is first pressed
    - **Delay Phase**: Brief pause (KEY_REPEAT_DELAY) before repeat begins
    - **Repeat Phase**: Continuous movement at faster intervals (KEY_REPEAT_INTERVAL)

    This mimics standard keyboard repeat behavior found in text editors and ensures
    both precise single-tile movement and smooth multi-tile traversal.

    Integration:
    -----------
    Works alongside the main InputHandler by:
    - Receiving movement key state from InputHandler.movement_keys
    - Generating MoveIntent objects when timing conditions are met
    - Allowing other input to be processed normally by the main handler

    The separation allows movement to have special timing logic while keeping
    other game actions (combat, menus, etc.) as immediate single-press responses.
    """
    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.next_move_time: float = 0.0
        self.is_first_move_of_burst: bool = True

    def generate_intent(
        self, movement_keys: set[tcod.event.KeySym]
    ) -> MoveIntent | None:
        # Don't generate movement if player is dead
        if not self.controller.gw.player.health.is_alive():
            return None

        dx, dy = 0, 0
        if (
            tcod.event.KeySym.UP in movement_keys
            or tcod.event.KeySym.k in movement_keys
        ):
            dy -= 1
        if (
            tcod.event.KeySym.DOWN in movement_keys
            or tcod.event.KeySym.j in movement_keys
        ):
            dy += 1
        if (
            tcod.event.KeySym.LEFT in movement_keys
            or tcod.event.KeySym.h in movement_keys
        ):
            dx -= 1
        if (
            tcod.event.KeySym.RIGHT in movement_keys
            or tcod.event.KeySym.l in movement_keys
        ):
            dx += 1

        if dx == 0 and dy == 0:
            self.is_first_move_of_burst = True
            return None

        current_time = time.perf_counter()

        if self.is_first_move_of_burst:
            self.is_first_move_of_burst = False
            self.next_move_time = current_time + config.MOVEMENT_KEY_REPEAT_DELAY
            return MoveIntent(self.controller, self.controller.gw.player, dx, dy)

        if current_time >= self.next_move_time:
            self.next_move_time = current_time + config.MOVEMENT_KEY_REPEAT_INTERVAL
            return MoveIntent(self.controller, self.controller.gw.player, dx, dy)

        return None
