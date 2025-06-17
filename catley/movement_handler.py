from __future__ import annotations

import time
from typing import TYPE_CHECKING

import tcod.event

from catley.game.actions.movement import MoveIntent

if TYPE_CHECKING:
    from catley.controller import Controller


class MovementInputHandler:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.KEY_REPEAT_DELAY: float = 0.25
        self.KEY_REPEAT_INTERVAL: float = 0.05
        self.next_move_time: float = 0.0
        self.is_first_move_of_burst: bool = True

    def generate_intent(
        self, movement_keys: set[tcod.event.KeySym]
    ) -> MoveIntent | None:
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
            self.next_move_time = current_time + self.KEY_REPEAT_DELAY
            return MoveIntent(self.controller, self.controller.gw.player, dx, dy)

        if current_time >= self.next_move_time:
            self.next_move_time = current_time + self.KEY_REPEAT_INTERVAL
            return MoveIntent(self.controller, self.controller.gw.player, dx, dy)

        return None
