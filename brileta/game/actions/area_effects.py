"""
Area effect actions for weapons and abilities that affect multiple tiles.

Handles explosions, area-of-effect attacks, and other abilities that impact
multiple targets or tiles simultaneously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import GameIntent

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actors import Character
    from brileta.game.items.item_core import Item


class AreaEffectIntent(GameIntent):
    """Intent for executing an item's area effect."""

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        target_x: int,
        target_y: int,
        weapon: Item,
    ) -> None:
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.target_x = target_x
        self.target_y = target_y
        self.weapon = weapon
