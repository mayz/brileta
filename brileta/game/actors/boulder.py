"""Boulder actor definition.

Boulders are static environmental blockers represented as bare actors.
They do not have health, inventory, AI, or energy components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors
from brileta.types import WorldTileCoord

from .core import Actor

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld

_BOULDER_COLOR: colors.Color = (132, 134, 138)


class Boulder(Actor):
    """A static boulder that blocks movement and grants adjacent cover."""

    _GROUND_ANCHOR_Y = 0.82

    def __init__(
        self,
        x: WorldTileCoord,
        y: WorldTileCoord,
        game_world: GameWorld | None = None,
        ch: str = "#",
        color: colors.Color = _BOULDER_COLOR,
        visual_scale: float = 1.0,
    ) -> None:
        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name="Boulder",
            game_world=game_world,
            blocks_movement=True,
            shadow_height=2,
            visual_scale=visual_scale,
            cover_bonus=2,
            sprite_ground_anchor_y=self._GROUND_ANCHOR_Y,
        )
