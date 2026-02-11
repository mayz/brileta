"""Shared displacement mechanics for stunt executors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors
from brileta.environment.tile_types import (
    get_tile_hazard_info,
    get_tile_type_name_by_id,
)
from brileta.events import MessageEvent, publish_event
from brileta.game.actors import Character
from brileta.game.actors.status_effects import OffBalanceEffect
from brileta.game.enums import StepBlock
from brileta.util.dice import roll_d
from brileta.util.pathfinding import probe_step

if TYPE_CHECKING:
    from brileta.controller import Controller


def attempt_displacement(
    controller: Controller, defender: Character, dx: int, dy: int
) -> bool:
    """Attempt to move a defender one tile and resolve displacement side effects."""
    game_map = controller.gw.game_map
    dest_x = defender.x + dx
    dest_y = defender.y + dy

    block = probe_step(game_map, controller.gw, dest_x, dest_y)
    match block:
        case None:
            defender.move(dx, dy, controller)
            _check_hazard_landing(controller, defender, dest_x, dest_y)
            return True
        case StepBlock.WALL | StepBlock.CLOSED_DOOR:
            _handle_wall_impact(defender)
            return False
        case StepBlock.BLOCKED_BY_ACTOR | StepBlock.BLOCKED_BY_CONTAINER:
            blocking_actor = controller.gw.get_actor_at_location(dest_x, dest_y)
            _handle_actor_collision(defender, blocking_actor)
            return False
        case _:  # OUT_OF_BOUNDS or any future variant
            return False


def _handle_wall_impact(defender: Character) -> None:
    """Apply impact effects when a displaced character hits a wall."""
    impact_damage = roll_d(4)
    defender.take_damage(impact_damage, damage_type="impact")
    defender.status_effects.apply_status_effect(OffBalanceEffect())

    msg = f"{defender.name} slams into the wall for {impact_damage} damage!"
    publish_event(MessageEvent(msg, colors.WHITE))


def _handle_actor_collision(defender: Character, blocking_actor: object) -> None:
    """Apply collision effects when a displaced character hits another actor."""
    defender.status_effects.apply_status_effect(OffBalanceEffect())

    if isinstance(blocking_actor, Character):
        blocking_actor.status_effects.apply_status_effect(OffBalanceEffect())
        publish_event(
            MessageEvent(
                f"{defender.name} collides with {blocking_actor.name}! "
                f"Both are off-balance!",
                colors.LIGHT_BLUE,
            )
        )
        return

    publish_event(
        MessageEvent(
            f"{defender.name} collides with something and is thrown off-balance!",
            colors.LIGHT_BLUE,
        )
    )


def _check_hazard_landing(
    controller: Controller, defender: Character, x: int, y: int
) -> None:
    """Log hazard landing feedback; hazard damage is applied by turn processing."""
    game_map = controller.gw.game_map
    tile_id = int(game_map.tiles[x, y])
    damage_dice, _damage_type = get_tile_hazard_info(tile_id)

    if damage_dice:
        tile_name = get_tile_type_name_by_id(tile_id)
        publish_event(
            MessageEvent(
                f"{defender.name} lands in the {tile_name.lower()}!",
                colors.ORANGE,
            )
        )
