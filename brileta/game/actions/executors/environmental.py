"""Executors for environmental damage actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors
from brileta.events import ActorDeathEvent, MessageEvent, publish_event
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.actors import Character
from brileta.types import WorldTilePos

if TYPE_CHECKING:
    from brileta.game.actions.environmental import EnvironmentalDamageIntent


class EnvironmentalDamageExecutor(ActionExecutor):
    """Executes environmental damage intents from hazards like fire and radiation.

    This executor handles passive environmental damage sources that don't have
    an attacker or weapon, such as fires, radiation zones, acid pools, and
    falling damage. For weapon-based area effects, use WeaponAreaEffectExecutor.
    """

    def execute(self, intent: EnvironmentalDamageIntent) -> GameActionResult | None:  # type: ignore[override]
        """Execute environmental damage by finding actors at coordinates."""
        if intent.damage_amount <= 0:
            return GameActionResult(succeeded=False)

        # Find all actors at the affected coordinates
        affected_actors: list[tuple[Character, WorldTilePos]] = []
        for coord in intent.affected_coords:
            x, y = coord
            actors_here = intent.controller.gw.actor_spatial_index.get_at_point(x, y)
            valid_actors = [
                (actor, coord)
                for actor in actors_here
                if (
                    isinstance(actor, Character)
                    and actor.health
                    and actor.health.is_alive()
                    and actor is not intent.source_actor
                )
            ]
            affected_actors.extend(valid_actors)

        if not affected_actors:
            # No living characters to damage
            return GameActionResult(succeeded=True)

        # Apply damage to each affected actor
        for actor, _coord in affected_actors:
            self._apply_environmental_damage(intent, actor)

        return GameActionResult(succeeded=True)

    def _apply_environmental_damage(
        self, intent: EnvironmentalDamageIntent, actor: Character
    ) -> None:
        """Apply environmental damage to a single actor and log appropriate messages."""
        # Apply the damage
        actor.take_damage(intent.damage_amount, damage_type=intent.damage_type)

        # Log damage message with environmental context
        damage_type_desc = intent.damage_type if intent.damage_type != "normal" else ""
        damage_type_text = f" {damage_type_desc}" if damage_type_desc else ""

        message = (
            f"{actor.name} takes {intent.damage_amount}{damage_type_text} "
            f"damage from {intent.source_description}."
        )
        if actor.health and actor.health.is_alive():
            message += f" ({actor.name} has {actor.health.hp} HP left.)"

        # Use orange color for environmental damage (matches AreaEffectExecutor pattern)
        publish_event(MessageEvent(message, colors.ORANGE))

        # Handle death
        if actor.health and not actor.health.is_alive():
            publish_event(MessageEvent(f"{actor.name} has been killed!", colors.RED))
            publish_event(ActorDeathEvent(actor))
