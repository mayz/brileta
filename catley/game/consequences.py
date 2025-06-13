from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actors import Actor, Character, ai, components
from catley.game.enums import Disposition, OutcomeTier
from catley.game.game_world import GameWorld

if TYPE_CHECKING:
    from catley.game.items.item_core import Item


@dataclass
class Consequence:
    """Represents a post-action consequence."""

    type: str
    target: Actor | None = None
    data: dict[str, Any] = field(default_factory=dict)


class AttackConsequenceGenerator:
    """Generate consequences for :class:`AttackAction`.

    Starting with combat validates the approach before expanding to other systems.
    """

    def generate(
        self,
        attacker: Character,
        weapon: Item,
        outcome_tier: OutcomeTier,
    ) -> list[Consequence]:
        consequences = [
            # Any attack generates a noise that alerts NPCs within 5 tiles.
            Consequence(
                type="noise_alert",
                data={"source": attacker, "radius": 5},
            )
        ]

        if outcome_tier == OutcomeTier.CRITICAL_FAILURE:
            # Right now, on a critical failure, the attacker just drops their
            # weapon. Obviously, lots of other consequences are possible, but this
            # validates the system.
            consequences.append(
                Consequence(
                    type="weapon_drop", target=attacker, data={"weapon": weapon}
                )
            )

        return consequences


class ConsequenceHandler:
    """Apply consequences to the game world."""

    def __init__(self) -> None:
        pass

    def apply_consequence(self, consequence: Consequence) -> None:
        if consequence.type == "weapon_drop":
            self._apply_weapon_drop(consequence.target, consequence.data.get("weapon"))
        elif consequence.type == "noise_alert":
            self._apply_noise_alert(
                consequence.data.get("source"),
                consequence.data.get("radius", 5),
            )

    def _apply_weapon_drop(self, actor: Actor | None, weapon: Item | None) -> None:
        if (
            not actor
            or not isinstance(actor, Character)
            or not weapon
            or not weapon.can_materialize
        ):
            return
        inv = actor.inventory
        if inv is None:
            return
        active_slot = inv.active_weapon_slot
        removed = inv.unequip_slot(active_slot)
        if removed is None:
            return
        gw = actor.gw
        if gw is None:
            return

        ground_actor = self._spawn_dropped_weapon(actor, weapon)
        assert ground_actor.inventory is not None
        inv_comp = cast(components.InventoryComponent, ground_actor.inventory)
        inv_comp.add_to_inventory(weapon)
        publish_event(MessageEvent(f"{actor.name} drops {weapon.name}!", colors.ORANGE))

    def _spawn_dropped_weapon(self, actor: Character, weapon: Item) -> Actor:
        gw = cast(GameWorld, actor.gw)
        ground_actor = Actor(
            x=actor.x,
            y=actor.y,
            ch="%",
            color=colors.WHITE,
            name=f"Dropped {weapon.name}",
            game_world=gw,
            blocks_movement=False,
            inventory=components.InventoryComponent(components.StatsComponent()),
        )
        gw.add_actor(ground_actor)
        return ground_actor

    def _apply_noise_alert(self, source: Actor | None, radius: int) -> None:
        if not source or source.gw is None:
            return
        gw = source.gw

        # Use the spatial index for an efficient radius query.
        nearby_actors = gw.actor_spatial_index.get_in_radius(source.x, source.y, radius)

        for actor in nearby_actors:
            if actor is source:
                continue
            if not isinstance(actor, Character):
                continue
            if not isinstance(actor.ai, ai.DispositionBasedAI):
                continue
            if actor.ai.disposition != Disposition.HOSTILE:
                actor.ai.disposition = Disposition.HOSTILE
                publish_event(
                    MessageEvent(
                        f"{actor.name} is alerted by the noise!", colors.ORANGE
                    )
                )
