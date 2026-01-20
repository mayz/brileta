from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from catley import colors
from catley.events import CombatInitiatedEvent, MessageEvent, publish_event
from catley.game.actors import Actor, Character, ai
from catley.game.actors.status_effects import OffBalanceEffect
from catley.game.enums import Disposition, OutcomeTier
from catley.game.items.properties import WeaponProperty
from catley.util.dice import Dice, roll_d

if TYPE_CHECKING:
    from catley.game.items.item_core import Item


@dataclass
class Consequence:
    """Represents a post-action consequence."""

    type: str
    target: Actor | None = None
    data: dict[str, Any] = field(default_factory=dict)


class AttackConsequenceGenerator:
    """Generate consequences for :class:`AttackIntent`.

    Starting with combat validates the approach before expanding to other systems.
    """

    def generate(
        self,
        attacker: Character,
        weapon: Item,
        outcome_tier: OutcomeTier,
    ) -> list[Consequence]:
        consequences: list[Consequence] = []

        # Silent weapons don't generate noise alerts
        if not self._is_silent_weapon(weapon):
            consequences.append(
                Consequence(
                    type="noise_alert",
                    data={"source": attacker, "radius": 5},
                )
            )

        if outcome_tier == OutcomeTier.CRITICAL_FAILURE:
            # On critical failure, randomly select among possible consequences:
            # - weapon_drop (40%): Most punishing, drop equipped weapon
            # - self_injury (35%): Fumble causes attacker to hurt themselves
            # - off_balance (25%): Attacker has disadvantage on next action
            roll = random.random()
            if roll < 0.40:
                consequences.append(
                    Consequence(
                        type="weapon_drop", target=attacker, data={"weapon": weapon}
                    )
                )
            elif roll < 0.75:  # 0.40 + 0.35
                consequences.append(
                    Consequence(
                        type="self_injury", target=attacker, data={"weapon": weapon}
                    )
                )
            else:  # remaining 0.25
                consequences.append(Consequence(type="off_balance", target=attacker))

        return consequences

    def _is_silent_weapon(self, weapon: Item) -> bool:
        """Check if weapon has SILENT property (doesn't alert nearby enemies)."""
        return WeaponProperty.SILENT in weapon.get_weapon_properties()


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
        elif consequence.type == "self_injury":
            self._apply_self_injury(consequence.target, consequence.data.get("weapon"))
        elif consequence.type == "off_balance":
            self._apply_off_balance(consequence.target)

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

        gw.spawn_ground_item(weapon, actor.x, actor.y)
        publish_event(MessageEvent(f"{actor.name} drops {weapon.name}!", colors.ORANGE))

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
                # Trigger combat mode if player caused the noise
                if source == gw.player:
                    publish_event(
                        CombatInitiatedEvent(
                            attacker=source,
                            defender=actor,
                        )
                    )

    def _apply_self_injury(self, target: Actor | None, weapon: Item | None) -> None:
        """Apply self-injury consequence from a fumbled attack.

        The attacker hurts themselves with their own weapon - a fumbled swing,
        ricochet, or mishandling the weapon. Deals half the weapon's damage dice.
        """
        if not target or not isinstance(target, Character) or not weapon:
            return

        # Get damage dice from the weapon (prefer ranged, then melee)
        damage_dice = self._get_weapon_damage_dice(weapon)
        if damage_dice is None:
            return

        # Roll half damage (e.g., d8 weapon = d4 self-damage)
        half_sides = max(2, damage_dice.sides // 2)
        damage = roll_d(half_sides)

        health = target.health
        if health is not None:
            health.hp -= damage

        publish_event(
            MessageEvent(
                f"{target.name} fumbles and hurts themselves for {damage} damage!",
                colors.ORANGE,
            )
        )

    def _apply_off_balance(self, target: Actor | None) -> None:
        """Apply off-balance consequence from a fumbled attack.

        The recoil or awkward motion throws the attacker off balance,
        giving them disadvantage on their next action.
        """
        if not target or not isinstance(target, Character):
            return

        target.status_effects.apply_status_effect(OffBalanceEffect())
        publish_event(
            MessageEvent(f"{target.name} is thrown off balance!", colors.ORANGE)
        )

    def _get_weapon_damage_dice(self, weapon: Item) -> Dice | None:
        """Get the damage dice from a weapon, preferring ranged over melee."""
        if weapon.ranged_attack:
            return weapon.ranged_attack.damage_dice
        if weapon.melee_attack:
            return weapon.melee_attack.damage_dice
        return None
