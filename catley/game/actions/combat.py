"""
Combat actions for direct actor-vs-actor confrontations.

Handles attack resolution, damage calculation, weapon reloading, and all
combat-related mechanics including hit/miss determination and effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.action_plan import ActionPlan, ApproachStep, IntentStep
from catley.game.actions.base import GameIntent
from catley.game.actors import Character
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from catley.controller import Controller


class AttackIntent(GameIntent):
    """Intent for one :class:`Actor` to attack another, or shoot at a tile.

    This object only stores attack parameters. Execution occurs via
    :class:`~catley.game.actions.executors.combat.AttackExecutor` through the
    :class:`~catley.game.turn_manager.TurnManager`.

    Supports two modes:
    - Actor target: defender is set, target_x/target_y are None
    - Tile target: defender is None, target_x/target_y are set
    """

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character | None = None,
        weapon: Item | None = None,
        attack_mode: str | None = None,
        target_x: int | None = None,
        target_y: int | None = None,
    ) -> None:
        """Create a combat intent.

        Args:
            controller: Game controller providing context.
            attacker: The character performing the attack.
            defender: The target actor (None for tile shots).
            weapon: Optional weapon to use. Defaults to the attacker's active
                weapon or bare hands.
            attack_mode: Explicit attack mode ("melee" or "ranged"). ``None``
                preserves the old behaviour of automatically choosing based on
                distance.
            target_x: X coordinate for tile shots (None for actor targets).
            target_y: Y coordinate for tile shots (None for actor targets).
        """

        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender
        self.weapon = weapon
        self.attack_mode = attack_mode
        self.target_x = target_x
        self.target_y = target_y


class ReloadIntent(GameIntent):
    """Intent for reloading a ranged weapon.

    Execution is performed by
    :class:`~catley.game.actions.executors.combat.ReloadExecutor`.
    """

    def __init__(self, controller: Controller, actor: Character, weapon: Item) -> None:
        super().__init__(controller, actor)
        self.weapon = weapon

        # Type narrowing.
        self.actor: Character


# =============================================================================
# Action Plans for Combat
# =============================================================================

MeleeAttackPlan = ActionPlan(
    name="Melee Attack",
    requires_target=True,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=AttackIntent,
            params=lambda ctx: {
                "attacker": ctx.actor,
                "defender": ctx.target_actor,
                "weapon": ctx.weapon,
                "attack_mode": "melee",
            },
        ),
    ],
)
