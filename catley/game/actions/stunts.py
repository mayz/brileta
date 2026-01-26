"""
Combat stunts - physical maneuvers that manipulate enemy positioning.

Stunts are non-weapon combat actions like Push, Trip, Disarm, and Grapple.
They use opposed stat checks and can create environmental interactions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.action_plan import ActionPlan, ApproachStep, IntentStep
from catley.game.actions.base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class PushIntent(GameIntent):
    """Intent to push an adjacent enemy one tile away.

    Resolution uses Strength vs Strength opposed check. On success, the
    defender is moved one tile directly away from the attacker. Environmental
    interactions occur if pushed into hazards, walls, or other actors.

    Execution is performed by
    :class:`~catley.game.actions.executors.stunts.PushExecutor`.
    """

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character,
    ) -> None:
        """Create a push intent.

        Args:
            controller: Game controller providing context.
            attacker: The character performing the push.
            defender: The target character being pushed.
        """
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender


class TripIntent(GameIntent):
    """Intent to trip an adjacent enemy, causing them to fall prone.

    Resolution uses Agility vs Agility opposed check. Trip is the reliable
    knockdown option - any success applies TrippedEffect (skips 2 turns),
    unlike Push which only causes knockdown on critical success.

    Execution is performed by
    :class:`~catley.game.actions.executors.stunts.TripExecutor`.
    """

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character,
    ) -> None:
        """Create a trip intent.

        Args:
            controller: Game controller providing context.
            attacker: The character performing the trip.
            defender: The target character being tripped.
        """
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender


class KickIntent(GameIntent):
    """Intent to kick an adjacent enemy, dealing damage and pushing them back.

    Resolution uses Strength vs Agility opposed check. Kick combines damage
    with repositioning - it's the offensive alternative to Push. Unlike Push,
    Kick deals d4 damage on any success, and can trip the target on critical.

    Execution is performed by
    :class:`~catley.game.actions.executors.stunts.KickExecutor`.
    """

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character,
    ) -> None:
        """Create a kick intent.

        Args:
            controller: Game controller providing context.
            attacker: The character performing the kick.
            defender: The target character being kicked.
        """
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender


class PunchIntent(GameIntent):
    """Intent to punch an adjacent enemy with bare fists.

    Deals d3 damage using Strength vs Agility opposed check. When used via
    the ActionPlan system, holstering is handled by a separate HolsterWeaponIntent
    step that precedes the punch.

    Execution is performed by
    :class:`~catley.game.actions.executors.stunts.PunchExecutor`.
    """

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        defender: Character,
    ) -> None:
        """Create a punch intent.

        Args:
            controller: Game controller providing context.
            attacker: The character performing the punch.
            defender: The target character being punched.
        """
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.defender = defender


class HolsterWeaponIntent(GameIntent):
    """Intent to holster the currently equipped weapon.

    Moves the active weapon from the equipment slot to inventory. This is
    used as a preparatory step in action plans that require unarmed combat
    (e.g., PunchPlan).

    Execution is performed by
    :class:`~catley.game.actions.executors.stunts.HolsterWeaponExecutor`.
    """

    def __init__(self, controller: Controller, actor: Character) -> None:
        """Create a holster weapon intent.

        Args:
            controller: Game controller providing context.
            actor: The character holstering their weapon.
        """
        super().__init__(controller, actor)
        self.holsterer = actor


# =============================================================================
# Action Plans for Stunts
# =============================================================================

PunchPlan = ActionPlan(
    name="Punch",
    requires_target=True,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=HolsterWeaponIntent,
            params=lambda ctx: {"actor": ctx.actor},
            skip_if=lambda ctx: ctx.actor.inventory.get_active_item() is None,
        ),
        IntentStep(
            intent_class=PunchIntent,
            params=lambda ctx: {
                "attacker": ctx.actor,
                "defender": ctx.target_actor,
            },
        ),
    ],
)

KickPlan = ActionPlan(
    name="Kick",
    requires_target=True,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=KickIntent,
            params=lambda ctx: {
                "attacker": ctx.actor,
                "defender": ctx.target_actor,
            },
        ),
    ],
)

TripPlan = ActionPlan(
    name="Trip",
    requires_target=True,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=TripIntent,
            params=lambda ctx: {
                "attacker": ctx.actor,
                "defender": ctx.target_actor,
            },
        ),
    ],
)

PushPlan = ActionPlan(
    name="Push",
    requires_target=True,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=PushIntent,
            params=lambda ctx: {
                "attacker": ctx.actor,
                "defender": ctx.target_actor,
            },
        ),
    ],
)
