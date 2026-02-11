"""
Executors for combat stunt actions like Push, Trip, Disarm, and Grapple.

Stunts are physical maneuvers that manipulate enemy positioning and status
using opposed stat checks rather than weapon attacks.
"""

from __future__ import annotations

from brileta import colors
from brileta.constants.combat import CombatConstants as Combat
from brileta.events import (
    FloatingTextEvent,
    FloatingTextValence,
    MessageEvent,
    publish_event,
)
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.executors.base import ActionExecutor
from brileta.game.actions.executors.displacement import attempt_displacement
from brileta.game.actions.stunts import (
    HolsterWeaponIntent,
    KickIntent,
    PunchIntent,
    PushIntent,
    TripIntent,
)
from brileta.game.actors.ai import escalate_hostility
from brileta.game.actors.status_effects import (
    OffBalanceEffect,
    StaggeredEffect,
    TrippedEffect,
)
from brileta.game.enums import OutcomeTier
from brileta.util.dice import roll_d


class PushExecutor(ActionExecutor[PushIntent]):
    """Executes push stunt intents.

    Push is a Strength vs Strength opposed check that moves an adjacent target
    one tile directly away from the attacker. Environmental interactions occur
    when pushed into walls, hazards, or other actors.
    """

    def execute(self, intent: PushIntent) -> GameActionResult | None:
        """Execute a push attempt.

        Resolution:
        - Critical Success (nat 20): Target pushed + TrippedEffect
        - Success: Target pushed 1 tile + StaggeredEffect
        - Partial Success (tie): Target pushed, attacker gains OffBalanceEffect
        - Failure: Attacker gains OffBalanceEffect
        - Critical Failure (nat 1): Attacker gains TrippedEffect
        """
        # 1. Validate adjacency (Chebyshev distance must be 1, includes diagonals)
        dx = intent.defender.x - intent.attacker.x
        dy = intent.defender.y - intent.attacker.y
        if max(abs(dx), abs(dy)) != 1:
            publish_event(
                MessageEvent(
                    f"{intent.defender.name} is not adjacent!",
                    colors.RED,
                )
            )
            return GameActionResult(succeeded=False, block_reason="not_adjacent")

        # 2. Perform opposed Strength check
        attacker_strength = intent.attacker.stats.strength
        defender_strength = intent.defender.stats.strength

        # Get resolution modifiers from status effects (advantage/disadvantage)
        resolution_args = intent.attacker.modifiers.get_resolution_modifiers("strength")

        resolver = intent.controller.create_resolver(
            ability_score=attacker_strength,
            roll_to_exceed=defender_strength + Combat.D20_DC_BASE,
            has_advantage=resolution_args.get("has_advantage", False),
            has_disadvantage=resolution_args.get("has_disadvantage", False),
        )
        result = resolver.resolve(intent.attacker, intent.defender)

        # 3. Handle outcome tiers
        match result.outcome_tier:
            case OutcomeTier.CRITICAL_SUCCESS:
                pushed = attempt_displacement(
                    intent.controller, intent.defender, dx, dy
                )
                intent.defender.status_effects.apply_status_effect(TrippedEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                publish_event(
                    FloatingTextEvent(
                        text="PUSHED",
                        target_actor_id=intent.defender.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.defender.x,
                        world_y=intent.defender.y,
                    )
                )
                if pushed:
                    msg = f"Critical! {atk_name} shoves {def_name} to the ground!"
                else:
                    # Couldn't push (edge of map), but still trip them
                    msg = f"Critical! {atk_name} knocks {def_name} to the ground!"
                publish_event(MessageEvent(msg, colors.YELLOW))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.SUCCESS:
                pushed = attempt_displacement(
                    intent.controller, intent.defender, dx, dy
                )
                # Apply StaggeredEffect - defender is disoriented and skips
                # their next action. This prevents them from immediately
                # walking back to their original position.
                intent.defender.status_effects.apply_status_effect(StaggeredEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                publish_event(
                    FloatingTextEvent(
                        text="PUSHED",
                        target_actor_id=intent.defender.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.defender.x,
                        world_y=intent.defender.y,
                    )
                )
                if pushed:
                    msg = f"{atk_name} shoves {def_name} back!"
                    publish_event(MessageEvent(msg, colors.WHITE))
                else:
                    msg = f"{atk_name} pushes {def_name} but they have nowhere to go!"
                    publish_event(MessageEvent(msg, colors.LIGHT_GREY))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.PARTIAL_SUCCESS:
                pushed = attempt_displacement(
                    intent.controller, intent.defender, dx, dy
                )
                intent.attacker.status_effects.apply_status_effect(OffBalanceEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                if pushed:
                    msg = f"{atk_name} shoves {def_name} back but stumbles!"
                else:
                    msg = f"{atk_name} pushes {def_name} but both end up off-balance!"
                publish_event(MessageEvent(msg, colors.LIGHT_BLUE))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.FAILURE:
                intent.attacker.status_effects.apply_status_effect(OffBalanceEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                publish_event(
                    FloatingTextEvent(
                        text="FAILED",
                        target_actor_id=intent.attacker.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.attacker.x,
                        world_y=intent.attacker.y,
                    )
                )
                msg = f"{atk_name} fails to push {def_name} and stumbles off-balance!"
                publish_event(MessageEvent(msg, colors.GREY))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=False, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.CRITICAL_FAILURE:
                intent.attacker.status_effects.apply_status_effect(TrippedEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                msg = f"Critical miss! {atk_name} trips trying to push {def_name}!"
                publish_event(MessageEvent(msg, colors.ORANGE))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=False, duration_ms=Combat.STUNT_DURATION_MS
                )

        # Fallback (should never reach)
        return GameActionResult(succeeded=False, duration_ms=Combat.STUNT_DURATION_MS)


class TripExecutor(ActionExecutor[TripIntent]):
    """Executes trip stunt intents.

    Trip is an Agility vs Agility opposed check that causes the target to
    fall prone. Unlike Push, Trip does not move the target but reliably
    applies TrippedEffect on any success (not just critical).
    """

    def execute(self, intent: TripIntent) -> GameActionResult | None:
        """Execute a trip attempt.

        Resolution:
        - Critical Success (nat 20): Target tripped + 1d4 impact damage
        - Success: Target gains TrippedEffect (skips 2 turns)
        - Partial Success (tie): Target tripped, attacker gains OffBalanceEffect
        - Failure: Attacker gains OffBalanceEffect
        - Critical Failure (nat 1): Attacker gains TrippedEffect
        """
        # 1. Validate adjacency (Chebyshev distance must be 1, includes diagonals)
        dx = intent.defender.x - intent.attacker.x
        dy = intent.defender.y - intent.attacker.y
        if max(abs(dx), abs(dy)) != 1:
            publish_event(
                MessageEvent(
                    f"{intent.defender.name} is not adjacent!",
                    colors.RED,
                )
            )
            return GameActionResult(succeeded=False, block_reason="not_adjacent")

        # 2. Perform opposed Agility check
        attacker_agility = intent.attacker.stats.agility
        defender_agility = intent.defender.stats.agility

        # Get resolution modifiers from status effects (advantage/disadvantage)
        resolution_args = intent.attacker.modifiers.get_resolution_modifiers("agility")

        resolver = intent.controller.create_resolver(
            ability_score=attacker_agility,
            roll_to_exceed=defender_agility + Combat.D20_DC_BASE,
            has_advantage=resolution_args.get("has_advantage", False),
            has_disadvantage=resolution_args.get("has_disadvantage", False),
        )
        result = resolver.resolve(intent.attacker, intent.defender)

        # 3. Handle outcome tiers
        atk_name = intent.attacker.name
        def_name = intent.defender.name

        match result.outcome_tier:
            case OutcomeTier.CRITICAL_SUCCESS:
                # Trip + bonus damage from hard landing
                intent.defender.status_effects.apply_status_effect(TrippedEffect())
                publish_event(
                    FloatingTextEvent(
                        text="TRIPPED",
                        target_actor_id=intent.defender.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.defender.x,
                        world_y=intent.defender.y,
                    )
                )
                impact_damage = roll_d(4)
                intent.defender.take_damage(impact_damage, damage_type="impact")
                msg = (
                    f"Critical! {atk_name} sweeps {def_name}'s legs out! "
                    f"{def_name} hits the ground hard for {impact_damage} damage!"
                )
                publish_event(MessageEvent(msg, colors.YELLOW))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.SUCCESS:
                intent.defender.status_effects.apply_status_effect(TrippedEffect())
                publish_event(
                    FloatingTextEvent(
                        text="TRIPPED",
                        target_actor_id=intent.defender.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.defender.x,
                        world_y=intent.defender.y,
                    )
                )
                msg = f"{atk_name} trips {def_name}! They fall prone!"
                publish_event(MessageEvent(msg, colors.WHITE))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.PARTIAL_SUCCESS:
                # Target tripped but attacker stumbles
                intent.defender.status_effects.apply_status_effect(TrippedEffect())
                intent.attacker.status_effects.apply_status_effect(OffBalanceEffect())
                publish_event(
                    FloatingTextEvent(
                        text="TRIPPED",
                        target_actor_id=intent.defender.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.defender.x,
                        world_y=intent.defender.y,
                    )
                )
                msg = f"{atk_name} trips {def_name} but stumbles in the process!"
                publish_event(MessageEvent(msg, colors.LIGHT_BLUE))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.FAILURE:
                # Trip attempt fails - attacker stumbles
                intent.attacker.status_effects.apply_status_effect(OffBalanceEffect())
                publish_event(
                    FloatingTextEvent(
                        text="FAILED",
                        target_actor_id=intent.attacker.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.attacker.x,
                        world_y=intent.attacker.y,
                    )
                )
                msg = f"{atk_name} fails to trip {def_name} and stumbles!"
                publish_event(MessageEvent(msg, colors.GREY))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=False, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.CRITICAL_FAILURE:
                # Attacker overextends and falls
                intent.attacker.status_effects.apply_status_effect(TrippedEffect())
                msg = f"Critical miss! {atk_name} trips over their own feet!"
                publish_event(MessageEvent(msg, colors.ORANGE))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=False, duration_ms=Combat.STUNT_DURATION_MS
                )

        # Fallback (should never reach)
        return GameActionResult(succeeded=False, duration_ms=Combat.STUNT_DURATION_MS)


class KickExecutor(ActionExecutor[KickIntent]):
    """Executes kick stunt intents.

    Kick is a Strength vs Agility opposed check that deals damage and pushes
    the target back. Unlike Push, Kick always deals d4 damage on success.
    It's the offensive alternative to Push - trading control for damage.
    """

    def execute(self, intent: KickIntent) -> GameActionResult | None:
        """Execute a kick attempt.

        Resolution:
        - Critical Success: d4 damage + pushed 1 tile + TrippedEffect
        - Success: d4 damage + pushed 1 tile
        - Partial Success: d4 damage + pushed 1 tile, attacker OffBalanceEffect
        - Failure: Attacker gains OffBalanceEffect
        - Critical Failure: Attacker gains TrippedEffect
        """
        # 1. Validate adjacency (Chebyshev distance must be 1, includes diagonals)
        dx = intent.defender.x - intent.attacker.x
        dy = intent.defender.y - intent.attacker.y
        if max(abs(dx), abs(dy)) != 1:
            publish_event(
                MessageEvent(
                    f"{intent.defender.name} is not adjacent!",
                    colors.RED,
                )
            )
            return GameActionResult(succeeded=False, block_reason="not_adjacent")

        # 2. Perform opposed Strength vs Agility check
        attacker_strength = intent.attacker.stats.strength
        defender_agility = intent.defender.stats.agility

        # Get resolution modifiers from status effects (advantage/disadvantage)
        resolution_args = intent.attacker.modifiers.get_resolution_modifiers("strength")

        resolver = intent.controller.create_resolver(
            ability_score=attacker_strength,
            roll_to_exceed=defender_agility + Combat.D20_DC_BASE,
            has_advantage=resolution_args.get("has_advantage", False),
            has_disadvantage=resolution_args.get("has_disadvantage", False),
        )
        result = resolver.resolve(intent.attacker, intent.defender)

        # 3. Handle outcome tiers
        atk_name = intent.attacker.name
        def_name = intent.defender.name

        match result.outcome_tier:
            case OutcomeTier.CRITICAL_SUCCESS:
                # Damage + push + trip
                damage = roll_d(4)
                intent.defender.take_damage(damage, damage_type="impact")
                pushed = attempt_displacement(
                    intent.controller, intent.defender, dx, dy
                )
                intent.defender.status_effects.apply_status_effect(TrippedEffect())
                publish_event(
                    FloatingTextEvent(
                        text="KICKED",
                        target_actor_id=intent.defender.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.defender.x,
                        world_y=intent.defender.y,
                    )
                )
                if pushed:
                    msg = (
                        f"Critical! {atk_name} kicks {def_name} to the ground "
                        f"for {damage} damage!"
                    )
                else:
                    msg = (
                        f"Critical! {atk_name} kicks {def_name} down "
                        f"for {damage} damage!"
                    )
                publish_event(MessageEvent(msg, colors.YELLOW))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.SUCCESS:
                # Damage + push
                damage = roll_d(4)
                intent.defender.take_damage(damage, damage_type="impact")
                pushed = attempt_displacement(
                    intent.controller, intent.defender, dx, dy
                )
                publish_event(
                    FloatingTextEvent(
                        text="KICKED",
                        target_actor_id=intent.defender.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.defender.x,
                        world_y=intent.defender.y,
                    )
                )
                if pushed:
                    msg = f"{atk_name} kicks {def_name} back for {damage} damage!"
                else:
                    msg = f"{atk_name} kicks {def_name} for {damage} damage!"
                publish_event(MessageEvent(msg, colors.WHITE))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.PARTIAL_SUCCESS:
                # Damage + push, but attacker stumbles
                damage = roll_d(4)
                intent.defender.take_damage(damage, damage_type="impact")
                pushed = attempt_displacement(
                    intent.controller, intent.defender, dx, dy
                )
                intent.attacker.status_effects.apply_status_effect(OffBalanceEffect())
                publish_event(
                    FloatingTextEvent(
                        text="KICKED",
                        target_actor_id=intent.defender.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.defender.x,
                        world_y=intent.defender.y,
                    )
                )
                if pushed:
                    msg = (
                        f"{atk_name} kicks {def_name} back for {damage} damage "
                        f"but stumbles!"
                    )
                else:
                    msg = (
                        f"{atk_name} kicks {def_name} for {damage} damage "
                        f"but loses balance!"
                    )
                publish_event(MessageEvent(msg, colors.LIGHT_BLUE))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=True, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.FAILURE:
                # Miss - attacker stumbles
                intent.attacker.status_effects.apply_status_effect(OffBalanceEffect())
                publish_event(
                    FloatingTextEvent(
                        text="MISSED",
                        target_actor_id=intent.attacker.actor_id,
                        valence=FloatingTextValence.NEGATIVE,
                        world_x=intent.attacker.x,
                        world_y=intent.attacker.y,
                    )
                )
                msg = f"{atk_name} misses the kick and stumbles off-balance!"
                publish_event(MessageEvent(msg, colors.GREY))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=False, duration_ms=Combat.STUNT_DURATION_MS
                )

            case OutcomeTier.CRITICAL_FAILURE:
                # Attacker falls
                intent.attacker.status_effects.apply_status_effect(TrippedEffect())
                msg = f"Critical miss! {atk_name} slips and falls trying to kick!"
                publish_event(MessageEvent(msg, colors.ORANGE))
                escalate_hostility(intent.attacker, intent.defender, intent.controller)
                return GameActionResult(
                    succeeded=False, duration_ms=Combat.STUNT_DURATION_MS
                )

        # Fallback (should never reach)
        return GameActionResult(succeeded=False, duration_ms=Combat.STUNT_DURATION_MS)


class HolsterWeaponExecutor(ActionExecutor[HolsterWeaponIntent]):
    """Executes holster weapon intents.

    Moves the active weapon from the equipment slot to inventory. Used as a
    preparatory step in action plans requiring unarmed combat (e.g., PunchPlan).
    """

    def execute(self, intent: HolsterWeaponIntent) -> GameActionResult | None:
        """Execute a holster weapon action.

        If a weapon is equipped, unequips it to inventory and emits a message.
        If no weapon is equipped, succeeds silently (no-op).
        """
        actor = intent.holsterer
        active_weapon = actor.inventory.get_active_item()

        if active_weapon is None:
            # No weapon to holster - succeed silently, no delay needed
            return GameActionResult(succeeded=True)

        # Holster the weapon (move to inventory) - this always succeeds
        # because equipped items already count toward capacity
        actor.inventory.unequip_to_inventory(actor.inventory.active_slot)
        publish_event(
            MessageEvent(
                f"{actor.name} holsters their {active_weapon.name}.",
                colors.WHITE,
            )
        )
        return GameActionResult(succeeded=True, duration_ms=Combat.HOLSTER_DURATION_MS)


class PunchExecutor(ActionExecutor[PunchIntent]):
    """Executes punch intents.

    Performs an unarmed punch attack using Fists (d3 damage). When used via
    the ActionPlan system, holstering is handled by a separate HolsterWeaponIntent
    step that precedes the punch.
    """

    def execute(self, intent: PunchIntent) -> GameActionResult | None:
        """Execute a punch attack with Fists.

        Holstering is handled by a separate HolsterWeaponIntent step when using
        the ActionPlan system. This executor assumes the attacker is ready to punch.
        """
        from brileta.game.items.item_types import FISTS_TYPE

        atk_name = intent.attacker.name
        def_name = intent.defender.name

        # 1. Validate adjacency (target may have moved since plan started)
        dx = intent.defender.x - intent.attacker.x
        dy = intent.defender.y - intent.attacker.y
        if max(abs(dx), abs(dy)) != 1:
            publish_event(
                MessageEvent(
                    f"{def_name} is not adjacent!",
                    colors.RED,
                )
            )
            return GameActionResult(succeeded=False, block_reason="not_adjacent")

        # 2. Get Fists weapon
        fists = FISTS_TYPE.create()

        # 3. Perform attack roll
        attack = fists.melee_attack
        assert attack is not None

        # Get attacker's strength for the attack
        attacker_stat = intent.attacker.stats.strength
        defender_defense = intent.defender.stats.agility

        # Get resolution modifiers
        resolution_args = intent.attacker.modifiers.get_resolution_modifiers("strength")

        resolver = intent.controller.create_resolver(
            ability_score=attacker_stat,
            roll_to_exceed=defender_defense + Combat.D20_DC_BASE,
            has_advantage=resolution_args.get("has_advantage", False),
            has_disadvantage=resolution_args.get("has_disadvantage", False),
        )
        result = resolver.resolve(intent.attacker, intent.defender)

        # 4. Handle outcome based on outcome tier
        # Hit: SUCCESS, CRITICAL_SUCCESS, PARTIAL_SUCCESS
        # Miss: FAILURE, CRITICAL_FAILURE
        if result.outcome_tier in (
            OutcomeTier.SUCCESS,
            OutcomeTier.CRITICAL_SUCCESS,
            OutcomeTier.PARTIAL_SUCCESS,
        ):
            # Roll damage using fists
            damage = attack.damage_dice.roll()
            intent.defender.take_damage(damage, damage_type="impact")

            publish_event(
                FloatingTextEvent(
                    text=f"-{damage}",
                    target_actor_id=intent.defender.actor_id,
                    valence=FloatingTextValence.NEGATIVE,
                    world_x=intent.defender.x,
                    world_y=intent.defender.y,
                )
            )

            if result.outcome_tier == OutcomeTier.CRITICAL_SUCCESS:
                msg = (
                    f"Critical hit! {atk_name} punches {def_name} for {damage} damage!"
                )
                publish_event(MessageEvent(msg, colors.YELLOW))
            else:
                msg = f"{atk_name} punches {def_name} for {damage} damage!"
                publish_event(MessageEvent(msg, colors.WHITE))

            escalate_hostility(intent.attacker, intent.defender, intent.controller)
            return GameActionResult(
                succeeded=True, duration_ms=Combat.PUNCH_DURATION_MS
            )

        # Miss
        publish_event(
            FloatingTextEvent(
                text="MISS",
                target_actor_id=intent.defender.actor_id,
                valence=FloatingTextValence.NEUTRAL,
                world_x=intent.defender.x,
                world_y=intent.defender.y,
            )
        )
        msg = f"{atk_name} swings at {def_name} but misses!"
        publish_event(MessageEvent(msg, colors.GREY))

        escalate_hostility(intent.attacker, intent.defender, intent.controller)
        return GameActionResult(succeeded=False, duration_ms=Combat.PUNCH_DURATION_MS)
