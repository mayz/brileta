"""
Executors for combat stunt actions like Push, Trip, Disarm, and Grapple.

Stunts are physical maneuvers that manipulate enemy positioning and status
using opposed stat checks rather than weapon attacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.events import CombatInitiatedEvent, MessageEvent, publish_event
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor
from catley.game.actors.status_effects import (
    OffBalanceEffect,
    StaggeredEffect,
    TrippedEffect,
)
from catley.game.enums import OutcomeTier
from catley.util.dice import roll_d

if TYPE_CHECKING:
    from catley.game.actions.stunts import PushIntent


class PushExecutor(ActionExecutor):
    """Executes push stunt intents.

    Push is a Strength vs Strength opposed check that moves an adjacent target
    one tile directly away from the attacker. Environmental interactions occur
    when pushed into walls, hazards, or other actors.
    """

    def execute(self, intent: PushIntent) -> GameActionResult | None:  # type: ignore[override]
        """Execute a push attempt.

        Resolution:
        - Critical Success (nat 20): Target pushed + TrippedEffect
        - Success: Target pushed 1 tile
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
            roll_to_exceed=defender_strength + 10,
            has_advantage=resolution_args.get("has_advantage", False),
            has_disadvantage=resolution_args.get("has_disadvantage", False),
        )
        result = resolver.resolve(intent.attacker, intent.defender)

        # 3. Handle outcome tiers
        match result.outcome_tier:
            case OutcomeTier.CRITICAL_SUCCESS:
                pushed = self._attempt_push(intent, dx, dy)
                intent.defender.status_effects.apply_status_effect(TrippedEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                if pushed:
                    msg = f"Critical! {atk_name} shoves {def_name} to the ground!"
                else:
                    # Couldn't push (edge of map), but still trip them
                    msg = f"Critical! {atk_name} knocks {def_name} to the ground!"
                publish_event(MessageEvent(msg, colors.YELLOW))
                self._update_ai_disposition(intent)
                return GameActionResult(succeeded=True)

            case OutcomeTier.SUCCESS:
                pushed = self._attempt_push(intent, dx, dy)
                # Apply StaggeredEffect - defender is disoriented and skips
                # their next action. This prevents them from immediately
                # walking back to their original position.
                intent.defender.status_effects.apply_status_effect(StaggeredEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                if pushed:
                    msg = f"{atk_name} shoves {def_name} back!"
                    publish_event(MessageEvent(msg, colors.WHITE))
                else:
                    msg = f"{atk_name} pushes {def_name} but they have nowhere to go!"
                    publish_event(MessageEvent(msg, colors.LIGHT_GREY))
                self._update_ai_disposition(intent)
                return GameActionResult(succeeded=True)

            case OutcomeTier.PARTIAL_SUCCESS:
                pushed = self._attempt_push(intent, dx, dy)
                intent.attacker.status_effects.apply_status_effect(OffBalanceEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                if pushed:
                    msg = f"{atk_name} shoves {def_name} back but stumbles!"
                else:
                    msg = f"{atk_name} pushes {def_name} but both end up off-balance!"
                publish_event(MessageEvent(msg, colors.LIGHT_BLUE))
                self._update_ai_disposition(intent)
                return GameActionResult(succeeded=True)

            case OutcomeTier.FAILURE:
                intent.attacker.status_effects.apply_status_effect(OffBalanceEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                msg = f"{atk_name} fails to push {def_name} and stumbles off-balance!"
                publish_event(MessageEvent(msg, colors.GREY))
                self._update_ai_disposition(intent)
                return GameActionResult(succeeded=False)

            case OutcomeTier.CRITICAL_FAILURE:
                intent.attacker.status_effects.apply_status_effect(TrippedEffect())
                atk_name = intent.attacker.name
                def_name = intent.defender.name
                msg = f"Critical miss! {atk_name} trips trying to push {def_name}!"
                publish_event(MessageEvent(msg, colors.ORANGE))
                self._update_ai_disposition(intent)
                return GameActionResult(succeeded=False)

        # Fallback (should never reach)
        return GameActionResult(succeeded=False)

    def _attempt_push(self, intent: PushIntent, dx: int, dy: int) -> bool:
        """Attempt to push the defender in the given direction.

        Handles environmental interactions:
        - Wall collision: 1d4 impact damage + OffBalanceEffect, no movement
        - Actor collision: Both actors get OffBalanceEffect, no movement
        - Hazard tile: Movement succeeds, hazard damage applied via turn system
        - Clear tile: Movement succeeds

        Args:
            intent: The push intent with attacker, defender, and controller.
            dx: X direction of push (from attacker to defender).
            dy: Y direction of push (from attacker to defender).

        Returns:
            True if the defender was moved, False otherwise.
        """
        game_map = intent.controller.gw.game_map
        dest_x = intent.defender.x + dx
        dest_y = intent.defender.y + dy

        # Check map boundaries
        if not (0 <= dest_x < game_map.width and 0 <= dest_y < game_map.height):
            return False

        # Check for wall collision
        if not game_map.walkable[dest_x, dest_y]:
            self._handle_wall_impact(intent)
            return False

        # Check for actor collision
        blocking_actor = intent.controller.gw.get_actor_at_location(dest_x, dest_y)
        if blocking_actor and blocking_actor.blocks_movement:
            self._handle_actor_collision(intent, blocking_actor)
            return False

        # Clear destination - execute the push
        intent.defender.move(dx, dy, intent.controller)

        # Check if pushed onto hazard and log a message
        self._check_hazard_landing(intent, dest_x, dest_y)

        return True

    def _handle_wall_impact(self, intent: PushIntent) -> None:
        """Handle a defender being pushed into a wall.

        Deals 1d4 impact damage and applies OffBalanceEffect.
        """
        impact_damage = roll_d(4)
        intent.defender.take_damage(impact_damage, damage_type="impact")
        intent.defender.status_effects.apply_status_effect(OffBalanceEffect())

        def_name = intent.defender.name
        msg = f"{def_name} slams into the wall for {impact_damage} damage!"
        publish_event(MessageEvent(msg, colors.WHITE))

    def _handle_actor_collision(
        self, intent: PushIntent, blocking_actor: object
    ) -> None:
        """Handle a defender being pushed into another actor.

        Both the defender and the blocking actor become Off-Balance.
        """
        from catley.game.actors import Character

        intent.defender.status_effects.apply_status_effect(OffBalanceEffect())

        # Apply OffBalance to the blocking actor if it's a Character
        if isinstance(blocking_actor, Character):
            blocking_actor.status_effects.apply_status_effect(OffBalanceEffect())
            publish_event(
                MessageEvent(
                    f"{intent.defender.name} collides with {blocking_actor.name}! "
                    f"Both are off-balance!",
                    colors.LIGHT_BLUE,
                )
            )
        else:
            publish_event(
                MessageEvent(
                    f"{intent.defender.name} is pushed but collides with something!",
                    colors.LIGHT_BLUE,
                )
            )

    def _check_hazard_landing(self, intent: PushIntent, x: int, y: int) -> None:
        """Log a message if the defender landed on a hazard tile.

        The actual hazard damage is applied by the turn system when
        the defender's turn ends on the hazard tile.
        """
        from catley.environment.tile_types import (
            get_tile_hazard_info,
            get_tile_type_name_by_id,
        )

        game_map = intent.controller.gw.game_map
        tile_id = int(game_map.tiles[x, y])
        damage_dice, _damage_type = get_tile_hazard_info(tile_id)

        if damage_dice:
            tile_name = get_tile_type_name_by_id(tile_id)
            publish_event(
                MessageEvent(
                    f"{intent.defender.name} lands in the {tile_name.lower()}!",
                    colors.ORANGE,
                )
            )

    def _update_ai_disposition(self, intent: PushIntent) -> None:
        """Update AI disposition if player pushed an NPC.

        Being pushed (or having someone attempt to push you) is treated as an
        aggressive act. Non-hostile NPCs become hostile when the player attempts
        to push them, regardless of whether the push succeeds.
        """
        from catley.game.actors import ai
        from catley.game.enums import Disposition

        # Only trigger for player pushing NPC
        if intent.attacker != intent.controller.gw.player:
            return

        # Check if defender has disposition-based AI
        if not isinstance(intent.defender.ai, ai.DispositionBasedAI):
            return

        # Only change disposition if not already hostile
        if intent.defender.ai.disposition == Disposition.HOSTILE:
            return

        # Make hostile
        intent.defender.ai.disposition = Disposition.HOSTILE
        publish_event(
            MessageEvent(
                f"{intent.defender.name} becomes hostile after being pushed!",
                colors.ORANGE,
            )
        )
        # Trigger auto-entry into combat mode
        publish_event(
            CombatInitiatedEvent(
                attacker=intent.attacker,
                defender=intent.defender,
            )
        )
