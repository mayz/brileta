from __future__ import annotations

from typing import TYPE_CHECKING, cast

from brileta.game import ranges
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.stunts import KickIntent, PunchIntent, PushIntent, TripIntent
from brileta.game.actors import Character
from brileta.game.items.properties import WeaponProperty

from .action_context import ActionContext, ActionContextBuilder
from .action_factory import ActionFactory
from .action_formatters import ActionFormatter
from .types import ActionCategory, ActionOption, ActionRequirement

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.items.capabilities import MeleeAttack, RangedAttack
    from brileta.game.items.item_core import Item


class CombatActionDiscovery:
    """Discover combat-related actions."""

    def __init__(
        self,
        context_builder: ActionContextBuilder,
        factory: ActionFactory,
        formatter: ActionFormatter,
    ) -> None:
        self.context_builder = context_builder
        self.factory = factory
        self.formatter = formatter

    # Public API -----------------------------------------------------

    def get_player_combat_actions(
        self,
        controller: Controller,
        actor: Character,
        target: Character | None = None,
    ) -> list[ActionOption]:
        """Get combat actions for action-centric UI.

        Returns combat actions based on capability, not immediate executability.
        Actions are shown if the actor has the capability to perform them:
        - Melee attacks: always shown if actor has a melee weapon
        - Ranged attacks: only shown if a target is within weapon range
        - Push: always shown (approach will be handled on execution)

        When a melee action or Push is selected on a non-adjacent target, the
        combat mode will pathfind to an adjacent tile before executing.

        If a specific target is provided, probabilities will be calculated
        against that target (only for adjacent targets). Otherwise, probabilities
        are left as None.

        Used by combat mode's action-centric panel where the player selects
        an action first, then clicks to execute it.

        Args:
            controller: Game controller.
            actor: The acting character (usually the player).
            target: Optional specific target for probability calculation.

        Returns:
            List of ActionOptions for combat and stunt actions.
        """
        options: list[ActionOption] = []
        context = self.context_builder.build_context(controller, actor)

        # All attacks always shown. Range/adjacency is validated at targeting
        # time, not at listing time, so the player can select an action before
        # finding a target.
        options = list(self.get_all_combat_actions(controller, actor, context, target))

        # Always add Push - approach handled on execution
        # Calculate push probability if target is provided (regardless of distance,
        # since player will approach before executing)
        push_prob: float | None = None
        if target is not None:
            push_prob = self._calculate_opposed_probability(
                controller, actor, target, "strength", "strength"
            )

        options.append(
            ActionOption(
                id="push",
                name="Push",
                description="Shove target 1 tile away. Strength vs Strength.",
                category=ActionCategory.STUNT,
                action_class=PushIntent,
                requirements=[ActionRequirement.TARGET_ACTOR],
                static_params={},
                success_probability=push_prob,
            )
        )

        # Always add Trip - approach handled on execution
        trip_prob: float | None = None
        if target is not None:
            trip_prob = self._calculate_opposed_probability(
                controller, actor, target, "agility", "agility"
            )

        options.append(
            ActionOption(
                id="trip",
                name="Trip",
                description="Knock target prone. Agility vs Agility.",
                category=ActionCategory.STUNT,
                action_class=TripIntent,
                requirements=[ActionRequirement.TARGET_ACTOR],
                static_params={},
                success_probability=trip_prob,
            )
        )

        # Always add Kick - approach handled on execution
        # Kick uses Strength vs Agility and deals d4 damage + push
        kick_prob: float | None = None
        if target is not None:
            kick_prob = self._calculate_opposed_probability(
                controller, actor, target, "strength", "agility"
            )

        options.append(
            ActionOption(
                id="kick",
                name="Kick",
                description="Kick target for d4 damage + push. Strength vs Agility.",
                category=ActionCategory.STUNT,
                action_class=KickIntent,
                requirements=[ActionRequirement.TARGET_ACTOR],
                static_params={},
                success_probability=kick_prob,
            )
        )

        # Always add Punch - always available, uses ActionPlan system.
        # PunchPlan handles approach, holster (if weapon equipped), then punch.
        punch_prob: float | None = None
        if target is not None:
            punch_prob = self._calculate_opposed_probability(
                controller, actor, target, "strength", "agility"
            )

        # Determine description based on whether weapon is equipped
        active_weapon = actor.inventory.get_active_item()
        if active_weapon is not None:
            punch_desc = f"Holster {active_weapon.name}, then punch next turn."
        else:
            punch_desc = "Punch target for d3 damage. Strength vs Agility."

        options.append(
            ActionOption(
                id="punch",
                name="Punch",
                description=punch_desc,
                category=ActionCategory.STUNT,
                action_class=PunchIntent,
                requirements=[ActionRequirement.TARGET_ACTOR],
                static_params={},
                success_probability=punch_prob if active_weapon is None else None,
            )
        )

        # Sort by category and priority so PREFERRED attacks appear first
        options.sort(key=lambda a: (a.category.value, self._get_action_priority(a)))
        return options

    def _get_action_priority(self, action: ActionOption) -> int:
        """Get sort priority for an action. Lower values sort first.

        Priority order:
        - 0: PREFERRED attacks (weapon's intended use, e.g., Shoot for a pistol)
        - 1: Regular attacks (no special property)
        - 2: IMPROVISED attacks (not designed as weapon, e.g., Pistol-whip)
        """
        weapon = action.static_params.get("weapon")
        attack_mode = action.static_params.get("attack_mode")
        if weapon is None or attack_mode is None:
            return 1
        attack = None
        if attack_mode == "melee" and weapon.melee_attack:
            attack = weapon.melee_attack
        elif attack_mode == "ranged" and weapon.ranged_attack:
            attack = weapon.ranged_attack
        if attack is None:
            return 1
        if WeaponProperty.PREFERRED in attack.properties:
            return 0
        if WeaponProperty.IMPROVISED in attack.properties:
            return 2
        return 1

    def _has_adjacent_enemy(self, actor: Character, context: ActionContext) -> bool:
        """Check if any valid combat target is adjacent to the actor.

        Args:
            actor: The acting character.
            context: Action context containing nearby actors.

        Returns:
            True if at least one valid enemy is adjacent (distance == 1).
        """
        for target in context.nearby_actors:
            if (
                target != actor
                and isinstance(target, Character)
                and target.health
                and target.health.is_alive()
            ):
                distance = ranges.calculate_distance(
                    actor.x, actor.y, target.x, target.y
                )
                if distance == 1:
                    return True
        return False

    def _has_enemy_in_weapon_range(
        self, actor: Character, weapon: Item, context: ActionContext
    ) -> bool:
        """Check if any enemy is within the weapon's ranged attack range.

        Args:
            actor: The acting character.
            weapon: The weapon to check range for.
            context: Action context containing nearby actors.

        Returns:
            True if at least one valid enemy is within the weapon's range.
        """
        if not weapon.ranged_attack:
            return False

        for target in context.nearby_actors:
            if (
                target != actor
                and isinstance(target, Character)
                and target.health
                and target.health.is_alive()
            ):
                dist = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)
                range_cat = ranges.get_range_category(dist, weapon)
                range_mods = ranges.get_range_modifier(weapon, range_cat)
                if range_mods is not None:
                    return True
        return False

    def discover_combat_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Return all combat options available to the actor."""
        return self.get_all_combat_actions(controller, actor, context)

    def discover_stunt_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Discover stunt actions (Push, Trip, etc.) available to the actor.

        Stunts are physical maneuvers that require adjacency and use opposed
        stat checks rather than weapon attacks.
        """
        options: list[ActionOption] = []

        # Find adjacent enemies for stunt targeting
        for target in context.nearby_actors:
            if (
                target == actor
                or not isinstance(target, Character)
                or not target.health
                or not target.health.is_alive()
            ):
                continue

            distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)

            # Push requires adjacency (distance == 1)
            if distance == 1:
                push_prob = self._calculate_opposed_probability(
                    controller, actor, target, "strength", "strength"
                )
                options.append(
                    ActionOption(
                        id=f"push-{target.name}-{target.x}-{target.y}",
                        name=f"Push {target.name}",
                        description="Shove target 1 tile away. Strength vs Strength.",
                        category=ActionCategory.STUNT,
                        action_class=PushIntent,
                        requirements=[],  # Target already specified
                        static_params={"defender": target},
                        success_probability=push_prob,
                    )
                )

            # Trip requires adjacency (distance == 1)
            if distance == 1:
                trip_prob = self._calculate_opposed_probability(
                    controller, actor, target, "agility", "agility"
                )
                options.append(
                    ActionOption(
                        id=f"trip-{target.name}-{target.x}-{target.y}",
                        name=f"Trip {target.name}",
                        description="Knock target prone. Agility vs Agility.",
                        category=ActionCategory.STUNT,
                        action_class=TripIntent,
                        requirements=[],  # Target already specified
                        static_params={"defender": target},
                        success_probability=trip_prob,
                    )
                )

        return options

    def _calculate_opposed_probability(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
        attacker_stat: str,
        defender_stat: str,
    ) -> float:
        """Calculate success probability for an opposed stat check.

        Args:
            controller: Game controller for creating resolver.
            actor: The character attempting the action.
            target: The target character.
            attacker_stat: Stat name for the attacker (e.g., "strength").
            defender_stat: Stat name for the defender (e.g., "strength").

        Returns:
            Probability of success as a float in [0.0, 1.0].
        """
        resolution_modifiers = actor.modifiers.get_resolution_modifiers(attacker_stat)
        has_advantage = resolution_modifiers.get("has_advantage", False)
        has_disadvantage = resolution_modifiers.get("has_disadvantage", False)

        attacker_score = getattr(actor.stats, attacker_stat)
        defender_score = getattr(target.stats, defender_stat)

        resolver = controller.create_resolver(
            ability_score=attacker_score,
            roll_to_exceed=defender_score + 10,
            has_advantage=has_advantage,
            has_disadvantage=has_disadvantage,
        )
        return resolver.calculate_success_probability()

    def get_all_combat_actions(
        self,
        controller: Controller,
        actor: Character,
        context: ActionContext,
        target: Character | None = None,
    ) -> list[ActionOption]:
        """Discover all combat actions without any UI presentation logic.

        Only shows actions for the currently active weapon, not all equipped weapons.
        This keeps the action panel focused and less overwhelming.

        Args:
            controller: Game controller.
            actor: The acting character.
            context: Action context containing nearby actors.
            target: Optional specific target for probability calculation.
                    If None, actions are returned with success_probability=None.
        """
        options: list[ActionOption] = []
        active_weapon = actor.inventory.get_active_item()
        equipped_weapons = [active_weapon] if active_weapon else []

        if not equipped_weapons:
            from brileta.game.items.item_types import FISTS_TYPE

            equipped_weapons = [FISTS_TYPE.create()]

        for weapon in equipped_weapons:
            # Melee attacks - calculate probability regardless of distance since
            # player will approach before executing
            melee_prob: float | None = None
            if target is not None and weapon.melee_attack:
                melee_prob = self.context_builder.calculate_combat_probability(
                    controller, actor, target, "strength"
                )
            if weapon.melee_attack:
                verb = weapon.melee_attack._spec.verb.capitalize()
                options.append(
                    ActionOption(
                        id=f"melee-{weapon.name}",
                        name=verb,
                        description=f"{verb} a target with {weapon.name}",
                        category=ActionCategory.COMBAT,
                        action_class=AttackIntent,
                        requirements=[ActionRequirement.TARGET_ACTOR],
                        static_params={"weapon": weapon, "attack_mode": "melee"},
                        success_probability=melee_prob,
                    )
                )

            # Ranged attacks
            ranged_prob: float | None = None
            if (
                target is not None
                and weapon.ranged_attack
                and weapon.ranged_attack.current_ammo > 0
            ):
                dist = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)
                range_cat = ranges.get_range_category(dist, weapon)
                range_mods = ranges.get_range_modifier(weapon, range_cat)
                if range_mods is not None:
                    ranged_prob = self.context_builder.calculate_combat_probability(
                        controller,
                        actor,
                        target,
                        "observation",
                        range_mods,
                    )

            if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                verb = weapon.ranged_attack._spec.verb.capitalize()
                options.append(
                    ActionOption(
                        id=f"ranged-{weapon.name}",
                        name=verb,
                        description=f"{verb} a target with {weapon.name}",
                        category=ActionCategory.COMBAT,
                        action_class=AttackIntent,
                        requirements=[ActionRequirement.TARGET_ACTOR],
                        static_params={"weapon": weapon, "attack_mode": "ranged"},
                        success_probability=ranged_prob,
                    )
                )

        return options

    def get_combat_options_for_target(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
        context: ActionContext,
    ) -> list[ActionOption]:
        """Get combat options specifically for a given target.

        Only shows actions for the currently active weapon, not all equipped weapons.

        DEPRECATED: used by the old state machine, will be removed in Task 3.
        """
        options: list[ActionOption] = []
        active_weapon = actor.inventory.get_active_item()
        equipped_weapons = [active_weapon] if active_weapon else []

        if not equipped_weapons:
            from brileta.game.items.item_types import FISTS_TYPE

            equipped_weapons = [FISTS_TYPE.create()]

        gm = controller.gw.game_map
        if (
            target == actor
            or not isinstance(target, Character)
            or not target.stats
            or not target.health
            or not target.health.is_alive()
            or not gm.visible[target.x, target.y]
            or not ranges.has_line_of_sight(gm, actor.x, actor.y, target.x, target.y)
        ):
            return options

        distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)

        for weapon in equipped_weapons:
            if weapon.melee_attack and distance == 1:
                prob = self.context_builder.calculate_combat_probability(
                    controller, actor, target, "strength"
                )

                melee = cast("MeleeAttack", weapon.melee_attack)
                verb = melee._spec.verb
                display_text = f"{verb.title()} {target.name} with {weapon.name}"
                action_id = f"melee-{weapon.name}-{target.name}"

                options.append(
                    ActionOption(
                        id=action_id,
                        name=display_text,
                        display_text=display_text,
                        description=f"Close combat attack using {weapon.name}",
                        category=ActionCategory.COMBAT,
                        action_class=AttackIntent,
                        requirements=[ActionRequirement.TARGET_ACTOR],
                        static_params={"weapon": weapon, "attack_mode": "melee"},
                        success_probability=prob,
                        execute=lambda w=weapon, t=target: (
                            self.factory.create_melee_attack(controller, actor, t, w)
                        ),
                    )
                )

            if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                range_cat = ranges.get_range_category(distance, weapon)
                range_mods = ranges.get_range_modifier(weapon, range_cat)
                if range_mods is None:
                    ranged_cap = cast("RangedAttack", weapon.ranged_attack)
                    verb = ranged_cap._spec.verb
                    action_id = f"ranged-{weapon.name}-{target.name}"
                    out_of_range = (
                        self.formatter.get_attack_display_name(
                            weapon, "ranged", target.name
                        )
                        + " (OUT OF RANGE)"
                    )
                    options.append(
                        ActionOption(
                            id=action_id,
                            name=out_of_range,
                            display_text=out_of_range,
                            description=(
                                f"Target is beyond {weapon.name}'s maximum range"
                            ),
                            category=ActionCategory.COMBAT,
                            action_class=AttackIntent,
                            requirements=[ActionRequirement.TARGET_ACTOR],
                            static_params={"weapon": weapon, "attack_mode": "ranged"},
                            success_probability=0.0,
                        )
                    )
                    continue

                prob = self.context_builder.calculate_combat_probability(
                    controller,
                    actor,
                    target,
                    "observation",
                    range_mods,
                )

                ranged_cap = cast("RangedAttack", weapon.ranged_attack)
                verb = ranged_cap._spec.verb
                display_name = self.formatter.get_attack_display_name(
                    weapon, "ranged", target.name
                )
                action_id = f"ranged-{weapon.name}-{target.name}"

                options.append(
                    ActionOption(
                        id=action_id,
                        name=display_name,
                        display_text=display_name,
                        description=f"Ranged attack at {range_cat} range",
                        category=ActionCategory.COMBAT,
                        action_class=AttackIntent,
                        requirements=[ActionRequirement.TARGET_ACTOR],
                        static_params={"weapon": weapon, "attack_mode": "ranged"},
                        success_probability=prob,
                        cost_description=None,
                        execute=lambda w=weapon, t=target: (
                            self.factory.create_ranged_attack(controller, actor, t, w)
                        ),
                    )
                )

        return options

    def get_all_terminal_combat_actions(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Generate a flat list of every possible end combat action.

        Only shows actions for the currently active weapon, not all equipped weapons.

        DEPRECATED: used by the old state machine, will be removed in Task 3.
        """
        context = self.context_builder.build_context(controller, actor)

        options: list[ActionOption] = []
        # Note: The weapon filtering is handled by get_combat_options_for_target(),
        # but we keep this check for the empty fallback case.
        active_weapon = actor.inventory.get_active_item()
        equipped_weapons = [active_weapon] if active_weapon else []

        if not equipped_weapons:
            from brileta.game.items.item_types import FISTS_TYPE

            equipped_weapons = [FISTS_TYPE.create()]

        for target in context.nearby_actors:
            if (
                target == actor
                or not isinstance(target, Character)
                or not target.health.is_alive()
            ):
                continue

            options.extend(
                self.get_combat_options_for_target(controller, actor, target, context)
            )

        return options
