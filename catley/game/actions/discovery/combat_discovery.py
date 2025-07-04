from __future__ import annotations

from typing import TYPE_CHECKING, cast

from catley.game import ranges
from catley.game.actions.combat import AttackIntent
from catley.game.actors import Character

from .action_context import ActionContext, ActionContextBuilder
from .action_factory import ActionFactory
from .action_formatters import ActionFormatter
from .types import ActionCategory, ActionOption, ActionRequirement

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.items.capabilities import MeleeAttack, RangedAttack


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
    def discover_combat_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Return all combat options available to the actor."""
        return self.get_all_combat_actions(controller, actor, context)

    def get_all_combat_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Discover all combat actions without any UI presentation logic."""
        options: list[ActionOption] = []
        equipped_weapons = [w for w in actor.inventory.attack_slots if w is not None]

        if not equipped_weapons:
            from catley.game.items.item_types import FISTS_TYPE

            equipped_weapons = [FISTS_TYPE.create()]

        # Choose a representative target for probability calculations
        representative: Character | None = None
        closest_dist: float | None = None
        for potential in context.nearby_actors:
            if (
                isinstance(potential, Character)
                and potential.health
                and potential.health.is_alive()
            ):
                dist = ranges.calculate_distance(
                    actor.x, actor.y, potential.x, potential.y
                )
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist
                    representative = potential

        for weapon in equipped_weapons:
            # Melee attacks
            melee_prob: float | None = None
            if representative is not None:
                dist = ranges.calculate_distance(
                    actor.x, actor.y, representative.x, representative.y
                )
                if weapon.melee_attack and dist == 1:
                    melee_prob = self.context_builder.calculate_combat_probability(
                        controller, actor, representative, "strength"
                    )
            if weapon.melee_attack:
                options.append(
                    ActionOption(
                        id=f"melee-{weapon.name}",
                        name=f"Melee attack with {weapon.name}",
                        description=f"Close combat attack using {weapon.name}",
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
                weapon.ranged_attack
                and weapon.ranged_attack.current_ammo > 0
                and representative is not None
            ):
                dist = ranges.calculate_distance(
                    actor.x, actor.y, representative.x, representative.y
                )
                range_cat = ranges.get_range_category(dist, weapon)
                range_mods = ranges.get_range_modifier(weapon, range_cat)
                if range_mods is not None:
                    ranged_prob = self.context_builder.calculate_combat_probability(
                        controller,
                        actor,
                        representative,
                        "observation",
                        range_mods,
                    )

            if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                options.append(
                    ActionOption(
                        id=f"ranged-{weapon.name}",
                        name=f"Ranged attack with {weapon.name}",
                        description=f"Use {weapon.name} for a ranged attack",
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

        DEPRECATED: used by the old state machine, will be removed in Task 3.
        """
        options: list[ActionOption] = []
        equipped_weapons = [w for w in actor.inventory.attack_slots if w is not None]

        if not equipped_weapons:
            from catley.game.items.item_types import FISTS_TYPE

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
                action_id = f"melee-{verb.title()}-{target.name}"

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
                        execute=lambda w=weapon,
                        t=target: self.factory.create_melee_attack(
                            controller, actor, t, w
                        ),
                    )
                )

            if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                range_cat = ranges.get_range_category(distance, weapon)
                range_mods = ranges.get_range_modifier(weapon, range_cat)
                if range_mods is None:
                    ranged_cap = cast("RangedAttack", weapon.ranged_attack)
                    verb = ranged_cap._spec.verb
                    action_id = f"ranged-{verb.title()}-{target.name}"
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
                ammo_warning = ""
                current_ammo = weapon.ranged_attack.current_ammo
                if current_ammo == 1:
                    ammo_warning = " (LAST SHOT!)"
                elif current_ammo <= 3:
                    ammo_warning = " (Low Ammo)"

                display_name += ammo_warning
                action_id = f"ranged-{verb.title()}-{target.name}"

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
                        execute=lambda w=weapon,
                        t=target: self.factory.create_ranged_attack(
                            controller, actor, t, w
                        ),
                    )
                )

        return options

    def get_all_terminal_combat_actions(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Generate a flat list of every possible end combat action.

        DEPRECATED: used by the old state machine, will be removed in Task 3.
        """
        context = self.context_builder.build_context(controller, actor)

        options: list[ActionOption] = []
        equipped_weapons = [w for w in actor.inventory.attack_slots if w is not None]

        if not equipped_weapons:
            from catley.game.items.item_types import FISTS_TYPE

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
