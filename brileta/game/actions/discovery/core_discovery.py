"""Core action discovery coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from brileta.game.actors import Character

from .action_context import ActionContext, ActionContextBuilder
from .action_factory import ActionFactory
from .types import ActionCategory, ActionOption

if TYPE_CHECKING:
    from brileta.controller import Controller


class ActionDiscovery:
    """Main system for discovering available actions."""

    TOP_LEVEL_CATEGORIES: ClassVar[dict[str, list[ActionCategory]]] = {
        "Attack...": [ActionCategory.COMBAT],
        "Stunts...": [ActionCategory.STUNT],
        "Interact with Environment...": [ActionCategory.ENVIRONMENT],
        "Use Item...": [ActionCategory.ITEMS],
        "Social Actions...": [ActionCategory.SOCIAL],
    }

    def __init__(self) -> None:
        from .action_formatters import ActionFormatter
        from .combat_discovery import CombatActionDiscovery
        from .environment_discovery import EnvironmentActionDiscovery
        from .item_discovery import ItemActionDiscovery

        self.context_builder = ActionContextBuilder()
        self.factory = ActionFactory()
        self.formatter = ActionFormatter()
        self.combat_discovery = CombatActionDiscovery(
            self.context_builder, self.factory, self.formatter
        )
        self.item_discovery = ItemActionDiscovery(self.factory, self.formatter)
        self.environment_discovery = EnvironmentActionDiscovery(
            self.factory, self.formatter
        )

    @staticmethod
    def get_probability_descriptor(probability: float) -> tuple[str, str]:
        from .action_formatters import ActionFormatter

        return ActionFormatter.get_probability_descriptor(probability)

    def get_available_options(
        self, controller: Controller, actor: Character, sort_by_relevance: bool = True
    ) -> list[ActionOption]:
        context = self.context_builder.build_context(controller, actor)
        all_actions: list[ActionOption] = []
        all_actions.extend(
            self.combat_discovery.discover_combat_actions(controller, actor, context)
        )
        all_actions.extend(
            self.combat_discovery.discover_stunt_actions(controller, actor, context)
        )
        all_actions.extend(
            self.item_discovery.discover_item_actions(controller, actor, context)
        )
        all_actions.extend(
            self.environment_discovery.discover_environment_actions(
                controller, actor, context
            )
        )
        if sort_by_relevance:
            all_actions = self._sort_by_relevance(all_actions, context)
        return all_actions

    def get_all_available_actions(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Get all available actions without UI categorization or sorting."""
        context = self.context_builder.build_context(controller, actor)
        all_actions: list[ActionOption] = []
        all_actions.extend(
            self.combat_discovery.discover_combat_actions(controller, actor, context)
        )
        all_actions.extend(
            self.combat_discovery.discover_stunt_actions(controller, actor, context)
        )
        all_actions.extend(
            self.item_discovery.discover_item_actions(controller, actor, context)
        )
        all_actions.extend(
            self.environment_discovery.discover_environment_actions(
                controller, actor, context
            )
        )
        return all_actions

    def get_options_for_category(
        self, controller: Controller, actor: Character, category: ActionCategory
    ) -> list[ActionOption]:
        all_options = self.get_available_options(controller, actor)
        return [option for option in all_options if option.category == category]

    def get_options_for_target(
        self, controller: Controller, actor: Character, target: Character
    ) -> list[ActionOption]:
        """Get actions available when targeting a specific character.

        In combat mode: Returns full combat actions (weapon attacks, stunts).
        Outside combat: Returns an "Attack" gateway action that enters combat mode,
                       plus any social actions (Talk, etc.).
        """
        context = self.context_builder.build_context(controller, actor)
        options: list[ActionOption] = []

        if controller.is_combat_mode():
            # Inside combat - show full combat actions
            options.extend(
                self.combat_discovery.get_combat_options_for_target(
                    controller, actor, target, context
                )
            )
            # Stunt actions (Push, etc.) - filter to only this target
            options.extend(
                stunt
                for stunt in self.combat_discovery.discover_stunt_actions(
                    controller, actor, context
                )
                if stunt.static_params.get("defender") == target
            )
        else:
            # Outside combat - show "Attack" gateway, stunts, and social actions
            # Only show Attack and stunts if target is a valid combat target
            if target.health and target.health.is_alive() and target is not actor:
                options.append(self._create_attack_gateway_action(controller))
                options.append(self._create_push_action(controller, actor, target))
                options.append(self._create_trip_action(controller, actor, target))

            # Add social actions (Talk)
            options.append(self._create_talk_action(controller, actor, target))

        return options

    def _create_attack_gateway_action(self, controller: Controller) -> ActionOption:
        """Create the 'Attack...' gateway action that enters combat mode.

        The ellipsis signals to the player that this action leads to more
        choices (combat mode entry with weapon/target selection) rather than
        executing immediately like Talk or Shove.
        """

        def enter_combat() -> bool:
            controller.enter_combat_mode()
            return True

        return ActionOption(
            id="attack-gateway",
            name="Attack...",
            description="Enter combat mode to select an attack",
            category=ActionCategory.COMBAT,
            action_class=None,
            requirements=[],
            static_params={},
            execute=enter_combat,
        )

    def _create_talk_action(
        self, controller: Controller, actor: Character, target: Character
    ) -> ActionOption:
        """Create a 'Talk' action for social interaction."""
        from brileta.game.actions.social import TalkIntent, TalkPlan

        def talk() -> bool:
            controller.start_plan(
                actor,
                TalkPlan,
                target_actor=target,
                target_position=(target.x, target.y),
            )
            return True

        return ActionOption(
            id="talk",
            name="Talk",
            description=f"Approach and speak with {target.name}",
            category=ActionCategory.SOCIAL,
            action_class=TalkIntent,
            requirements=[],
            static_params={"target": target},
            execute=talk,
        )

    def _create_push_action(
        self, controller: Controller, actor: Character, target: Character
    ) -> ActionOption:
        """Create a 'Push' stunt action.

        Push shoves the target one tile away. If not adjacent, the actor will
        approach the target first. If the target is non-hostile, this triggers
        hostility and auto-enters combat mode.
        """
        from brileta.game.actions.stunts import PushIntent, PushPlan

        def push() -> bool:
            controller.start_plan(
                actor,
                PushPlan,
                target_actor=target,
                target_position=(target.x, target.y),
            )
            return True

        return ActionOption(
            id="push",
            name="Push",
            description="Shove target 1 tile away. Strength vs Strength.",
            category=ActionCategory.STUNT,
            action_class=PushIntent,
            requirements=[],
            static_params={"defender": target},
            execute=push,
        )

    def _create_trip_action(
        self, controller: Controller, actor: Character, target: Character
    ) -> ActionOption:
        """Create a 'Trip' stunt action.

        Trip knocks the target prone. If not adjacent, the actor will approach
        the target first. If the target is non-hostile, this triggers hostility
        and auto-enters combat mode.
        """
        from brileta.game.actions.stunts import TripIntent, TripPlan

        def trip() -> bool:
            controller.start_plan(
                actor,
                TripPlan,
                target_actor=target,
                target_position=(target.x, target.y),
            )
            return True

        return ActionOption(
            id="trip",
            name="Trip",
            description="Knock target prone. Agility vs Agility.",
            category=ActionCategory.STUNT,
            action_class=TripIntent,
            requirements=[],
            static_params={"defender": target},
            execute=trip,
        )

    def _sort_by_relevance(
        self, options: list[ActionOption], context: ActionContext
    ) -> list[ActionOption]:
        def relevance_score(option: ActionOption) -> int:
            score = 0
            if context.in_combat and option.category in (
                ActionCategory.COMBAT,
                ActionCategory.STUNT,
            ):
                score += 100
            success_prob = option.success_probability
            if success_prob is not None:
                score += int(success_prob * 50)
            if option.hotkey:
                score += 20
            return score

        return sorted(options, key=relevance_score, reverse=True)

    # ------------------------------------------------------------------
    # Backwards compatibility wrappers for old API
    # DEPRECATED: These will be removed in Task 3

    def _build_context(self, controller: Controller, actor: Character) -> ActionContext:
        """DEPRECATED wrapper for context building."""
        return self.context_builder.build_context(controller, actor)

    def _calculate_combat_probability(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
        stat_name: str,
        range_modifiers: dict | None = None,
    ) -> float:
        """DEPRECATED helper for combat probability."""
        return self.context_builder.calculate_combat_probability(
            controller, actor, target, stat_name, range_modifiers
        )

    def _get_all_terminal_combat_actions(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """DEPRECATED wrapper for legacy combat actions."""
        return self.combat_discovery.get_all_terminal_combat_actions(controller, actor)

    def _get_combat_options(
        self,
        controller: Controller,
        actor: Character,
        context: ActionContext,
        target: Character | None = None,
    ) -> list[ActionOption]:
        """DEPRECATED wrapper for combat options."""
        return self.combat_discovery.get_all_combat_actions(
            controller, actor, context, target
        )

    def _get_combat_options_for_target(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
        context: ActionContext,
    ) -> list[ActionOption]:
        """DEPRECATED wrapper for target combat options."""
        return self.combat_discovery.get_combat_options_for_target(
            controller, actor, target, context
        )

    def _get_inventory_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """DEPRECATED wrapper for inventory options."""
        return self.item_discovery._get_inventory_options(controller, actor, context)

    def _get_recovery_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """DEPRECATED wrapper for recovery options."""
        return self.item_discovery._get_recovery_options(controller, actor, context)

    def _get_environment_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """DEPRECATED wrapper for environment options."""
        return self.environment_discovery._get_environment_options(
            controller,
            actor,
            context,
        )
