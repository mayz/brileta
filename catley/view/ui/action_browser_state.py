from __future__ import annotations

import functools
import string
from typing import TYPE_CHECKING, cast

from catley.game import ranges
from catley.game.actions.base import GameAction
from catley.game.actions.discovery import ActionDiscovery
from catley.game.actions.discovery.core_discovery import (
    ActionCategory,
    ActionOption,
)
from catley.game.actors import Character
from catley.game.items.capabilities import MeleeAttack, RangedAttack
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from catley.controller import Controller


class ActionBrowserStateMachine:
    """State machine for navigating the action browser UI."""

    def __init__(self, action_discovery: ActionDiscovery) -> None:
        self.action_discovery = action_discovery
        self.ui_state = "main"
        self.selected_target: Character | None = None
        self.selected_weapon: Item | None = None
        self.selected_attack_mode: str | None = None

    # ------------------------------------------------------------------
    def get_options_for_current_state(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        context = self.action_discovery.context_builder.build_context(controller, actor)
        all_actions: list[ActionOption] = []
        all_actions.extend(
            self.action_discovery.combat_discovery.discover_combat_actions(
                controller, actor, context
            )
        )
        all_actions.extend(
            self.action_discovery.item_discovery.discover_item_actions(
                controller, actor, context
            )
        )
        all_actions.extend(
            self.action_discovery.environment_discovery.discover_environment_actions(
                controller, actor, context
            )
        )

        match self.ui_state:
            case "main":
                options = self._get_main_options(
                    controller, actor, context, all_actions
                )
                options = self.action_discovery._sort_by_relevance(options, context)
            case "submenu_items":
                options = self._get_submenu_options(
                    controller, actor, context, ActionCategory.ITEMS, all_actions
                )
            case "submenu_environment":
                options = self._get_submenu_options(
                    controller, actor, context, ActionCategory.ENVIRONMENT, all_actions
                )
            case "submenu_social":
                options = self._get_submenu_options(
                    controller, actor, context, ActionCategory.SOCIAL, all_actions
                )
            case "select_attack_approach":
                options = self._get_attack_approach_options(controller, actor, context)
            case "select_target":
                options = self._get_target_selection_options(controller, actor, context)
            case "select_weapon":
                options = self._get_weapon_selection_options(controller, actor, context)
            case "weapons_for_target":
                options = self._get_weapons_for_selected_target(
                    controller, actor, context
                )
            case "targets_for_weapon":
                options = self._get_targets_for_selected_weapon(
                    controller, actor, context
                )
            case _:
                self.ui_state = "main"
                options = self._get_main_options(
                    controller, actor, context, all_actions
                )
                options = self.action_discovery._sort_by_relevance(options, context)
        return options

    def handle_back_navigation(self, controller: Controller) -> bool:
        return self._go_back(controller)

    def set_ui_state(self, new_state: str, **kwargs) -> bool:
        return self._set_ui_state(new_state, **kwargs)

    def reset_state(self) -> None:
        self.ui_state = "main"
        self.selected_target = None
        self.selected_weapon = None
        self.selected_attack_mode = None

    # ------------------------------------------------------------------
    # Internal state helpers
    def _get_main_options(
        self,
        controller: Controller,
        actor: Character,
        context,
        all_actions: list[ActionOption],
    ) -> list[ActionOption]:
        by_category: dict[ActionCategory, list[ActionOption]] = {}
        for act in all_actions:
            by_category.setdefault(act.category, []).append(act)

        options: list[ActionOption] = []
        for label, cats in ActionDiscovery.TOP_LEVEL_CATEGORIES.items():
            if any(by_category.get(cat) for cat in cats):
                if cats[0] == ActionCategory.COMBAT:
                    execute = functools.partial(
                        self._set_ui_state, "select_attack_approach"
                    )
                else:
                    execute = functools.partial(
                        self._set_ui_state, f"submenu_{cats[0].name.lower()}"
                    )
                options.append(
                    ActionOption(
                        id=label,
                        name=label,
                        description="",
                        category=cats[0],
                        action_class=cast(type[GameAction], type(None)),
                        requirements=[],
                        static_params={},
                        execute=execute,
                    )
                )
        return options

    def _get_submenu_options(
        self,
        controller: Controller,
        actor: Character,
        context,
        target_category: ActionCategory,
        all_actions: list[ActionOption],
    ) -> list[ActionOption]:
        filtered = [act for act in all_actions if act.category == target_category]

        return [
            ActionOption(
                id="back",
                name="\u2190 Back",
                description="Return to previous screen",
                category=target_category,
                action_class=cast(type[GameAction], type(None)),
                requirements=[],
                static_params={},
                hotkey="escape",
                execute=functools.partial(self._go_back, controller),
            ),
            *filtered,
        ]

    def _get_attack_approach_options(
        self, controller: Controller, actor: Character, context
    ) -> list[ActionOption]:
        options: list[ActionOption] = []

        if context.nearby_actors:
            options.append(
                ActionOption(
                    id="approach-target",
                    name="Attack a target...",
                    description="Select target, then choose how to attack",
                    category=ActionCategory.COMBAT,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                    static_params={},
                    hotkey="t",
                    execute=functools.partial(self._set_ui_state, "select_target"),
                )
            )
            options.append(
                ActionOption(
                    id="approach-weapon",
                    name="Attack with a weapon...",
                    description="Select weapon/attack mode, then choose target",
                    category=ActionCategory.COMBAT,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                    static_params={},
                    hotkey="w",
                    execute=functools.partial(self._set_ui_state, "select_weapon"),
                )
            )

        options.insert(
            0,
            ActionOption(
                id="back",
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
                action_class=cast(type[GameAction], type(None)),
                requirements=[],
                static_params={},
                hotkey="escape",
                execute=functools.partial(self._go_back, controller),
            ),
        )
        return options

    def _get_target_selection_options(
        self, controller: Controller, actor: Character, context
    ) -> list[ActionOption]:
        options: list[ActionOption] = [
            ActionOption(
                id="back",
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
                action_class=cast(type[GameAction], type(None)),
                requirements=[],
                static_params={},
                hotkey="escape",
                execute=functools.partial(self._go_back, controller),
            )
        ]
        letters = string.ascii_lowercase
        sorted_targets = sorted(
            context.nearby_actors,
            key=lambda t: ranges.calculate_distance(actor.x, actor.y, t.x, t.y),
        )
        for i, target in enumerate(sorted_targets):
            if target == actor or not isinstance(target, Character):
                continue
            if not target.health.is_alive():
                continue
            distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)
            desc = "adjacent" if distance == 1 else "close" if distance <= 3 else "far"
            options.append(
                ActionOption(
                    id=f"target-{target.name}",
                    name=f"{target.name} ({desc})",
                    description=f"Attack {target.name}",
                    category=ActionCategory.COMBAT,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                    static_params={},
                    hotkey=letters[i] if i < len(letters) else None,
                    execute=functools.partial(
                        self._set_ui_state, "weapons_for_target", target=target
                    ),
                )
            )
        return options

    def _get_weapon_selection_options(
        self, controller: Controller, actor: Character, context
    ) -> list[ActionOption]:
        options: list[ActionOption] = [
            ActionOption(
                id="back",
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
                action_class=cast(type[GameAction], type(None)),
                requirements=[],
                static_params={},
                hotkey="escape",
                execute=functools.partial(self._go_back, controller),
            )
        ]
        equipped_weapons = [w for w in actor.inventory.attack_slots if w is not None]
        if not equipped_weapons:
            from catley.game.items.item_types import FISTS_TYPE

            equipped_weapons = [FISTS_TYPE.create()]

        option_index = 0
        letters = string.ascii_lowercase
        for weapon in equipped_weapons:
            if weapon.melee_attack:
                has_adjacent = any(
                    t
                    for t in context.nearby_actors
                    if t != actor
                    and isinstance(t, Character)
                    and t.health.is_alive()
                    and ranges.calculate_distance(actor.x, actor.y, t.x, t.y) == 1
                )
                if has_adjacent:
                    melee = cast(MeleeAttack, weapon.melee_attack)
                    verb = melee._spec.verb
                    options.append(
                        ActionOption(
                            id=f"weapon-melee-{weapon.name}",
                            name=f"{weapon.name} ({verb.title()})",
                            description=f"Melee attack using {weapon.name}",
                            category=ActionCategory.COMBAT,
                            action_class=cast(type[GameAction], type(None)),
                            requirements=[],
                            static_params={},
                            hotkey=letters[option_index]
                            if option_index < len(letters)
                            else None,
                            execute=functools.partial(
                                self._set_ui_state,
                                "targets_for_weapon",
                                weapon=weapon,
                                attack_mode="melee",
                            ),
                        )
                    )
                    option_index += 1
            if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                has_ranged_targets = any(
                    t
                    for t in context.nearby_actors
                    if t != actor
                    and isinstance(t, Character)
                    and t.health.is_alive()
                    and ranges.get_range_category(
                        ranges.calculate_distance(actor.x, actor.y, t.x, t.y), weapon
                    )
                    != "out_of_range"
                )
                if has_ranged_targets:
                    ranged = cast(RangedAttack, weapon.ranged_attack)
                    verb = ranged._spec.verb
                    options.append(
                        ActionOption(
                            id=f"weapon-ranged-{weapon.name}",
                            name=f"{weapon.name} ({verb.title()})",
                            description=f"Ranged attack using {weapon.name}",
                            category=ActionCategory.COMBAT,
                            action_class=cast(type[GameAction], type(None)),
                            requirements=[],
                            static_params={},
                            hotkey=letters[option_index]
                            if option_index < len(letters)
                            else None,
                            execute=functools.partial(
                                self._set_ui_state,
                                "targets_for_weapon",
                                weapon=weapon,
                                attack_mode="ranged",
                            ),
                        )
                    )
                    option_index += 1
        return options

    def _get_weapons_for_selected_target(
        self, controller: Controller, actor: Character, context
    ) -> list[ActionOption]:
        if not self.selected_target:
            self.ui_state = "select_target"
            return []

        options = [
            ActionOption(
                id="back",
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
                action_class=cast(type[GameAction], type(None)),
                requirements=[],
                static_params={},
                hotkey="escape",
                execute=functools.partial(self._go_back, controller),
            )
        ]
        options.extend(
            self.action_discovery.combat_discovery.get_combat_options_for_target(
                controller, actor, self.selected_target, context
            )
        )
        return options

    def _get_targets_for_selected_weapon(
        self, controller: Controller, actor: Character, context
    ) -> list[ActionOption]:
        weapon = self.selected_weapon
        mode = self.selected_attack_mode
        if not weapon or not mode:
            self.ui_state = "select_weapon"
            return []

        options: list[ActionOption] = [
            ActionOption(
                id="back",
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
                action_class=cast(type[GameAction], type(None)),
                requirements=[],
                static_params={},
                hotkey="escape",
                execute=functools.partial(self._go_back, controller),
            )
        ]
        sorted_targets = sorted(
            context.nearby_actors,
            key=lambda t: ranges.calculate_distance(actor.x, actor.y, t.x, t.y),
        )
        for target in sorted_targets:
            if target == actor or not isinstance(target, Character):
                continue
            if not target.health.is_alive():
                continue
            gm = controller.gw.game_map
            if not gm.visible[target.x, target.y]:
                continue
            if not ranges.has_line_of_sight(gm, actor.x, actor.y, target.x, target.y):
                continue
            distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)
            if mode == "melee" and weapon.melee_attack and distance == 1:
                prob = (
                    self.action_discovery.context_builder.calculate_combat_probability(
                        controller, actor, target, "strength"
                    )
                )
                options.append(
                    ActionOption(
                        id=f"melee-{weapon.name}-{target.name}",
                        name=self.action_discovery.formatter.get_attack_display_name(
                            weapon, "melee", target.name
                        ),
                        description=f"Close combat attack using {weapon.name}",
                        category=ActionCategory.COMBAT,
                        action_class=cast(type[GameAction], type(None)),
                        requirements=[],
                        static_params={},
                        success_probability=prob,
                        execute=lambda t=target,
                        w=weapon: self.action_discovery.factory.create_melee_attack(
                            controller, actor, t, w
                        ),
                    )
                )
            if (
                mode == "ranged"
                and weapon.ranged_attack
                and weapon.ranged_attack.current_ammo > 0
            ):
                range_cat = ranges.get_range_category(distance, weapon)
                range_mods = ranges.get_range_modifier(weapon, range_cat)
                if range_mods is None:
                    out_of_range = (
                        self.action_discovery.formatter.get_attack_display_name(
                            weapon, "ranged", target.name
                        )
                        + " (OUT OF RANGE)"
                    )
                    options.append(
                        ActionOption(
                            id=f"ranged-{weapon.name}-{target.name}",
                            name=out_of_range,
                            description=(
                                f"Target is beyond {weapon.name}'s maximum range"
                            ),
                            category=ActionCategory.COMBAT,
                            action_class=cast(type[GameAction], type(None)),
                            requirements=[],
                            static_params={},
                            success_probability=0.0,
                        )
                    )
                    continue
                prob = (
                    self.action_discovery.context_builder.calculate_combat_probability(
                        controller, actor, target, "observation", range_mods
                    )
                )
                options.append(
                    ActionOption(
                        id=f"ranged-{weapon.name}-{target.name}",
                        name=self.action_discovery.formatter.get_attack_display_name(
                            weapon, "ranged", target.name
                        ),
                        description=f"Ranged attack at {range_cat} range",
                        category=ActionCategory.COMBAT,
                        action_class=cast(type[GameAction], type(None)),
                        requirements=[],
                        static_params={},
                        success_probability=prob,
                        execute=lambda t=target,
                        w=weapon: self.action_discovery.factory.create_ranged_attack(
                            controller, actor, t, w
                        ),
                    )
                )
        return options

    # Navigation helpers
    def _go_back(self, controller: Controller) -> bool:
        if self.ui_state in [
            "submenu_items",
            "submenu_environment",
            "submenu_social",
        ]:
            self.ui_state = "main"
            self.selected_target = None
            self.selected_weapon = None
            self.selected_attack_mode = None
        elif self.ui_state == "select_attack_approach":
            self.ui_state = "main"
        elif self.ui_state == "select_target":
            self.ui_state = "select_attack_approach"
            self.selected_target = None
        elif self.ui_state == "select_weapon":
            self.ui_state = "select_attack_approach"
            self.selected_weapon = None
            self.selected_attack_mode = None
        elif self.ui_state == "weapons_for_target":
            self.ui_state = "select_target"
            self.selected_weapon = None
            self.selected_attack_mode = None
        elif self.ui_state == "targets_for_weapon":
            self.ui_state = "select_weapon"
            self.selected_target = None
        else:
            self.ui_state = "main"
            self.selected_target = None
            self.selected_weapon = None
            self.selected_attack_mode = None
        return False

    def _set_ui_state(self, new_state: str, **kwargs) -> bool:
        self.ui_state = new_state
        if "target" in kwargs:
            self.selected_target = kwargs["target"]
        if "weapon" in kwargs:
            self.selected_weapon = kwargs["weapon"]
        if "attack_mode" in kwargs:
            self.selected_attack_mode = kwargs["attack_mode"]
        return False
