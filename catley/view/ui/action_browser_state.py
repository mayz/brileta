from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from catley.game import ranges
from catley.game.actions.base import GameAction
from catley.game.actions.combat import AttackAction
from catley.game.actions.discovery import ActionDiscovery
from catley.game.actions.discovery.core_discovery import (
    ActionCategory,
    ActionOption,
)
from catley.game.actions.discovery.types import ActionRequirement, CombatIntentCache
from catley.game.actors import Character
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from catley.controller import Controller


class ActionBrowserStateMachine:
    """Generic requirement-driven state machine for the action browser."""

    def __init__(self, action_discovery: ActionDiscovery) -> None:
        self.action_discovery = action_discovery
        self.ui_state: str = "main"
        self.current_action_option: ActionOption | None = None
        self.fulfilled_requirements: dict[ActionRequirement, Any] = {}
        # Legacy combat workflow selections
        self.selected_target: Character | None = None
        self.selected_weapon: Item | None = None
        self.selected_attack_mode: str | None = None

    # ------------------------------------------------------------------
    def get_options_for_current_state(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Return the list of action options for the current state."""
        if self.current_action_option is not None:
            next_requirement = self._get_next_unfulfilled_requirement()
            if next_requirement is None:
                self._execute_completed_action(controller, actor)
                return []
            return self._get_options_for_requirement(
                controller, actor, next_requirement
            )

        match self.ui_state:
            case "main":
                return self._get_main_menu_options(controller, actor)
            case "combat":
                return self._get_combat_category_options(controller, actor)
            case "select_target":
                return self._get_target_selection_options(controller, actor)
            case "select_weapon":
                return self._get_weapon_selection_options(controller, actor)
            case "weapons_for_target":
                return self._get_weapons_for_selected_target(controller, actor)
            case "targets_for_weapon":
                return self._get_targets_for_selected_weapon(controller, actor)
            case "items":
                return self._get_items_category_options(controller, actor)
            case "environment":
                return self._get_environment_category_options(controller, actor)
            case "social":
                return self._get_social_category_options(controller, actor)
            case _:
                self.ui_state = "main"
                return self._get_main_menu_options(controller, actor)

    # ------------------------------------------------------------------
    def handle_back_navigation(self) -> bool:
        return self._go_back()

    def reset_state(self) -> None:
        self.current_action_option = None
        self.fulfilled_requirements = {}
        self.ui_state = "main"
        self.selected_target = None
        self.selected_weapon = None
        self.selected_attack_mode = None

    def set_current_action(self, action_option: ActionOption) -> bool:
        self.current_action_option = action_option
        self.fulfilled_requirements = {}
        self.selected_target = None
        self.selected_weapon = None
        self.selected_attack_mode = None
        return False

    # ------------------------------------------------------------------
    # Menu option generation helpers
    def _get_main_menu_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Show top-level categories that actually contain actions."""
        all_actions = self.action_discovery.get_all_available_actions(controller, actor)
        categories_present = {opt.category for opt in all_actions}

        options: list[ActionOption] = []
        for label, cats in ActionDiscovery.TOP_LEVEL_CATEGORIES.items():
            if any(cat in categories_present for cat in cats):
                cat_state = cats[0].name.lower()
                options.append(
                    ActionOption(
                        id=f"category-{cat_state}",
                        name=label,
                        description=f"Browse {cat_state} actions",
                        category=cats[0],
                        action_class=cast(type[GameAction], type(None)),
                        requirements=[],
                        execute=lambda s=cat_state: self._set_ui_state(s),
                    )
                )

        return options

    def _get_combat_category_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Show T/W choice directly instead of intermediate step."""
        context = self.action_discovery.context_builder.build_context(controller, actor)
        options = [self._get_back_option()]

        if context.nearby_actors:
            options.append(
                ActionOption(
                    id="attack-target-first",
                    name="Attack a target...",
                    description="Select target, then choose how to attack",
                    category=ActionCategory.COMBAT,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                    hotkey="t",
                    execute=lambda: self._set_ui_state("select_target"),
                )
            )

        if self._has_usable_weapons(controller, actor, context):
            options.append(
                ActionOption(
                    id="attack-weapon-first",
                    name="Attack with weapon...",
                    description="Select weapon/attack mode, then choose target",
                    category=ActionCategory.COMBAT,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                    hotkey="w",
                    execute=lambda: self._set_ui_state("select_weapon"),
                )
            )

        return options

    def _get_items_category_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        options = [self._get_back_option()]
        options.extend(
            self.action_discovery.get_options_for_category(
                controller, actor, ActionCategory.ITEMS
            )
        )
        return options

    def _get_environment_category_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        options = [self._get_back_option()]
        options.extend(
            self.action_discovery.get_options_for_category(
                controller, actor, ActionCategory.ENVIRONMENT
            )
        )
        return options

    def _get_social_category_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        options = [self._get_back_option()]
        options.extend(
            self.action_discovery.get_options_for_category(
                controller, actor, ActionCategory.SOCIAL
            )
        )
        return options

    def _has_usable_weapons(
        self, controller: Controller, actor: Character, context
    ) -> bool:
        """Return True if the actor has any usable weapons in the current context."""
        equipped_weapons = [w for w in actor.inventory.attack_slots if w]
        if not equipped_weapons:
            return True  # Fists always available

        for weapon in equipped_weapons:
            if weapon.melee_attack and any(
                ranges.calculate_distance(actor.x, actor.y, t.x, t.y) == 1
                for t in context.nearby_actors
                if t != actor and t.health.is_alive()
            ):
                return True
            if (
                weapon.ranged_attack
                and weapon.ranged_attack.current_ammo > 0
                and any(
                    ranges.get_range_modifier(
                        weapon,
                        ranges.get_range_category(
                            ranges.calculate_distance(actor.x, actor.y, t.x, t.y),
                            weapon,
                        ),
                    )
                    is not None
                    for t in context.nearby_actors
                    if t != actor and t.health.is_alive()
                )
            ):
                return True
        return False

    def _get_target_selection_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Show targets for selection (target-first workflow)."""
        context = self.action_discovery.context_builder.build_context(controller, actor)
        options = [self._get_back_option()]

        valid_targets = [
            t
            for t in context.nearby_actors
            if t != actor
            and isinstance(t, Character)
            and t.health.is_alive()
            and controller.gw.game_map.visible[t.x, t.y]
        ]

        for target in sorted(
            valid_targets,
            key=lambda t: ranges.calculate_distance(actor.x, actor.y, t.x, t.y),
        ):
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
                    execute=lambda t=target: self._select_target_then_weapon(t),
                )
            )

        return options

    def _select_target_then_weapon(self, target: Character) -> bool:
        """Target selected, now show valid weapons for that target."""
        self.selected_target = target
        self.ui_state = "weapons_for_target"
        return False

    def _get_weapon_selection_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Show weapons for selection (weapon-first workflow)."""
        context = self.action_discovery.context_builder.build_context(controller, actor)
        options = [self._get_back_option()]

        equipped_weapons = [w for w in actor.inventory.attack_slots if w]
        if not equipped_weapons:
            from catley.game.items.item_types import FISTS_TYPE

            equipped_weapons = [FISTS_TYPE.create()]

        for weapon in equipped_weapons:
            if weapon.melee_attack and any(
                ranges.calculate_distance(actor.x, actor.y, t.x, t.y) == 1
                for t in context.nearby_actors
                if t != actor and t.health.is_alive()
            ):
                options.append(
                    ActionOption(
                        id=f"weapon-melee-{weapon.name}",
                        name=f"{weapon.name} (Melee)",
                        description=f"Melee attack using {weapon.name}",
                        category=ActionCategory.COMBAT,
                        action_class=cast(type[GameAction], type(None)),
                        requirements=[],
                        execute=lambda w=weapon: self._select_weapon_then_target(
                            w, "melee"
                        ),
                    )
                )

            if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                has_valid_targets = any(
                    ranges.get_range_modifier(
                        weapon,
                        ranges.get_range_category(
                            ranges.calculate_distance(actor.x, actor.y, t.x, t.y),
                            weapon,
                        ),
                    )
                    is not None
                    for t in context.nearby_actors
                    if t != actor and t.health.is_alive()
                )
                if has_valid_targets:
                    options.append(
                        ActionOption(
                            id=f"weapon-ranged-{weapon.name}",
                            name=f"{weapon.name} (Ranged)",
                            description=f"Ranged attack using {weapon.name}",
                            category=ActionCategory.COMBAT,
                            action_class=cast(type[GameAction], type(None)),
                            requirements=[],
                            execute=lambda w=weapon: self._select_weapon_then_target(
                                w, "ranged"
                            ),
                        )
                    )

        return options

    def _select_weapon_then_target(self, weapon: Item, mode: str) -> bool:
        """Weapon selected, now choose a target."""
        self.selected_weapon = weapon
        self.selected_attack_mode = mode
        self.ui_state = "targets_for_weapon"
        return False

    def _get_weapons_for_selected_target(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Show weapons valid for the chosen target."""
        if not self.selected_target:
            self.ui_state = "select_target"
            return []

        context = self.action_discovery.context_builder.build_context(controller, actor)
        options = [self._get_back_option()]
        options.extend(
            self.action_discovery.combat_discovery.get_combat_options_for_target(
                controller, actor, self.selected_target, context
            )
        )
        return options

    def _get_targets_for_selected_weapon(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        """Show targets valid for the chosen weapon and mode."""
        weapon = self.selected_weapon
        mode = self.selected_attack_mode
        if not weapon or not mode:
            self.ui_state = "select_weapon"
            return []

        context = self.action_discovery.context_builder.build_context(controller, actor)
        options = [self._get_back_option()]
        sorted_targets = sorted(
            context.nearby_actors,
            key=lambda t: ranges.calculate_distance(actor.x, actor.y, t.x, t.y),
        )
        gm = controller.gw.game_map
        for target in sorted_targets:
            if target == actor or not isinstance(target, Character):
                continue
            if not target.health.is_alive():
                continue
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
                        success_probability=prob,
                        execute=lambda t=target, w=weapon: (
                            self.action_discovery.factory.create_melee_attack(
                                controller,
                                actor,
                                t,
                                w,
                            )
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
                            success_probability=0.0,
                        )
                    )
                    continue
                prob = (
                    self.action_discovery.context_builder.calculate_combat_probability(
                        controller,
                        actor,
                        target,
                        "observation",
                        range_mods,
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
                        success_probability=prob,
                        execute=lambda t=target, w=weapon: (
                            self.action_discovery.factory.create_ranged_attack(
                                controller,
                                actor,
                                t,
                                w,
                            )
                        ),
                    )
                )

        return options

    def _get_back_option(self) -> ActionOption:
        return ActionOption(
            id="back",
            name="\u2190 Back",
            description="Return to previous screen",
            category=ActionCategory.MOVEMENT,
            action_class=cast(type[GameAction], type(None)),
            requirements=[],
            execute=lambda: self._go_back(),
        )

    def _get_next_unfulfilled_requirement(self) -> ActionRequirement | None:
        if not self.current_action_option:
            return None

        priority = [
            ActionRequirement.TARGET_ACTOR,
            ActionRequirement.ITEM_FROM_INVENTORY,
            ActionRequirement.TARGET_TILE,
        ]

        for req_type in priority:
            if (
                req_type in self.current_action_option.requirements
                and req_type not in self.fulfilled_requirements
            ):
                return req_type

        for requirement in self.current_action_option.requirements:
            if requirement not in self.fulfilled_requirements:
                return requirement
        return None

    def _get_options_for_requirement(
        self, controller: Controller, actor: Character, requirement: ActionRequirement
    ) -> list[ActionOption]:
        options = [
            ActionOption(
                id="back",
                name="\u2190 Back",
                description="Return to action selection",
                category=ActionCategory.MOVEMENT,
                action_class=cast(type[GameAction], type(None)),
                requirements=[],
                execute=lambda: self._go_back(),
            )
        ]

        match requirement:
            case ActionRequirement.TARGET_ACTOR:
                options.extend(self._get_target_actor_options(controller, actor))
            case ActionRequirement.TARGET_TILE:
                options.extend(self._get_target_tile_options(controller, actor))
            case ActionRequirement.ITEM_FROM_INVENTORY:
                options.extend(self._get_inventory_item_options(controller, actor))
            case _:
                print(f"Warning: Unknown requirement type: {requirement}")

        if len(options) == 1:
            options.append(
                ActionOption(
                    id="none",
                    name="(no valid options)",
                    description="",
                    category=ActionCategory.MOVEMENT,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                )
            )

        return options

    def _execute_completed_action(
        self, controller: Controller, actor: Character
    ) -> None:
        if (
            not self.current_action_option
            or not self.current_action_option.action_class
        ):
            return

        all_params = dict(self.current_action_option.static_params)

        action_param_mapping = {
            "AttackAction": {ActionRequirement.TARGET_ACTOR: "defender"},
            "OpenDoorAction": {ActionRequirement.TARGET_TILE: ["x", "y"]},
            "CloseDoorAction": {ActionRequirement.TARGET_TILE: ["x", "y"]},
            "DEFAULT": {
                ActionRequirement.TARGET_ACTOR: "target",
                ActionRequirement.TARGET_TILE: "target_tile",
                ActionRequirement.ITEM_FROM_INVENTORY: "item",
            },
        }

        mapping = action_param_mapping.get(
            self.current_action_option.action_class.__name__,
            action_param_mapping["DEFAULT"],
        )

        for req_type, value in self.fulfilled_requirements.items():
            # Validate requirement value before applying
            if req_type is ActionRequirement.TARGET_ACTOR and (
                not value.health.is_alive()
                or not controller.gw.game_map.visible[value.x, value.y]
            ):
                print("Target no longer valid")
                self.reset_state()
                return
            if (
                req_type is ActionRequirement.ITEM_FROM_INVENTORY
                and value not in actor.inventory
            ):
                print("Item no longer available")
                self.reset_state()
                return

            param = mapping.get(req_type)
            if isinstance(param, list):
                try:
                    all_params[param[0]] = value[0]
                    all_params[param[1]] = value[1]
                except Exception:
                    pass
            elif param:
                all_params[param] = value

        try:
            action_instance = self.current_action_option.action_class(
                controller, actor, **all_params
            )
        except TypeError as exc:  # pragma: no cover - runtime feedback only
            print(f"Action init error: {exc}")
            self.reset_state()
            return

        controller.queue_action(action_instance)
        if isinstance(action_instance, AttackAction):
            weapon = action_instance.weapon
            attack_mode = action_instance.attack_mode
            if weapon and attack_mode:
                controller.combat_intent_cache = CombatIntentCache(
                    weapon=weapon,
                    attack_mode=attack_mode,
                    target=action_instance.defender,
                )
        else:
            controller.combat_intent_cache = None

        self.reset_state()

    # ------------------------------------------------------------------
    # Requirement-specific option builders
    def _get_target_actor_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        context = self.action_discovery.context_builder.build_context(controller, actor)
        options: list[ActionOption] = []

        gm = controller.gw.game_map
        for target in context.nearby_actors:
            if target == actor or not isinstance(target, Character):
                continue
            if not target.health.is_alive():
                continue
            if not gm.visible[target.x, target.y]:
                continue

            distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)
            desc = "adjacent" if distance == 1 else "close" if distance <= 3 else "far"

            options.append(
                ActionOption(
                    id=f"select-target-{target.name}",
                    name=f"{target.name} ({desc})",
                    description=f"Select {target.name} as target",
                    category=ActionCategory.COMBAT,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                    execute=lambda t=target: self._fulfill_requirement(
                        ActionRequirement.TARGET_ACTOR, t
                    ),
                )
            )

        return options

    def _get_target_tile_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        options: list[ActionOption] = []

        gm = controller.gw.game_map
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            tx, ty = actor.x + dx, actor.y + dy
            if 0 <= tx < gm.width and 0 <= ty < gm.height and gm.visible[tx, ty]:
                options.append(
                    ActionOption(
                        id=f"select-tile-{tx}-{ty}",
                        name=f"Tile ({tx}, {ty})",
                        description=f"Select tile at ({tx}, {ty})",
                        category=ActionCategory.ENVIRONMENT,
                        action_class=cast(type[GameAction], type(None)),
                        requirements=[],
                        execute=lambda x=tx, y=ty: self._fulfill_requirement(
                            ActionRequirement.TARGET_TILE, (x, y)
                        ),
                    )
                )

        return options

    def _get_inventory_item_options(
        self, controller: Controller, actor: Character
    ) -> list[ActionOption]:
        options: list[ActionOption] = []
        for item in actor.inventory:
            if not hasattr(item, "name"):
                continue
            options.append(
                ActionOption(
                    id=f"select-item-{item.name}",
                    name=f"{item.name}",
                    description=f"Select {item.name}",
                    category=ActionCategory.ITEMS,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                    execute=lambda i=item: self._fulfill_requirement(
                        ActionRequirement.ITEM_FROM_INVENTORY, i
                    ),
                )
            )
        return options

    # ------------------------------------------------------------------
    # State management helpers
    def _fulfill_requirement(self, requirement: ActionRequirement, value: Any) -> bool:
        self.fulfilled_requirements[requirement] = value
        return False

    def _go_back(self) -> bool:
        if self.fulfilled_requirements:
            last_requirement = list(self.fulfilled_requirements.keys())[-1]
            del self.fulfilled_requirements[last_requirement]
        elif self.current_action_option is not None:
            self.current_action_option = None
            self.fulfilled_requirements = {}
        elif self.ui_state in ["items", "environment", "social", "combat"]:
            self.ui_state = "main"
            self.selected_target = None
            self.selected_weapon = None
            self.selected_attack_mode = None
        elif self.ui_state == "select_target":
            self.ui_state = "combat"
            self.selected_target = None
        elif self.ui_state == "select_weapon":
            self.ui_state = "combat"
            self.selected_weapon = None
            self.selected_attack_mode = None
        elif self.ui_state == "weapons_for_target":
            self.ui_state = "select_target"
            self.selected_weapon = None
            self.selected_attack_mode = None
        elif self.ui_state == "targets_for_weapon":
            self.ui_state = "select_weapon"
            self.selected_target = None
        elif self.ui_state != "main":
            self.ui_state = "main"
        else:
            self.reset_state()
        return False

    def _set_ui_state(self, new_state: str) -> bool:
        self.ui_state = new_state
        return False
