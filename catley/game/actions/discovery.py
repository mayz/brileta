"""
Action Discovery System.

A unified interface for finding and presenting available actions.

This system bridges the gap between having rich game mechanics and players
being able to discover them.

There are three interaction paradigms for action discovery:
- Quick targeting (hotkeys)
- Context-aware action browser (see catley.modes.action_browser)
- Actor-specific menus
"""

from __future__ import annotations

import functools
import string
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from catley import colors
from catley.game import ranges
from catley.game.actions.base import GameAction
from catley.game.actions.combat import AttackAction, ReloadAction
from catley.game.actions.recovery import (
    ComfortableSleepAction,
    RestAction,
    SleepAction,
    UseConsumableAction,
    is_safe_location,
)
from catley.game.actors import Character
from catley.game.enums import Disposition
from catley.game.items.item_core import Item

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.items.capabilities import MeleeAttack, RangedAttack


class ActionCategory(Enum):
    COMBAT = auto()
    MOVEMENT = auto()
    ITEMS = auto()
    ENVIRONMENT = auto()
    SOCIAL = auto()


@dataclass
class ActionOption:
    """Represents an action choice the player can select from a menu."""

    name: str  # Display name like "Pistol-whip with Pistol"
    description: str  # Detailed description
    category: ActionCategory
    hotkey: str | None = None  # Optional keyboard shortcut
    success_probability: float | None = None  # 0.0-1.0 if applicable
    cost_description: str | None = None  # "Uses 1 ammo", "Takes 2 turns"
    execute: Callable[[], GameAction | None] | None = None  # How to perform the action
    enabled: bool = True
    color: colors.Color | None = None

    @property
    def display_text(self) -> str:
        """Formatted text for menus."""
        text = self.name
        if self.success_probability is not None:
            descriptor, _color = ActionDiscovery.get_probability_descriptor(
                self.success_probability
            )
            text += f" ({descriptor})"
        if self.cost_description:
            text += f" - {self.cost_description}"
        return text


@dataclass
class ActionContext:
    """Context information for action discovery."""

    # Location context
    tile_x: int
    tile_y: int

    # What's nearby
    nearby_actors: list[Character]
    items_on_ground: list  # Items at current location

    # Player state
    in_combat: bool = False
    selected_actor: Character | None = None

    # UI state
    interaction_mode: str = "normal"  # "normal", "targeting", "inventory"


class ActionDiscovery:
    """Main system for discovering available actions."""

    def __init__(self) -> None:
        # Multi-screen state for the action browser workflow
        self.ui_state: str = "main"
        self.selected_weapon: Item | None = None
        self.selected_attack_mode: str | None = None
        self.selected_target: Character | None = None

    @staticmethod
    def get_probability_descriptor(probability: float) -> tuple[str, str]:
        """Convert probability to qualitative descriptor and color."""
        from catley import config

        for max_prob, descriptor, color in config.PROBABILITY_DESCRIPTORS:
            if probability <= max_prob:
                return (descriptor, color)

        return ("Unknown", "white")

    def get_available_options(
        self, controller: Controller, actor: Character, sort_by_relevance: bool = True
    ) -> list[ActionOption]:
        """Get all available action options for the actor in current context."""

        # Build context
        context = self._build_context(controller, actor)

        # Collect options from all game systems
        all_options = []
        all_options.extend(self._get_combat_options(controller, actor, context))
        all_options.extend(self._get_inventory_options(controller, actor, context))
        all_options.extend(self._get_movement_options(controller, actor, context))
        all_options.extend(self._get_recovery_options(controller, actor, context))
        all_options.extend(self._get_environment_options(controller, actor, context))
        all_options.extend(self._get_social_options(controller, actor, context))

        if sort_by_relevance:
            all_options = self._sort_by_relevance(all_options, context)

        return all_options

    def get_options_for_category(
        self, controller: Controller, actor: Character, category: ActionCategory
    ) -> list[ActionOption]:
        """Get action options filtered by category."""
        all_options = self.get_available_options(controller, actor)
        return [option for option in all_options if option.category == category]

    def get_options_for_target(
        self, controller: Controller, actor: Character, target: Character
    ) -> list[ActionOption]:
        """Get all action options that can be performed on a specific target."""
        # Build context but focus on the specific target
        context = self._build_context(controller, actor)

        # Only get combat and social options that involve this target
        options = []
        options.extend(
            self._get_combat_options_for_target(controller, actor, target, context)
        )
        options.extend(
            self._get_social_options_for_target(controller, actor, target, context)
        )

        return options

    def _build_context(self, controller: Controller, actor: Character) -> ActionContext:
        """Build action context from current game state."""

        gm = controller.gw.game_map
        # Find nearby actors using the spatial index for efficiency.
        potential_actors = controller.gw.actor_spatial_index.get_in_radius(
            actor.x, actor.y, radius=15
        )

        nearby_actors: list[Character] = []
        for other_actor in potential_actors:
            if other_actor == actor:
                continue
            if not isinstance(other_actor, Character):
                continue
            if not other_actor.health.is_alive():
                continue

            # Radius filtering is done already; still verify visibility and
            # line of sight.
            if (
                0 <= other_actor.x < gm.width
                and 0 <= other_actor.y < gm.height
                and gm.visible[other_actor.x, other_actor.y]
                and ranges.has_line_of_sight(
                    gm, actor.x, actor.y, other_actor.x, other_actor.y
                )
            ):
                nearby_actors.append(other_actor)

        # Get items on ground
        items_on_ground = controller.gw.get_pickable_items_at_location(actor.x, actor.y)

        # Determine if in combat (any hostile nearby)
        in_combat = any(
            getattr(getattr(other, "ai", None), "disposition", None)
            == Disposition.HOSTILE
            for other in nearby_actors
        )

        selected_actor = controller.gw.selected_actor
        selected_actor = (
            selected_actor if isinstance(selected_actor, Character) else None
        )
        return ActionContext(
            tile_x=actor.x,
            tile_y=actor.y,
            nearby_actors=nearby_actors,
            items_on_ground=items_on_ground,
            in_combat=in_combat,
            selected_actor=selected_actor,
        )

    def _calculate_combat_probability(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
        stat_name: str,
        range_modifiers: dict[str, bool] | None = None,
    ) -> float:
        """Calculate combat success probability with all modifiers combined."""

        resolution_modifiers = actor.modifiers.get_resolution_modifiers(stat_name)
        has_advantage = (
            range_modifiers and range_modifiers.get("has_advantage", False)
        ) or resolution_modifiers.get("has_advantage", False)
        has_disadvantage = (
            range_modifiers and range_modifiers.get("has_disadvantage", False)
        ) or resolution_modifiers.get("has_disadvantage", False)

        resolver = controller.create_resolver(
            ability_score=getattr(actor.stats, stat_name),
            roll_to_exceed=target.stats.agility + 10,
            has_advantage=has_advantage,
            has_disadvantage=has_disadvantage,
        )
        return resolver.calculate_success_probability()

    def _get_attack_display_name(
        self, weapon: Item, attack_mode: str, target_name: str
    ) -> str:
        """Generate attack display name using the weapon's verb."""
        if attack_mode == "melee" and weapon.melee_attack:
            melee = cast("MeleeAttack", weapon.melee_attack)
            verb = melee._spec.verb
            return f"{verb.title()} {target_name} with {weapon.name}"
        if attack_mode == "ranged" and weapon.ranged_attack:
            ranged = cast("RangedAttack", weapon.ranged_attack)
            verb = ranged._spec.verb
            return f"{verb.title()} {target_name} with {weapon.name}"
        return f"Attack {target_name} with {weapon.name}"

    def _get_combat_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Get combat-related options based on the current UI state."""

        if self.ui_state == "main":
            return self.get_combat_approach_options(controller, actor, context)
        if self.ui_state == "target_select":
            return self.get_target_selection_options(controller, actor, context)
        if self.ui_state == "weapon_select":
            return self.get_weapon_selection_options(controller, actor, context)
        if self.ui_state == "weapon_for_target":
            return self._get_weapons_for_selected_target(controller, actor, context)
        if self.ui_state == "target_for_weapon":
            return self._get_targets_for_selected_weapon(controller, actor, context)

        options: list[ActionOption] = []
        equipped_weapons = [w for w in actor.inventory.attack_slots if w is not None]

        if not equipped_weapons:
            from catley.game.items.item_types import FISTS_TYPE

            equipped_weapons = [FISTS_TYPE.create()]

        # Get all nearby targetable actors
        for target in context.nearby_actors:
            if target == actor:
                continue
            if not isinstance(target, Character):
                continue
            if not target.stats or not target.health or not target.health.is_alive():
                continue
            gm = controller.gw.game_map
            if not gm.visible[target.x, target.y]:
                continue
            if not ranges.has_line_of_sight(gm, actor.x, actor.y, target.x, target.y):
                continue

            distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)

            for weapon in equipped_weapons:
                # Melee attacks
                if weapon.melee_attack and distance == 1:
                    prob = self._calculate_combat_probability(
                        controller, actor, target, "strength"
                    )

                    options.append(
                        ActionOption(
                            name=self._get_attack_display_name(
                                weapon, "melee", target.name
                            ),
                            description=f"Close combat attack using {weapon.name}",
                            category=ActionCategory.COMBAT,
                            hotkey="m" if len(options) == 0 else None,
                            success_probability=prob,
                            execute=functools.partial(
                                self._create_melee_attack,
                                controller,
                                actor,
                                target,
                                weapon,
                            ),
                        )
                    )

                # Ranged attacks - show if weapon has ranged capability and ammo
                if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                    range_cat = ranges.get_range_category(distance, weapon)
                    range_mods = ranges.get_range_modifier(weapon, range_cat)
                    if range_mods is None:
                        options.append(
                            ActionOption(
                                name=self._get_attack_display_name(
                                    weapon, "ranged", target.name
                                )
                                + " (OUT OF RANGE)",
                                description=(
                                    f"Target is beyond {weapon.name}'s maximum range"
                                ),
                                category=ActionCategory.COMBAT,
                                enabled=False,
                                color=colors.GREY,
                                success_probability=0.0,
                            )
                        )
                        continue

                    prob = self._calculate_combat_probability(
                        controller,
                        actor,
                        target,
                        "observation",
                        range_mods,
                    )

                    ammo_warning = ""
                    current_ammo = weapon.ranged_attack.current_ammo
                    if current_ammo == 1:
                        ammo_warning = " (LAST SHOT!)"
                    elif current_ammo <= 3:
                        ammo_warning = " (Low ammo)"
                    ammo_cost = (
                        f"Uses 1 {weapon.ranged_attack.ammo_type} ammo{ammo_warning}"
                    )
                    options.append(
                        ActionOption(
                            name=self._get_attack_display_name(
                                weapon, "ranged", target.name
                            ),
                            description=f"Ranged attack at {range_cat} range",
                            category=ActionCategory.COMBAT,
                            hotkey="r"
                            if not any(opt.hotkey == "r" for opt in options)
                            else None,
                            success_probability=prob,
                            cost_description=ammo_cost,
                            execute=functools.partial(
                                self._create_ranged_attack,
                                controller,
                                actor,
                                target,
                                weapon,
                            ),
                        )
                    )

                # Area effect attacks
                if weapon.area_effect:
                    prob = self._calculate_combat_probability(
                        controller, actor, target, "observation"
                    )
                    options.append(
                        ActionOption(
                            name=f"Spray area around {target.name} with {weapon.name}",
                            description="Area attack affecting multiple targets",
                            category=ActionCategory.COMBAT,
                            success_probability=prob,
                            execute=functools.partial(
                                self._create_area_attack,
                                controller,
                                actor,
                                target.x,
                                target.y,
                                weapon,
                            ),
                        )
                    )

        # Weapon-specific special actions
        name_counts = {
            w.name: sum(1 for x in equipped_weapons if x.name == w.name)
            for w in equipped_weapons
        }
        options.extend(
            ActionOption(
                name=f"Reload {weapon.name}"
                + (f" (Slot {i + 1})" if name_counts.get(weapon.name, 0) > 1 else ""),
                description=(
                    f"Reload {weapon.name} with {weapon.ranged_attack.ammo_type} ammo"
                ),
                category=ActionCategory.COMBAT,
                hotkey="R" if not any(opt.hotkey == "R" for opt in options) else None,
                execute=functools.partial(
                    self._create_reload_action, controller, actor, weapon
                ),
            )
            for i, weapon in enumerate(equipped_weapons)
            if (
                weapon.ranged_attack
                and weapon.ranged_attack.current_ammo < weapon.ranged_attack.max_ammo
            )
        )

        return options

    def _get_combat_options_for_target(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
        context: ActionContext,
    ) -> list[ActionOption]:
        """Get combat options specifically for a given target."""
        options: list[ActionOption] = []
        equipped_weapons = [w for w in actor.inventory.attack_slots if w is not None]

        if not equipped_weapons:
            from catley.game.items.item_types import FISTS_TYPE

            equipped_weapons = [FISTS_TYPE.create()]

        gm = controller.gw.game_map
        if (
            not isinstance(target, Character)
            or not target.stats
            or not target.health
            or not target.health.is_alive()
            or not gm.visible[target.x, target.y]
            or not ranges.has_line_of_sight(gm, actor.x, actor.y, target.x, target.y)
        ):
            return options

        distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)

        for weapon in equipped_weapons:
            # Melee attacks
            if weapon.melee_attack and distance == 1:
                prob = self._calculate_combat_probability(
                    controller, actor, target, "strength"
                )

                options.append(
                    ActionOption(
                        name=self._get_attack_display_name(
                            weapon, "melee", target.name
                        ),
                        description=f"Close combat attack using {weapon.name}",
                        category=ActionCategory.COMBAT,
                        success_probability=prob,
                        execute=functools.partial(
                            self._create_melee_attack, controller, actor, target, weapon
                        ),
                    )
                )

            # Ranged attacks - show if weapon has ranged capability and ammo
            if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                range_cat = ranges.get_range_category(distance, weapon)
                range_mods = ranges.get_range_modifier(weapon, range_cat)
                if range_mods is None:
                    options.append(
                        ActionOption(
                            name=self._get_attack_display_name(
                                weapon, "ranged", target.name
                            )
                            + " (OUT OF RANGE)",
                            description=(
                                f"Target is beyond {weapon.name}'s maximum range"
                            ),
                            category=ActionCategory.COMBAT,
                            enabled=False,
                            color=colors.GREY,
                            success_probability=0.0,
                        )
                    )
                    continue

                prob = self._calculate_combat_probability(
                    controller,
                    actor,
                    target,
                    "observation",
                    range_mods,
                )

                ammo_warning = ""
                current_ammo = weapon.ranged_attack.current_ammo
                if current_ammo == 1:
                    ammo_warning = " (LAST SHOT!)"
                elif current_ammo <= 3:
                    ammo_warning = " (Low ammo)"
                cost_desc = (
                    f"Uses 1 {weapon.ranged_attack.ammo_type} ammo{ammo_warning}"
                )

                options.append(
                    ActionOption(
                        name=self._get_attack_display_name(
                            weapon, "ranged", target.name
                        ),
                        description=f"Ranged attack at {range_cat} range",
                        category=ActionCategory.COMBAT,
                        success_probability=prob,
                        cost_description=cost_desc,
                        execute=functools.partial(
                            self._create_ranged_attack,
                            controller,
                            actor,
                            target,
                            weapon,
                        ),
                    )
                )

        return options

    def _get_inventory_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Get all inventory and item-related action options."""
        options = []

        # Pickup actions
        if context.items_on_ground:
            options.append(
                ActionOption(
                    name=f"Pick up items ({len(context.items_on_ground)})",
                    description="Pickup items from the ground",
                    category=ActionCategory.ITEMS,
                    hotkey="g",
                    execute=functools.partial(self._open_pickup_menu, controller),
                )
            )

        # Equipment switching - only show when NOT in combat
        if not context.in_combat:
            for i, item in enumerate(actor.inventory.attack_slots):
                if i != actor.inventory.active_weapon_slot and item:
                    options.append(
                        ActionOption(
                            name=f"Switch to {item.name}",
                            description=f"Equip {item.name} as active weapon",
                            category=ActionCategory.ITEMS,
                            hotkey=str(i + 1),
                            execute=functools.partial(
                                self._switch_weapon, controller, actor, i
                            ),
                        )
                    )

        # Consumable items
        options.extend(
            ActionOption(
                name=f"Use {item.name}",
                description=f"Consume {item.name}",
                category=ActionCategory.ITEMS,
                execute=functools.partial(UseConsumableAction, controller, actor, item),
            )
            for item in actor.inventory
            if isinstance(item, Item) and item.consumable_effect
        )

        return options

    def _get_recovery_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Get rest and sleep related options."""

        options: list[ActionOption] = []
        safe, _ = is_safe_location(actor)

        if actor.health.ap < actor.health.max_ap and safe:
            options.append(
                ActionOption(
                    name="Rest",
                    description="Recover armor points",
                    category=ActionCategory.ENVIRONMENT,
                    execute=functools.partial(RestAction, controller, actor),
                )
            )

        needs_sleep = (
            actor.health.hp < actor.health.max_hp
            or actor.modifiers.get_exhaustion_count() > 0
        )

        if needs_sleep and safe:
            options.append(
                ActionOption(
                    name="Sleep",
                    description="Sleep to restore HP and ease exhaustion",
                    category=ActionCategory.ENVIRONMENT,
                    execute=functools.partial(SleepAction, controller, actor),
                )
            )

        if actor.modifiers.get_exhaustion_count() > 0 and safe:
            options.append(
                ActionOption(
                    name="Comfortable Sleep",
                    description="Remove all exhaustion and restore HP",
                    category=ActionCategory.ENVIRONMENT,
                    execute=functools.partial(
                        ComfortableSleepAction, controller, actor
                    ),
                )
            )

        return options

    def _get_movement_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Get all movement and positioning action options."""
        # TODO: Add movement options like:
        # - Move to specific location
        # - Take cover behind object
        # - Climb/jump actions
        # - Sneak movement

        return []

    def _get_environment_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Get all environment interaction action options."""

        from catley.environment import tile_types
        from catley.game.actions.environment import CloseDoorAction, OpenDoorAction

        options: list[ActionOption] = []

        gm = controller.gw.game_map

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            tx = actor.x + dx
            ty = actor.y + dy
            if not (0 <= tx < gm.width and 0 <= ty < gm.height):
                continue
            tile_id = gm.tiles[tx, ty]
            if tile_id == tile_types.TILE_TYPE_ID_DOOR_CLOSED:  # type: ignore[attr-defined]
                options.append(
                    ActionOption(
                        name="Open Door",
                        description="Open the door",
                        category=ActionCategory.ENVIRONMENT,
                        execute=functools.partial(
                            OpenDoorAction, controller, actor, tx, ty
                        ),
                    )
                )
            elif tile_id == tile_types.TILE_TYPE_ID_DOOR_OPEN:  # type: ignore[attr-defined]
                options.append(
                    ActionOption(
                        name="Close Door",
                        description="Close the door",
                        category=ActionCategory.ENVIRONMENT,
                        execute=functools.partial(
                            CloseDoorAction, controller, actor, tx, ty
                        ),
                    )
                )

        return options

    def _get_social_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Get all social interaction action options."""
        # TODO: Add social options like:
        # - Talk to NPCs
        # - Pickpocket
        # - Intimidate
        # - Trade

        return []

    def _get_social_options_for_target(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
        context: ActionContext,
    ) -> list[ActionOption]:
        """Get social options specifically for a given target."""
        # TODO: Add target-specific social options

        return []

    def _sort_by_relevance(
        self, options: list[ActionOption], context: ActionContext
    ) -> list[ActionOption]:
        """Sort action options by relevance to current situation."""

        def relevance_score(option: ActionOption) -> int:
            score = 0

            # Combat actions are more relevant when in combat
            if context.in_combat and option.category == ActionCategory.COMBAT:
                score += 100

            # Actions with higher success probability are more relevant
            success_prob = option.success_probability
            if success_prob is not None:
                score += int(success_prob * 50)

            # Actions with hotkeys are more accessible
            if option.hotkey:
                score += 20

            return score

        return sorted(options, key=relevance_score, reverse=True)

    # Action creation helpers

    def _create_melee_attack(self, controller, actor, target, weapon) -> AttackAction:
        """Helper to build a melee `AttackAction`."""
        return AttackAction(controller, actor, target, weapon, attack_mode="melee")

    def _create_ranged_attack(self, controller, actor, target, weapon) -> AttackAction:
        """Helper to build a ranged `AttackAction`."""
        return AttackAction(controller, actor, target, weapon, attack_mode="ranged")

    def _create_area_attack(self, controller, actor, target_x, target_y, weapon):
        """Helper to build an area effect attack."""
        from catley.game.actions.area_effects import AreaEffectAction

        return AreaEffectAction(
            controller.gw.game_map,
            controller.gw.actors,
            actor,
            target_x,
            target_y,
            weapon,
        )

    def _create_reload_action(self, controller, actor, weapon) -> ReloadAction:
        return ReloadAction(controller, actor, weapon)

    def _open_pickup_menu(self, controller):
        # This doesn't return a GameAction, but triggers UI
        # The ActionBrowserMode will need to handle this specially
        return

    def _switch_weapon(self, controller, actor, slot) -> None:
        # This is an immediate state change, not a GameAction
        actor.inventory.switch_to_weapon_slot(slot)

    # ------------------------------------------------------------------
    # Multi-screen navigation helpers

    def get_combat_approach_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Return high-level combat approach choices."""

        options: list[ActionOption] = []

        if context.nearby_actors:
            options.append(
                ActionOption(
                    name="Attack a target...",
                    description="Select target, then choose how to attack",
                    category=ActionCategory.COMBAT,
                    hotkey="t",
                    execute=functools.partial(
                        self._enter_target_selection_mode, controller
                    ),
                )
            )
            options.append(
                ActionOption(
                    name="Attack with a weapon...",
                    description="Select weapon/attack mode, then choose target",
                    category=ActionCategory.COMBAT,
                    hotkey="w",
                    execute=functools.partial(
                        self._enter_weapon_selection_mode, controller
                    ),
                )
            )

        return options

    def get_target_selection_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Options for selecting a target in the target-first workflow."""

        options: list[ActionOption] = [
            ActionOption(
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
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
            if distance == 1:
                desc = "adjacent"
            elif distance <= 3:
                desc = "close"
            else:
                desc = "far"

            options.append(
                ActionOption(
                    name=f"{target.name} ({desc})",
                    description=f"Attack {target.name}",
                    category=ActionCategory.COMBAT,
                    hotkey=letters[i] if i < len(letters) else None,
                    execute=functools.partial(
                        self._target_selected, controller, target
                    ),
                )
            )

        return options

    def get_weapon_selection_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Options for selecting weapon and mode in the weapon-first workflow."""

        options: list[ActionOption] = [
            ActionOption(
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
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
                melee = cast("MeleeAttack", weapon.melee_attack)
                verb = melee._spec.verb
                options.append(
                    ActionOption(
                        name=f"{verb.title()} with {weapon.name}",
                        description=f"Melee attack using {weapon.name}",
                        category=ActionCategory.COMBAT,
                        hotkey=letters[option_index]
                        if option_index < len(letters)
                        else None,
                        execute=functools.partial(
                            self._weapon_mode_selected, controller, weapon, "melee"
                        ),
                    )
                )
                option_index += 1

            if weapon.ranged_attack and weapon.ranged_attack.current_ammo > 0:
                ranged = cast("RangedAttack", weapon.ranged_attack)
                verb = ranged._spec.verb
                options.append(
                    ActionOption(
                        name=f"{verb.title()} with {weapon.name}",
                        description=f"Ranged attack using {weapon.name}",
                        category=ActionCategory.COMBAT,
                        hotkey=letters[option_index]
                        if option_index < len(letters)
                        else None,
                        execute=functools.partial(
                            self._weapon_mode_selected, controller, weapon, "ranged"
                        ),
                    )
                )
                option_index += 1

        return options

    def _enter_target_selection_mode(self, controller) -> None:
        self.ui_state = "target_select"

    def _enter_weapon_selection_mode(self, controller) -> None:
        self.ui_state = "weapon_select"

    def _target_selected(self, controller, target: Character) -> None:
        self.selected_target = target
        self.ui_state = "weapon_for_target"

    def _weapon_mode_selected(self, controller, weapon: Item, attack_mode: str) -> None:
        self.selected_weapon = weapon
        self.selected_attack_mode = attack_mode
        self.ui_state = "target_for_weapon"

    def _get_weapons_for_selected_target(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        if not self.selected_target:
            self.ui_state = "target_select"
            return []

        options = [
            ActionOption(
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
                hotkey="escape",
                execute=functools.partial(self._go_back, controller),
            )
        ]

        options.extend(
            self._get_combat_options_for_target(
                controller, actor, self.selected_target, context
            )
        )

        return options

    def _get_targets_for_selected_weapon(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        weapon = self.selected_weapon
        mode = self.selected_attack_mode
        if not weapon or not mode:
            self.ui_state = "weapon_select"
            return []

        options: list[ActionOption] = [
            ActionOption(
                name="\u2190 Back",
                description="Return to previous screen",
                category=ActionCategory.COMBAT,
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
                prob = self._calculate_combat_probability(
                    controller, actor, target, "strength"
                )
                options.append(
                    ActionOption(
                        name=self._get_attack_display_name(
                            weapon, "melee", target.name
                        ),
                        description=f"Close combat attack using {weapon.name}",
                        category=ActionCategory.COMBAT,
                        success_probability=prob,
                        execute=functools.partial(
                            self._create_melee_attack, controller, actor, target, weapon
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
                    options.append(
                        ActionOption(
                            name=self._get_attack_display_name(
                                weapon, "ranged", target.name
                            )
                            + " (OUT OF RANGE)",
                            description=(
                                f"Target is beyond {weapon.name}'s maximum range"
                            ),
                            category=ActionCategory.COMBAT,
                            enabled=False,
                            color=colors.GREY,
                            success_probability=0.0,
                        )
                    )
                    continue

                prob = self._calculate_combat_probability(
                    controller, actor, target, "observation", range_mods
                )

                ammo_warning = ""
                current_ammo = weapon.ranged_attack.current_ammo
                if current_ammo == 1:
                    ammo_warning = " (LAST SHOT!)"
                elif current_ammo <= 3:
                    ammo_warning = " (Low ammo)"
                cost_desc = (
                    f"Uses 1 {weapon.ranged_attack.ammo_type} ammo{ammo_warning}"
                )

                options.append(
                    ActionOption(
                        name=self._get_attack_display_name(
                            weapon, "ranged", target.name
                        ),
                        description=f"Ranged attack at {range_cat} range",
                        category=ActionCategory.COMBAT,
                        success_probability=prob,
                        cost_description=cost_desc,
                        execute=functools.partial(
                            self._create_ranged_attack,
                            controller,
                            actor,
                            target,
                            weapon,
                        ),
                    )
                )

        return options

    def _go_back(self, controller) -> None:
        if self.ui_state in ["target_select", "weapon_select"]:
            self.ui_state = "main"
        elif self.ui_state == "weapon_for_target":
            self.ui_state = "target_select"
            self.selected_target = None
        elif self.ui_state == "target_for_weapon":
            self.ui_state = "weapon_select"
            self.selected_weapon = None
            self.selected_attack_mode = None
        else:
            self.ui_state = "main"
            self.selected_target = None
            self.selected_weapon = None
            self.selected_attack_mode = None
