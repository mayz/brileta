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
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from catley.game import range_system
from catley.game.actions.base import GameAction
from catley.game.actions.combat import AttackAction, ReloadAction
from catley.game.actors import Character
from catley.util import dice

if TYPE_CHECKING:
    from catley.controller import Controller


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

    @property
    def display_text(self) -> str:
        """Formatted text for menus."""
        text = self.name
        if self.success_probability is not None:
            percentage = int(self.success_probability * 100)
            text += f" ({percentage}%)"
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

        # Find nearby actors (within reasonable interaction range)
        nearby_actors = []
        for other_actor in controller.gw.actors:
            if other_actor == actor:
                continue
            distance = range_system.calculate_distance(
                actor.x, actor.y, other_actor.x, other_actor.y
            )
            if distance <= 15:  # Reasonable interaction range
                nearby_actors.append(other_actor)

        # Get items on ground
        items_on_ground = controller.gw.get_pickable_items_at_location(actor.x, actor.y)

        # Determine if in combat (any hostile nearby)
        in_combat = any(
            hasattr(other, "ai")
            and hasattr(other.ai, "disposition")
            and other.ai.disposition.name == "HOSTILE"
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

    def _get_combat_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        """Get all combat-related action options."""
        options = []
        weapon = actor.inventory.get_active_weapon()

        if not weapon:
            return options

        # Get all nearby targetable actors
        for target in context.nearby_actors:
            if target == actor:
                continue

            distance = range_system.calculate_distance(
                actor.x, actor.y, target.x, target.y
            )

            # Melee attacks
            if weapon.melee_attack and distance == 1:
                # Calculate success probability
                prob = dice.calculate_check_roll_success_probability(
                    actor.stats.strength,
                    target.stats.agility + 10,  # Opposed roll target
                )

                options.append(
                    ActionOption(
                        name=f"Melee {target.name} with {weapon.name}",
                        description=f"Close combat attack using {weapon.name}",
                        category=ActionCategory.COMBAT,
                        hotkey="m" if len(options) == 0 else None,
                        success_probability=prob,
                        execute=functools.partial(
                            self._create_melee_attack, controller, actor, target, weapon
                        ),
                    )
                )

            # Ranged attacks
            if weapon.ranged_attack and distance > 1:
                range_cat = range_system.get_range_category(distance, weapon)
                range_mods = range_system.get_range_modifier(weapon, range_cat)

                if range_mods is not None:  # In range
                    # Calculate probability with range modifiers
                    prob = dice.calculate_check_roll_success_probability(
                        actor.stats.observation,
                        target.stats.agility + 10,
                        has_advantage=range_mods.get("has_advantage", False),
                        has_disadvantage=range_mods.get("has_disadvantage", False),
                    )

                    # FIXME: Unclear that we need to show this most of the time.
                    #        Leaving it commented out for now.
                    # ammo_cost = f"Uses 1 {weapon.ranged_attack.ammo_type} ammo"
                    ammo_cost = ""
                    options.append(
                        ActionOption(
                            name=f"Shoot {target.name} with {weapon.name}",
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

        # Weapon-specific special actions
        if (
            weapon.ranged_attack
            and weapon.ranged_attack.current_ammo < weapon.ranged_attack.max_ammo
        ):
            options.append(
                ActionOption(
                    name=f"Reload {weapon.name}",
                    description=(
                        f"Reload {weapon.name} with "
                        f"{weapon.ranged_attack.ammo_type} ammo"
                    ),
                    category=ActionCategory.COMBAT,
                    hotkey="R",
                    execute=functools.partial(
                        self._create_reload_action, controller, actor, weapon
                    ),
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
        options = []
        weapon = actor.inventory.get_active_weapon()

        if not weapon:
            return options

        distance = range_system.calculate_distance(actor.x, actor.y, target.x, target.y)

        # Melee attacks
        if weapon.melee_attack and distance == 1:
            prob = dice.calculate_check_roll_success_probability(
                actor.stats.strength, target.stats.agility + 10
            )

            options.append(
                ActionOption(
                    name=f"Melee attack with {weapon.name}",
                    description=f"Close combat attack using {weapon.name}",
                    category=ActionCategory.COMBAT,
                    success_probability=prob,
                    execute=functools.partial(
                        self._create_melee_attack, controller, actor, target, weapon
                    ),
                )
            )

        # Ranged attacks
        if weapon.ranged_attack and distance > 1:
            range_cat = range_system.get_range_category(distance, weapon)
            range_mods = range_system.get_range_modifier(weapon, range_cat)

            if range_mods is not None:
                prob = dice.calculate_check_roll_success_probability(
                    actor.stats.observation,
                    target.stats.agility + 10,
                    has_advantage=range_mods.get("has_advantage", False),
                    has_disadvantage=range_mods.get("has_disadvantage", False),
                )

                options.append(
                    ActionOption(
                        name=f"Ranged attack with {weapon.name}",
                        description=f"Ranged attack at {range_cat} range",
                        category=ActionCategory.COMBAT,
                        success_probability=prob,
                        cost_description=(
                            f"Uses 1 {weapon.ranged_attack.ammo_type} ammo"
                        ),
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

        # Equipment switching
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

        from catley.game.actions.environment import CloseDoorAction, OpenDoorAction
        from catley.world import tile_types

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
        return AttackAction(controller, actor, target, weapon)

    def _create_ranged_attack(self, controller, actor, target, weapon) -> AttackAction:
        return AttackAction(controller, actor, target, weapon)

    def _create_reload_action(self, controller, actor, weapon) -> ReloadAction:
        return ReloadAction(controller, actor, weapon)

    def _open_pickup_menu(self, controller):
        # This doesn't return a GameAction, but triggers UI
        # The ActionBrowserMode will need to handle this specially
        return None

    def _switch_weapon(self, controller, actor, slot) -> None:
        # This is an immediate state change, not a GameAction
        actor.inventory.switch_to_weapon_slot(slot)
