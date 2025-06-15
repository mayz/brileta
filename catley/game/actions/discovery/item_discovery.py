from __future__ import annotations

from typing import TYPE_CHECKING, cast

from catley.game.actions.base import GameAction
from catley.game.actions.combat import ReloadAction
from catley.game.actions.recovery import (
    ComfortableSleepAction,
    RestAction,
    SleepAction,
    UseConsumableAction,
    is_safe_location,
)
from catley.game.actors import Character
from catley.game.items.item_core import Item

from .action_context import ActionContext
from .action_factory import ActionFactory
from .action_formatters import ActionFormatter
from .types import ActionCategory, ActionOption

if TYPE_CHECKING:
    from catley.controller import Controller


class ItemActionDiscovery:
    """Discover item-related actions."""

    def __init__(self, factory: ActionFactory, formatter: ActionFormatter) -> None:
        self.factory = factory
        self.formatter = formatter

    # ------------------------------------------------------------------
    # Public API
    def discover_item_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        return self._get_inventory_options(controller, actor, context)

    # ------------------------------------------------------------------
    # Discovery helpers
    def _get_all_item_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        options: list[ActionOption] = []

        equipped_weapons = [w for w in actor.inventory.attack_slots if w is not None]
        options.extend(
            ActionOption(
                id=f"reload-{weapon.name}",
                name=f"Reload {weapon.name}",
                description=(
                    f"Reload {weapon.name} with {weapon.ranged_attack.ammo_type} ammo"
                ),
                category=ActionCategory.ITEMS,
                action_class=ReloadAction,
                requirements=[],
                static_params={"weapon": weapon},
                execute=lambda w=weapon: self.factory.create_reload_action(
                    controller, actor, w
                ),
            )
            for weapon in equipped_weapons
            if (
                weapon.ranged_attack
                and weapon.ranged_attack.current_ammo < weapon.ranged_attack.max_ammo
            )
        )

        options.extend(
            ActionOption(
                id=f"use-{item.name}",
                name=f"Use {item.name}",
                description=f"Consume {item.name}",
                category=ActionCategory.ITEMS,
                action_class=UseConsumableAction,
                requirements=[],
                static_params={"item": item},
                execute=lambda i=item: self.factory.create_use_consumable_action(
                    controller, actor, i
                ),
            )
            for item in actor.inventory
            if isinstance(item, Item) and item.consumable_effect
        )

        return options

    def _get_inventory_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        options = self._get_all_item_actions(controller, actor, context)

        # Pickup actions
        if context.items_on_ground:
            options.append(
                ActionOption(
                    id="pickup",
                    name=f"Pick up items ({len(context.items_on_ground)})",
                    description="Pickup items from the ground",
                    category=ActionCategory.ITEMS,
                    action_class=cast(type[GameAction], type(None)),
                    requirements=[],
                    static_params={},
                    hotkey="g",
                    execute=lambda: self._open_pickup_menu(controller),
                )
            )

        # Equipment switching - only show when NOT in combat
        if not context.in_combat:
            for i, item in enumerate(actor.inventory.attack_slots):
                if i != actor.inventory.active_weapon_slot and item:
                    options.append(
                        ActionOption(
                            id=f"switch-{item.name}",
                            name=f"Switch to {item.name}",
                            description=f"Equip {item.name} as active weapon",
                            category=ActionCategory.ITEMS,
                            action_class=cast(type[GameAction], type(None)),
                            requirements=[],
                            static_params={},
                            hotkey=str(i + 1),
                            execute=lambda slot=i: self._switch_weapon(actor, slot),
                        )
                    )

        return options

    def _get_recovery_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        options: list[ActionOption] = []
        safe, _ = is_safe_location(actor)

        if actor.health.ap < actor.health.max_ap and safe:
            options.append(
                ActionOption(
                    id="rest",
                    name="Rest",
                    description="Recover armor points",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=RestAction,
                    requirements=[],
                    static_params={},
                    execute=lambda: RestAction(controller, actor),
                )
            )

        needs_sleep = (
            actor.health.hp < actor.health.max_hp
            or actor.modifiers.get_exhaustion_count() > 0
        )

        if needs_sleep and safe:
            options.append(
                ActionOption(
                    id="sleep",
                    name="Sleep",
                    description="Sleep to restore HP and ease exhaustion",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=SleepAction,
                    requirements=[],
                    static_params={},
                    execute=lambda: SleepAction(controller, actor),
                )
            )

        if actor.modifiers.get_exhaustion_count() > 0 and safe:
            options.append(
                ActionOption(
                    id="comfort_sleep",
                    name="Comfortable Sleep",
                    description="Remove all exhaustion and restore HP",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=ComfortableSleepAction,
                    requirements=[],
                    static_params={},
                    execute=lambda: ComfortableSleepAction(controller, actor),
                )
            )

        return options

    # ------------------------------------------------------------------
    # Internal utilities
    def _open_pickup_menu(self, controller: Controller) -> None:
        return None

    def _switch_weapon(self, actor: Character, slot: int) -> None:
        actor.inventory.switch_to_weapon_slot(slot)
