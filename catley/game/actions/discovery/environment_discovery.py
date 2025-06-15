from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actors import Character

from .action_context import ActionContext
from .action_factory import ActionFactory
from .action_formatters import ActionFormatter
from .core_discovery import ActionCategory, ActionOption

if TYPE_CHECKING:
    from catley.controller import Controller


class EnvironmentActionDiscovery:
    """Discover environment interaction actions."""

    def __init__(self, factory: ActionFactory, formatter: ActionFormatter) -> None:
        self.factory = factory
        self.formatter = formatter

    # ------------------------------------------------------------------
    # Public API
    def discover_environment_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        options: list[ActionOption] = []
        options.extend(self._get_environment_options(controller, actor, context))
        options.extend(self._get_movement_options(controller, actor, context))
        options.extend(self._get_recovery_actions(controller, actor, context))
        return options

    # ------------------------------------------------------------------
    # Discovery helpers
    def _get_all_environment_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        return []

    def _get_environment_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
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
                        id="open-door",
                        name="Open Door",
                        description="Open the door",
                        category=ActionCategory.ENVIRONMENT,
                        execute=lambda x=tx, y=ty: OpenDoorAction(
                            controller, actor, x, y
                        ),
                    )
                )
            elif tile_id == tile_types.TILE_TYPE_ID_DOOR_OPEN:  # type: ignore[attr-defined]
                options.append(
                    ActionOption(
                        id="close-door",
                        name="Close Door",
                        description="Close the door",
                        category=ActionCategory.ENVIRONMENT,
                        execute=lambda x=tx, y=ty: CloseDoorAction(
                            controller, actor, x, y
                        ),
                    )
                )

        return options

    def _get_movement_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        # TODO: Add movement-related actions
        return []

    def _get_recovery_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        from catley.game.actions.recovery import (
            ComfortableSleepAction,
            RestAction,
            SleepAction,
            is_safe_location,
        )

        options: list[ActionOption] = []
        safe, _ = is_safe_location(actor)

        if actor.health.ap < actor.health.max_ap and safe:
            options.append(
                ActionOption(
                    id="rest",
                    name="Rest",
                    description="Recover armor points",
                    category=ActionCategory.ENVIRONMENT,
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
                    execute=lambda: ComfortableSleepAction(controller, actor),
                )
            )

        return options
