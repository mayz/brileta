from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.environment import CloseDoorIntent, OpenDoorIntent
from catley.game.actions.recovery import (
    ComfortableSleepIntent,
    RestIntent,
    SleepIntent,
)
from catley.game.actors import Character

from .action_context import ActionContext
from .action_factory import ActionFactory
from .action_formatters import ActionFormatter
from .core_discovery import ActionCategory, ActionOption
from .types import ActionRequirement

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
    def _get_environment_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        from catley.environment import tile_types

        options: list[ActionOption] = []
        gm = controller.gw.game_map
        closed_doors: list[tuple[int, int]] = []
        open_doors: list[tuple[int, int]] = []

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            tx = actor.x + dx
            ty = actor.y + dy
            if not (0 <= tx < gm.width and 0 <= ty < gm.height):
                continue
            tile_id = gm.tiles[tx, ty]
            if tile_id == tile_types.TILE_TYPE_ID_DOOR_CLOSED:  # type: ignore[attr-defined]
                closed_doors.append((tx, ty))
            elif tile_id == tile_types.TILE_TYPE_ID_DOOR_OPEN:  # type: ignore[attr-defined]
                open_doors.append((tx, ty))

        # Handle closed doors
        if len(closed_doors) == 1:
            # Single door - create direct action
            door_x, door_y = closed_doors[0]
            options.append(
                ActionOption(
                    id="open-door-direct",
                    name="Open Door",
                    description="Open the adjacent door",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=OpenDoorIntent,
                    requirements=[],
                    static_params={"x": door_x, "y": door_y},
                )
            )
        elif len(closed_doors) > 1:
            # Multiple doors - require selection
            options.append(
                ActionOption(
                    id="open-door",
                    name="Open Door",
                    description="Choose a door to open",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=OpenDoorIntent,
                    requirements=[ActionRequirement.TARGET_TILE],
                    static_params={},
                )
            )

        # Handle open doors
        if len(open_doors) == 1:
            # Single door - create direct action
            door_x, door_y = open_doors[0]
            options.append(
                ActionOption(
                    id="close-door-direct",
                    name="Close Door",
                    description="Close the adjacent door",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=CloseDoorIntent,
                    requirements=[],
                    static_params={"x": door_x, "y": door_y},
                )
            )
        elif len(open_doors) > 1:
            # Multiple doors - require selection
            options.append(
                ActionOption(
                    id="close-door",
                    name="Close Door",
                    description="Choose a door to close",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=CloseDoorIntent,
                    requirements=[ActionRequirement.TARGET_TILE],
                    static_params={},
                )
            )

        return options

    def _get_movement_options(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        # Placeholder for future movement-related actions
        return []

    def _get_recovery_actions(
        self, controller: Controller, actor: Character, context: ActionContext
    ) -> list[ActionOption]:
        from catley.game.actions.recovery import is_safe_location

        options: list[ActionOption] = []
        safe, _ = is_safe_location(actor)

        if actor.health.ap < actor.health.max_ap and safe:
            options.append(
                ActionOption(
                    id="rest",
                    name="Rest",
                    description="Recover armor points",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=RestIntent,
                    requirements=[],
                    static_params={},
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
                    action_class=SleepIntent,
                    requirements=[],
                    static_params={},
                )
            )

        if actor.modifiers.get_exhaustion_count() > 0 and safe:
            options.append(
                ActionOption(
                    id="comfort_sleep",
                    name="Comfortable Sleep",
                    description="Remove all exhaustion and restore HP",
                    category=ActionCategory.ENVIRONMENT,
                    action_class=ComfortableSleepIntent,
                    requirements=[],
                    static_params={},
                )
            )

        return options
