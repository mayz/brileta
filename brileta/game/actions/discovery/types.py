"""Shared types for the action discovery system."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from brileta.game.actions.base import GameIntent
    from brileta.game.actors import Character
    from brileta.game.items.item_core import Item


@dataclass
class CombatIntentCache:
    """Cached combat parameters from the most recent attack."""

    weapon: Item
    attack_mode: str
    target: Character | None = None


class ActionCategory(Enum):
    COMBAT = auto()
    STUNT = auto()  # Physical maneuvers like Push, Trip, Disarm
    MOVEMENT = auto()
    ITEMS = auto()
    ENVIRONMENT = auto()
    SOCIAL = auto()


class TargetType(Enum):
    """Classification of targets for default action lookup.

    Used by the right-click default action system to determine what action
    to perform when the player right-clicks on different target types.
    """

    NPC = auto()  # Any character (disposition is a continuum)
    CONTAINER = auto()  # Searchable containers (crates, bookcases, etc.)
    DOOR_CLOSED = auto()  # Closed doors
    DOOR_OPEN = auto()  # Open doors
    ITEM_PILE = auto()  # Items on the ground
    FLOOR = auto()  # Walkable tile (for pathfinding)


class ActionRequirement(Enum):
    """Represents what an action needs from the user to be executed."""

    TARGET_ACTOR = auto()
    TARGET_TILE = auto()
    ITEM_FROM_INVENTORY = auto()


@dataclass
class ActionOption:
    """Represents an action choice the player can select from a menu.

    The action_class field is optional because some actions (like gateway
    actions or pathfind-and-execute patterns) use the execute callback
    instead of a direct action class instantiation.
    """

    id: str
    name: str
    description: str
    category: ActionCategory
    action_class: type[GameIntent] | None
    requirements: list[ActionRequirement] = field(default_factory=list)
    static_params: dict[str, Any] = field(default_factory=dict)
    hotkey: str | None = None
    display_text: str | None = None
    success_probability: float | None = None
    cost_description: str | None = None
    # Temporary backwards compatibility - will be removed in Task 3
    execute: Callable[[], GameIntent | bool | None] | None = None

    def __post_init__(self) -> None:
        if self.display_text is None:
            self.display_text = self.name

    @property
    def menu_text(self) -> str:
        from .action_formatters import ActionFormatter

        return ActionFormatter.format_menu_text(self)
