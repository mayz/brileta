"""Shared types for the action discovery system."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catley.game.actions.base import GameAction
    from catley.game.actors import Character
    from catley.game.items.item_core import Item


@dataclass
class CombatIntentCache:
    """Cached combat parameters from the most recent attack."""

    weapon: Item
    attack_mode: str
    target: Character | None = None


class ActionCategory(Enum):
    COMBAT = auto()
    MOVEMENT = auto()
    ITEMS = auto()
    ENVIRONMENT = auto()
    SOCIAL = auto()


@dataclass
class ActionOption:
    """Represents an action choice the player can select from a menu."""

    id: str
    name: str
    description: str
    category: ActionCategory
    hotkey: str | None = None
    display_text: str | None = None
    success_probability: float | None = None
    cost_description: str | None = None
    execute: Callable[[], GameAction | bool | None] | None = None

    def __post_init__(self) -> None:
        if self.display_text is None:
            self.display_text = self.name

    @property
    def menu_text(self) -> str:
        from .action_formatters import ActionFormatter

        return ActionFormatter.format_menu_text(self)
