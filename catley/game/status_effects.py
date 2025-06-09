from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .actors import Actor


@dataclass
class StatusEffect(abc.ABC):
    """Base class for temporary tactical modifiers that affect actors.

    Status effects are distinct from Condition:
    - Status effects: Immediate tactical modifiers, auto-expire, no inventory cost
    - Conditions: Persistent states with mechanical effects, consume inventory slots

    Status effects represent fleeting battlefield states like being off balance
    or temporarily boosted from adrenaline. They automatically expire without
    manual treatment and do not burden the actor's inventory.
    """

    name: str
    duration: int
    description: str = ""
    can_stack: bool = False

    @abc.abstractmethod
    def apply_on_start(self, actor: Actor) -> None:
        """Called when the effect is first applied."""

    @abc.abstractmethod
    def apply_turn_effect(self, actor: Actor) -> None:
        """Called each turn while the effect is active."""

    @abc.abstractmethod
    def remove_effect(self, actor: Actor) -> None:
        """Called when the effect is removed or expires."""

    def should_remove(self, actor: Actor) -> bool:
        """Return ``True`` if the effect should be removed this turn."""
        return self.duration == 0

    def apply_to_resolution(self, resolution_args: dict[str, bool]) -> dict[str, bool]:
        """Modify resolution arguments to apply this effect's influence."""
        return resolution_args


class OffBalanceEffect(StatusEffect):
    """Actor has disadvantage on their next action, then removes itself."""

    def __init__(self) -> None:
        super().__init__(
            name="Off Balance", duration=0, description="Disadvantage on next action"
        )

    def apply_on_start(self, actor: Actor) -> None:
        pass

    def apply_turn_effect(self, actor: Actor) -> None:
        pass

    def remove_effect(self, actor: Actor) -> None:
        pass

    def apply_to_resolution(self, resolution_args: dict[str, bool]) -> dict[str, bool]:
        """Apply disadvantage to resolution."""
        resolution_args["has_disadvantage"] = True
        return resolution_args


class FocusedEffect(StatusEffect):
    """Actor has advantage on their next action, then removes itself."""

    def __init__(self) -> None:
        super().__init__(
            name="Focused", duration=0, description="Advantage on next action"
        )

    def apply_on_start(self, actor: Actor) -> None:
        pass

    def apply_turn_effect(self, actor: Actor) -> None:
        pass

    def remove_effect(self, actor: Actor) -> None:
        pass

    def apply_to_resolution(self, resolution_args: dict[str, bool]) -> dict[str, bool]:
        """Apply advantage to resolution."""
        resolution_args["has_advantage"] = True
        return resolution_args


class TrippedEffect(StatusEffect):
    """Actor skips their next action opportunity, then removes itself."""

    def __init__(self) -> None:
        super().__init__(name="Tripped", duration=0, description="Skip next action")

    def apply_on_start(self, actor: Actor) -> None:
        pass

    def apply_turn_effect(self, actor: Actor) -> None:
        pass

    def remove_effect(self, actor: Actor) -> None:
        pass


class StrengthBoostEffect(StatusEffect):
    """Temporarily increase actor strength by 2 for a number of turns."""

    def __init__(self, duration: int = 3) -> None:
        super().__init__(
            name="Strength Boost",
            duration=duration,
            description="Strength increased by 2",
        )

    def apply_on_start(self, actor: Actor) -> None:
        stats = actor.stats
        if stats is not None:
            stats.strength += 2

    def apply_turn_effect(self, actor: Actor) -> None:
        pass

    def remove_effect(self, actor: Actor) -> None:
        stats = actor.stats
        if stats is not None:
            stats.strength -= 2
