from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .actors import Actor


@dataclass
class StatusEffect(abc.ABC):
    """Base class for temporary tactical modifiers that affect actors.

    Unlike :class:`~catley.game.conditions.Condition` instances, status effects
    are short lived and do not occupy inventory slots. They model fleeting
    battlefield states such as being momentarily off balance or benefiting from
    a rush of adrenaline. Each effect automatically counts down and removes
    itself when its ``duration`` expires.

    Attributes
    ----------
    name:
        Human readable name used when displaying the effect.
    duration:
        Remaining number of **rounds** this effect should stay active. This
        value is decremented in :meth:`Actor.update_turn` at the start of each
        round when it is greater than zero. Common values:

        * ``1``: Effect lasts until the actor's next turn, then is removed
        before they act again (useful for "next action" effects like
        Off Balance or Focused)
        * ``3``: Effect lasts for 3 rounds (e.g., temporary stat boosts)
        * ``-1``: Indefinite duration that must be removed manually
        (e.g., Encumbered status that depends on inventory weight)
    description:
        Short explanation shown in the UI.
    can_stack:
        Whether multiple instances of this effect may exist on the same actor.
        If ``False`` (the default), attempts to apply another instance will be
        ignored.
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
    """Actor has disadvantage on exactly one action.

    ``duration=1`` means the effect lasts until the actor's next turn begins,
    applies to that action, then is removed when turns advance again.
    """

    def __init__(self) -> None:
        # ``duration=1`` causes the effect to persist until the next call to
        # :meth:`Actor.update_turn`, ensuring it affects the very next action
        # taken.
        super().__init__(
            name="Off Balance",
            duration=1,
            description="Disadvantage on next action",
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
    """Grants advantage on exactly one action.

    Mirroring :class:`OffBalanceEffect`, ``duration=1`` causes the bonus to
    persist only until the actor's next turn begins.
    """

    def __init__(self) -> None:
        # ``duration=1`` keeps the effect active until the next call to
        # :meth:`Actor.update_turn` so that it applies once and then expires.
        super().__init__(
            name="Focused",
            duration=1,
            description="Advantage on next action",
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
    """Forces the actor to skip their next action opportunity."""

    # ``duration=1`` means the effect lasts until the actor's next turn begins
    # and prevents that action before expiring.
    def __init__(self) -> None:
        super().__init__(
            name="Tripped",
            duration=1,
            description="Skip next action",
        )

    def apply_on_start(self, actor: Actor) -> None:
        pass

    def apply_turn_effect(self, actor: Actor) -> None:
        pass

    def remove_effect(self, actor: Actor) -> None:
        pass

    def apply_to_resolution(self, resolution_args: dict[str, bool]) -> dict[str, bool]:
        """Prevent action entirely by marking it as blocked."""
        resolution_args["action_prevented"] = True
        return resolution_args


class StrengthBoostEffect(StatusEffect):
    """Temporarily raise strength by two points.

    The ``duration`` argument specifies how many turns the boost should last
    and is forwarded to the base class. When the counter reaches zero the
    additional strength is removed in :meth:`remove_effect`.
    """

    def __init__(self, duration: int = 3) -> None:
        # ``duration`` controls how many turns the bonus is active.
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


class EncumberedEffect(StatusEffect):
    """Actor has disadvantage on movement and defense due to carrying too much.

    This status persists until the actor's carried weight drops below the
    encumbrance threshold. ``duration=-1`` marks it as indefinite so the base
    class will never remove it automatically.
    """

    def __init__(self) -> None:
        # ``duration=-1`` indicates the effect lasts until explicitly removed.
        super().__init__(
            name="Encumbered",
            duration=-1,
            description="Disadvantage on movement and defense from carrying too much",
            can_stack=False,
        )

    def apply_on_start(self, actor: Actor) -> None:
        pass

    def apply_turn_effect(self, actor: Actor) -> None:
        pass

    def remove_effect(self, actor: Actor) -> None:
        pass

    def apply_to_resolution(self, resolution_args: dict[str, bool]) -> dict[str, bool]:
        """Apply disadvantage to movement and defense actions."""
        stat_name = resolution_args.get("stat_name")
        if stat_name == "agility":
            resolution_args["has_disadvantage"] = True
        return resolution_args
