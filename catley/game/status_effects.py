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
        Remaining number of **turns** this effect should stay active.  This
        value is decremented in :meth:`Actor.update_turn` whenever it is greater
        than zero.  A value of ``0`` means the effect will be removed on the
        next call to ``update_turn`` (useful for "next action" effects) while
        ``-1`` represents an indefinite duration that must be removed manually.
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

    The effect is constructed with ``duration=0`` so it is removed as soon as
    :meth:`Actor.update_turn` runs, ensuring the penalty applies only to the
    actor's very next action.
    """

    def __init__(self) -> None:
        # ``duration=0`` -> expire on next ``update_turn`` call.
        super().__init__(
            name="Off Balance",
            duration=0,
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

    As with :class:`OffBalanceEffect`, ``duration=0`` ensures the bonus is
    consumed and the effect removed the next time the actor's turn updates.
    """

    def __init__(self) -> None:
        # Duration of ``0`` means this effect lasts only until the next turn.
        super().__init__(
            name="Focused",
            duration=0,
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

    # Like the other single-turn effects, ``duration=0`` ensures it self-removes
    # right after blocking one action.

    def __init__(self) -> None:
        # ``duration=0`` -> expire after one call to ``update_turn``.
        super().__init__(
            name="Tripped",
            duration=0,
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
