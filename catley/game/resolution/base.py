from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catley.game.actors import Actor, Character
    from catley.game.items.item_core import Item


class ResolutionResult(abc.ABC):  # noqa: B024
    """
    Abstract base class for the result of any mechanical resolution.

    This class serves as a "marker interface." It has no methods or data
    of its own. Its purpose is to provide a common parent type for all
    specific result objects (e.g., D20ResolutionResult, TwoD6ResolutionResult).

    This allows systems like the CombatResolver to accept any kind of
    ResolutionResult and then use type checking (isinstance) to determine
    how to interpret its specific contents.
    """

    pass


class ResolutionSystem(abc.ABC):
    """
    Abstract base class for a system that resolves an action.

    This defines the common interface for any dice mechanic or other rule
    system used to determine the outcome of an attempted action. Concrete
    implementations (like a D20System or a TwoD6System) will implement
    the `resolve` method.
    """

    @abc.abstractmethod
    def resolve(
        self,
        actor: Character,
        target: Actor,
        weapon: Item | None = None,
        # Future-proofing: add a generic 'context' or 'modifiers' dict
        # situational_bonus: int = 0
    ) -> ResolutionResult:
        """
        Performs a check to determine the outcome of an action.

        Args:
            actor: The character performing the action.
            target: The actor or object being targeted by the action.
            weapon: The item being used for the action, if any. This can
                    provide properties that modify the resolution (e.g., a
                    'Scoped' property granting advantage).

        Returns:
            A ResolutionResult subclass instance (e.g., D20ResolutionResult)
            containing the specific outcome of the roll.
        """
        raise NotImplementedError
