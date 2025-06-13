from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

from catley.game.enums import OutcomeTier

if TYPE_CHECKING:
    from catley.game.actors import Actor, Character
    from catley.game.items.item_core import Item


@dataclass
class ResolutionResult(abc.ABC):  # noqa: B024 - no abstract methods defined
    """Base result data returned by a resolver implementation."""

    outcome_tier: OutcomeTier = OutcomeTier.FAILURE


class Resolver(abc.ABC):
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
        status_modifiers: dict[str, bool] | None = None,
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
            status_modifiers: Additional advantage/disadvantage flags derived
                from status effects. Keys are ``has_advantage`` and
                ``has_disadvantage``.

        Returns:
            A ResolutionResult subclass instance (e.g., D20ResolutionResult)
            containing the specific outcome of the roll.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_success_probability(
        self, additional_modifiers: dict[str, bool] | None = None
    ) -> float:
        """Calculate the probability of success for this resolver's configuration.

        Args:
            additional_modifiers: Advantage/disadvantage flags from status
                effects or situational factors. Keys are ``has_advantage`` and
                ``has_disadvantage``.

        Returns:
            Probability of success as a float in the range ``[0.0, 1.0]``.
        """
        raise NotImplementedError
