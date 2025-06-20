"""
Base classes for the Intent/Executor action system.

This module defines the core data structures for the game's action system, which
is built on the "Intent and Executor" pattern. This pattern decouples the
specification of an action from its implementation.

Core Components:
- GameIntent: A pure data object, or "blueprint," that specifies an actor's
  desired action (e.g., AttackIntent, MoveIntent). It contains no game logic.
  These are the "public API" of the action system.

- ActionExecutor: A specialist class that contains all the logic for a single
  type of Intent. It takes an Intent and applies its effects to the game world.
  Executors are the "private implementation" and should only be called by the
  ActionRouter.

- GameActionResult: A data object returned by an Executor that reports the
  mechanical outcome of an action, such as success, failure, or whether a
  Field of View update is required.

- ActionRouter: The central dispatcher that receives all GameIntents, looks up
  the correct ActionExecutor in a registry, and manages the execution flow. It
  is also responsible for arbitrating the results of actions to handle special
  cases and chained actions (e.g., a failed move becoming an attack).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from catley.game.actions.types import AnimationType
from catley.game.consequences import Consequence

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor
    from catley.view.animation import Animation


@dataclass
class GameActionResult:
    """
    A data object returned by an ActionExecutor to report an action's outcome.

    This class provides a structured report on the mechanical results of an
    executed intent. The ActionRouter inspects this result to determine
    what happens next, such as updating the player's field of view or
    triggering a new, chained action in response to a specific failure.
    """

    succeeded: bool = True
    should_update_fov: bool = False
    blocked_by: Any | None = None
    block_reason: str | None = None
    consequences: list[Consequence] = field(default_factory=list)


class GameIntent:
    """
    A pure data object representing an actor's intended action.

    Intents are the "blueprints" for actions. They are created by UI handlers or
    AI components to describe what an actor *wants* to do, but contain no
    execution logic themselves. They are the public-facing part of the action
    system and are passed to the ActionRouter for processing.
    """

    def __init__(self, controller: Controller, actor: Actor) -> None:
        """Initialize the intent with its context.

        Attributes:
            animation_type (AnimationType): Controls the action's timing behavior
                (INSTANT vs. WIND_UP) as part of the PPIAS system. Defaults to
                INSTANT for all common actions.
            windup_animation (Animation | None): The animation to play *before*
                a WIND_UP action is resolved. This is ignored for INSTANT actions.
        """
        self.controller = controller
        self.actor = actor
        self.animation_type: AnimationType = AnimationType.INSTANT
        self.windup_animation: Animation | None = None
