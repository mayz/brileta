"""Game Mode System - Behavioral overlays that change how the game works.

The game always has an active mode. ExploreMode is the default for normal
gameplay; CombatMode is for combat. Transitions are always mode-to-mode, never
mode-to-nothing. The Controller always delegates input to current_mode.handle_input().

Modes are appropriate when the interaction:
- Takes over the normal gameplay context
- Needs different input handling (including "ignore all input while animation plays")
- Is self-contained with its own state and entry/exit transitions

This includes brief interactions - not just sustained multi-turn contexts. A 3-second
lockpicking ceremony that shows an animation and auto-exits is architecturally
consistent with combat mode that you stay in for many turns. Use modes for:
- Sustained contexts: CombatMode (many turns of combat), ConversationMode (dialogue)
- Brief ceremonies: LockpickingMode (attempt + result), RepairMode (attempt + result)
- Resolution presentations: ResolutionCeremonyMode (skill check drama)

Modes are distinct from Overlays (like menus), which primarily display information.

The input handling priority is always: UI Overlays -> Active Mode -> Game Actions.
The lifecycle is managed by the Controller:
  enter() -> handle input/render -> transition to another mode.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from catley import input_events
from catley.game.actors import Character

if TYPE_CHECKING:
    from catley.controller import Controller


class Mode(abc.ABC):
    """Base class for all game behavioral modes.

    A Mode defines a temporary change in how the game interprets input and
    what is rendered on screen. It has direct hooks to render effects both
    within the world view and on the main UI console.
    """

    def __init__(self, controller: Controller) -> None:
        self.controller: Controller = controller
        self.active: bool = False

    @abc.abstractmethod
    def enter(self) -> None:
        """Initialize and activate the mode.

        Subclasses should always call super().enter() to set the active flag.
        """
        self.active = True

    @abc.abstractmethod
    def _exit(self) -> None:
        """Perform internal cleanup and deactivate the mode.

        This method should only be called by the Controller's exit_*_mode()
        methods, not directly. Subclasses should always call super()._exit().
        """
        self.active = False

    @abc.abstractmethod
    def handle_input(self, event: input_events.InputEvent) -> bool:
        """Handle input events with priority over other systems.

        Return True if the event was consumed, preventing it from being
        processed by overlays or the main input handler.
        """
        pass

    def render_world(self) -> None:  # noqa: B027
        """Render visual effects inside the game world's viewport.

        Use this for effects like highlighting actors, drawing targeting lines,
        or showing area-of-effect previews on the game map itself.
        """
        pass

    def update(self) -> None:  # noqa: B027
        """Perform per-frame updates.

        Useful for animations, state validation, or other logic that needs
        to run on every frame while the mode is active.
        """
        pass

    def on_actor_death(self, actor: Character) -> None:  # noqa: B027
        """Hook to react to an actor's death.

        Override this if the mode needs to track specific actors and must
        react if one of them is killed (e.g., by removing them from a list
        of valid targets).
        """
        pass
