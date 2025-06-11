from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event
from tcod.console import Console

from catley import colors
from catley.events import ActorDeathEvent, subscribe_to_event, unsubscribe_from_event
from catley.game import ranges
from catley.game.actions.combat import AttackAction
from catley.game.actors import Character
from catley.modes.base import Mode

if TYPE_CHECKING:
    from catley.controller import Controller


class TargetingMode(Mode):
    """Mode for targeting enemies and performing attacks"""

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.cursor_manager = controller.frame_manager.cursor_manager
        self.candidates: list[Character] = []
        self.current_index: int = 0
        self.last_targeted: Character | None = None

    def enter(self) -> None:
        """Enter targeting mode and find all valid targets"""
        super().enter()
        self.candidates = []

        # Subscribe to actor death events only while targeting
        subscribe_to_event(ActorDeathEvent, self._handle_actor_death_event)

        self.cursor_manager.set_active_cursor_type("crosshair")

        # Build initial candidate list
        self.update()

        # Prefer last targeted actor or currently selected one
        preferred_target = self.last_targeted or self.controller.gw.selected_actor
        if preferred_target and preferred_target in self.candidates:
            assert isinstance(preferred_target, Character)
            self.current_index = self.candidates.index(preferred_target)
            self.controller.gw.selected_actor = preferred_target
        elif self.candidates:
            self.current_index = 0
            self.controller.gw.selected_actor = self.candidates[0]

    def _exit(self) -> None:
        """Exit targeting mode.

        Called via :func:`Controller.exit_targeting_mode`.
        """
        # Remember who we were targeting
        if self.controller.gw.selected_actor:
            selected = self.controller.gw.selected_actor
            assert isinstance(selected, Character)
            self.last_targeted = selected

        self.candidates = []
        self.current_index = 0
        self.controller.gw.selected_actor = None

        self.cursor_manager.set_active_cursor_type("arrow")

        # Clean up event subscriptions
        unsubscribe_from_event(ActorDeathEvent, self._handle_actor_death_event)

        super()._exit()

    def _handle_actor_death_event(self, event: ActorDeathEvent) -> None:
        """Handle actor death events from the global event bus."""
        if self.active:
            self.on_actor_death(event.actor)

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle targeting mode input"""
        if not self.active:
            return False

        match event:
            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
                self.controller.exit_targeting_mode()
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.t):
                self.controller.exit_targeting_mode()
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.TAB):
                if self.candidates:
                    direction = -1 if (event.mod & tcod.event.Modifier.SHIFT) else 1
                    self._cycle_target(direction)
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN):
                if self._get_current_target():
                    target = self._get_current_target()
                    if isinstance(target, Character):
                        attack_action = AttackAction(
                            self.controller, self.controller.gw.player, target
                        )
                        self.controller.queue_action(attack_action)
                return True

        return False

    def render_ui(self, console: Console) -> None:
        """Render targeting mode UI text"""
        if not self.active:
            return

        # Render "TARGETING" status text in bottom margin area
        # Position it in an area that doesn't block game map or other panels
        status_text = "TARGETING"
        # Place it in the bottom area, but above equipment/message panels
        status_y = console.height - 10  # Above the UI panels
        status_x = console.width // 2 - len(status_text) // 2  # Centered horizontally
        console.print(status_x, status_y, status_text, fg=colors.RED)  # type: ignore[no-matching-overload]

    def render_world(self) -> None:
        """Render targeting highlights in world space"""
        if not self.active:
            return

        current_target = self._get_current_target()
        gw_panel = self.controller.frame_manager.world_panel

        for actor in self.candidates:
            if not self.controller.gw.game_map.visible[actor.x, actor.y]:
                continue
            if actor == current_target:
                gw_panel.highlight_actor(actor, (255, 0, 0), effect="pulse")
            else:
                gw_panel.highlight_actor(actor, (100, 0, 0), effect="solid")

    def update(self) -> None:
        """Rebuild target candidates each frame and maintain selection."""
        if not self.active:
            return

        previous_target = self._get_current_target()
        gw = self.controller.gw
        player = gw.player
        visible = gw.game_map.visible

        new_candidates: list[Character] = []
        # Only consider actors within the targeting radius using the spatial index
        potential_actors = gw.actor_spatial_index.get_in_radius(
            player.x, player.y, radius=15
        )
        for actor in potential_actors:
            if actor is player:
                continue

            if (
                isinstance(actor, Character)
                and actor.health.is_alive()
                and visible[actor.x, actor.y]
            ):
                new_candidates.append(actor)

        new_candidates.sort(key=self._calculate_distance_to_player)
        self.candidates = new_candidates

        if not self.candidates:
            self.current_index = 0
            self.controller.gw.selected_actor = None
            return

        if previous_target and previous_target in self.candidates:
            self.current_index = self.candidates.index(previous_target)
        else:
            self.current_index = 0

        self.controller.gw.selected_actor = self.candidates[self.current_index]

    def on_actor_death(self, actor: Character) -> None:
        """Handle actor death in targeting mode"""
        # Remove dead actors from targeting and update current target.
        if not self.active:
            return

        # Remove dead actors from candidates
        self.candidates = [
            actor for actor in self.candidates if actor.health.is_alive()
        ]

        # If current target is dead or no candidates left, exit targeting mode
        current_target = self._get_current_target()
        if not self.candidates or (
            current_target and not current_target.health.is_alive()
        ):
            self.controller.exit_targeting_mode()
            return

        # Adjust index if needed
        if self.current_index >= len(self.candidates):
            self.current_index = 0

        # Update selected actor
        if self.candidates:
            self.controller.gw.selected_actor = self.candidates[self.current_index]

    def _get_current_target(self) -> Character | None:
        """Get currently targeted actor"""
        if (
            self.active
            and self.candidates
            and 0 <= self.current_index < len(self.candidates)
        ):
            return self.candidates[self.current_index]
        return None

    def _cycle_target(self, direction: int = 1) -> None:
        """Cycle to next/previous target. direction: 1 for next, -1 for previous"""
        if not self.candidates:
            return

        self.current_index = (self.current_index + direction) % len(self.candidates)
        self.controller.gw.selected_actor = self.candidates[self.current_index]

    def _calculate_distance_to_player(self, actor) -> int:
        return ranges.calculate_distance(
            self.controller.gw.player.x, self.controller.gw.player.y, actor.x, actor.y
        )
