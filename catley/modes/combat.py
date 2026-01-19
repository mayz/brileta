from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event

from catley.events import ActorDeathEvent, subscribe_to_event, unsubscribe_from_event
from catley.game import ranges
from catley.game.actions.combat import AttackIntent
from catley.game.actors import Character
from catley.modes.base import Mode
from catley.modes.picker import PickerResult
from catley.view.ui.combat_indicator_overlay import CombatIndicatorOverlay

if TYPE_CHECKING:
    from catley.controller import Controller


class CombatMode(Mode):
    """Mode for deliberate combat - targeting enemies and performing attacks.

    This mode is entered when the player explicitly chooses to engage in combat,
    shifting the game away from combat-by-default toward deliberate interaction.
    """

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)

        assert controller.frame_manager is not None
        self.cursor_manager = controller.frame_manager.cursor_manager
        self.candidates: list[Character] = []
        self.current_index: int = 0
        self.last_targeted: Character | None = None
        self.combat_indicator_overlay = CombatIndicatorOverlay(controller)

    def enter(self) -> None:
        """Enter combat mode and find all valid targets.

        Immediately pushes PickerMode to handle target selection. CombatMode
        provides the combat context (highlights, valid targets) while PickerMode
        handles the actual click-to-select interaction.
        """
        super().enter()

        assert self.controller.overlay_system is not None
        self.candidates = []

        # Subscribe to actor death events only while in combat mode
        subscribe_to_event(ActorDeathEvent, self._handle_actor_death_event)

        # Note: Cursor management is now handled by PickerMode

        self.controller.overlay_system.show_overlay(self.combat_indicator_overlay)

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

        # Push PickerMode to handle target selection
        self._start_target_selection()

    def _exit(self) -> None:
        """Exit combat mode.

        Called via :func:`Controller.exit_combat_mode`.
        Note: Cursor management is handled by PickerMode.
        """
        # Remember who we were targeting
        if self.controller.gw.selected_actor:
            selected = self.controller.gw.selected_actor
            assert isinstance(selected, Character)
            self.last_targeted = selected

        self.candidates = []
        self.current_index = 0
        self.controller.gw.selected_actor = None

        # Note: Cursor is restored to arrow by PickerMode when it exits
        self.combat_indicator_overlay.hide()

        # Clean up event subscriptions
        unsubscribe_from_event(ActorDeathEvent, self._handle_actor_death_event)

        super()._exit()

    def _handle_actor_death_event(self, event: ActorDeathEvent) -> None:
        """Handle actor death events from the global event bus."""
        if self.active:
            self.on_actor_death(event.actor)

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle combat mode input.

        Combat-specific input is handled first. Click-to-attack is handled by
        PickerMode which is pushed on top of CombatMode.

        Unhandled input returns False to let the mode stack pass it to
        ExploreMode below.
        """
        if not self.active:
            return False

        # Handle targeting-specific input first
        # Note: Mouse clicks are handled by PickerMode (on top of this mode)
        match event:
            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
                self.controller.pop_mode()
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
                        attack_action = AttackIntent(
                            self.controller, self.controller.gw.player, target
                        )
                        self.controller.queue_action(attack_action)
                return True

        # Let the mode stack pass unhandled input to ExploreMode below
        return False

    def render_world(self) -> None:
        """Render targeting highlights in world space.

        When PickerMode is on top, it calls _render_combat_visuals() via the
        render_underneath callback. Skip rendering here to avoid double-rendering
        highlights.
        """
        if not self.active:
            return

        # When PickerMode is on top, it handles rendering via render_underneath.
        # Only render directly when CombatMode is the active (top) mode.
        if self.controller.active_mode is not self:
            return

        self._render_targeting_highlights()

    def update(self) -> None:
        """Rebuild target candidates each frame.

        Movement is handled by ExploreMode.update() which the controller
        calls for all modes in the stack.
        """
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
        """Handle actor death in combat mode."""
        # Remove dead actors from candidate list and update current target.
        if not self.active:
            return

        # Remove dead actors from candidates
        self.candidates = [
            candidate for candidate in self.candidates if candidate.health.is_alive()
        ]

        # If current target is dead or no candidates left, exit combat mode
        current_target = self._get_current_target()
        if not self.candidates or (
            current_target and not current_target.health.is_alive()
        ):
            self.controller.pop_mode()
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

    def _calculate_distance_to_player(self, actor: Character) -> int:
        return ranges.calculate_distance(
            self.controller.gw.player.x, self.controller.gw.player.y, actor.x, actor.y
        )

    # --- PickerMode integration ---

    def _start_target_selection(self) -> None:
        """Push PickerMode to handle target selection.

        PickerMode handles click-to-select while CombatMode provides the
        combat context (valid targets, highlights).
        """
        self.controller.picker_mode.start(
            on_select=self._on_target_selected,
            on_cancel=self._on_target_cancelled,
            valid_filter=self._is_valid_target,
            render_underneath=self._render_combat_visuals,
        )

    def _on_target_selected(self, result: PickerResult) -> None:
        """Handle target selection from PickerMode.

        Executes an attack on the selected target and re-pushes PickerMode
        for the next selection. This keeps the player in combat mode until
        they explicitly exit.
        """
        if (
            result.actor
            and isinstance(result.actor, Character)
            and result.actor in self.candidates
        ):
            attack_action = AttackIntent(
                self.controller, self.controller.gw.player, result.actor
            )
            self.controller.queue_action(attack_action)

        # Stay in combat mode - push picker again for next target
        self._start_target_selection()

    def _on_target_cancelled(self) -> None:
        """Handle cancel from PickerMode - exit combat mode entirely."""
        self.controller.pop_mode()

    def _is_valid_target(self, x: int, y: int) -> bool:
        """Check if a tile contains a valid attack target.

        Used as the valid_filter for PickerMode to restrict selection
        to tiles containing targetable enemies.
        """
        actor = self.controller.gw.get_actor_at_location(x, y)
        if actor is None:
            return False
        return actor in self.candidates

    def _render_combat_visuals(self) -> None:
        """Render combat highlights while PickerMode is active.

        Called by PickerMode's render_underneath to ensure combat highlights
        are visible during target selection.
        """
        self._render_targeting_highlights()

    def _render_targeting_highlights(self) -> None:
        """Render the targeting highlights for all candidates.

        Extracted from render_world() to be callable by PickerMode's
        render_underneath callback.
        """
        if self.controller.frame_manager is None:
            return

        current_target = self._get_current_target()
        gw_view = self.controller.frame_manager.world_view

        for actor in self.candidates:
            if not self.controller.gw.game_map.visible[actor.x, actor.y]:
                continue
            if actor == current_target:
                gw_view.highlight_actor(actor, (255, 0, 0), effect="pulse")
            else:
                gw_view.highlight_actor(actor, (100, 0, 0), effect="solid")
