from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event

from catley.events import ActorDeathEvent, subscribe_to_event, unsubscribe_from_event
from catley.game import ranges
from catley.game.actions.combat import AttackIntent
from catley.game.actors import Character
from catley.input_handler import Keys
from catley.modes.base import Mode
from catley.view.ui.targeting_indicator_overlay import TargetingIndicatorOverlay

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
        self.targeting_indicator_overlay = TargetingIndicatorOverlay(controller)

    def enter(self) -> None:
        """Enter combat mode and find all valid targets."""
        super().enter()

        assert self.controller.overlay_system is not None
        self.candidates = []

        # Subscribe to actor death events only while in combat mode
        subscribe_to_event(ActorDeathEvent, self._handle_actor_death_event)

        # Set the crosshair cursor to indicate combat mode
        self.cursor_manager.set_active_cursor_type("crosshair")

        self.controller.overlay_system.show_overlay(self.targeting_indicator_overlay)

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
        """Exit combat mode.

        Called via :func:`Controller.exit_combat_mode`.
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
        self.targeting_indicator_overlay.hide()

        # Clean up event subscriptions
        unsubscribe_from_event(ActorDeathEvent, self._handle_actor_death_event)

        super()._exit()

    def _handle_actor_death_event(self, event: ActorDeathEvent) -> None:
        """Handle actor death events from the global event bus."""
        if self.active:
            self.on_actor_death(event.actor)

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle combat mode input.

        Combat-specific input is handled first. Unhandled input falls back
        to ExploreMode for common functionality (movement, inventory, etc.).
        """
        if not self.active:
            return False

        # Handle targeting-specific input first
        match event:
            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
                self.controller.exit_combat_mode()
                return True

            case tcod.event.KeyDown(sym=Keys.KEY_T):
                self.controller.exit_combat_mode()
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

            case tcod.event.MouseButtonDown(button=tcod.event.MouseButton.LEFT):
                # Click-to-attack: find actor at click position and attack if valid
                clicked_actor = self._get_actor_at_mouse_position(event)
                if clicked_actor and clicked_actor in self.candidates:
                    attack_action = AttackIntent(
                        self.controller, self.controller.gw.player, clicked_actor
                    )
                    self.controller.queue_action(attack_action)
                    return True
                return True  # Consume click even if no valid target

        # Fall back to explore mode for everything else
        # (movement, inventory, reload, weapon switch, etc.)
        return self.controller.explore_mode.handle_input(event)

    def render_world(self) -> None:
        """Render targeting highlights in world space"""
        if not self.active:
            return

        assert self.controller.frame_manager is not None

        current_target = self._get_current_target()
        gw_view = self.controller.frame_manager.world_view

        for actor in self.candidates:
            if not self.controller.gw.game_map.visible[actor.x, actor.y]:
                continue
            if actor == current_target:
                gw_view.highlight_actor(actor, (255, 0, 0), effect="pulse")
            else:
                gw_view.highlight_actor(actor, (100, 0, 0), effect="solid")

    def update(self) -> None:
        """Rebuild target candidates and forward to explore mode for movement.

        CombatMode layers on top of ExploreMode, so we forward update()
        to allow movement while in combat.
        """
        if not self.active:
            return

        # Forward to explore mode for movement
        self.controller.explore_mode.update()

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
            actor for actor in self.candidates if actor.health.is_alive()
        ]

        # If current target is dead or no candidates left, exit targeting mode
        current_target = self._get_current_target()
        if not self.candidates or (
            current_target and not current_target.health.is_alive()
        ):
            self.controller.exit_combat_mode()
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

    def _get_actor_at_mouse_position(
        self, event: tcod.event.MouseButtonDown
    ) -> Character | None:
        """Convert mouse click position to world coordinates and find actor there."""
        assert self.controller.frame_manager is not None

        # Convert pixel coordinates to root console tile coordinates
        graphics = self.controller.graphics
        scale_x, scale_y = graphics.get_display_scale_factor()
        scaled_px_x = event.position.x * scale_x
        scaled_px_y = event.position.y * scale_y
        root_tile_x, root_tile_y = graphics.pixel_to_tile(scaled_px_x, scaled_px_y)

        # Convert root tile to world tile coordinates
        root_tile_pos = (int(root_tile_x), int(root_tile_y))
        world_tile_pos = (
            self.controller.frame_manager.get_world_coords_from_root_tile_coords(
                root_tile_pos
            )
        )

        if world_tile_pos is None:
            return None

        world_x, world_y = world_tile_pos
        gw = self.controller.gw

        # Check bounds and visibility
        if not (0 <= world_x < gw.game_map.width and 0 <= world_y < gw.game_map.height):
            return None
        if not gw.game_map.visible[world_x, world_y]:
            return None

        # Find actor at this position
        actor = gw.get_actor_at_location(world_x, world_y)
        if isinstance(actor, Character) and actor.health.is_alive():
            return actor
        return None
