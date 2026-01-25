from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.events import (
    ActorDeathEvent,
    MessageEvent,
    publish_event,
    subscribe_to_event,
    unsubscribe_from_event,
)
from catley.game import ranges
from catley.game.actions.base import GameIntent
from catley.game.actions.combat import AttackIntent
from catley.game.actions.discovery import ActionCategory, ActionOption
from catley.game.actions.stunts import KickIntent, PunchIntent, PushIntent, TripIntent
from catley.game.actors import Character
from catley.modes.base import Mode
from catley.modes.picker import PickerResult

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

        # Selected action for action-centric combat UI.
        # When in combat mode, the player first selects an action (Attack, Push, etc.)
        # then clicks on a target to execute it.
        self.selected_action: ActionOption | None = None

    def enter(self) -> None:
        """Enter combat mode and find all valid targets.

        Immediately pushes PickerMode to handle target selection. CombatMode
        provides the combat context (highlights, valid targets) while PickerMode
        handles the actual click-to-select interaction.
        """
        super().enter()

        self.candidates = []

        # Subscribe to actor death events only while in combat mode
        subscribe_to_event(ActorDeathEvent, self._handle_actor_death_event)

        # Set default action (Attack) for action-centric UI
        self._set_default_action()

        # Show combat tooltip overlay (if available - may not exist in tests)
        fm = self.controller.frame_manager
        if fm is not None and hasattr(fm, "combat_tooltip_overlay"):
            self.controller.overlay_system.show_overlay(fm.combat_tooltip_overlay)

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
        self._set_selected_action(None)  # Reset action selection on exit

        # Hide combat tooltip overlay (if available - may not exist in tests)
        fm = self.controller.frame_manager
        if fm is not None and hasattr(fm, "combat_tooltip_overlay"):
            tooltip = fm.combat_tooltip_overlay
            if tooltip.is_active:
                tooltip.hide()

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
        # Note: Target selection clicks are handled by PickerMode (on top of this mode)
        match event:
            case tcod.event.MouseButtonDown(button=tcod.event.MouseButton.LEFT):
                # Check if click is on action panel to select actions
                if self._try_select_action_by_click(event):
                    return True
                # Otherwise let PickerMode handle target selection

            case tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE):
                if self.controller.has_visible_hostiles():
                    publish_event(
                        MessageEvent(
                            "Standing down despite hostile presence.", colors.YELLOW
                        )
                    )
                self.controller.exit_combat_mode("manual_exit")
                return True

            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN):
                # Execute the currently selected action on the current target
                target = self._get_current_target()
                if target and isinstance(target, Character):
                    intent = self._create_intent_for_target(target)
                    if intent is not None:
                        self.controller.queue_action(intent)
                return True

            case tcod.event.KeyDown(sym=sym):
                # Check for action hotkey selection (a-z)
                key_char = chr(sym) if 97 <= sym <= 122 else None  # a-z
                if key_char and self._try_select_action_by_hotkey(key_char):
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
            self.controller.exit_combat_mode("all_enemies_dead")
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

        Executes the selected action on the target and re-pushes PickerMode
        for the next selection. This keeps the player in combat mode until
        they explicitly exit.
        """
        if (
            result.actor
            and isinstance(result.actor, Character)
            and result.actor in self.candidates
        ):
            # Create appropriate intent based on selected action
            intent = self._create_intent_for_target(result.actor)
            if intent is not None:
                self.controller.queue_action(intent)

        # Stay in combat mode - push picker again for next target
        self._start_target_selection()

    def _on_target_cancelled(self) -> None:
        """Handle cancel from PickerMode - exit combat mode entirely."""
        if self.controller.has_visible_hostiles():
            publish_event(
                MessageEvent("Standing down despite hostile presence.", colors.YELLOW)
            )
        self.controller.exit_combat_mode("cancelled")

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
        """Render shimmering glyph outlines on all targetable enemies.

        All visible candidates get the same shimmering outline effect - there's
        no visual distinction between "current target" and others since TAB
        cycling has been removed in favor of pure mouse targeting.
        """
        if self.controller.frame_manager is None:
            return

        gw_view = self.controller.frame_manager.world_view
        alpha = gw_view.get_shimmer_alpha()
        outline_color = colors.COMBAT_OUTLINE[:3]  # Extract RGB from RGBA

        for actor in self.candidates:
            if not self.controller.gw.game_map.visible[actor.x, actor.y]:
                continue
            gw_view.render_actor_outline(actor, outline_color, alpha)

    # --- Action-centric combat UI ---

    def _set_default_action(self) -> None:
        """Set the default combat action when entering combat mode.

        Selects the first action from discovery, which is already sorted by
        priority (PREFERRED attacks first).
        """
        from catley.game.actions.discovery import ActionDiscovery

        discovery = ActionDiscovery()
        actions = discovery.combat_discovery.get_player_combat_actions(
            self.controller, self.controller.gw.player
        )

        if actions:
            self._set_selected_action(actions[0])
        else:
            self._set_selected_action(None)

    def select_action(self, action: ActionOption) -> None:
        """Select a combat action to use on the next target click.

        Args:
            action: The action to select.
        """
        self._set_selected_action(action)

    def get_available_combat_actions(
        self, target: Character | None = None
    ) -> list[ActionOption]:
        """Get all combat actions available to the player.

        Returns actions like Attack (melee/ranged) and Push. These are
        action-centric (not tied to a specific target) for the combat
        mode action panel.

        If a target is provided, probabilities will be calculated against
        that specific target. Otherwise, probabilities are left as None.

        Also validates that the currently selected action is still available
        (e.g., after weapon switch). If the selected action becomes invalid,
        it resets to the first available action.

        Args:
            target: Optional target for probability calculation. If None,
                    actions are returned without probability values.
        """
        from catley.game.actions.discovery import ActionDiscovery

        discovery = ActionDiscovery()
        actions = discovery.combat_discovery.get_player_combat_actions(
            self.controller, self.controller.gw.player, target
        )

        # Validate current selection is still available (may have changed due to
        # weapon switch, running out of ammo, etc.)
        self._ensure_valid_selection(actions)

        return actions

    def _ensure_valid_selection(self, actions: list[ActionOption]) -> None:
        """Reset selection if current action is no longer available.

        Called after getting available actions to handle cases like weapon
        switching where the previously selected action may no longer exist.
        Also auto-selects first available action if none is currently selected,
        ensuring there's always a selection when valid actions exist.

        Args:
            actions: The current list of available combat actions.
        """
        if not actions:
            self._set_selected_action(None)
            return

        # Auto-select first action if none currently selected
        if self.selected_action is None:
            self._set_selected_action(actions[0])
            return

        # Check if current selection is still valid
        action_ids = {a.id for a in actions}
        if self.selected_action.id not in action_ids:
            # Weapon changed - select first action
            self._set_selected_action(actions[0])

    def _set_selected_action(self, action: ActionOption | None) -> None:
        """Assign selected action and refresh tooltip if it changed."""
        if self.selected_action is action:
            return

        self.selected_action = action
        self._invalidate_combat_tooltip()

    def _invalidate_combat_tooltip(self) -> None:
        """Invalidate combat tooltip overlay if it is active."""
        fm = self.controller.frame_manager
        if fm is None or not hasattr(fm, "combat_tooltip_overlay"):
            return

        tooltip = fm.combat_tooltip_overlay
        if tooltip.is_active:
            tooltip.invalidate()

    def _try_select_action_by_hotkey(self, key: str) -> bool:
        """Try to select a combat action by its hotkey.

        Uses the action panel's hotkey mappings as the source of truth, since
        the panel uses a sticky hotkey system that preserves assignments across
        frames. Falls back to index-based hotkey assignment in tests where
        the action panel may not be available.

        Args:
            key: The lowercase letter pressed (a-z).

        Returns:
            True if an action was selected, False otherwise.
        """
        # Try to get hotkey mappings from the action panel (source of truth)
        fm = self.controller.frame_manager
        if fm is not None and hasattr(fm, "action_panel_view"):
            hotkeys = fm.action_panel_view.get_hotkeys()
            if key in hotkeys:
                self.select_action(hotkeys[key])
                return True
            return False

        # Fallback for tests: assign hotkeys by index
        actions = self.get_available_combat_actions()
        hotkey_chars = "abcdefghijklmnopqrstuvwxyz"
        for i, action in enumerate(actions):
            if i < len(hotkey_chars) and hotkey_chars[i] == key:
                self.select_action(action)
                return True

        return False

    def _try_select_action_by_click(self, event: tcod.event.MouseButtonDown) -> bool:
        """Try to select a combat action by clicking on it in the action panel.

        Args:
            event: The mouse button down event.

        Returns:
            True if click was on action panel and an action was selected.
        """
        fm = self.controller.frame_manager
        if fm is None or not hasattr(fm, "action_panel_view"):
            return False
        action_panel_view = fm.action_panel_view

        # Convert raw pixel position to scaled pixel position
        graphics = self.controller.graphics
        scale_x, scale_y = graphics.get_display_scale_factor()
        scaled_px_x = int(event.position.x * scale_x)
        scaled_px_y = int(event.position.y * scale_y)

        # Calculate action panel's screen pixel bounds
        tile_width, tile_height = graphics.tile_dimensions
        panel_px_x = action_panel_view.x * tile_width
        panel_px_y = action_panel_view.y * tile_height
        panel_px_width = action_panel_view.width * tile_width
        panel_px_height = action_panel_view.height * tile_height

        # Check if click is within panel bounds
        if not (
            panel_px_x <= scaled_px_x < panel_px_x + panel_px_width
            and panel_px_y <= scaled_px_y < panel_px_y + panel_px_height
        ):
            return False

        # Convert to panel-relative pixel coordinates
        rel_px_x = scaled_px_x - panel_px_x
        rel_px_y = scaled_px_y - panel_px_y

        # Check if click hit an action
        action = action_panel_view.get_action_at_pixel(rel_px_x, rel_px_y)
        if action is not None:
            self.select_action(action)
            return True

        return False

    def _create_intent_for_target(self, target: Character) -> GameIntent | None:
        """Create the appropriate intent for the selected action and target.

        Uses the currently selected action to determine what intent to create.
        Falls back to AttackIntent if no action is selected.

        For melee attacks and Push, if the target is not adjacent, initiates
        pathfinding to an adjacent tile with the action as the final_intent.
        Returns None in this case (action will be queued on arrival).

        Args:
            target: The target character for the action.

        Returns:
            The appropriate GameIntent for immediate execution, or None if
            pathfinding was started (action queued for later) or if pathfinding
            failed.
        """
        player = self.controller.gw.player
        distance = ranges.calculate_distance(player.x, player.y, target.x, target.y)

        # Fall back to default attack if no action selected
        if self.selected_action is None:
            return self._handle_melee_intent(
                target, distance, AttackIntent(self.controller, player, target)
            )

        # Handle based on action category
        if self.selected_action.category == ActionCategory.COMBAT:
            weapon = self.selected_action.static_params.get("weapon")
            attack_mode = self.selected_action.static_params.get("attack_mode")
            intent = AttackIntent(
                self.controller, player, target, weapon=weapon, attack_mode=attack_mode
            )

            # Ranged attacks execute immediately regardless of distance
            if attack_mode == "ranged":
                return intent

            # Melee attacks may need approach first
            return self._handle_melee_intent(target, distance, intent)

        if self.selected_action.category == ActionCategory.STUNT:
            if self.selected_action.action_class == PushIntent:
                intent = PushIntent(self.controller, player, target)
                return self._handle_melee_intent(target, distance, intent)
            if self.selected_action.action_class == TripIntent:
                intent = TripIntent(self.controller, player, target)
                return self._handle_melee_intent(target, distance, intent)
            if self.selected_action.action_class == KickIntent:
                intent = KickIntent(self.controller, player, target)
                return self._handle_melee_intent(target, distance, intent)
            if self.selected_action.action_class == PunchIntent:
                # Punch uses the ActionPlan system - it handles approach, holster,
                # and punch as separate steps.
                self.controller.start_punch_plan(player, target)
                return None

        # Unhandled action type - fail loudly so we catch missing handlers
        raise ValueError(
            f"Unhandled action type in combat mode: "
            f"category={self.selected_action.category}, "
            f"action_class={self.selected_action.action_class}"
        )

    def _handle_melee_intent(
        self,
        target: Character,
        distance: int,
        intent: GameIntent,
    ) -> GameIntent | None:
        """Handle melee intent creation, with approach if needed.

        If adjacent, returns the intent for immediate execution.
        If not adjacent, starts pathfinding with the intent as final_intent.

        Args:
            target: The target character.
            distance: Pre-calculated distance to target.
            intent: The intent to execute (AttackIntent or PushIntent).

        Returns:
            The intent if adjacent and ready to execute, None if pathfinding
            was started or failed.
        """
        if distance == 1:
            # Adjacent - execute immediately
            return intent

        # Not adjacent - pathfind first, then execute
        player = self.controller.gw.player
        if self.controller.start_actor_pathfinding(
            player, (target.x, target.y), intent
        ):
            # Pathfinding started - action will be queued on arrival
            return None

        # Pathfinding failed - "Path is blocked" message already shown
        return None
