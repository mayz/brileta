from catley import colors
from catley.controller import Controller
from catley.game.actions import GameAction, MoveAction


class TurnManager:
    """Manages turn-based gameplay, actor energy, and action execution."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.player = self.controller.gw.player
        self._pending_action: GameAction | None = None

    def queue_action(self, action: GameAction) -> None:
        """Queue a game action to be processed on the next turn."""
        self._pending_action = action

    def process_unified_round(self) -> None:
        """Process a single round where all actors can act."""
        # Only process if there's actually a pending action to start the round
        if not self._pending_action:
            return

        # Let actors spend their accumulated points to take actions
        max_iterations = 50  # Prevent infinite loops
        iteration = 0

        # Give all actors energy once per round
        for actor in self.controller.gw.actors:
            actor.regenerate_energy()

        while iteration < max_iterations:
            someone_acted = False
            iteration += 1

            # Check each actor to see if they can act
            for actor in self.controller.gw.actors.copy():
                if actor.can_afford_action(self.controller.action_cost):
                    action = actor.get_next_action(self.controller)
                    # Always spend energy if you can afford to act. No hoarding!
                    actor.spend_energy(self.controller.action_cost)

                    if action:
                        self._execute_action(action)
                        someone_acted = True

                    if not self.player.health.is_alive():
                        # Player died, handle game over and stop processing
                        self.controller.message_log.add_message(
                            f"{self.controller.gw.player.name}"
                            " has been defeated! Game Over.",
                            colors.RED,
                        )
                        return

            # If nobody acted this round, we're done
            if not someone_acted:
                break

        # After the round, call update_turn for any passive effects
        for actor in self.controller.gw.actors:
            actor.update_turn(self.controller)

    def dequeue_player_action(self) -> GameAction | None:
        """Dequeue and return the pending player action."""
        action = self._pending_action
        self._pending_action = None
        return action

    def _execute_action(self, action: GameAction | None) -> None:
        """Execute an action immediately without turn processing."""
        if not action:
            return

        try:
            action.execute()
        except SystemExit:
            raise
        except Exception as e:
            error_message = f"Error executing action '{type(action).__name__}': {e!s}"
            self.controller.message_log.add_message(error_message, colors.RED)
            print(f"Unhandled exception during action execution: {e}")
            return

        # Update FOV only if the player moved
        if action.actor == self.controller.gw.player and isinstance(action, MoveAction):
            self.controller.update_fov()
