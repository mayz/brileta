"""
Movement actions for actors in the game world.

Handles actor movement including collision detection and automatic ramming
when attempting to move into occupied spaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.environment import tile_types
from catley.game.actions.base import GameAction, GameActionResult
from catley.game.actors import Character
from catley.game.items.capabilities import MeleeAttack
from catley.game.items.item_types import FISTS_TYPE
from catley.game.items.properties import WeaponProperty

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.items.item_core import Item


class MoveAction(GameAction):
    """Action for moving an actor on the game map."""

    def __init__(
        self, controller: Controller, actor: Character, dx: int, dy: int
    ) -> None:
        super().__init__(controller, actor)

        # Type narrowing.
        self.actor: Character

        self.game_map = controller.gw.game_map

        self.dx = dx
        self.dy = dy
        self.newx = self.actor.x + self.dx
        self.newy = self.actor.y + self.dy

    def _select_ram_weapon(self) -> Item:
        """Choose the best weapon for ramming into another actor."""
        # First, try the currently active weapon if it's suitable
        active_weapon = self.actor.inventory.get_active_weapon()
        if active_weapon and self._is_suitable_ram_weapon(active_weapon):
            return active_weapon

        # Fall back to any other equipped weapon that's suitable
        for weapon, _ in self.actor.inventory.get_equipped_items():
            if weapon != active_weapon and self._is_suitable_ram_weapon(weapon):
                return weapon

        return FISTS_TYPE.create()

    def _is_suitable_ram_weapon(self, weapon: Item) -> bool:
        """Check if a weapon is suitable for ramming."""
        attack_mode = weapon.get_preferred_attack_mode(distance=1)
        ranged_attack = weapon.ranged_attack
        has_preferred_ranged = (
            ranged_attack is not None
            and WeaponProperty.PREFERRED in ranged_attack.properties
        )
        return (
            isinstance(attack_mode, MeleeAttack)
            and WeaponProperty.IMPROVISED not in attack_mode.properties
            and not has_preferred_ranged
        )

    def execute(self) -> GameActionResult | None:
        # Import here to avoid circular import
        from catley.game.actions.combat import AttackAction

        # Prevent indexing errors by bounding to the map dimensions first.
        if not (
            0 <= self.newx < self.game_map.width
            and 0 <= self.newy < self.game_map.height
        ):
            return None

        tile_id = self.game_map.tiles[self.newx, self.newy]

        if tile_id == tile_types.TILE_TYPE_ID_DOOR_CLOSED:  # type: ignore[attr-defined]
            # Automatically open doors when bumped into.
            from catley.game.actions.environment import OpenDoorAction

            OpenDoorAction(
                self.controller,
                self.actor,
                self.newx,
                self.newy,
            ).execute()

        if not self.game_map.walkable[self.newx, self.newy]:
            return None

        # Check for a blocking actor using the spatial index for O(1) lookup.
        blocking_actor = self.controller.gw.get_actor_at_location(self.newx, self.newy)
        if blocking_actor and blocking_actor.blocks_movement:
            if (
                isinstance(blocking_actor, Character)
                and blocking_actor.health.is_alive()
            ):
                weapon = self._select_ram_weapon()
                AttackAction(
                    controller=self.controller,
                    attacker=self.actor,
                    defender=blocking_actor,
                    weapon=weapon,
                ).execute()
            return None  # Cannot move into blocking actor

        self.actor.move(self.dx, self.dy)
        return GameActionResult(should_update_fov=True)
