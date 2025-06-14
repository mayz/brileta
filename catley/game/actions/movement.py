"""
Movement actions for actors in the game world.

Handles actor movement including collision detection and automatic ramming
when attempting to move into occupied spaces.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from catley import colors
from catley.constants.movement import MovementConstants
from catley.environment import tile_types
from catley.events import MessageEvent, publish_event
from catley.game.actions.base import GameAction, GameActionResult
from catley.game.actors import Character
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
        """Choose the best melee weapon for accidental ramming.

        Ranged weapons are ignored entirely to avoid wasting ammo or using
        precision gear when simply bumping into an enemy.
        """

        active_weapon = self.actor.inventory.get_active_weapon()

        candidates: list[Item] = []
        if active_weapon and self._is_suitable_melee_for_ramming(active_weapon):
            candidates.append(active_weapon)

        for weapon, _ in self.actor.inventory.get_equipped_items():
            if weapon is not active_weapon and self._is_suitable_melee_for_ramming(
                weapon
            ):
                candidates.append(weapon)

        if not candidates:
            return FISTS_TYPE.create()

        # Prefer designed weapons over improvised ones.
        non_improvised = [
            w
            for w in candidates
            if WeaponProperty.IMPROVISED not in w.melee_attack.properties  # type: ignore[union-attr]
        ]
        if non_improvised:
            if active_weapon in non_improvised:
                return active_weapon  # type: ignore[return-value]
            return non_improvised[0]

        # All options are improvised; keep active weapon if possible
        if active_weapon in candidates:
            return active_weapon  # type: ignore[return-value]
        return candidates[0]

    def _is_suitable_melee_for_ramming(self, weapon: Item) -> bool:
        """Return True if the weapon can be used to ram in melee."""
        melee = weapon.melee_attack
        if melee is None:
            return False

        ranged = weapon.ranged_attack
        return not (
            ranged is not None and WeaponProperty.PREFERRED in ranged.properties
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

        # Check for stumbling if effective speed is reduced (mostly from exhaustion)
        if isinstance(self.actor, Character):
            speed_multiplier = self.actor.modifiers.get_movement_speed_multiplier()
            if speed_multiplier < MovementConstants.EXHAUSTION_STUMBLE_THRESHOLD:
                stumble_chance = (
                    1.0 - speed_multiplier
                ) * MovementConstants.EXHAUSTION_STUMBLE_MULTIPLIER
                if random.random() < stumble_chance:
                    publish_event(
                        MessageEvent(
                            f"{self.actor.name} stumbles from exhaustion!",
                            colors.YELLOW,
                        )
                    )
                    return None

        self.actor.move(self.dx, self.dy)
        return GameActionResult(should_update_fov=True)
