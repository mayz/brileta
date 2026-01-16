from __future__ import annotations

import abc
from typing import TYPE_CHECKING, TypeVar, cast

from catley.game.actors.conditions import Condition
from catley.game.enums import AreaType, BlendMode, ConsumableEffectType  # noqa: F401
from catley.game.items.properties import ItemProperty
from catley.util import dice

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor
    from catley.game.actors.components import ConditionsComponent


class AttackSpec(abc.ABC):  # noqa: B024
    """Defines the properties of a melee attack."""

    def __init__(
        self,
        damage_die: str,
        stat_name: str,
        properties: set[ItemProperty] | None = None,
    ) -> None:
        self.damage_dice = dice.Dice(damage_die)
        self.properties: set[ItemProperty] = (
            properties if properties is not None else set()
        )
        self.stat_name = stat_name

    def has_property(self, property_enum: ItemProperty) -> bool:
        """Check if this attack has the given property."""
        return property_enum in self.properties

    def has_any_property(self, *properties: ItemProperty) -> bool:
        """Check if this attack has any of the given properties."""
        return bool(self.properties & set(properties))

    def has_all_properties(self, *properties: ItemProperty) -> bool:
        """Check if this attack has all of the given properties."""
        return set(properties).issubset(self.properties)


class MeleeAttackSpec(AttackSpec):
    """Defines the properties of a melee attack."""

    def __init__(
        self,
        damage_die: str,
        properties: set[ItemProperty] | None = None,
        verb: str = "strike",
    ) -> None:
        super().__init__(damage_die, "strength", properties)
        self.verb = verb


class RangedAttackSpec(AttackSpec):
    """Defines the properties of a ranged attack."""

    def __init__(
        self,
        damage_die: str,
        ammo_type: str,
        max_ammo: int,
        optimal_range: int,
        max_range: int,
        properties: set[ItemProperty] | None = None,
        verb: str = "shoot",
    ):
        super().__init__(damage_die, "observation", properties)
        self.ammo_type = ammo_type
        self.max_ammo = max_ammo
        self.optimal_range = optimal_range
        self.max_range = max_range
        self.verb = verb


SpecType = TypeVar("SpecType", bound=AttackSpec)


class Attack[SpecType: AttackSpec](abc.ABC):
    """Interface for item components that can perform attacks."""

    def __init__(self, spec: SpecType) -> None:
        self._spec = spec

    @abc.abstractmethod
    def can_attack(
        self,
        attacker: Actor,
        target: Actor,
        distance: int,
        controller: Controller,
    ) -> bool:
        pass

    @abc.abstractmethod
    def perform_attack(
        self, attacker: Actor, target: Actor, controller: Controller
    ) -> None:
        """Execute the attack action."""
        pass

    @property
    def properties(self) -> set[ItemProperty]:
        """Get the properties of this attack."""
        return self._spec.properties

    @property
    def damage_dice(self) -> dice.Dice:
        """Get the damage dice for this attack."""
        return self._spec.damage_dice

    @property
    def stat_name(self) -> str:
        """Get the name of the ability score to use for this attack."""
        return self._spec.stat_name


class MeleeAttack(Attack[MeleeAttackSpec]):
    """Handles melee attacks for a specific item instance."""

    def __init__(self, spec: MeleeAttackSpec) -> None:
        super().__init__(spec)
        # Melee attacks are often stateless for the item itself,
        # but this structure allows adding state later (e.g., durability).

    def can_attack(
        self,
        attacker: Actor,
        target: Actor,
        distance: int,
        controller: Controller,
    ) -> bool:
        # Must be adjacent.
        return distance == 1

    def perform_attack(
        self, attacker: Actor, target: Actor, controller: Controller
    ) -> None:
        """Carry out the melee attack."""
        # Access definitions via self._spec.damage_dice, etc.
        # Actual melee attack logic will be implemented later
        pass


class RangedAttack(Attack[RangedAttackSpec]):
    """Handles ranged attacks for a specific item instance."""

    def __init__(self, spec: RangedAttackSpec) -> None:
        super().__init__(spec)
        # Start with full ammo.
        self.current_ammo: int = spec.max_ammo

    def can_attack(
        self,
        attacker: Actor,
        target: Actor,
        distance: int,
        controller: Controller,
    ) -> bool:
        return (
            self.current_ammo > 0
            and distance <= self._spec.max_range
            and self._has_line_of_sight(attacker, target, controller)
        )

    def _has_line_of_sight(self, attacker, target, controller) -> bool:
        # Will implement in Phase 3
        return True

    def perform_attack(
        self, attacker: Actor, target: Actor, controller: Controller
    ) -> None:
        """Carry out the ranged attack and reduce ammo."""
        # Actual ranged attack logic (Phase 4)
        if self.current_ammo > 0:
            self.current_ammo -= 1
        else:
            # Out of ammo should already be validated before calling.
            pass
        # Attack logic to be implemented

    def reload(self, rounds_to_add: int) -> int:
        needed = self._spec.max_ammo - self.current_ammo
        can_add = min(rounds_to_add, needed)
        self.current_ammo += can_add
        return can_add

    # Convenience properties
    @property
    def max_ammo(self) -> int:
        return self._spec.max_ammo

    @property
    def ammo_type(self) -> str:
        return self._spec.ammo_type

    @property
    def optimal_range(self) -> int:
        return self._spec.optimal_range

    @property
    def max_range(self) -> int:
        return self._spec.max_range

    @property
    def properties(self) -> set[ItemProperty]:
        return self._spec.properties

    @property
    def ammo_display(self) -> str:
        return f"[{self.current_ammo}/{self.max_ammo}]"


# === Area Effects ===
#
# Area effects can do damage (e.g., an explosion, a line of flame from a flamethrower,
# a cone bullet spray from a SMG), but the area effect system supports many other types
# of location-targeted capabilities beyond just damage.
#
# The `properties` system handles the specific mechanics (what the effect actually
# does), while the `AreaEffect` component handles the geometric and targeting aspects
# (where and how big the effect is). This makes the system very extensible - you can
# add new effect types just by adding new property handlers, without changing
# the core area effect architecture.
#
### Utility/Support Effects
#
# - **Smoke Grenade**:
#   Creates concealment area that blocks line of sight (no damage, smoke property)
# - **Flare**:
#   Creates illuminated area for visibility, may blind night-vision enemies
#   (light property)
# - **Healing Gas**:
#   Creates healing mist that restores HP to allies (negative damage, healing property)
#
### Area Denial Effects
#
# - **Caltrops**:
#   Creates hazardous area that damages/slows movement (minor damage,
#   movement_penalty property)
# - **Oil Slick**:
#   Makes area difficult to traverse, can be ignited later
#   (no damage, slippery + flammable properties)
#
### Electronic/Technical Effects
#
# - **EMP Device**: Temporarily disables electronic devices/robots in area
#   (no damage, electronics_disable property)
# - **Motion Sensor**: Creates detection field that alerts to movement
#   (no damage, detection property)
# - **Decontamination Spray**: Removes radiation effects from area and actors
#   (no damage, radiation_removal property)


class AreaEffectSpec:
    """Defines the properties of an area-of-effect capability."""

    def __init__(
        self,
        damage_die: str,
        area_type: AreaType,
        size: int,
        properties: set[ItemProperty] | None = None,
        damage_falloff: bool = True,
        requires_line_of_sight: bool = False,
        penetrates_walls: bool = False,
        friendly_fire: bool = True,
    ) -> None:
        self.damage_dice = dice.Dice(damage_die)
        self.area_type = area_type  # "circle", "line", "cone", "cross"
        self.size = size  # radius for circle, length for line, etc.
        self.properties: set[ItemProperty] = (
            properties if properties is not None else set()
        )
        self.damage_falloff = damage_falloff
        self.requires_line_of_sight = requires_line_of_sight
        self.penetrates_walls = penetrates_walls
        self.friendly_fire = friendly_fire


class AreaEffect:
    """
    Handles area-of-effect capabilities for a specific item instance.

    Defines the intrinsic capabilities of the item.
    - "I create a circle/cone/line of size X at a target location"
    - "I have damage falloff/penetration properties Y"
    - "I can target locations within range Z"

    *Does Not Handle*: Map-specific calculations, actor finding, damage application.
    Those belong in the WeaponAreaEffectExecutor.
    """

    def __init__(self, spec: AreaEffectSpec) -> None:
        self._spec = spec

    def can_target_location(
        self, attacker: Actor, target_x: int, target_y: int, controller: Controller
    ) -> bool:
        """Check if we can target the specified location based on
        weapon capabilities."""
        from catley.game import ranges

        game_map = controller.gw.game_map

        # Ensure the target tile is within the bounds of the map.
        if not (0 <= target_x < game_map.width and 0 <= target_y < game_map.height):
            return False

        # Determine distance from attacker to target.
        distance = ranges.calculate_distance(attacker.x, attacker.y, target_x, target_y)

        # Use a default range if we don't have additional info.
        max_range = 10
        if distance > max_range:
            return False

        # If the effect requires line of sight, verify it via the range system.
        if self._spec.requires_line_of_sight:
            return ranges.has_line_of_sight(
                game_map,
                attacker.x,
                attacker.y,
                target_x,
                target_y,
            )

        return True

    def can_create_effect(self, attacker, controller) -> bool:
        """Check if this item can create its area effect
        (e.g., has ammo, not broken)."""
        # Check if parent weapon has ammo (for things like SMG area spray)
        # Check if single-use item is still available (for grenades)
        return True

    # Convenience properties to access definition
    @property
    def damage_dice(self) -> dice.Dice:
        return self._spec.damage_dice

    @property
    def area_type(self) -> AreaType:
        return self._spec.area_type

    @property
    def size(self) -> int:
        return self._spec.size

    @property
    def properties(self) -> set[ItemProperty]:
        return self._spec.properties

    @property
    def damage_falloff(self) -> bool:
        return self._spec.damage_falloff

    @property
    def penetrates_walls(self) -> bool:
        return self._spec.penetrates_walls

    @property
    def friendly_fire(self) -> bool:
        return self._spec.friendly_fire

    @property
    def requires_line_of_sight(self) -> bool:
        return self._spec.requires_line_of_sight


# === Consumable Capability ===
class ConsumableEffectSpec:  # Definition Class
    """Defines the effect of a consumable item."""

    effect_type: ConsumableEffectType
    effect_value: int
    max_uses: int = 1
    target_condition_types: set[type[Condition]]

    def __init__(
        self,
        effect_type: ConsumableEffectType,
        effect_value: int,
        max_uses: int = 1,
        *,
        target_condition_types: set[type[Condition]] | None = None,
    ) -> None:
        self.effect_type = effect_type
        self.effect_value = effect_value
        self.max_uses = max_uses
        self.target_condition_types = target_condition_types or set()


class ConsumableEffect:  # Handler Class
    """Handles the state and use of a consumable item instance."""

    def __init__(self, spec: ConsumableEffectSpec) -> None:
        self._spec = spec
        self.uses_remaining = spec.max_uses

    def consume(self, target_actor: Actor, controller: Controller) -> bool:
        """Apply the consumable's effect to ``target_actor``."""

        from catley import colors
        from catley.events import MessageEvent, publish_event
        from catley.game.actors import conditions

        if self.uses_remaining <= 0:
            return False

        messages: list[str] = []
        health = target_actor.health

        # Apply healing effects
        if health and self._spec.effect_type == ConsumableEffectType.HEAL:
            before = health.hp
            health.heal(self._spec.effect_value)
            healed = health.hp - before
            if healed:
                messages.append(f"Restored {healed} HP")
        elif health and self._spec.effect_type == ConsumableEffectType.HEAL_HP:
            if health.hp < health.max_hp:
                healed = health.max_hp - health.hp
                health.hp = health.max_hp
                messages.append(f"Restored {healed} HP")
            else:
                messages.append("Already at full HP")
        elif self._spec.effect_type == ConsumableEffectType.POISON and health:
            damage = abs(self._spec.effect_value)
            target_actor.take_damage(damage)
            messages.append(f"Took {damage} poison damage")

        # Remove conditions in bulk if specified
        removed_conditions: list[conditions.Condition] = []
        if target_actor.conditions and self._spec.target_condition_types:
            removed_conditions = cast(
                "ConditionsComponent", target_actor.conditions
            ).remove_conditions_by_type(self._spec.target_condition_types)
            if removed_conditions:
                counts: dict[str, int] = {}
                for cond in removed_conditions:
                    name = type(cond).__name__
                    counts[name] = counts.get(name, 0) + 1
                parts = [
                    f"{count} {name}{'s' if count > 1 else ''}"
                    for name, count in counts.items()
                ]
                messages.append("Removed " + " and ".join(parts))

        if messages:
            publish_event(MessageEvent("; ".join(messages), colors.GREEN))
        else:
            publish_event(MessageEvent("Nothing happens", colors.YELLOW))

        self.uses_remaining -= 1
        return True

    @property
    def max_uses(self) -> int:
        return self._spec.max_uses


# === Ammo Pack Capability ===
class AmmoSpec:
    """Defines the type and capacity of an ammunition pack."""

    ammo_type: str
    capacity: int  # How many rounds this type of pack holds when full

    def __init__(self, ammo_type: str, capacity: int) -> None:
        self.ammo_type = ammo_type
        self.capacity = capacity


class Ammo:
    """Handles the state of a specific ammunition pack instance."""

    def __init__(self, spec: AmmoSpec) -> None:
        self._spec = spec
        self.rounds_left = spec.capacity  # This pack starts full

    def take_rounds(self, amount: int) -> int:
        taken = min(amount, self.rounds_left)
        self.rounds_left -= taken
        return taken

    @property
    def ammo_type(self) -> str:
        return self._spec.ammo_type

    @property
    def capacity(self) -> int:
        return self._spec.capacity


# === Outfit Capability ===
class OutfitSpec:
    """Defines the protection stats for an outfit (armor/clothing).

    This is the template that defines an outfit's stats. Similar to how
    MeleeAttackSpec/RangedAttackSpec work for weapons.

    Attributes:
        protection: Flat damage reduction (PR). 0 for regular clothes.
        max_ap: Maximum armor points. 0 for no-protection outfits.
    """

    def __init__(self, protection: int = 0, max_ap: int = 0) -> None:
        self.protection = protection
        self.max_ap = max_ap
