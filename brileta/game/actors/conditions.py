import abc
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from . import Actor

from brileta import colors
from brileta.events import MessageEvent, publish_event
from brileta.game.enums import InjuryLocation


class Condition(abc.ABC):  # noqa: B024
    """Base class for all conditions that can affect an actor and take up
    inventory space."""

    def __init__(
        self,
        name: str,
        description: str = "",
        display_color: colors.Color = colors.ORANGE,
    ) -> None:
        self.name = name
        self.description = description
        self.display_color = display_color  # Color used in menus

    def __str__(self) -> str:
        return self.name

    def apply_to_resolution(
        self, resolution_args: dict[str, bool | str]
    ) -> dict[str, bool | str]:
        """Modify resolution arguments if this condition has an effect."""
        return resolution_args

    def get_movement_cost_modifier(self) -> float:
        """Return a movement speed multiplier for this condition."""
        return 1.0

    def apply_turn_effect(self, actor: "Actor") -> None:
        """Apply any per-turn effects of this condition."""
        return


class Injury(Condition):
    """Represents a physical injury at a specific body location."""

    def __init__(
        self,
        injury_location: InjuryLocation,
        injury_type: str = "Generic Injury",
        description: str = "A physical wound.",
    ) -> None:
        self.injury_location = injury_location
        # Condition names should be short so they fit inside the status view.
        name = injury_type

        super().__init__(
            name=name,
            description=description,
            display_color=colors.RED,
        )

    def apply_to_resolution(
        self, resolution_args: dict[str, bool | str]
    ) -> dict[str, bool | str]:
        """Apply disadvantage based on injury location and action stat."""
        stat_name = resolution_args.get("stat_name")
        if stat_name is None:
            return resolution_args
        head_hit = self.injury_location == InjuryLocation.HEAD and stat_name in {
            "intelligence",
            "observation",
        }
        torso_hit = (
            self.injury_location == InjuryLocation.TORSO and stat_name == "toughness"
        )
        arm_hit = (
            self.injury_location in {InjuryLocation.LEFT_ARM, InjuryLocation.RIGHT_ARM}
            and stat_name == "strength"
        )
        if head_hit or torso_hit or arm_hit:
            resolution_args["has_disadvantage"] = True
        return resolution_args

    def get_movement_cost_modifier(self) -> float:
        """Return movement speed multiplier from this injury."""
        if self.injury_location in {InjuryLocation.LEFT_LEG, InjuryLocation.RIGHT_LEG}:
            return 0.75
        return 1.0


class Rads(Condition):
    """Represents one slot filled by radiation sickness."""

    def __init__(self) -> None:
        """Create a radiation sickness condition."""
        super().__init__(
            name="Rads",
            description="Suffering from radiation exposure.",
            display_color=colors.YELLOW,
        )

    def apply_to_resolution(
        self, resolution_args: dict[str, bool | str]
    ) -> dict[str, bool | str]:
        """Radiation exposure causes general weakness affecting all physical actions."""
        stat_name = resolution_args.get("stat_name")
        if stat_name in {"strength", "toughness", "agility"}:
            resolution_args["has_disadvantage"] = True
        return resolution_args


class Sickness(Condition):
    """Represents one slot filled by a sickness like poison, venom, or disease.

    Sickness conditions deal damage each turn and impose stat disadvantages.
    They can optionally expire after a set number of turns (``remaining_turns``).
    When ``remaining_turns`` is ``None`` the sickness persists until cured.
    """

    # Per-turn damage and message color for each sickness type.
    _EFFECTS: ClassVar[dict[str, tuple[int, str | None, colors.Color]]] = {
        #                    dmg  damage_type   msg_color
        # damage_type=None uses Character.take_damage() default handling.
        "Poisoned": (1, None, colors.GREEN),
        "Venom": (2, None, colors.GREEN),
        "Radiation Sickness": (1, "radiation", colors.YELLOW),
    }

    def __init__(
        self,
        sickness_type: str = "General Sickness",
        description: str = "Afflicted by an illness.",
        remaining_turns: int | None = None,
    ) -> None:
        """Create a sickness condition with a specific type.

        Args:
            sickness_type: Label like "Poisoned", "Venom", or "Disease".
            description: Flavor text shown in the UI.
            remaining_turns: Turns until the sickness expires on its own.
                ``None`` means permanent (must be cured).
        """
        super().__init__(
            name=f"Sickness: {sickness_type}",
            description=description,
            display_color=colors.GREEN,
        )  # A sickly green
        self.sickness_type = sickness_type
        self.remaining_turns = remaining_turns

    def apply_turn_effect(self, actor: "Actor") -> None:
        """Apply ongoing sickness effects each turn.

        Deals type-specific damage, then decrements the remaining duration.
        When the duration reaches zero the sickness is removed automatically.
        """
        effect = self._EFFECTS.get(self.sickness_type)
        if effect is not None:
            damage, damage_type, msg_color = effect
            if damage_type is None:
                actor.take_damage(damage)
            else:
                actor.take_damage(damage, damage_type=damage_type)
            publish_event(
                MessageEvent(
                    f"{actor.name} suffers from {self.sickness_type}! (-{damage} HP)",
                    msg_color,
                )
            )

        # Tick down duration and auto-remove when expired.
        if self.remaining_turns is not None:
            self.remaining_turns -= 1
            if self.remaining_turns <= 0 and actor.conditions is not None:
                actor.conditions.remove_condition(self)
                publish_event(
                    MessageEvent(
                        f"{actor.name} recovers from {self.sickness_type}.",
                        colors.WHITE,
                    )
                )

    def apply_to_resolution(
        self, resolution_args: dict[str, bool | str]
    ) -> dict[str, bool | str]:
        """Apply sickness penalties to action resolution."""
        stat_name = resolution_args.get("stat_name")
        if stat_name is None:
            return resolution_args

        disadvantaged = (
            (self.sickness_type == "Poisoned" and stat_name == "toughness")
            or (self.sickness_type == "Venom" and stat_name == "agility")
            or (
                self.sickness_type == "Disease"
                and stat_name in {"strength", "toughness", "agility"}
            )
            or (
                self.sickness_type == "Radiation Sickness"
                and stat_name == "intelligence"
            )
        )

        if disadvantaged:
            resolution_args["has_disadvantage"] = True
        return resolution_args


class Exhaustion(Condition):
    """Represents one slot filled by exhaustion."""

    def __init__(self) -> None:
        """Create an exhaustion condition."""
        super().__init__(
            name="Exhaustion",
            description="Physically and mentally drained.",
            display_color=colors.LIGHT_BLUE,
        )
