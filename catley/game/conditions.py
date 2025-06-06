import abc

from catley import colors


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


class Injury(Condition):
    """Represents a physical injury."""

    def __init__(
        self,
        injury_type: str = "Generic Injury",
        description: str = "A physical wound.",
    ):
        super().__init__(
            name=f"Injury: {injury_type}",
            description=description,
            display_color=colors.RED,
        )


class Rads(Condition):
    """Represents one slot filled by radiation sickness."""

    def __init__(self) -> None:
        """Create a radiation sickness condition."""
        super().__init__(
            name="Rads",
            description="Suffering from radiation exposure.",
            display_color=colors.YELLOW,
        )


class Sickness(Condition):
    """Represents one slot filled by a sickness like poison, venom, or disease."""

    def __init__(
        self,
        sickness_type: str = "General Sickness",
        description: str = "Afflicted by an illness.",
    ) -> None:
        """Create a sickness condition with a specific type."""
        # sickness_type could be "Poisoned", "Diseased", "Venom"
        super().__init__(
            name=f"Sickness: {sickness_type}",
            description=description,
            display_color=colors.GREEN,
        )  # A sickly green


class Exhaustion(Condition):
    """Represents one slot filled by exhaustion."""

    def __init__(self) -> None:
        """Create an exhaustion condition."""
        super().__init__(
            name="Exhaustion",
            description="Physically and mentally drained.",
            display_color=colors.LIGHT_BLUE,
        )
