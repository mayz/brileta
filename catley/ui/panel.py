import abc


class Panel(abc.ABC):
    """Abstract base class for UI panels."""

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.visible = True

    @abc.abstractmethod
    def draw(self) -> None:
        """Draw this panel using resources provided in constructor."""
        pass

    def set_position(self, x: int, y: int) -> None:
        """Update panel position."""
        self.x = x
        self.y = y

    def set_size(self, width: int, height: int) -> None:
        """Update panel size."""
        self.width = width
        self.height = height

    def show(self) -> None:
        """Make panel visible."""
        self.visible = True

    def hide(self) -> None:
        """Make panel invisible."""
        self.visible = False
