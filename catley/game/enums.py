from enum import Enum, auto


class ItemSize(Enum):
    """Size categories for items, used for inventory management."""

    TINY = auto()
    NORMAL = auto()
    BIG = auto()
    HUGE = auto()


class Disposition(Enum):
    HOSTILE = auto()  # Will attack/flee.
    UNFRIENDLY = auto()
    WARY = auto()
    APPROACHABLE = auto()
    FRIENDLY = auto()
    ALLY = auto()
