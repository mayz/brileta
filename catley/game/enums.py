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


class OutcomeTier(Enum):
    """Represents the broad outcome tiers for action resolution.

    This 5-tier system provides granular results for all game actions,
    enabling varied and interesting consequences beyond simple pass/fail.

    From best to worst outcome:
    - CRITICAL_SUCCESS: Exceptional success with additional benefits
    - SUCCESS: Full successful outcome as intended
    - PARTIAL_SUCCESS: Limited success with complications or reduced effectiveness
    - FAILURE: Standard unsuccessful outcome
    - CRITICAL_FAILURE: Catastrophic failure with additional negative consequences

    This system allows actions to have nuanced outcomes
    while remaining universal across all resolution mechanics.
    """

    CRITICAL_FAILURE = auto()
    FAILURE = auto()
    PARTIAL_SUCCESS = auto()
    SUCCESS = auto()
    CRITICAL_SUCCESS = auto()
