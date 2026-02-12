"""Goal-backed NPC behaviors.

Each module pairs an action (the utility-system entry point) with its
corresponding goal (the multi-turn state machine).
"""

from .flee import FleeAction, FleeGoal
from .patrol import PatrolAction, PatrolGoal
from .wander import WanderAction, WanderGoal

__all__ = [
    "FleeAction",
    "FleeGoal",
    "PatrolAction",
    "PatrolGoal",
    "WanderAction",
    "WanderGoal",
]
