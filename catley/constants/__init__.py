"""Constants packages for implementation details.

These are distinct from config.py which contains user-configurable settings.
Constants here are implementation details that need descriptive names.
"""

from .combat import CombatConstants
from .view import ViewConstants

__all__ = [
    "CombatConstants",
    "ViewConstants",
]
