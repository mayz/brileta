"""Constants packages for implementation details.

These are distinct from config.py which contains user-configurable settings.
Constants here are implementation details that need descriptive names.
"""

from .combat import CombatConstants
from .physics import PhysicsConstants
from .rendering import RenderingConstants
from .ui import UIConstants

__all__ = [
    "CombatConstants",
    "PhysicsConstants",
    "RenderingConstants",
    "UIConstants",
]
