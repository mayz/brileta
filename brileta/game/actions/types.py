#
# FILE: brileta/game/actions/types.py
#

"""Module for defining action-related types, such as animation styles.

This module establishes the core concepts for how game actions are resolved
and presented visually, forming the foundation of the PPIAS (Player-Priority,
Interruptible Animation System).

The Golden Rule of Animation & Resolution
-----------------------------------------
The `AnimationType` enum defined here is central to the game's real-time feel
and turn-based integrity. It dictates how the game loop handles player actions,
separating the mechanical outcome from the visual presentation.

Every game action falls into one of two categories, each with a strict rule
for how it is processed:

1.  `INSTANT` (The Default)
    -   **What it is:** The standard for any common, fast action.
    -   **How it works:** Input -> Resolve Instantly -> Animate Result.
        The game immediately calculates the outcome of the action (e.g., a move
        is completed, damage is dealt). The animation that follows is a purely
        cosmetic, interruptible replay of that already-decided result.
    -   **Player Experience:** The game feels instantaneous. Mashing keys
        results in rapid action resolution because the animation phase can be
        skipped.
    -   **Examples:** Movement, standard melee/ranged attacks, simple item use.

2.  `WIND_UP` (For Committed Actions)
    -   **What it is:** Used for special actions that require a commitment of
        time, where the outcome is not known until the action is complete.
    -   **How it works:** Input -> Play Wind-up Animation -> Resolve on Finish.
        The game plays an animation *first*. Only if this animation completes
        without being cancelled does the action's mechanical outcome resolve.
    -   **Player Experience:** The action feels weighty and deliberate. It
        communicates risk and commitment, as the player is locked in for a
        short period.
    -   **Examples:** Picking a lock, disarming a trap, casting a complex spell,
        using a powerful "charged" attack.
"""

from __future__ import annotations

from enum import Enum, auto


class AnimationType(Enum):
    """Specifies how an action's resolution is timed relative to its animation.

    This is the core of the PPIAS system, determining whether an action
    resolves instantly (and the animation is a cosmetic replay) or only after
    a wind-up animation completes.
    """

    INSTANT = auto()
    """
    The action's mechanical outcome is resolved instantly upon input.
    The subsequent animation is purely cosmetic and can be interrupted by the
    player's next action. This is the default for most actions.
    (e.g., Movement, basic attacks).
    """

    WIND_UP = auto()
    """
    The action must play a "wind-up" animation *before* its mechanical
    outcome is resolved. If the player interrupts the animation, the action is
    cancelled and does not occur. This is for "committed" actions like
    lockpicking or casting a complex spell.
    """
