"""Constants for movement and exhaustion systems."""


class MovementConstants:
    """Constants for movement and exhaustion systems."""

    # Exhaustion penalty rates
    EXHAUSTION_SPEED_REDUCTION_PER_STACK = (
        0.9  # 10% speed reduction per exhaustion stack
    )
    EXHAUSTION_ENERGY_REDUCTION_PER_STACK = (
        0.9  # 10% energy reduction per exhaustion stack
    )

    # Encumbrance speed penalty
    # Speed multiplier per slot over capacity: 0.85^slots_over
    # 1 over: 0.85, 2 over: 0.72, 3 over: 0.61, 4 over: 0.52
    ENCUMBRANCE_SPEED_BASE = 0.85
