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

    # --- Approach movement curve ---
    # Duration slows as the actor gets closer, making long-range travel snappy
    # and near-target movement deliberate.
    APPROACH_MIN_DURATION_MS = 70
    APPROACH_MAX_DURATION_MS = 140
    APPROACH_FAR_DISTANCE = 4  # Distance at which movement is fastest
    APPROACH_DISTANCE_SPAN = 3  # Range over which the curve interpolates
