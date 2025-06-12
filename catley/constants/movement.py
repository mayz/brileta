"""Constants for movement and exhaustion systems."""


class MovementConstants:
    """Constants for movement and exhaustion systems."""

    # Exhaustion stumbling thresholds
    EXHAUSTION_STUMBLE_THRESHOLD = 0.7  # 30% speed reduction threshold for stumbling
    EXHAUSTION_STUMBLE_MULTIPLIER = 0.3  # Scaling factor for stumble chance calculation

    # Exhaustion penalty rates
    EXHAUSTION_SPEED_REDUCTION_PER_STACK = (
        0.9  # 10% speed reduction per exhaustion stack
    )
    EXHAUSTION_ENERGY_REDUCTION_PER_STACK = (
        0.9  # 10% energy reduction per exhaustion stack
    )
