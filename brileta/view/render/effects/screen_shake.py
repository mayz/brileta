import math

from brileta.types import DeltaTime


class ScreenShake:
    """Smooth sub-tile screen shake using multi-frequency sine wave oscillation.

    The shake amplitude (intensity) controls how far the screen moves in tiles,
    while the oscillation frequencies create organic, non-repetitive motion.
    """

    # Oscillation frequencies in Hz
    PRIMARY_FREQ = 18.0  # Main oscillation - fast enough to feel impactful
    SECONDARY_FREQ = 24.0  # Adds texture without dominating

    # Phase offset between X and Y axes for 2D motion (radians)
    PHASE_OFFSET = math.pi / 3  # 60 degrees - creates elliptical motion

    def __init__(self) -> None:
        """Initialize screen shake state."""
        self.intensity = 0.0  # Maximum amplitude in tiles
        self.duration = 0.0
        self.time_remaining = 0.0
        self.elapsed_time = 0.0  # Tracks time for oscillation continuity

    def trigger(self, intensity: float, duration: DeltaTime) -> None:
        """Trigger a screen shake.

        Args:
            intensity: Maximum amplitude of shake in tiles (0.0-0.3 typical).
            duration: Length of the shake in seconds.
        """
        # Use the strongest current shake if multiple shakes overlap
        self.intensity = max(self.intensity, intensity)
        self.duration = max(self.duration, duration)
        self.time_remaining = max(self.time_remaining, duration)
        # Don't reset elapsed_time - continuous oscillation feels more natural

    def update(self, delta_time: DeltaTime) -> tuple[float, float]:
        """Update and return current shake offset as float tile coordinates.

        Uses multi-frequency sine waves for smooth, organic motion that
        fades out quadratically over the shake duration.
        """
        if self.time_remaining <= 0:
            return 0.0, 0.0

        self.time_remaining -= delta_time
        self.elapsed_time += delta_time

        # Check again after decrementing - shake may have just expired
        if self.time_remaining <= 0:
            return 0.0, 0.0

        # Quadratic fade-out for natural decay
        fade_factor = (self.time_remaining / self.duration) ** 2
        amplitude = self.intensity * fade_factor

        # Multi-frequency oscillation: primary wave + secondary texture
        t = self.elapsed_time * 2.0 * math.pi

        # X axis: blend of primary and secondary frequencies
        offset_x = amplitude * (
            0.7 * math.sin(self.PRIMARY_FREQ * t)
            + 0.3 * math.sin(self.SECONDARY_FREQ * t * 1.3)
        )

        # Y axis: same frequencies but phase-shifted for 2D motion
        offset_y = amplitude * (
            0.7 * math.sin(self.PRIMARY_FREQ * t + self.PHASE_OFFSET)
            + 0.3 * math.sin(self.SECONDARY_FREQ * t * 1.1 + self.PHASE_OFFSET)
        )

        return offset_x, offset_y

    def is_active(self) -> bool:
        """Check if shake is currently active."""
        return self.time_remaining > 0
