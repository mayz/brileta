import random

from catley.types import DeltaTime


class ScreenShake:
    def __init__(self) -> None:
        """Initialize screen shake state."""
        self.intensity = 0.0
        self.duration = 0.0
        self.time_remaining = 0.0

    def trigger(self, intensity: float, duration: DeltaTime) -> None:
        """Trigger a screen shake.

        Args:
            intensity: Probability of a shake offset occurring (0.0-1.0).
            duration: Length of the shake in seconds.
        """
        # Use the strongest current shake if multiple shakes overlap
        self.intensity = max(self.intensity, intensity)
        self.duration = max(self.duration, duration)
        self.time_remaining = max(self.time_remaining, duration)

    def update(self, delta_time: DeltaTime) -> tuple[int, int]:
        """Update and return current shake offset as tile coordinates."""
        if self.time_remaining <= 0:
            return 0, 0

        self.time_remaining -= delta_time

        # Fade out over time
        fade_factor = (self.time_remaining / self.duration) ** 2
        current_intensity = self.intensity * fade_factor

        # Work in tile coordinates - either 0 or ±1 tile most of the time
        if random.random() < current_intensity:  # Use intensity as probability
            return random.choice([-1, 1]), random.choice([-1, 1])

        return 0, 0

    def is_active(self) -> bool:
        """Check if shake is currently active."""
        return self.time_remaining > 0
