"""A system to control time and framerate."""

import statistics
import time
from collections import deque

from brileta.types import DeltaTime

# Number of frame time samples to track.
_FPS_SAMPLE_SIZE = 256


class Clock:
    """Measure framerate performance and sync to a given framerate."""

    def __init__(self) -> None:
        self.last_time = time.perf_counter()
        self.last_delta_time: DeltaTime = DeltaTime(0.0)
        self.time_samples: deque[float] = deque(maxlen=_FPS_SAMPLE_SIZE)
        self.drift_time = 0.0

    def tick(self) -> DeltaTime:
        """
        Measures the time since the last tick, updates FPS, and returns the delta.

        This method is for event-driven loops that handle their own
        timing and do not need the clock to enforce an FPS cap by sleeping.
        """
        current_time = time.perf_counter()
        delta_time = DeltaTime(max(0, current_time - self.last_time))
        self.last_time = current_time
        self.last_delta_time = delta_time
        self.time_samples.append(delta_time)
        return delta_time

    def sync(self, fps: float | None = None) -> DeltaTime:
        """
        Sync to a given framerate and return the delta time.

        This method is for polling-based loops that need the clock to enforce
        an FPS cap by sleeping. It then calls tick() to perform
        the time measurement.
        """
        if fps is not None and fps > 0:
            desired_frame_time = 1 / fps
            target_time = self.last_time + desired_frame_time - self.drift_time
            sleep_time = max(0, target_time - time.perf_counter() - 0.001)
            if sleep_time:
                time.sleep(sleep_time)
            # busy-wait for the remaining time
            while (drift_time := time.perf_counter() - target_time) < 0:
                pass
            self.drift_time = min(drift_time, desired_frame_time)

        # After potentially sleeping, call tick() to do the actual time measurement.
        return self.tick()

    @property
    def last_fps(self) -> float:
        """The FPS of the most recent frame."""
        if not self.time_samples or self.time_samples[-1] == 0:
            return 0
        return 1 / self.time_samples[-1]

    @property
    def mean_fps(self) -> float:
        """The FPS of the sampled frames overall."""
        if not self.time_samples:
            return 0
        try:
            return 1 / statistics.fmean(self.time_samples)
        except ZeroDivisionError:
            return 0
