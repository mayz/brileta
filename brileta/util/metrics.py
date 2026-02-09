import abc

import numpy as np

from brileta.util import rng

_rng = rng.get("util.metrics")


class StatsVar(abc.ABC):
    """Abstract base class for tracking statistics of a variable over time."""

    @abc.abstractmethod
    def record(self, value: float) -> None:
        """Record a new sample value."""
        pass

    @property
    @abc.abstractmethod
    def sample_count(self) -> int:
        pass

    @abc.abstractmethod
    def _get_valid_samples(self) -> np.ndarray:
        """Get the valid samples as a numpy array. Subclasses must implement this."""
        pass

    @property
    def p50(self) -> float:
        """Get the 50th percentile (median) value."""
        return self.get_percentiles()[0]

    @property
    def p95(self) -> float:
        """Get the 95th percentile value."""
        return self.get_percentiles()[1]

    @property
    def p99(self) -> float:
        """Get the 99th percentile value."""
        return self.get_percentiles()[2]

    def get_percentiles(self) -> tuple[float, float, float]:
        """Return (p50, p95, p99) as a tuple of floats."""
        valid = self._get_valid_samples()
        if len(valid) == 0:
            return (0.0, 0.0, 0.0)

        p50, p95, p99 = np.percentile(valid, [50, 95, 99])
        return (float(p50), float(p95), float(p99))

    def get_percentiles_string(self) -> str:
        p50, p95, p99 = self.get_percentiles()
        return f"p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}"


class MostRecentNVar(StatsVar):
    """Track statistics for the most recent N samples."""

    def __init__(self, num_samples: int = 1000) -> None:
        self.num_samples = num_samples
        self.samples = np.zeros(num_samples, dtype=np.float32)
        self.count = 0
        self.write_index = 0

    def record(self, value: float):
        """Record a new sample value."""
        self.samples[self.write_index] = value
        self.write_index = (self.write_index + 1) % self.num_samples
        self.count += 1

    def _get_valid_samples(self) -> np.ndarray:
        """Get the valid samples as a numpy array."""
        if self.count <= self.num_samples:
            # Haven't wrapped yet
            return self.samples[: self.count]

        # Have wrapped - need to reconstruct order
        return np.concatenate(
            [self.samples[self.write_index :], self.samples[: self.write_index]]
        )

    @property
    def sample_count(self) -> int:
        return min(self.count, self.num_samples)


class CumulativeVar(StatsVar):
    """Track statistics for all-time."""

    def __init__(self, num_samples: int = 1000) -> None:
        self.num_samples = num_samples
        self.samples = np.zeros(num_samples, dtype=np.float32)
        self.count = 0
        self.sum_value = 0.0

    def record(self, value: float):
        # Update aggregate stats
        self.count += 1
        self.sum_value += value

        # Reservoir sampling
        if self.count <= self.num_samples:
            self.samples[self.count - 1] = value
        else:
            # Random replacement
            j = _rng.randint(0, self.count)
            if j < self.num_samples:
                self.samples[j] = value

    def _get_valid_samples(self) -> np.ndarray:
        """Get the valid samples as a numpy array."""
        return self.samples[: min(self.count, self.num_samples)]

    @property
    def sample_count(self) -> int:
        return self.count
