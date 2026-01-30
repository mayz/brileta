"""Atmospheric layer data models and animation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AtmosphericLayerConfig:
    """Configuration for a single atmospheric layer (cloud shadows or mist)."""

    name: str = "unnamed"
    enabled: bool = True
    blend_mode: str = "darken"
    strength: float = 0.2
    tint_color: tuple[int, int, int] = (180, 190, 210)
    noise_scale: float = 0.08
    noise_threshold_low: float = 0.25
    noise_threshold_high: float = 0.75
    drift_direction: tuple[float, float] = (1.0, 0.3)
    # Direction the visible pattern should move in world tiles (+x = right, +y = down).
    drift_speed: float = 0.15
    # Drift speed in tiles per second (pattern motion in world space).
    turbulence_strength: float = 0.0
    turbulence_scale: float = 0.15
    turbulence_speed: float = 0.03
    disable_when_overcast: bool = False
    sky_exposure_threshold: float = 0.8


@dataclass
class AtmosphericConfig:
    """Top-level atmospheric configuration containing all layers and shared state."""

    cloud_coverage: float = 0.5
    layers: list[AtmosphericLayerConfig] = field(default_factory=list)

    @classmethod
    def create_default(cls) -> AtmosphericConfig:
        """Create config with standard cloud shadows + ground mist layers."""
        return cls(
            cloud_coverage=0.8,
            layers=[
                AtmosphericLayerConfig(
                    name="cloud_shadows",
                    enabled=True,
                    blend_mode="darken",
                    strength=0.8,
                    tint_color=(140, 150, 165),
                    noise_scale=0.03,
                    noise_threshold_low=0.46,
                    noise_threshold_high=0.62,
                    drift_direction=(1.0, 0.2),
                    drift_speed=0.01,
                    turbulence_strength=0.0,
                    disable_when_overcast=True,
                    sky_exposure_threshold=0.85,
                ),
                AtmosphericLayerConfig(
                    name="ground_mist",
                    enabled=True,
                    blend_mode="lighten",
                    strength=0.085,
                    tint_color=(238, 242, 246),
                    noise_scale=0.13,
                    noise_threshold_low=0.15,
                    noise_threshold_high=0.85,
                    drift_direction=(0.2, 1.0),
                    drift_speed=0.04,
                    turbulence_strength=0.7,
                    turbulence_scale=0.16,
                    turbulence_speed=0.03,
                    disable_when_overcast=False,
                    sky_exposure_threshold=0.85,
                ),
            ],
        )


@dataclass
class LayerAnimationState:
    """Per-layer animation state updated each frame."""

    drift_offset_x: float = 0.0
    drift_offset_y: float = 0.0
    turbulence_offset: float = 0.0


class AtmosphericLayerSystem:
    """Manages atmospheric layer animation and rendering state."""

    def __init__(self, config: AtmosphericConfig) -> None:
        self.config = config
        self._animation_states: dict[str, LayerAnimationState] = {}
        self._total_time: float = 0.0

        for layer in config.layers:
            self._animation_states[layer.name] = LayerAnimationState()

    def update(self, delta_time: float) -> None:
        """Update animation state for all layers."""
        self._total_time += delta_time

        for layer in self.config.layers:
            if not layer.enabled:
                continue

            state = self._animation_states[layer.name]

            effective_speed = layer.drift_speed
            dx, dy = layer.drift_direction
            length = math.hypot(dx, dy)
            if length > 0:
                dx, dy = dx / length, dy / length

            state.drift_offset_x += dx * effective_speed * delta_time
            state.drift_offset_y += dy * effective_speed * delta_time
            state.drift_offset_x = state.drift_offset_x % 100.0
            state.drift_offset_y = state.drift_offset_y % 100.0

            if layer.turbulence_strength > 0:
                state.turbulence_offset += layer.turbulence_speed * delta_time
                state.turbulence_offset = state.turbulence_offset % 100.0

    def get_active_layers(
        self,
    ) -> list[tuple[AtmosphericLayerConfig, LayerAnimationState]]:
        """Return list of (config, state) for layers that should render."""
        active_layers: list[tuple[AtmosphericLayerConfig, LayerAnimationState]] = []

        for layer in self.config.layers:
            if not layer.enabled:
                continue
            if layer.disable_when_overcast and self.config.cloud_coverage >= 1.0:
                continue
            active_layers.append((layer, self._animation_states[layer.name]))

        return active_layers

    def set_cloud_coverage(self, coverage: float) -> None:
        """Update cloud coverage (0.0 = clear, 1.0 = overcast)."""
        self.config.cloud_coverage = max(0.0, min(1.0, coverage))
