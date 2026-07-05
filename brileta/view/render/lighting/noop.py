"""A lighting system that does nothing.

Used in headless contexts (tests, the sim harness, any run with no real GPU to
render to) where the game's semantic state must advance but nothing is drawn.
Constructing a real :class:`~brileta.backends.wgpu.gpu_lighting.GPULightingSystem`
spins up a wgpu render pipeline (adapter request, shader compilation) that costs
hundreds of milliseconds and produces pixels no one looks at. This satisfies the
:class:`LightingSystem` contract with pure no-ops so the world runs without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta.view.render.lighting.base import LightingSystem

if TYPE_CHECKING:
    import numpy as np

    from brileta.game.game_world import GameWorld
    from brileta.game.lights import LightSource
    from brileta.types import FixedTimestep
    from brileta.util.coordinates import Rect


class NoOpLightingSystem(LightingSystem):
    """Satisfies the LightingSystem interface without computing any lighting."""

    def __init__(self, game_world: GameWorld) -> None:
        super().__init__(game_world)

    def update(self, fixed_timestep: FixedTimestep) -> None:
        return

    def compute_lightmap(self, viewport_bounds: Rect) -> np.ndarray | None:
        # No lighting data; callers treat None as "nothing to composite".
        return None

    def on_light_added(self, light: LightSource) -> None:
        return

    def on_light_removed(self, light: LightSource) -> None:
        return

    def on_light_moved(self, light: LightSource) -> None:
        return
