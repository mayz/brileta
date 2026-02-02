"""Abstract base class for generation layers.

Each layer in the pipeline implements the GenerationLayer interface and
transforms the GenerationContext in some way - adding tiles, regions,
buildings, or other map features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import GenerationContext


class GenerationLayer(ABC):
    """Abstract base class for map generation layers.

    Layers are applied sequentially by the PipelineGenerator. Each layer
    receives a GenerationContext and modifies it in place.

    Subclasses must implement the apply() method to perform their specific
    generation logic.
    """

    @abstractmethod
    def apply(self, ctx: GenerationContext) -> None:
        """Apply this layer's generation logic to the context.

        This method should modify the context in place. It may:
        - Modify tiles (ctx.tiles)
        - Add/modify regions (ctx.regions, ctx.tile_to_region_id)
        - Add buildings (ctx.buildings)
        - Use ctx.rng for random decisions

        Args:
            ctx: The generation context to modify.
        """
        raise NotImplementedError
