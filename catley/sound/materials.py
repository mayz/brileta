"""Audio material resolution for impact sounds.

This module provides the mapping between game entities (actors, tiles) and
the impact sounds that should play when they are hit by projectiles or
melee attacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.enums import ImpactMaterial

if TYPE_CHECKING:
    from catley.game.actors import Character


# Material to sound ID mapping
IMPACT_SOUND_MAP: dict[ImpactMaterial, str] = {
    ImpactMaterial.FLESH: "impact_flesh",
    ImpactMaterial.METAL: "impact_metal",
    ImpactMaterial.STONE: "impact_stone",
    ImpactMaterial.WOOD: "impact_wood",
}


class AudioMaterialResolver:
    """Resolves impact materials for actors and tiles.

    This class provides static methods to determine what material an entity
    is made of, which in turn determines what impact sound should play when
    it is struck.
    """

    @staticmethod
    def resolve_actor_material(actor: Character) -> ImpactMaterial:
        """Get impact material for an actor.

        Currently returns FLESH for all actors. When armor items are
        implemented, this can check equipped armor material to return
        METAL for heavily armored targets.

        Args:
            actor: The character that was hit.

        Returns:
            The impact material for sound selection.
        """
        # Future: check actor.inventory for equipped armor with material property
        # For now, all biological creatures are flesh
        return ImpactMaterial.FLESH

    @staticmethod
    def resolve_tile_material(tile_type_id: int) -> ImpactMaterial:
        """Get impact material for a tile type by querying the tile data.

        Args:
            tile_type_id: The numeric ID of the tile type.

        Returns:
            The impact material for sound selection.
        """
        from catley.environment.tile_types import get_tile_material

        return get_tile_material(tile_type_id)


def get_impact_sound_id(material: ImpactMaterial) -> str:
    """Get the sound ID for an impact material.

    Args:
        material: The impact material type.

    Returns:
        The sound definition ID to use for this material.
    """
    return IMPACT_SOUND_MAP[material]
