import tcod
import tcod.constants

from .model import Model


class FieldOfView:
    def __init__(self, model: Model) -> None:
        self.fov_algorithm: int = tcod.constants.FOV_SYMMETRIC_SHADOWCAST
        self.fov_light_walls = True
        self.fov_radius = 15

        self.model = model
        self.fov_needs_recomputing = True

        m = model.game_map
        self.fov_map: tcod.map.Map = tcod.map.Map(m.width, m.height, order="F")

        # Using ~ for logical NOT
        self.fov_map.walkable[...] = ~m.tile_blocked
        self.fov_map.transparent[...] = ~m.tile_blocks_sight

    def contains(self, x: int, y: int) -> bool:
        return self.fov_map.fov[x, y]

    def recompute_if_needed(self) -> bool:
        """Recompute the FOV if the 'fov_needs_recomputing' field is True.

        After recomputing, the field is reset to False.

        Returns:
          Whether the FOV was recomputed.
        """
        if not self.fov_needs_recomputing:
            return False

        # Recompute the FOV map.
        p = self.model.player
        self.fov_map.compute_fov(
            p.x, p.y, self.fov_radius, self.fov_light_walls, self.fov_algorithm
        )

        self.fov_needs_recomputing = False
        return True
