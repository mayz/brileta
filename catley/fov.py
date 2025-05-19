import tcod
from model import Model


class FieldOfView:
    def __init__(self, model: Model):
        self.fov_algorithm: int = tcod.constants.FOV_SYMMETRIC_SHADOWCAST
        self.fov_light_walls = True
        self.fov_radius = 5

        self.model = model
        self.fov_needs_recomputing = True

        m = model.game_map
        self.fov_map: tcod.map.Map = tcod.map.Map(m.width, m.height, order="F")

        for x in range(m.width):
            for y in range(m.height):
                self.fov_map.transparent[x, y] = not m.tiles[x][y].blocks_sight
                self.fov_map.walkable[x, y] = not m.tiles[x][y].blocked

    def contains(self, x: int, y: int) -> bool:
        """DOCME"""
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
