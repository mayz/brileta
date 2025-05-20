import abc
import random
import time

import colors
import tcod
import tcod.event
from fov import FieldOfView
from model import Entity, Model
from render import Renderer


class Controller:
    def __init__(self):
        self.screen_width = 80
        self.screen_height = 50

        self.bar_width = 20
        self.panel_height = 7
        self.panel_y = self.screen_height - self.panel_height

        self.map_width = 80
        self.map_height = 43

        self.max_room_size = 20
        self.min_room_size = 6
        self.max_num_rooms = 30

        self.model = Model(self.map_width, self.map_height)

        rooms = self.model.game_map.make_map(
            self.max_num_rooms,
            self.min_room_size,
            self.max_room_size,
            self.map_width,
            self.map_height,
        )
        first_room = rooms[0]
        self.model.player.x, self.model.player.y = first_room.center()

        # Initialize FOV after map is created but before renderer
        self.fov = FieldOfView(self.model)

        # Create renderer after FOV is initialized
        self.renderer = Renderer(
            self.screen_width, self.screen_height, self.model, self.fov
        )

        # Place NPC in a random room that's not the first room
        if len(rooms) > 1:
            npc_room = random.choice(rooms[1:])  # Skip the first room where player is
            npc_x, npc_y = npc_room.center()
            npc = Entity(
                npc_x,
                npc_y,
                ord("T"),
                colors.RED,
                model=self.model,
                blocks_movement=True,
            )
            self.model.entities.append(npc)

        # Ensure initial FOV computation
        self.fov.fov_needs_recomputing = True

        # For handling input events in run_game_loop().
        self.event_handler = EventHandler(self)

        # Add timing variables for animation
        self.last_frame_time = time.time()
        self.frame_time = 1.0 / 60  # Target 60 FPS

    def run_game_loop(self):
        while True:
            current_time = time.time()
            delta_time = current_time - self.last_frame_time

            # Handle real-time updates
            if delta_time >= self.frame_time:
                self.last_frame_time = current_time
                # Update animations and real-time effects
                self.model.lighting.update(delta_time)
                # Always render after real-time updates
                self.renderer.render_all()

            # Handle input events (turn-based)
            for event in tcod.event.get():
                self.event_handler.dispatch(event)


class Action(abc.ABC):
    @abc.abstractmethod
    def execute(self) -> None:
        raise NotImplementedError()


class MoveAction(Action):
    def __init__(self, controller: Controller, entity: Entity, dx: int, dy: int):
        self.controller = controller
        self.game_map = controller.model.game_map
        self.entity = entity

        self.dx = dx
        self.dy = dy
        self.newx = self.entity.x + self.dx
        self.newy = self.entity.y + self.dy

    def execute(self) -> None:
        if self.game_map.tiles[self.newx][self.newy].blocked:
            return

        # Check for blocking entities
        for entity in self.controller.model.entities:
            if (
                entity.blocks_movement
                and entity.x == self.newx
                and entity.y == self.newy
            ):
                return  # Cannot move into blocking entity

        self.entity.move(self.dx, self.dy)
        self.controller.fov.fov_needs_recomputing = True


class ToggleFullscreenAction(Action):
    def __init__(self, context: tcod.context.Context):
        self.context = context

    def execute(self) -> None:
        self.context.present(self.context.console, keep_aspect=True)


class QuitAction(Action):
    def execute(self) -> None:
        raise SystemExit()


class EventHandler:
    def __init__(self, controller: Controller):
        self.controller = controller
        self.game_map = controller.model.game_map
        self.p = controller.model.player

    def dispatch(self, event: tcod.event.Event) -> None:
        action = self.handle_event(event)
        if action:
            action.execute()

    def handle_event(self, event: tcod.event.Event) -> Action | None:
        match event:
            case tcod.event.Quit():
                return QuitAction()

            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.q)
            ):
                return QuitAction()

            case tcod.event.KeyDown(sym=tcod.event.KeySym.UP):
                return MoveAction(self.controller, self.p, 0, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN):
                return MoveAction(self.controller, self.p, 0, 1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT):
                return MoveAction(self.controller, self.p, -1, 0)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.RIGHT):
                return MoveAction(self.controller, self.p, 1, 0)

            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN, mod=mod) if (
                mod & tcod.event.Modifier.ALT
            ):
                return ToggleFullscreenAction(self.controller.renderer.context)

            case _:
                return None


def main() -> None:
    controller = Controller()
    controller.run_game_loop()


if __name__ == "__main__":
    main()
