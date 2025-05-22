import abc
import random

import colors
import items
import tcod
import tcod.event
from clock import Clock
from fov import FieldOfView
from model import Actor, Entity, Model
from render import Renderer


class Controller:
    def __init__(self) -> None:
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

        # Initialize clock for frame timing
        self.clock = Clock()
        self.target_fps = 60

        # Create renderer after FOV is initialized
        self.renderer = Renderer(
            self.screen_width, self.screen_height, self.model, self.fov, self.clock
        )

        # Place NPC in a random room that's not the first room
        if len(rooms) > 1:
            npc_room = random.choice(rooms[1:])  # Skip the first room where player is
            npc_x, npc_y = npc_room.center()
            npc = Actor(
                x=npc_x,
                y=npc_y,
                ch=ord("T"),
                color=colors.RED,
                max_hp=10,
                max_ap=3,
                model=self.model,
                blocks_movement=True,
            )
            npc.equipped_weapon = items.LEAD_PIPE
            self.model.entities.append(npc)

        # Ensure initial FOV computation
        self.fov.fov_needs_recomputing = True

        # For handling input events in run_game_loop().
        self.event_handler = EventHandler(self)

    def run_game_loop(self) -> None:
        while True:
            # Process any pending events
            for event in tcod.event.get():
                self.event_handler.dispatch(event)

            # Update using clock's delta time
            delta_time = self.clock.sync(fps=self.target_fps)

            # Update animations and render
            self.model.lighting.update(delta_time)
            self.renderer.render_all()


class Action(abc.ABC):
    @abc.abstractmethod
    def execute(self) -> None:
        raise NotImplementedError()


class MoveAction(Action):
    def __init__(
        self, controller: Controller, entity: Entity, dx: int, dy: int
    ) -> None:
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
    def __init__(self, context: tcod.context.Context) -> None:
        self.context = context

    def execute(self) -> None:
        self.context.present(self.context.console, keep_aspect=True)


class QuitAction(Action):
    def execute(self) -> None:
        raise SystemExit()


class EventHandler:
    def __init__(self, controller: Controller) -> None:
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
