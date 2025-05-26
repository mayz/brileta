import random

import colors
import conditions
import items
import tcod.event
from clock import Clock
from event_handler import EventHandler
from fov import FieldOfView
from menu_system import MenuSystem
from message_log import MessageLog
from model import Model, WastoidActor
from render import Renderer
from tcod.console import Console
from tcod.context import Context


class Controller:
    """
    Orchestrates the main game loop and connects all other systems
    (Model, Renderer, EventHandler, MenuSystem, MessageLog).

    Holds instances of the major game components and provides a central point
    of access for them. Responsible for high-level game flow, such as processing
    player actions and then triggering updates for all other game entities.
    """

    def __init__(self, context: Context, root_console: Console) -> None:
        self.context = context
        self.root_console = root_console

        self.help_height = 1  # One line for help text

        self.map_width = 80
        self.map_height = 43

        self.max_room_size = 20
        self.min_room_size = 6
        self.max_num_rooms = 3

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

        # Initialize Message Log
        self.message_log = MessageLog()

        # Initialize FOV after map is created but before renderer
        self.fov = FieldOfView(self.model)

        # Initialize clock for frame timing
        self.clock = Clock()
        self.target_fps = 60

        # Initialize menu system
        self.menu_system = MenuSystem(self)

        # Create renderer after FOV is initialized
        self.renderer = Renderer(
            screen_width=self.root_console.width,
            screen_height=self.root_console.height,
            model=self.model,
            fov=self.fov,
            clock=self.clock,
            message_log=self.message_log,
            menu_system=self.menu_system,
            context=self.context,
            root_console=self.root_console,
        )

        self.model.player.inventory.append(
            conditions.Injury(injury_type="Sprained Ankle")
        )
        self.model.player.inventory.append(items.SLEDGEHAMMER)

        self._add_npc()

        # Ensure initial FOV computation
        self.fov.fov_needs_recomputing = True

        # For handling input events in run_game_loop().
        self.event_handler = EventHandler(self)

    def _add_npc(self) -> None:
        """Add an NPC to the map."""
        # Place NPC right next to the player.
        npc_x = self.model.player.x
        npc_y = self.model.player.y
        choice = random.randint(1, 4)
        match choice:
            case 1:
                npc_x += 2
            case 2:
                npc_x -= 2
            case 3:
                npc_y += 2
            case 4:
                npc_y -= 2
        npc = WastoidActor(
            x=npc_x,
            y=npc_y,
            ch="T",
            name="Trog",
            color=colors.DARK_GREY,
            model=self.model,
            blocks_movement=True,
            weirdness=3,
            strength=3,
            toughness=3,
            intelligence=-3,
        )
        npc.equipped_weapon = items.SLEDGEHAMMER
        self.model.entities.append(npc)

    def run_game_loop(self) -> None:
        while True:
            # Process any pending events
            for event in tcod.event.get():
                event_with_tile_coords = self.context.convert_event(event)
                self.event_handler.dispatch(event_with_tile_coords)

            # Update using clock's delta time
            delta_time = self.clock.sync(fps=self.target_fps)

            # Update animations and render
            self.model.lighting.update(delta_time)
            self.renderer.render_all()
