import random

import actions
import colors
import conditions
import items
import tcod
import tcod.event
from clock import Clock
from fov import FieldOfView
from menu_system import MenuSystem
from message_log import MessageLog
from model import Actor, Model
from render import Renderer


class Controller:
    def __init__(self) -> None:
        self.screen_width = 80
        self.screen_height = 50

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
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            model=self.model,
            fov=self.fov,
            clock=self.clock,
            message_log=self.message_log,
            menu_system=self.menu_system,  # Pass menu system to renderer
        )

        self.model.player.inventory.append(
            conditions.Injury(injury_type="Sprained Ankle")
        )
        self.model.player.inventory.append(items.DAGGER)

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
        npc = Actor(
            x=npc_x,
            y=npc_y,
            ch="T",
            name="Trog",
            color=colors.RED,
            max_hp=10,
            max_ap=0,  # Hulkification = too big for armor
            model=self.model,
            blocks_movement=True,
        )
        npc.equipped_weapon = items.LEAD_PIPE
        self.model.entities.append(npc)

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


class EventHandler:
    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.game_map = controller.model.game_map
        self.p = controller.model.player

    def dispatch(self, event: tcod.event.Event) -> None:
        # First, try to handle the event with the menu system
        if self.controller.menu_system.handle_input(event):
            return  # Event was consumed by menu system

        # If no menu handled it, process normal game actions
        action = self.handle_event(event)
        if action:
            try:
                action.execute()
            except SystemExit:  # Allow SystemExit to propagate for quitting
                raise
            except Exception as e:
                error_message = f"Error: {str(e)}"
                self.controller.message_log.add_message(error_message, colors.RED)
                print(f"Unhandled exception during action execution: {e}")

    def handle_event(self, event: tcod.event.Event) -> actions.Action | None:
        # Don't process game actions if menus are active
        if self.controller.menu_system.has_active_menus():
            return None

        match event:
            case tcod.event.Quit():
                return actions.QuitAction()

            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.ESCAPE)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.q)
            ):
                return actions.QuitAction()

            # Movement keys (Arrows and VIM)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.UP)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.k)
            ):
                return actions.MoveAction(self.controller, self.p, 0, -1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.DOWN)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.j)
            ):
                return actions.MoveAction(self.controller, self.p, 0, 1)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.LEFT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.h)
            ):
                return actions.MoveAction(self.controller, self.p, -1, 0)
            case (
                tcod.event.KeyDown(sym=tcod.event.KeySym.RIGHT)
                | tcod.event.KeyDown(sym=tcod.event.KeySym.l)
            ):
                return actions.MoveAction(self.controller, self.p, 1, 0)

            # Diagonal movement (numpad or vi-keys)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_7):  # Up-left
                return actions.MoveAction(self.controller, self.p, -1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_9):  # Up-right
                return actions.MoveAction(self.controller, self.p, 1, -1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_1):  # Down-left
                return actions.MoveAction(self.controller, self.p, -1, 1)
            case tcod.event.KeyDown(sym=tcod.event.KeySym.KP_3):  # Down-right
                return actions.MoveAction(self.controller, self.p, 1, 1)

            # Menu keys
            case tcod.event.KeyDown(sym=tcod.event.KeySym.i):
                # Quick access to inventory
                from menu_system import InventoryMenu

                inventory_menu = InventoryMenu(self.controller)
                self.controller.menu_system.show_menu(inventory_menu)
                return None

            # Help Menu: '?'
            # tcod.event.KeySym.QUESTION is typically generated for '?'
            # This pattern handles '?' (if sym is KeySym.QUESTION),
            # or '?' (if sym is KeySym.SLASH and SHIFT modifier is present).
            case tcod.event.KeyDown(sym=key_sym, mod=key_mod) if (
                key_sym == tcod.event.KeySym.QUESTION
                or (
                    key_sym == tcod.event.KeySym.SLASH
                    and (key_mod & tcod.event.Modifier.SHIFT)
                )
            ):
                from menu_system import HelpMenu

                help_menu = HelpMenu(self.controller)
                self.controller.menu_system.show_menu(help_menu)
                return None

            case tcod.event.KeyDown(sym=tcod.event.KeySym.g):
                player_x, player_y = self.p.x, self.p.y
                if self.controller.model.has_pickable_items_at_location(
                    player_x, player_y
                ):
                    from menu_system import PickupMenu

                    pickup_menu = PickupMenu(self.controller, (player_x, player_y))
                    self.controller.menu_system.show_menu(pickup_menu)
                # If no items, pressing 'g' does nothing.
                return None

            case tcod.event.KeyDown(sym=tcod.event.KeySym.RETURN, mod=mod) if (
                mod & tcod.event.Modifier.ALT
            ):
                return actions.ToggleFullscreenAction(self.controller.renderer.context)

            case _:
                return None


def main() -> None:
    controller = Controller()
    controller.run_game_loop()


if __name__ == "__main__":
    main()
