from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Self, cast
from unittest.mock import MagicMock, patch

import numpy as np

from brileta import colors, config
from brileta.app import App
from brileta.controller import Controller, GameWorld
from brileta.environment.generators import GeneratedMapData
from brileta.environment.map import GameMap, MapRegion
from brileta.environment.tile_types import TileTypeID
from brileta.game.actors import NPC, PC, Actor, Character
from brileta.game.enums import ItemSize
from brileta.game.item_spawner import ItemSpawner
from brileta.game.items.item_core import Item, ItemType
from brileta.game.turn_manager import TurnManager
from brileta.types import ActorId, DeltaTime, WorldTileCoord
from brileta.util.spatial import SpatialHashGrid
from brileta.view.render.graphics import GraphicsContext


class DummyOverlay:
    """Minimal overlay stub that matches OverlaySystem expectations."""

    def __init__(self, *, interactive: bool = False) -> None:
        self.is_active = False
        self.is_interactive = interactive

    def show(self) -> None:
        self.is_active = True

    def hide(self) -> None:
        self.is_active = False

    def handle_input(self, _event: object) -> bool:
        return False

    def draw(self) -> None:
        return None

    def present(self) -> None:
        return None

    def invalidate(self) -> None:
        return None


def dt(seconds: float) -> DeltaTime:
    """Build a ``DeltaTime`` value for test call sites."""
    return DeltaTime(seconds)


def reset_actor_id_counter(start: int = 1) -> None:
    """Reset Actor._next_actor_id for test isolation."""
    Actor._next_actor_id = ActorId(start)


def make_item(name: str = "Test Item", size: ItemSize = ItemSize.NORMAL) -> Item:
    """Create a simple test item with the given name and size."""
    return Item(ItemType(name=name, description="A test item", size=size))


def _make_renderer(tile_height: int = 16) -> GraphicsContext:
    """Create a mock GraphicsContext for testing UI components."""
    renderer = MagicMock(spec=GraphicsContext)
    renderer.tile_dimensions = (8, tile_height)
    renderer.console_width_tiles = 80
    renderer.console_height_tiles = 50
    renderer.sdl_renderer = MagicMock()
    renderer.root_console = MagicMock()

    # Mock console_render - return new object each time
    renderer.console_render = MagicMock()
    renderer.console_render.render.side_effect = lambda console: MagicMock()

    # Mock upload_texture - return new object each time
    renderer.sdl_renderer.upload_texture.side_effect = lambda pixels: MagicMock()

    return renderer


class DummyGameWorld(GameWorld):
    """A lightweight, standalone dummy GameWorld for testing."""

    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        *,
        game_map: GameMap | None = None,
        actors: Sequence[Actor] | None = None,
        create_player: bool = False,
        **kwargs: object,
    ) -> None:
        # Avoid heavy GameWorld initialization.
        if game_map is None:
            tiles = np.full(
                (width, height),
                TileTypeID.FLOOR,
                dtype=np.uint8,
                order="F",
            )
            regions: dict[int, MapRegion] = {}
            tile_to_region_id = np.full((width, height), -1, dtype=np.int16, order="F")
            map_data = GeneratedMapData(
                tiles=tiles, regions=regions, tile_to_region_id=tile_to_region_id
            )
            game_map = GameMap(width, height, map_data)
            game_map.visible[:] = True
            game_map.transparent[:] = True
        self.game_map = game_map
        self.game_map.gw = self

        self.actor_spatial_index = SpatialHashGrid(cell_size=16)
        self.actors: list[Actor] = []
        # Registry for O(1) actor lookup by actor_id.
        self._actor_id_registry: dict[ActorId, Actor] = {}

        self.item_spawner = ItemSpawner(self)

        # Add initial actors through the proper lifecycle method.
        if actors:
            for actor in actors:
                self.add_actor(actor)

        self.player: Character | None = None
        self.selected_actor: Actor | None = None
        self.items: dict[tuple[int, int], list] = {}

        # New lighting system architecture - Phase 1 scaffolding
        self.lights: list = []
        self.lighting_system = None

        # Mouse position for hover tracking
        self.mouse_tile_location_on_map: tuple[int, int] | None = None

        if create_player:
            player = PC(0, 0, "@", colors.WHITE, "Player", game_world=self)
            self.player = player
            self.add_actor(player)

    def add_actor(self, actor: Actor) -> None:
        """Adds an actor to the list and the spatial index."""
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)
        self._actor_id_registry[actor.actor_id] = actor

    def remove_actor(self, actor: Actor) -> None:
        """Removes an actor from the list and the spatial index."""
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            pass
        self._actor_id_registry.pop(actor.actor_id, None)

    def get_actor_by_id(self, actor_id: ActorId) -> Actor | None:
        """Look up an actor by its ``actor_id`` in O(1) time."""
        return self._actor_id_registry.get(actor_id)

    def get_pickable_items_at_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> list:
        """Return items stored at ``(x, y)``."""
        return self.items.get((x, y), [])

    def get_actor_at_location(
        self, x: WorldTileCoord, y: WorldTileCoord
    ) -> Actor | None:
        """Return an actor at a location, prioritizing blockers."""
        actors_at_point = self.actor_spatial_index.get_at_point(x, y)
        if not actors_at_point:
            return None
        for actor in actors_at_point:
            if getattr(actor, "blocks_movement", False):
                return actor
        return actors_at_point[0]

    def spawn_ground_item(
        self, item: Item, x: WorldTileCoord, y: WorldTileCoord, **kwargs
    ) -> Actor:
        return self.item_spawner.spawn_item(item, x, y, **kwargs)

    def spawn_ground_items(
        self, items: list[Item], x: WorldTileCoord, y: WorldTileCoord
    ) -> Actor:
        return self.item_spawner.spawn_multiple(items, x, y)

    def add_light(self, light) -> None:
        """Add a light source to the world."""
        self.lights.append(light)
        if self.lighting_system is not None:
            self.lighting_system.on_light_added(light)

    def remove_light(self, light) -> None:
        """Remove a light source from the world."""
        try:
            self.lights.remove(light)
            if self.lighting_system is not None:
                self.lighting_system.on_light_removed(light)
        except ValueError:
            pass

    def get_global_lights(self) -> list:
        """Return global lights for controller sun helpers."""
        from brileta.game.lights import GlobalLight

        return [light for light in self.lights if isinstance(light, GlobalLight)]


@dataclass
class DummyController(Controller):
    """Lightweight controller for AI tests that bypasses full Controller init."""

    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.action_cost = 100


def make_ai_world(
    npc_x: int = 3,
    npc_y: int = 0,
    npc_hp_damage: int = 0,
    disposition: int = -75,
    map_size: int = 80,
) -> tuple[DummyController, Character, NPC]:
    """Create a test world with player at origin and NPC at given position."""
    gw = DummyGameWorld(width=map_size, height=map_size)
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        npc_x,
        npc_y,
        "g",
        colors.RED,
        "Enemy",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    if disposition != 0:
        npc.ai.modify_disposition(player, disposition)
    controller = DummyController(gw)

    if npc_hp_damage > 0:
        npc.take_damage(npc_hp_damage)

    return controller, player, npc


def get_controller_with_player_and_map() -> Controller:
    """Return a fully initialized ``Controller`` using dummy SDL context."""
    return _build_dummy_controller(use_dummy_world=False)


def get_controller_with_dummy_world() -> Controller:
    """Return a Controller with a DummyGameWorld and no GPU lighting system."""
    return _build_dummy_controller(use_dummy_world=True)


class DummyApp(App):
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def run(self) -> None:  # pragma: no cover - stub
        pass

    def prepare_for_new_frame(self) -> None:  # pragma: no cover - stub
        pass

    def present_frame(self) -> None:  # pragma: no cover - stub
        pass

    def toggle_fullscreen(self) -> None:  # pragma: no cover - stub
        pass

    def _exit_backend(self) -> None:  # pragma: no cover - stub
        pass


class DummyGraphicsContext:
    def __init__(self, *_args, **_kwargs) -> None:
        self._coordinate_converter = SimpleNamespace(
            pixel_to_tile=lambda x, y: (x, y), tile_to_pixel=lambda x, y: (x, y)
        )
        self.resource_manager = None
        self.root_console = SimpleNamespace(
            width=config.SCREEN_WIDTH, height=config.SCREEN_HEIGHT
        )

    def create_canvas(self, transparent: bool = True) -> Any:
        return MagicMock()

    @property
    def coordinate_converter(self):
        return self._coordinate_converter

    @property
    def tile_dimensions(self):
        return (16, 16)

    @property
    def console_width_tiles(self):
        return config.SCREEN_WIDTH

    @property
    def console_height_tiles(self):
        return config.SCREEN_HEIGHT

    def clear_console(self, *_a, **_kw) -> None:  # pragma: no cover - stub
        pass

    def blit_console(self, *_a, **_kw) -> None:  # pragma: no cover - stub
        pass

    def present_frame(self, *_a, **_kw) -> None:  # pragma: no cover - stub
        pass

    # Add minimal implementations for all abstract methods
    def get_display_scale_factor(self):
        return (1.0, 1.0)

    def render_effects_layer(self, *_a, **_kw):
        pass

    def render_particles(self, *_a, **_kw):
        pass

    def update_dimensions(self):
        pass

    def console_to_screen_coords(self, *_a):
        return (0, 0)

    def screen_to_console_coords(self, *_a):
        return (0, 0)

    def draw_actor_smooth(self, *_a, **_kw):
        pass

    def draw_particles_smooth(self, *_a, **_kw):
        pass

    def draw_texture_at_screen_pos(self, *_a, **_kw):
        pass

    def draw_texture_at_tile_pos(self, *_a, **_kw):
        pass

    def texture_from_console(self, *_a, **_kw):
        return None

    def render_glyph_buffer_to_texture(self, *_a, **_kw):
        return None

    def texture_from_numpy(self, *_a, **_kw):
        return None

    def texture_from_surface(self, *_a, **_kw):
        return None

    def reset_texture_mods(self, *_a, **_kw):
        pass

    def draw_debug_rect(self, *_a, **_kw):
        pass

    def release_texture(self, *_a, **_kw):
        pass


class DummyFrameManager:
    def __init__(self, controller: Controller, graphics: Any = None) -> None:
        self.controller = controller
        self.graphics = graphics or controller.graphics
        self.cursor_manager = SimpleNamespace(
            mouse_pixel_x=0,
            mouse_pixel_y=0,
            update_mouse_position=lambda *_a, **_kw: None,
            set_active_cursor_type=lambda *_a, **_kw: None,
        )
        self.action_panel_view = SimpleNamespace(
            x=0,
            y=0,
            width=0,
            height=0,
            get_hotkeys=lambda: self._build_hotkeys(),
            update_hover_from_pixel=lambda *_a, **_kw: False,
            execute_at_pixel=lambda *_a, **_kw: False,
            get_action_at_pixel=lambda *_a, **_kw: None,
            invalidate_cache=lambda: None,
        )
        self.equipment_view = SimpleNamespace(
            x=0,
            y=0,
            width=0,
            height=0,
            set_hover_row=lambda *_a, **_kw: None,
            is_row_in_active_slot=lambda *_a, **_kw: False,
            handle_click=lambda *_a, **_kw: False,
        )
        self.dev_console_overlay = DummyOverlay()
        self.combat_tooltip_overlay = DummyOverlay()
        self.world_view = SimpleNamespace(
            _render_selection_and_hover_outlines=lambda: None
        )

    def _build_hotkeys(self) -> dict[str, Any]:
        """Build a-z hotkey mappings from available combat actions."""
        combat_mode = getattr(self.controller, "combat_mode", None)
        if combat_mode is None:
            return {}

        actions = combat_mode.get_available_combat_actions()
        hotkey_chars = "abcdefghijklmnopqrstuvwxyz"
        return {
            hotkey_chars[i]: action
            for i, action in enumerate(actions)
            if i < len(hotkey_chars)
        }

    def get_visible_bounds(self) -> None:
        return None

    def get_world_coords_from_root_tile_coords(
        self, root_tile_pos: tuple[int, int]
    ) -> tuple[int, int]:
        return root_tile_pos

    def render_frame(self, *_a, **_kw) -> None:  # pragma: no cover - stub
        pass


class DummyAtlas:
    def __init__(self) -> None:
        self.p = SimpleNamespace(texture=None)
        self.tileset = None
        self._renderer = SimpleNamespace()


class DummyContext:
    def __init__(self) -> None:
        self.sdl_renderer = SimpleNamespace()
        self.sdl_atlas = DummyAtlas()

    def __enter__(self) -> Self:  # pragma: no cover - context stub
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # pragma: no cover
        pass


def _build_dummy_controller(*, use_dummy_world: bool) -> Controller:
    """Build a controller with shared mock object classes for tests."""
    context = DummyContext()

    with ExitStack() as stack:
        stack.enter_context(patch("brileta.controller.FrameManager", DummyFrameManager))
        stack.enter_context(
            patch("brileta.controller.InputHandler", lambda *_a, **_kw: None)
        )
        if use_dummy_world:
            stack.enter_context(
                patch(
                    "brileta.controller.GameWorld",
                    lambda *args, **kwargs: DummyGameWorld(
                        *args, **kwargs, create_player=True
                    ),
                )
            )
            stack.enter_context(
                patch(
                    "brileta.controller.config.BACKEND",
                    SimpleNamespace(lighting="none"),
                )
            )

        app = DummyApp()
        graphics = DummyGraphicsContext(context, None, (16, 16))
        return Controller(app, cast(GraphicsContext, graphics))


def reset_dummy_controller(controller: Controller) -> None:
    """Reset a DummyGameWorld-backed controller to a clean state between tests.

    This resets mode stack, event subscriptions, player state, and removes all
    actors except the player. Use with module-scoped controller fixtures to
    avoid expensive re-initialization while ensuring test isolation.
    """
    from brileta.events import (
        CombatInitiatedEvent,
        reset_event_bus_for_testing,
        subscribe_to_event,
    )

    reset_event_bus_for_testing()
    subscribe_to_event(CombatInitiatedEvent, controller._on_combat_initiated)

    # Some tests temporarily null this to exercise guard clauses; restore the
    # test contract before reusing the shared controller fixture.
    if controller.frame_manager is None:
        controller.frame_manager = DummyFrameManager(controller, controller.graphics)

    controller.mode_stack = [controller.explore_mode]
    controller.explore_mode.enter()
    controller.explore_mode.movement_keys.clear()
    controller.combat_mode._set_selected_action(None)
    controller.explore_mode.active = True
    controller.combat_mode.active = False
    controller.picker_mode.active = False
    controller.selected_target = None
    controller.hovered_actor = None
    controller.picker_mode._on_select = None
    controller.picker_mode._on_cancel = None
    controller.picker_mode._valid_filter = None
    controller.picker_mode._render_underneath = None

    player = controller.gw.player
    if player is not None:
        player.active_plan = None
        queued_actions = getattr(player, "queued_actions", None)
        if queued_actions is not None:
            queued_actions.clear()
        # Reset inventory to prevent equipped items from leaking between tests
        player.inventory._stored_items.clear()
        player.inventory.ready_slots = [None] * len(player.inventory.ready_slots)
        player.inventory.active_slot = 0
        player.inventory._equipped_outfit = None

    for actor in list(controller.gw.actors):
        if actor is not player:
            controller.gw.remove_actor(actor)
