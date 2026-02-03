from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import patch

import numpy as np
import tcod.context
from tcod.console import Console

from catley import colors, config
from catley.app import App
from catley.controller import Controller, GameWorld
from catley.environment.generators import GeneratedMapData
from catley.environment.map import GameMap, MapRegion
from catley.environment.tile_types import TileTypeID
from catley.game.actors import PC, Actor, Character
from catley.game.item_spawner import ItemSpawner
from catley.game.items.item_core import Item
from catley.types import WorldTileCoord
from catley.util.spatial import SpatialHashGrid


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
        # Registry for O(1) actor lookup by Python object id.
        self._actor_id_registry: dict[int, Actor] = {}

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
        self._actor_id_registry[id(actor)] = actor

    def remove_actor(self, actor: Actor) -> None:
        """Removes an actor from the list and the spatial index."""
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            pass
        self._actor_id_registry.pop(id(actor), None)

    def get_actor_by_id(self, actor_id: int) -> Actor | None:
        """Look up an actor by its Python object ID in O(1) time."""
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


def get_controller_with_player_and_map() -> Controller:
    """Return a fully initialized ``Controller`` using dummy SDL context."""

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
            self.root_console = SimpleNamespace(
                width=config.SCREEN_WIDTH, height=config.SCREEN_HEIGHT
            )

        def create_canvas(self, transparent: bool = True) -> Any:
            from unittest.mock import MagicMock

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
            self.graphics = graphics or getattr(controller, "graphics", None)
            self.cursor_manager = SimpleNamespace(
                update_mouse_position=lambda *_a, **_kw: None,
                set_active_cursor_type=lambda *_a, **_kw: None,
            )

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

    context = DummyContext()
    root_console = Console(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, order="F")
    with (
        patch("catley.controller.FrameManager", DummyFrameManager),
        patch("catley.controller.InputHandler", lambda c, a: None),
    ):
        from typing import cast

        from catley.view.render.graphics import GraphicsContext

        app = DummyApp()
        graphics = DummyGraphicsContext(
            cast(tcod.context.Context, context), root_console, (16, 16)
        )
        return Controller(app, cast(GraphicsContext, graphics))


def get_controller_with_dummy_world() -> Controller:
    """Return a Controller with a DummyGameWorld and no GPU lighting system."""

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
            self.root_console = SimpleNamespace(
                width=config.SCREEN_WIDTH, height=config.SCREEN_HEIGHT
            )

        def create_canvas(self, transparent: bool = True) -> Any:
            from unittest.mock import MagicMock

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
            self.graphics = graphics or getattr(controller, "graphics", None)
            self.cursor_manager = SimpleNamespace(
                update_mouse_position=lambda *_a, **_kw: None,
                set_active_cursor_type=lambda *_a, **_kw: None,
            )

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

    context = DummyContext()
    root_console = Console(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, order="F")
    with (
        patch("catley.controller.FrameManager", DummyFrameManager),
        patch("catley.controller.InputHandler", lambda c, a: None),
        patch(
            "catley.controller.GameWorld",
            lambda *args, **kwargs: DummyGameWorld(*args, **kwargs, create_player=True),
        ),
        patch("catley.controller.config.BACKEND", SimpleNamespace(lighting="none")),
    ):
        from typing import cast

        from catley.view.render.graphics import GraphicsContext

        app = DummyApp()
        graphics = DummyGraphicsContext(
            cast(tcod.context.Context, context), root_console, (16, 16)
        )
        return Controller(app, cast(GraphicsContext, graphics))


def reset_dummy_controller(controller: Controller) -> None:
    """Reset a DummyGameWorld-backed controller to a clean state between tests.

    This resets mode stack, event subscriptions, player state, and removes all
    actors except the player. Use with module-scoped controller fixtures to
    avoid expensive re-initialization while ensuring test isolation.
    """
    from catley.events import (
        CombatInitiatedEvent,
        reset_event_bus_for_testing,
        subscribe_to_event,
    )

    reset_event_bus_for_testing()
    subscribe_to_event(CombatInitiatedEvent, controller._on_combat_initiated)

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

    for actor in list(controller.gw.actors):
        if actor is not player:
            controller.gw.remove_actor(actor)
