from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from brileta.app import App
from brileta.game.resolution.base import ResolutionSystem
from brileta.sound import SoundSystem

if TYPE_CHECKING:
    from brileta.game.action_plan import ActionPlan
    from brileta.game.actions.discovery import CombatIntentCache
    from brileta.game.items.item_core import Item
    from brileta.view.render.effects.atmospheric import AtmosphericLayerSystem

from . import colors, config
from .environment.fov import compute_fov
from .events import (
    CombatEndedEvent,
    CombatInitiatedEvent,
    MessageEvent,
    SoundEvent,
    publish_event,
    subscribe_to_event,
)
from .game.actions.base import GameIntent
from .game.actions.discovery import ActionDiscovery
from .game.actions.types import AnimationType
from .game.actors import Actor
from .game.actors.core import Character
from .game.clock import compute_sky_lighting
from .game.enums import CombatEndReason
from .game.game_world import GameWorld
from .game.lights import DirectionalLight, DynamicLight
from .game.turn_manager import TurnManager
from .input_handler import InputHandler
from .modes.base import Mode
from .modes.combat import CombatMode
from .modes.explore import ExploreMode
from .modes.picker import PickerMode
from .types import (
    DeltaTime,
    FixedTimestep,
    InterpolationAlpha,
    MapDecorationSeed,
    RandomSeed,
    SoundId,
    SpriteUV,
    saturate,
)
from .util import rng
from .util.clock import Clock
from .util.coordinates import Rect, WorldTilePos
from .util.live_vars import (
    MetricSpec,
    live_variable_registry,
    record_time_live_variable,
)
from .util.message_log import MessageLog
from .view.animation import AnimationManager
from .view.frame_manager import FrameManager
from .view.presentation import PresentationManager
from .view.render.effects.rain import RainConfig, compute_rain_density
from .view.render.graphics import GraphicsContext
from .view.ui.overlays import OverlaySystem

logger = logging.getLogger(__name__)

_RAIN_LIGHT_AMBIENT_SOUND_ID: SoundId = "rain_light"
_RAIN_HEAVY_AMBIENT_SOUND_ID: SoundId = "rain_heavy"
# Keep light rain texture present across the full density range; the heavy
# layer fades in over it as density increases.
_RAIN_LIGHT_LAYER_LEVEL = 0.62
_RAIN_HEAVY_LAYER_MAX_LEVEL = 1.0
# Indoors should be muffled but still audible.
_RAIN_INDOOR_VOLUME_FLOOR = 0.35
_RAIN_AMBIENT_DISABLE_FADE_SECONDS = 0.4

# Quantized global-lighting state used to gate lightmap invalidation:
# (sun angles or None, sun intensity, ambient, sun color, cloud coverage or None).
_SkyKey = tuple[tuple[float, float] | None, float, float, colors.Color, float | None]


# Controller-level metrics.
_CONTROLLER_METRICS: list[MetricSpec] = [
    MetricSpec("time.logic_ms", "One fixed logic step", 500),
    MetricSpec("time.logic.animation_ms", "Animation updates", 500),
    MetricSpec("time.logic.action_ms", "Action processing", 500),
    MetricSpec("time.logic.actor_snapshot_ms", "Actor position snapshot loop", 500),
    MetricSpec("time.fov_ms", "FOV compute + explored merge", 500),
    # Sprite atlas generation (one-shot at map load, window=1 to keep the value).
    MetricSpec("time.sprites.generate_ms", "Sprite generation (CPU)", 1),
    MetricSpec("time.sprites.atlas_pack_ms", "Atlas packing + GPU upload", 1),
    MetricSpec("time.sprites.total_ms", "Total sprite atlas setup", 1),
]
live_variable_registry.register_metrics(_CONTROLLER_METRICS)


class Controller:
    """
    The central coordinator of the game's logic, state, and subsystems.

    Orchestrates the main application flow by managing a fixed-timestep game loop with
    interpolation. This architecture ensures two key things:

    - Deterministic Game Logic: All game state updates (NPC actions, status effects,
      etc.) occur in discrete, fixed-duration steps (e.g., 60 times per second). This
      makes the simulation predictable and independent of the rendering speed or
      hardware performance.
    - Smooth Visuals: Rendering happens as fast as the hardware allows (up to a target
      FPS cap), and is decoupled from the logic rate. Smooth animations and movement
      are achieved by interpolating object positions and states between the last logic
      step and the current one.

    Key Responsibilities:
    - Loop Management: Owns the game clock and uses an "accumulator" to manage the
      fixed-timestep loop, dispatching calls to `update_logic_step`.
    - System Orchestration: Initializes and holds references to all major subsystems,
      including the `GameWorld` (model), `FrameManager` (view), `InputHandler`, and
      `TurnManager`.
    - Input Processing: Immediately processes player input intents each frame to ensure
      zero-latency responsiveness, separate from the fixed logic loop.
    - Mode Management: Controls the active game mode (e.g., explore, combat) which
      alters player input and rendering.
    - Factory for Core Systems: Provides factory methods for creating essential game
      systems like `ResolutionSystem`.
    """

    gw: GameWorld
    frame_manager: FrameManager
    input_handler: InputHandler
    turn_manager: TurnManager
    explore_mode: ExploreMode
    combat_mode: CombatMode
    picker_mode: PickerMode
    mode_stack: list[Mode]

    def __init__(self, app: App, graphics: GraphicsContext) -> None:
        self.app = app
        self.graphics = graphics
        self.coordinate_converter = graphics.coordinate_converter
        self._systems_initialized = False

        # Systems that persist across world regeneration
        self.action_discovery = ActionDiscovery()
        self.message_log = MessageLog()
        self.overlay_system = OverlaySystem(self)

        # Animation manager for handling movement and effects
        self.animation_manager = AnimationManager()

        # Presentation manager for staggered combat feedback
        self.presentation_manager = PresentationManager()

        # Initialize sound system
        self._initialize_sound_system()

        self.combat_intent_cache: CombatIntentCache | None = None

        # Fixed timestep game loop implementation
        # This separates rendering (variable framerate) from game logic (fixed rate)
        self.clock = Clock()
        self.target_fps = config.TARGET_FPS  # Visual framerate cap

        # Game logic runs at exactly 60Hz regardless of visual framerate
        # This ensures consistent physics, movement, and game timing
        # 16.67ms per logic step
        self.fixed_timestep: FixedTimestep = FixedTimestep(1.0 / 60.0)

        # Accumulator tracks excess frame time to "catch up" logic steps
        # When accumulator >= fixed_timestep, we run one logic update
        self.accumulator = 0.0

        # Death spiral protection: Never run more than this many logic steps per frame
        # Prevents performance collapse when rendering framerate drops below timestep
        self.max_logic_steps_per_frame = 5

        self.last_input_time: float | None = None
        self.action_count_for_latency_metric: int = 0

        # Subscribe to combat initiation events for auto-entry
        subscribe_to_event(CombatInitiatedEvent, self._on_combat_initiated)

        # Selection-based target tracking for click-to-select interaction.
        # selected_target is sticky (persists until explicitly deselected by
        # clicking empty ground). ActionPanel displays actions for selected_target.
        self.selected_target: Actor | None = None

        # Hover target tracking for visual feedback (outline only, no game state).
        # Updated each frame based on mouse position, used for subtle hover highlight.
        self.hovered_actor: Actor | None = None

        # Track the current world seed for debugging/reproducibility
        self._current_seed: RandomSeed = None
        self._rain_enabled: bool = False
        # Global lighting is composed each logic step from the day/night clock
        # (the clear-sky base) plus rain (a dim/cool modifier). _last_sky_key
        # gates the write so the lightmap only re-invalidates when the composed
        # result visibly changes.
        self._last_sky_key: _SkyKey | None = None
        # Clear-weather cloud level, captured while rain is active so it can be
        # restored when rain stops (the atmospheric system carries its own
        # non-rain baseline that the rain overcast level overrides).
        self._atmospheric_cloud_baseline: float | None = None
        # Ambient rain audio recomputes only when spacing or player region changes.
        self._rain_last_ambient_mix_key: tuple[tuple[float, float], int] | None = None

        # Create the initial game world (uses config.RANDOM_SEED)
        # This sets self.gw, lighting_system, player torch, and sun
        self.new_world()

        # Create frame manager after world exists (needs gw.lighting_system)
        self.frame_manager = FrameManager(self, graphics)
        self.input_handler = InputHandler(app, self)

        # Create turn manager after world exists (accesses gw.player via property)
        self.turn_manager = TurnManager(self)

        # Wire up actor-change callbacks now that turn_manager exists.
        self.gw.on_actors_changed = self.turn_manager.invalidate_cache
        self.gw.on_actor_removed = self.turn_manager.on_actor_removed

        # Initialize mode system - game always has an active mode
        # Modes are organized in a stack. ExploreMode is always at the bottom.
        # Other modes (CombatMode, PickerMode, etc.) push on top and pop when done.
        # Created after world exists because modes access controller.gw
        self.explore_mode = ExploreMode(self)
        self.combat_mode = CombatMode(self)
        self.picker_mode = PickerMode(self)
        self.mode_stack: list[Mode] = [self.explore_mode]

        # Global pause: halts simulation (player + NPC movement, animations)
        # while leaving mouse hover/selection working. Toggled with SPACE.
        self.paused: bool = False

        self._systems_initialized = True

        # Register game.seed live variable for debugging/reproducibility
        live_variable_registry.register(
            "game.seed",
            getter=lambda: self._current_seed,
            description="Current world generation seed.",
        )

        # Register live vars for the sun's angles. The angles live on the
        # DirectionalLight; the controller wires them to the registry and
        # handles lighting cache invalidation on change.
        live_variable_registry.register(
            "sun.azimuth",
            getter=lambda: sun.azimuth_degrees if (sun := self._get_sun()) else 0.0,
            setter=lambda v: self._set_sun_angle(azimuth=v),
            description="Sun direction in degrees (0=N, 90=E, 180=S, 270=W).",
            display_decimals=1,
            value_range=(0.0, 360.0),
        )
        live_variable_registry.register(
            "sun.elevation",
            getter=lambda: sun.elevation_degrees if (sun := self._get_sun()) else 0.0,
            setter=lambda v: self._set_sun_angle(elevation=v),
            description="Sun elevation above horizon in degrees (0-90).",
            display_decimals=1,
            value_range=(0.0, 90.0),
        )

        self._register_clock_live_variables()

        self._set_rain_enabled(bool(config.RAIN_ENABLED))
        self._register_rain_live_variables()

        # ---- AI Debug Overlay ----
        # Toggle + hovered-actor variables for inspecting NPC utility scoring.
        # When ai.debug is false the variables are unwatched and invisible.
        self._ai_debug_enabled = False
        self._ai_force_hostile = False
        self._register_ai_debug_variables()

        # Enter explore mode now that all systems are initialized
        self.explore_mode.enter()

    def new_world(self, seed: RandomSeed = None) -> None:
        """Create or recreate the game world.

        This method handles both initial world creation (called from __init__)
        and runtime regeneration (called from dev console). It uses the same
        code path in both cases.

        Args:
            seed: Random seed for world generation. If None on first call,
                uses config.RANDOM_SEED. If None on subsequent calls, generates
                a new random seed.
        """
        from .backends.wgpu.gpu_lighting import GPULightingSystem

        # Determine the seed to use
        if seed is None:
            if not self._systems_initialized:
                # First call (from __init__) - use config seed
                seed = config.RANDOM_SEED
            else:
                # Subsequent call with no seed - generate random one
                seed = rng.generate_seed()

        # If seed is still None (config.RANDOM_SEED is None), generate a
        # human-readable seed so game.seed always shows a reproducible value.
        if seed is None:
            seed = rng.generate_seed()

        # Set the global random state
        random.seed(seed)
        self._current_seed = seed

        # Create new game world and set up lighting
        self.gw = GameWorld(
            config.MAP_WIDTH,
            config.MAP_HEIGHT,
            generator_type=config.MAP_GENERATOR_TYPE,
            seed=seed,
        )

        # Invalidate the turn manager's energy cache when actors are added/removed
        # so dynamically spawned NPCs participate in the action economy.
        # The turn manager doesn't exist yet on the very first call from __init__,
        # but actors added during world construction are picked up by the initial
        # cache build anyway.
        if self._systems_initialized:
            self.gw.on_actors_changed = self.turn_manager.invalidate_cache
            self.gw.on_actor_removed = self.turn_manager.on_actor_removed

        # Initialize GPU lighting system
        self.gw.lighting_system = GPULightingSystem(self.gw, self.graphics)

        # Generate procedural sprites for trees and boulders; upload to GPU atlas.
        self._generate_environmental_sprites()

        # Create the player's torch - automatically disabled in sunlit outdoor areas
        self._player_torch = DynamicLight.create_player_torch(self.gw.player)
        self._player_torch_active = True
        self.gw.add_light(self._player_torch)

        # Create the sun if enabled (uses config defaults)
        if config.SUN_ENABLED:
            self.gw.add_light(DirectionalLight.create_sun())

        # Show seed so the player can reproduce this world
        publish_event(MessageEvent(f"Seed: {seed}", colors.WHITE))

        # Reset dependent systems if this is a regeneration (not first call)
        if self._systems_initialized:
            self._reset_for_new_world()

    def _generate_environmental_sprites(self) -> None:
        """Generate procedural sprites for environmental and character actors.

        Pre-generates all tree, boulder, and humanoid character sprites into
        CPU memory, computes the smallest power-of-two atlas that can hold
        them (clamped to the GPU's hardware limit), then packs and uploads
        in one pass.  This means the atlas is right-sized per map - a small
        map gets a small atlas, a large map gets a larger one, and map
        transitions simply drop the old atlas and create a fresh one.
        """
        from brileta.backends.wgpu.sprite_atlas import compute_atlas_size
        from brileta.game.actors.boulder import Boulder
        from brileta.game.actors.core import Character
        from brileta.game.actors.trees import Tree
        from brileta.sprites.boulders import (
            archetype_for_position as boulder_archetype_for_position,
        )
        from brileta.sprites.boulders import (
            generate_boulder_sprite_for_position,
        )
        from brileta.sprites.boulders import (
            shadow_height_for_archetype as boulder_shadow_height,
        )
        from brileta.sprites.boulders import (
            visual_scale_with_height_jitter as boulder_visual_scale_with_height_jitter,
        )
        from brileta.sprites.characters import (
            CHARACTER_POSE_COUNT,
            HUMANOID_GLYPHS,
            character_sprite_seed,
            generate_character_pose_set,
        )
        from brileta.sprites.common import sprite_visual_scale_for_shadow_height
        from brileta.sprites.trees import (
            generate_tree_sprite_for_position,
        )
        from brileta.sprites.trees import (
            visual_scale_with_height_jitter as tree_visual_scale_with_height_jitter,
        )
        from brileta.util.parallel import parallel_map

        trees: list[Tree] = []
        boulders: list[Boulder] = []
        characters: list[Character] = []
        for a in self.gw.actors:
            if isinstance(a, Tree):
                trees.append(a)
            elif isinstance(a, Boulder):
                boulders.append(a)
            elif isinstance(a, Character) and a.ch in HUMANOID_GLYPHS:
                characters.append(a)
        # Need at least one environmental sprite to justify atlas creation.
        # Character sprites piggyback on the atlas - when no environmental
        # sprites exist, characters fall back to their text glyphs.
        if not trees and not boulders:
            return

        map_seed: MapDecorationSeed = int(self.gw.game_map.decoration_seed)

        with record_time_live_variable("time.sprites.total_ms"):
            # Phase 1: Pre-generate all sprites into CPU memory so we can
            # measure the total area before creating the GPU texture.
            with record_time_live_variable("time.sprites.generate_ms"):
                tree_sprites = parallel_map(
                    generate_tree_sprite_for_position,
                    [t.x for t in trees],
                    [t.y for t in trees],
                    [map_seed] * len(trees),
                    [t.tree_type for t in trees],
                )

                boulder_archetypes = [
                    boulder_archetype_for_position(b.x, b.y, map_seed) for b in boulders
                ]
                for boulder, archetype in zip(
                    boulders, boulder_archetypes, strict=True
                ):
                    boulder.shadow_height = boulder_shadow_height(archetype)

                boulder_sprites = parallel_map(
                    generate_boulder_sprite_for_position,
                    [b.x for b in boulders],
                    [b.y for b in boulders],
                    [map_seed] * len(boulders),
                    boulder_archetypes,
                )

                # Character sprites: four directional poses per humanoid actor,
                # seeded from actor_id so the appearance is stable across movement.
                char_seeds = [
                    character_sprite_seed(c.actor_id, map_seed) for c in characters
                ]
                char_pose_sprites = [generate_character_pose_set(s) for s in char_seeds]

            # Phase 2: Compute the atlas size and create it.
            flat_char_sprites = [
                sprite for pose_set in char_pose_sprites for sprite in pose_set
            ]
            all_sprites = tree_sprites + boulder_sprites + flat_char_sprites
            gpu_max = self.graphics.gpu_max_texture_dimension_2d
            atlas_side = compute_atlas_size(all_sprites, gpu_max)

            atlas = self.graphics.create_sprite_atlas(atlas_side, atlas_side)
            if atlas is None:
                logger.debug(
                    "Graphics backend has no sprite atlas support; "
                    "actors will use fallback glyph rendering."
                )
                return

            # Phase 3: Bulk-pack all sprites via shelf packing, then flush
            # to the GPU in a single write_texture call.
            with record_time_live_variable("time.sprites.atlas_pack_ms"):
                uvs = atlas.pack_all(all_sprites)

                # Assign UVs and visual scales back to the actors.
                # UV list order: [trees..., boulders..., characters...].
                n_trees = len(trees)
                for i, tree in enumerate(trees):
                    uv = uvs[i]
                    if uv is not None:
                        tree.sprite_uv = uv
                        tree.visual_scale = tree_visual_scale_with_height_jitter(
                            tree_sprites[i],
                            tree.shadow_height,
                            tree.x,
                            tree.y,
                            map_seed,
                        )

                for j, boulder in enumerate(boulders):
                    uv = uvs[n_trees + j]
                    if uv is not None:
                        boulder.sprite_uv = uv
                        boulder.visual_scale = boulder_visual_scale_with_height_jitter(
                            boulder_sprites[j],
                            boulder.shadow_height,
                            boulder.x,
                            boulder.y,
                            map_seed,
                        )

                n_boulders = len(boulders)
                pose_count = CHARACTER_POSE_COUNT
                for k, character in enumerate(characters):
                    uv_offset = n_trees + n_boulders + (k * pose_count)
                    resolved_pose_uvs: list[SpriteUV] = []
                    for pose_i in range(pose_count):
                        uv = uvs[uv_offset + pose_i]
                        if uv is None:
                            resolved_pose_uvs = []
                            break
                        resolved_pose_uvs.append(uv)

                    if len(resolved_pose_uvs) == pose_count:
                        pose_uvs = tuple(resolved_pose_uvs)
                        character.character_sprite_uvs = pose_uvs
                        character.sprite_uv = pose_uvs[0]
                        character.visual_scale = sprite_visual_scale_for_shadow_height(
                            char_pose_sprites[k][0],
                            character.shadow_height,
                        )

                atlas.flush()

        if atlas.texture is not None:
            self.graphics.set_sprite_atlas_texture(atlas.texture)

        logger.info(
            "Sprite atlas: %dx%d (%d tree + %d boulder + %d character"
            " pose sprites"
            " across %d characters,"
            " %d allocations)",
            atlas_side,
            atlas_side,
            len(trees),
            len(boulders),
            len(characters) * CHARACTER_POSE_COUNT,
            len(characters),
            atlas.allocated_count,
        )

    def _reset_for_new_world(self) -> None:
        """Reset all systems that depend on the game world.

        Called by new_world() when regenerating (not on first creation).
        """
        # Reset turn manager state
        self.turn_manager.reset()

        # Clear all animations (actors no longer exist)
        self.animation_manager.clear()

        # Reset presentation manager
        self.presentation_manager.clear()

        # Clear controller state
        self.selected_target = None
        self.hovered_actor = None
        self.combat_intent_cache = None

        # Close all menus/overlays (they hold stale references to old world objects)
        self.overlay_system.hide_all_overlays()

        # Reset mode stack to just explore mode, calling _exit() on each
        while len(self.mode_stack) > 1:
            old_mode = self.mode_stack.pop()
            old_mode._exit()

        # Reset explore mode state
        self.explore_mode.movement_keys.clear()

        # Never carry a paused state into a freshly generated world.
        self.paused = False

        # A fresh world has a fresh clock and a sun at config defaults; force the
        # day/night cycle to re-apply so the new sun matches the clock at once.
        self._last_sky_key = None

        # Update frame manager's world view with new lighting system
        wv = self.frame_manager.world_view
        wv.lighting_system = self.gw.lighting_system
        self._get_rain_config().enabled = self._rain_enabled
        if self._rain_enabled:
            # new_world() creates a fresh atmospheric system; re-capture its
            # clear-weather cloud baseline so disabling rain later restores the
            # right value. The sun re-composes on the next logic step.
            atmospheric = self._get_atmospheric_system()
            if atmospheric is not None:
                self._atmospheric_cloud_baseline = float(
                    atmospheric.config.cloud_coverage
                )
            self._rain_last_ambient_mix_key = None
            self._update_rain_ambient_audio()

        # Clear world view caches
        wv.particle_system.clear()
        wv.decal_system.clear()
        wv.floating_text_manager.clear()
        wv._texture_cache.clear()
        wv._active_background_texture = None
        wv._light_overlay_texture = None

        # Reset minimap so it rebuilds from the new map data.
        self.frame_manager.mini_map_view.reset_for_new_world()

        # Reset viewport to follow new player
        wv.viewport_system.camera.set_position(self.gw.player.x, self.gw.player.y)

        # Initialize FOV for new world
        self.update_fov()

    def _get_sun(self) -> DirectionalLight | None:
        """Return the first DirectionalLight in the game world, if any."""
        for light in self.gw.lights:
            if isinstance(light, DirectionalLight):
                return light
        return None

    def _set_sun_angle(
        self,
        azimuth: float | None = None,
        elevation: float | None = None,
    ) -> None:
        """Update the sun's angles, recompute direction, and invalidate caches."""
        sun = self._get_sun()
        if sun:
            sun.set_angles(azimuth=azimuth, elevation=elevation)
        if self.gw.lighting_system:
            self.gw.lighting_system.on_global_light_changed()

    def _register_clock_live_variables(self) -> None:
        """Register dev-console variables for the day/night clock.

        Closures resolve ``self.gw.clock`` lazily so they always target the
        current world's clock, even after a map reload replaces the GameWorld.
        """
        live_variable_registry.register(
            "clock.time",
            getter=self._format_clock_time,
            description="Current game time of day (HH:MM, read-only).",
        )
        live_variable_registry.register(
            "clock.day_progress",
            getter=lambda: self.gw.clock.time_of_day,
            setter=lambda v: self._set_time_of_day(float(v)),
            description="Progress through the day, 0-1 (0=midnight, 0.5=noon).",
            display_decimals=3,
            value_range=(0.0, 1.0),
        )
        live_variable_registry.register(
            "clock.day_length_seconds",
            getter=lambda: self.gw.clock.day_length_seconds,
            setter=lambda v: setattr(
                self.gw.clock, "day_length_seconds", max(1.0, float(v))
            ),
            description="Real seconds per full day-night cycle (lower = faster).",
            display_decimals=0,
            value_range=(1.0, 3600.0),
        )

    def _format_clock_time(self) -> str:
        """Return the current game time as a 24-hour HH:MM string."""
        minutes_in_day = 24 * 60
        total = round(self.gw.clock.time_of_day * minutes_in_day) % minutes_in_day
        return f"{total // 60:02d}:{total % 60:02d}"

    def _set_time_of_day(self, value: float) -> None:
        """Scrub the clock to a time of day and re-light immediately."""
        self.gw.clock.time_of_day = value % 1.0
        # Force a fresh compose so the change shows at once, even while paused.
        self._last_sky_key = None
        self._update_global_lighting(DeltaTime(0.0))

    def _update_global_lighting(self, dt: DeltaTime) -> None:
        """Advance the clock and compose the sun/ambient from day/night + weather.

        This is the single owner of the sun's intensity and color. The day/night
        clock supplies the clear-sky base (angles, intensity, color, ambient);
        rain, when active, dims and cools that base and thickens cloud cover.
        Because both flow through here, the cycle and weather layer together
        instead of overwriting each other.

        The result is quantized into a key and only written when it visibly
        changes, so the lightmap re-invalidates a few times a second as the sun
        sweeps - not every step, and not at all while the sky holds.
        """
        if config.DAY_NIGHT_CYCLE_ENABLED and config.SUN_ENABLED:
            self.gw.clock.advance(dt)
            sky = compute_sky_lighting(self.gw.clock.time_of_day)
            angles: tuple[float, float] | None = (
                sky.azimuth_degrees,
                sky.elevation_degrees,
            )
            base_intensity = sky.sun_intensity
            base_color = sky.sun_color
            ambient = sky.ambient_light
        else:
            # Static sun: leave its angles (config default or dev-console)
            # untouched and use the configured clear-sky values as the base.
            angles = None
            base_intensity = config.SUN_INTENSITY
            base_color = config.SUN_COLOR
            ambient = config.AMBIENT_LIGHT_LEVEL

        # Weather modifier: rain dims/cools the base and thickens cloud cover.
        intensity = base_intensity
        color = base_color
        cloud_coverage: float | None = None
        if self._rain_enabled:
            rain_config = self._get_rain_config()
            if rain_config.enabled:
                density = saturate(float(compute_rain_density(rain_config)))
                dim = colors.lerp(
                    config.RAIN_SUN_DIM_RANGE[0], config.RAIN_SUN_DIM_RANGE[1], density
                )
                intensity = base_intensity * dim
                blend = colors.lerp(
                    config.RAIN_SUN_COLOR_BLEND_RANGE[0],
                    config.RAIN_SUN_COLOR_BLEND_RANGE[1],
                    density,
                )
                color = colors.lerp_color(base_color, config.RAIN_SUN_COOL_COLOR, blend)
                cloud_coverage = colors.lerp(
                    config.RAIN_CLOUD_COVERAGE_RANGE[0],
                    config.RAIN_CLOUD_COVERAGE_RANGE[1],
                    density,
                )

        # Angles change continuously; round them in the key so the lightmap
        # invalidates only when the sun has moved a perceptible amount.
        rounded_angles = (
            None if angles is None else (round(angles[0], 1), round(angles[1], 1))
        )
        sky_key: _SkyKey = (
            rounded_angles,
            round(intensity, 3),
            round(ambient, 3),
            color,
            None if cloud_coverage is None else round(cloud_coverage, 3),
        )
        if sky_key == self._last_sky_key:
            return
        self._last_sky_key = sky_key

        sun = self._get_sun()
        if sun is not None:
            if angles is not None:
                sun.set_angles(azimuth=angles[0], elevation=angles[1])
            sun.intensity = intensity
            sun.color = color

        if cloud_coverage is not None:
            atmospheric = self._get_atmospheric_system()
            if (
                atmospheric is not None
                and abs(atmospheric.config.cloud_coverage - cloud_coverage) > 1e-6
            ):
                atmospheric.set_cloud_coverage(cloud_coverage)

        if self.gw.lighting_system is not None:
            self.gw.lighting_system.ambient_light = ambient
            self.gw.lighting_system.on_global_light_changed()

    def _get_rain_config(self) -> RainConfig:
        """Return WorldView rain config, attaching defaults when absent."""
        world_view = getattr(self.frame_manager, "world_view", None)
        if world_view is None:
            return RainConfig.from_config()
        rain_config = getattr(world_view, "rain_config", None)
        if rain_config is None:
            rain_config = RainConfig.from_config()
            world_view.rain_config = rain_config
        return rain_config

    def _get_atmospheric_system(self) -> AtmosphericLayerSystem | None:
        """Return the active atmospheric layer system when available."""
        world_view = getattr(self.frame_manager, "world_view", None)
        if world_view is None:
            return None
        return getattr(world_view, "atmospheric_system", None)

    def _get_player_sky_exposure(self) -> float:
        """Return sky exposure at the player's current location."""
        player = getattr(self.gw, "player", None)
        if player is None:
            return 1.0
        region = self.gw.game_map.get_region_at((player.x, player.y))
        if region is None:
            return 1.0
        return saturate(float(region.sky_exposure))

    def _stop_rain_ambient_audio(self) -> None:
        """Fade out and stop rain ambient layers."""
        self.sound_system.stop_ambient_loop(
            _RAIN_LIGHT_AMBIENT_SOUND_ID,
            fade_out_seconds=_RAIN_AMBIENT_DISABLE_FADE_SECONDS,
        )
        self.sound_system.stop_ambient_loop(
            _RAIN_HEAVY_AMBIENT_SOUND_ID,
            fade_out_seconds=_RAIN_AMBIENT_DISABLE_FADE_SECONDS,
        )
        self._rain_last_ambient_mix_key = None

    def _get_player_region_id(self) -> int:
        """Return region ID at the player's tile, or -1 when unavailable."""
        player = getattr(self.gw, "player", None)
        if player is None:
            return -1

        game_map = self.gw.game_map
        if not (0 <= player.x < game_map.width and 0 <= player.y < game_map.height):
            return -1
        return int(game_map.tile_to_region_id[player.x, player.y])

    def _update_rain_ambient_audio(self) -> None:
        """Update rain ambient layer volumes from density and player exposure."""
        rain_config = self._get_rain_config()
        if not self._rain_enabled or not rain_config.enabled:
            return

        spacing_key = (rain_config.stream_spacing, rain_config.drop_spacing)
        region_id = self._get_player_region_id()
        mix_key = (spacing_key, region_id)
        if mix_key == self._rain_last_ambient_mix_key:
            return
        self._rain_last_ambient_mix_key = mix_key

        density = saturate(float(compute_rain_density(rain_config)))
        sky_exposure = self._get_player_sky_exposure()
        exposure_gain = colors.lerp(_RAIN_INDOOR_VOLUME_FLOOR, 1.0, sky_exposure)

        # Drizzle = light only. Downpour = heavy dominates with light underneath.
        light_layer_volume = _RAIN_LIGHT_LAYER_LEVEL * exposure_gain
        heavy_layer_volume = _RAIN_HEAVY_LAYER_MAX_LEVEL * density * exposure_gain

        self.sound_system.play_ambient_loop(
            _RAIN_LIGHT_AMBIENT_SOUND_ID,
            volume=light_layer_volume,
        )
        if heavy_layer_volume <= 0.001:
            self.sound_system.stop_ambient_loop(_RAIN_HEAVY_AMBIENT_SOUND_ID)
        else:
            self.sound_system.play_ambient_loop(
                _RAIN_HEAVY_AMBIENT_SOUND_ID,
                volume=heavy_layer_volume,
            )

    def _set_rain_enabled(self, enabled: bool) -> None:
        """Toggle rain visuals and synchronize weather-driven lighting."""
        rain_config = self._get_rain_config()
        rain_enabled = bool(enabled)
        rain_config.enabled = rain_enabled
        if rain_enabled == self._rain_enabled:
            return

        self._rain_enabled = rain_enabled
        self._rain_last_ambient_mix_key = None
        atmospheric = self._get_atmospheric_system()
        if rain_enabled:
            # Remember the clear-weather cloud level so disabling rain restores
            # it; the compose step overrides the atmospheric coverage while wet.
            if atmospheric is not None:
                self._atmospheric_cloud_baseline = float(
                    atmospheric.config.cloud_coverage
                )
            self._update_rain_ambient_audio()
        else:
            self._stop_rain_ambient_audio()
            if atmospheric is not None and self._atmospheric_cloud_baseline is not None:
                atmospheric.set_cloud_coverage(self._atmospheric_cloud_baseline)
            self._atmospheric_cloud_baseline = None

        # Re-compose the sun/ambient now so the toggle takes effect immediately
        # (dt=0 leaves the clock where it is). Resetting the key forces a write.
        self._last_sky_key = None
        self._update_global_lighting(DeltaTime(0.0))

    def _register_rain_live_variables(self) -> None:
        """Register live variables for rain rendering controls.

        Closures resolve the rain config lazily via _get_rain_config() so that
        live variables always target the current WorldView's config, even after
        a map reload replaces the WorldView instance.
        """
        live_variable_registry.register(
            "rain.enabled",
            getter=lambda: self._rain_enabled,
            setter=lambda v: self._set_rain_enabled(bool(v)),
            description="Toggle rain overlay.",
        )
        live_variable_registry.register(
            "rain.intensity",
            getter=lambda: self._get_rain_config().intensity,
            setter=lambda v: setattr(
                self._get_rain_config(), "intensity", saturate(float(v))
            ),
            description="Rain alpha intensity (0-1).",
            display_decimals=2,
            value_range=(0.0, 1.0),
        )
        live_variable_registry.register(
            "rain.angle",
            getter=lambda: self._get_rain_config().angle,
            setter=lambda v: setattr(
                self._get_rain_config(),
                "angle",
                max(
                    -float(config.RAIN_ANGLE_MAX_ABS_RAD),
                    min(float(config.RAIN_ANGLE_MAX_ABS_RAD), float(v)),
                ),
            ),
            description="Baseline rain tilt in radians (wind varies around this).",
            display_decimals=2,
            value_range=(
                -float(config.RAIN_ANGLE_MAX_ABS_RAD),
                float(config.RAIN_ANGLE_MAX_ABS_RAD),
            ),
        )
        live_variable_registry.register(
            "rain.speed",
            getter=lambda: self._get_rain_config().drop_speed,
            setter=lambda v: setattr(
                self._get_rain_config(), "drop_speed", max(8.0, min(120.0, float(v)))
            ),
            description="Rain drop speed in tiles/second.",
            display_decimals=2,
            value_range=(8.0, 120.0),
        )
        live_variable_registry.register(
            "rain.drop_length",
            getter=lambda: self._get_rain_config().drop_length,
            setter=lambda v: setattr(
                self._get_rain_config(), "drop_length", max(0.05, float(v))
            ),
            description="Rain drop length in tiles.",
            display_decimals=2,
            value_range=(0.05, 2.5),
        )
        live_variable_registry.register(
            "rain.drop_spacing",
            getter=lambda: self._get_rain_config().drop_spacing,
            setter=lambda v: setattr(
                self._get_rain_config(), "drop_spacing", max(0.1, float(v))
            ),
            description="Primary along-fall drop spacing control (tiles).",
            display_decimals=2,
            value_range=(0.1, 4.0),
        )
        live_variable_registry.register(
            "rain.stream_spacing",
            getter=lambda: self._get_rain_config().stream_spacing,
            setter=lambda v: setattr(
                self._get_rain_config(), "stream_spacing", max(0.01, min(0.6, float(v)))
            ),
            description="Average lateral spacing between nearby drops (tiles).",
            display_decimals=2,
            value_range=(0.01, 0.6),
        )

    # ------------------------------------------------------------------
    # AI Debug Overlay
    # ------------------------------------------------------------------

    def _register_ai_debug_variables(self) -> None:
        """Register live variables for inspecting NPC utility scoring.

        Registers ai.debug (toggle), ai.hovered.action, ai.hovered.scores,
        ai.hovered.goal, ai.hovered.routine, ai.hovered.personality,
        ai.hovered.disposition_to_player, ai.hovered.disposition_to_target,
        and ai.hovered.threat_level.
        The display variables report on the click-selected NPC only (it stays
        pinned as it moves); hovering has no effect. When ai.debug is toggled on
        they are watched so they appear in the debug stats overlay; toggled off,
        they are unwatched. The variables keep the ``ai.hovered.*`` names for
        continuity even though the subject is now the selected NPC.
        """
        from brileta.game.actors.ai import AIComponent
        from brileta.game.actors.core import NPC

        _AI_VAR_NAMES = (
            "ai.hovered.action",
            "ai.hovered.scores",
            "ai.hovered.goal",
            "ai.hovered.routine",
            "ai.hovered.personality",
            "ai.hovered.disposition_to_player",
            "ai.hovered.disposition_to_target",
            "ai.hovered.threat_level",
        )

        def _debug_subject() -> Actor | None:
            """The NPC the debug overlay reports on: the click-selected one.

            Driven solely by the sticky click selection (selected_target), so
            the readout stays pinned to one NPC as it and others move around.
            Hovering deliberately has no effect - otherwise an NPC wandering
            under the cursor would hijack the display.
            """
            return self.selected_target

        def _get_ai() -> AIComponent | None:
            """Get the AI component for the debug subject, if applicable.

            Returns None for dead actors (stale cached scores are irrelevant).
            """
            actor = _debug_subject()
            if not isinstance(actor, NPC):
                return None
            if not actor.health.is_alive():
                return None
            return actor.ai

        # -- ai.debug toggle --

        def _set_ai_debug(value: object) -> None:
            self._ai_debug_enabled = bool(value)
            for name in _AI_VAR_NAMES:
                if self._ai_debug_enabled:
                    live_variable_registry.watch(name)
                else:
                    live_variable_registry.unwatch(name)

        live_variable_registry.register(
            "ai.debug",
            getter=lambda: self._ai_debug_enabled,
            setter=_set_ai_debug,
            formatter=lambda v: "on" if v else "off",
            description="Toggle AI debug overlay for the click-selected NPC.",
        )

        # -- ai.force_hostile toggle --
        # When true, AIComponent overrides disposition to hostile for all NPCs.
        # Useful for testing hostile behaviors (patrol, flee, attack) on any NPC.

        live_variable_registry.register(
            "ai.force_hostile",
            getter=lambda: self._ai_force_hostile,
            setter=lambda v: setattr(self, "_ai_force_hostile", bool(v)),
            formatter=lambda v: "on" if v else "off",
            description="Force all NPCs to use hostile AI regardless of disposition.",
        )

        # -- ai.hovered.action --

        def _get_action() -> str:
            if not self._ai_debug_enabled:
                return "---"
            ai = _get_ai()
            if ai is None or ai.last_chosen_action is None:
                return "---"
            return ai.last_chosen_action

        live_variable_registry.register(
            "ai.hovered.action",
            getter=_get_action,
            description="Action the selected NPC chose this tick.",
        )

        # -- ai.hovered.scores --
        # Indent for continuation lines so they're visually grouped
        # under the variable name.
        _SCORE_INDENT = "  "

        def _get_scores() -> str:
            if not self._ai_debug_enabled:
                return "---"
            ai = _get_ai()
            if ai is None or not ai.last_scores:
                return "---"
            parts: list[str] = []
            for s in sorted(ai.last_scores, key=lambda x: -x.final_score):
                if s.persistence_bonus > 0:
                    parts.append(
                        f"{s.display_name}: {s.final_score:.2f} "
                        f"(base {s.base_score:.2f} + persist {s.persistence_bonus:.2f})"
                    )
                else:
                    parts.append(f"{s.display_name}: {s.final_score:.2f}")
            # First entry on the same line as the variable name,
            # remaining entries indented on their own lines.
            return ("\n" + _SCORE_INDENT).join(parts)

        live_variable_registry.register(
            "ai.hovered.scores",
            getter=_get_scores,
            description="All action scores for the selected NPC, sorted descending.",
        )

        # -- ai.hovered.goal --

        def _get_goal() -> str:
            if not self._ai_debug_enabled:
                return "---"
            actor = _debug_subject()
            if not isinstance(actor, NPC):
                return "---"
            goal = actor.current_goal
            if goal is None:
                return "None"
            return (
                f"{type(goal).__name__} {goal.state.name} progress={goal.progress:.1f}"
            )

        live_variable_registry.register(
            "ai.hovered.goal",
            getter=_get_goal,
            description="Active goal of the selected NPC.",
        )

        # -- ai.hovered.routine --

        def _get_routine() -> str:
            if not self._ai_debug_enabled:
                return "---"
            actor = _debug_subject()
            if not isinstance(actor, NPC):
                return "---"
            # An NPC with neither anchor nor home has no routine: creatures,
            # hostiles, and the unanchored fraction of residents that wander.
            if actor.anchor_pos is None and actor.home_pos is None:
                return "none (wanders)"
            from brileta.game.actors.ai.behaviors.routine import routine_target

            target = routine_target(self.gw.clock.time_of_day, actor)
            at_work = actor.anchor_pos is not None and target == actor.anchor_pos
            phase = "work" if at_work else "home"
            return (
                f"{phase} -> {target} | anchor={actor.anchor_pos} "
                f"home={actor.home_pos} off={actor.routine_offset:+.2f}"
            )

        live_variable_registry.register(
            "ai.hovered.routine",
            getter=_get_routine,
            description=(
                "Daily-routine state: current phase, target tile, "
                "anchor/home, and schedule offset."
            ),
        )

        # -- ai.hovered.personality --

        def _get_personality() -> str:
            if not self._ai_debug_enabled:
                return "---"
            actor = _debug_subject()
            if not isinstance(actor, NPC):
                return "---"
            p = actor.personality
            # Compact OCEAN readout: each trait on the 0-10 scale (5 = average),
            # in canonical O-C-E-A-N order so it reads left to right.
            return (
                f"O{p.openness} C{p.conscientiousness} E{p.extraversion} "
                f"A{p.agreeableness} N{p.neuroticism}"
            )

        live_variable_registry.register(
            "ai.hovered.personality",
            getter=_get_personality,
            description=(
                "OCEAN personality of the selected NPC (0-10, 5=average): "
                "Openness, Conscientiousness, Extraversion, Agreeableness, "
                "Neuroticism."
            ),
        )

        # -- ai.hovered.disposition_to_player --

        def _get_disposition_to_player() -> str:
            if not self._ai_debug_enabled:
                return "---"
            ai = _get_ai()
            if ai is None:
                return "---"
            return str(ai.disposition_toward(self.gw.player))

        live_variable_registry.register(
            "ai.hovered.disposition_to_player",
            getter=_get_disposition_to_player,
            description=("Numeric disposition (-100 to +100) toward the player."),
        )

        # -- ai.hovered.disposition_to_target --

        def _get_disposition_to_target() -> str:
            if not self._ai_debug_enabled:
                return "---"
            ai = _get_ai()
            if ai is None:
                return "---"
            target = (
                self.gw.get_actor_by_id(ai.last_target_actor_id)
                if ai.last_target_actor_id is not None
                else None
            )
            if target is None:
                return "---"
            return str(ai.disposition_toward(target))

        live_variable_registry.register(
            "ai.hovered.disposition_to_target",
            getter=_get_disposition_to_target,
            description=(
                "Numeric disposition (-100 to +100) toward the NPC's current AI target."
            ),
        )

        # -- ai.hovered.threat_level --

        def _get_threat_level() -> str:
            if not self._ai_debug_enabled:
                return "---"
            ai = _get_ai()
            if ai is None or ai.last_threat_level is None:
                return "---"
            return f"{ai.last_threat_level:.2f}"

        live_variable_registry.register(
            "ai.hovered.threat_level",
            getter=_get_threat_level,
            description=(
                "Relationship-aware threat level (proximity * hostility) "
                "for the selected NPC's current AI target."
            ),
        )

    def update_fov(self) -> None:
        """Recompute the visible area based on the player's point of view."""
        with record_time_live_variable("time.fov_ms"):
            self.gw.game_map.visible[:] = compute_fov(
                self.gw.game_map.transparent,
                (self.gw.player.x, self.gw.player.y),
                radius=config.FOV_RADIUS,
            )

            # If a tile is "visible" it should be added to "explored"
            self.gw.game_map.explored |= self.gw.game_map.visible
            self.gw.game_map.exploration_revision += 1

        # Auto-toggle torch based on ambient lighting (disable in sunlit outdoor areas)
        self._update_player_torch()

    def _update_player_torch(self) -> None:
        """Enable/disable player torch based on current location's sky exposure.

        In well-lit outdoor areas (high sky exposure), the torch is unnecessary
        and would wash out sun shadows, so we automatically disable it.

        Uses hysteresis to prevent flickering at doorways: torch turns OFF when
        sky_exposure >= 0.7 (clearly outdoors) but only turns back ON when
        sky_exposure <= 0.3 (clearly indoors).
        """
        # Check sky exposure at player's position
        region = self.gw.game_map.get_region_at((self.gw.player.x, self.gw.player.y))
        if region is None:
            return

        sky_exposure = region.sky_exposure

        # Hysteresis thresholds to prevent flickering at boundaries
        turn_off_threshold = 0.7  # Clearly outdoors
        turn_on_threshold = 0.3  # Clearly indoors

        if self._player_torch_active and sky_exposure >= turn_off_threshold:
            # Disable torch in sunlit outdoor areas
            self.gw.remove_light(self._player_torch)
            self._player_torch_active = False
        elif not self._player_torch_active and sky_exposure <= turn_on_threshold:
            # Re-enable torch when entering dark areas
            self.gw.add_light(self._player_torch)
            self._player_torch_active = True

    def get_visible_bounds(self) -> Rect | None:
        """Return the world-space bounds currently visible in the world view."""
        return self.frame_manager.get_visible_bounds()

    def process_player_input(self) -> None:
        """Process all pending player input.

        Should be called once per visual frame before the logic step loop
        to ensure responsiveness. All active modes get update() called to
        support layered behavior (e.g., movement while in combat/picker mode).
        """
        assert self.overlay_system is not None and self.input_handler is not None

        # Paused: skip all player movement generation, queued actions, autopilot.
        # Mouse hover/selection still works (driven by input_handler, not here).
        if self.paused:
            return

        # Let all modes in the stack update (bottom-to-top)
        # This allows ExploreMode to generate movement intents even when
        # CombatMode or PickerMode is on top.
        if not self.overlay_system.has_interactive_overlays():
            for mode in self.mode_stack:
                mode.update()

        # Check for other queued player actions (manual input from UI/keys)
        if self.turn_manager.has_pending_actions():
            player_action = self.turn_manager.dequeue_player_action()
            if player_action:
                # Manual action cancels any active plan and presentation timing.
                # Player input should feel immediately responsive.
                self.stop_plan(self.gw.player)
                self.turn_manager.clear_presentation_timing()
                self.animation_manager.interrupt_player_animations(self.gw.player)
                self._execute_player_action_immediately(player_action)

        # Check for autopilot actions if no manual input.
        # Autopilot respects presentation timing (duration_ms) to create paced movement.
        # No separate rate limiting needed - duration_ms controls the pacing.
        has_movement_keys = bool(self.explore_mode.movement_keys)
        presentation_complete = self.turn_manager.is_presentation_complete()
        if (
            not has_movement_keys
            and presentation_complete
            and not self.turn_manager.has_pending_actions()
            and self.turn_manager.is_player_turn_available()
        ):
            # Get autopilot action - check active_plan first, then pathfinding_goal
            if self.gw.player.active_plan is not None:
                autopilot_action = self.turn_manager._get_intent_from_plan(
                    self.gw.player
                )
            else:
                # Fall back to pathfinding_goal (legacy system)
                autopilot_action = self.gw.player.get_next_action(self)

            if autopilot_action:
                if self.gw.player.energy.can_afford(config.ACTION_COST):
                    self._execute_player_action_immediately(autopilot_action)
                else:
                    # Can't afford action - time passes but no movement
                    self.turn_manager.on_player_action()

    def update_logic_step(self) -> None:
        """
        Runs one fixed-step of game logic for animations and non-player actors.
        Called by the App's fixed-step loop.
        """
        # Paused: freeze the whole simulation (animations, NPCs, lighting, sound).
        if self.paused:
            return

        with record_time_live_variable("time.logic_ms"):
            with record_time_live_variable("time.logic.actor_snapshot_ms"):
                # Reset only actors that moved since the prior logic step.
                self.gw.reset_pending_actor_position_snapshots()

            with record_time_live_variable("time.logic.animation_ms"):
                # ANIMATIONS: Update frame-independent systems
                # Ensures consistent animation timing regardless of FPS
                self.animation_manager.update(self.fixed_timestep)

            self._update_global_lighting(DeltaTime(self.fixed_timestep))

            if self.gw.lighting_system is not None:
                self.gw.lighting_system.update(self.fixed_timestep)

            # Update sound system with player position
            if self.gw.player:
                self.sound_system.update(
                    self.gw.player.x,
                    self.gw.player.y,
                    self.gw.actor_spatial_index,
                    DeltaTime(self.fixed_timestep),
                    game_map=self.gw.game_map,
                )
            self._update_rain_ambient_audio()

            # Update presentation manager (dispatches staggered combat feedback)
            self.presentation_manager.update(self.fixed_timestep)

            with record_time_live_variable("time.logic.action_ms"):
                # Explore mode: NPCs accrue time-driven energy every step so the
                # world keeps moving while the player idles. Combat energy stays
                # player-action-driven, so we skip ambient accrual there.
                if not self.is_combat_mode():
                    self.turn_manager.accumulate_ambient_energy(self.fixed_timestep)

                # GAME LOGIC: Process all NPC actions for this step
                # This is where the core game simulation happens
                self._process_all_available_npc_actions()

    def render_visual_frame(self, alpha: InterpolationAlpha) -> None:
        """Renders one visual frame with interpolation. Called by the App."""
        assert self.frame_manager is not None
        with record_time_live_variable("time.render.cpu_ms"):
            # Uses alpha to smoothly blend between prev_* and current
            self.frame_manager.render_frame(alpha)

    def _execute_player_action_immediately(self, action: GameIntent) -> None:
        """Execute a player action immediately on the current frame.

        This is the core of the RAF system - player actions are never queued
        or delayed. They are processed instantly when detected.

        Args:
            action: The player's GameIntent to execute immediately
        """
        # Handle wind-up actions specially (preserve existing PPIAS behavior)
        if action.animation_type == AnimationType.WIND_UP and action.windup_animation:
            self.animation_manager.add(action.windup_animation)
            # For wind-up actions, we still need to wait for completion
            # This preserves the existing wind-up behavior from PPIAS
            # TODO: This may be simplified further in future phases
            return

        # For INSTANT actions, execute immediately.
        # CRITICAL: Check prevention BEFORE update_turn() to match NPC behavior.
        # If checked after, 1-duration effects (Staggered) would already expire,
        # allowing the player to act when they should be blocked.
        if self.gw.player.status_effects.is_action_prevented():
            # Player's action is blocked - still counts as their turn
            self.gw.player.update_turn(self)
            self.gw.player.energy.spend(config.ACTION_COST)
            publish_event(MessageEvent("You cannot act!", colors.RED))
            self.invalidate_combat_tooltip()
            return

        # Process turn effects for the acting player
        self.gw.player.update_turn(self)

        # Execute the player's action (using execute_player_intent for plan advancement)
        self.turn_manager.execute_player_intent(action)
        self.gw.player.energy.spend(config.ACTION_COST)

        # Check for terrain hazard damage after the player completes their action
        self.turn_manager._apply_terrain_hazard(self.gw.player)

        # Update FOV after player action (important for movement)
        self.update_fov()

        # RAF: Trigger immediate NPC scheduling based on the world state change
        self.turn_manager.on_player_action()

        # Process all ready NPCs immediately. This ensures NPCs get their turns
        # even during held-key movement, where presentation timing would otherwise
        # starve them (player resets timer every 70ms, NPCs never see it expire).
        self.turn_manager.process_all_ready_npcs_immediately()

        self.invalidate_combat_tooltip()

        # Refresh hovered actor in case actors moved into/out of the mouse position
        self.update_hovered_actor(self.gw.mouse_tile_location_on_map)

    def update_hovered_actor(self, mouse_pos: WorldTilePos | None) -> None:
        """Update the hovered actor based on mouse position.

        This is used for visual feedback only (subtle hover outline).
        Does not affect game state or ActionPanel.
        """
        if mouse_pos is None:
            self.hovered_actor = None
            return

        mouse_x, mouse_y = mouse_pos
        self.hovered_actor = self._get_visible_actor_at_tile(mouse_x, mouse_y)

    def _get_visible_actor_at_tile(self, x: int, y: int) -> Actor | None:
        """Return the first visible actor at a tile, if any."""
        gm = self.gw.game_map
        if not (0 <= x < gm.width and 0 <= y < gm.height):
            return None
        if not gm.visible[x, y]:
            return None

        actors_at_tile = self.gw.actor_spatial_index.get_at_point(x, y)
        if not actors_at_tile:
            return None

        for actor in sorted(actors_at_tile, key=self._actor_sort_key):
            if actor is not self.gw.player:
                return actor
        return None

    @staticmethod
    def _actor_sort_key(actor: Actor) -> tuple[int, int, str]:
        """Deterministic ordering for multiple actors on the same tile."""
        return (actor.y, actor.x, actor.name)

    def _process_all_available_npc_actions(self) -> None:
        """Process NPCs who can currently afford actions this logic step.

        Combat mode: process one NPC per tick, gated on presentation timing, so
        reactions are sequenced and readable (player can follow cause and effect).

        Explore mode: process all ready NPCs immediately, ungated by presentation
        timing, so ambient wandering NPCs act in parallel on their own rhythms
        instead of being serialized into a one-per-second metronome.
        """
        if self.is_combat_mode():
            # Wait for current action's presentation to complete before NPCs act
            if not self.turn_manager.is_presentation_complete():
                return
            self.turn_manager.process_all_npc_reactions()
        else:
            self.turn_manager.process_all_ready_npcs_immediately()

    def queue_action(self, action: GameIntent) -> None:
        """
        Queue a game action to be processed on the next turn.

        Use this when you want an action to consume a turn and trigger
        the unified round system (giving all actors a chance to act).

        Args:
            action: The intent to queue for execution
        """
        # Paused: drop the command instead of deferring it. Input dispatch stays
        # ungated (so SPACE can unpause), but no player action may commit to the
        # world while paused, otherwise it would fire the instant we resume.
        # getattr tolerates lightweight test doubles that skip Controller.__init__.
        if getattr(self, "paused", False):
            return
        self.turn_manager.queue_action(action)

    def toggle_pause(self) -> None:
        """Toggle the global pause state and refresh the paused indicator."""
        self.paused = not self.paused
        overlay = getattr(self.frame_manager, "paused_indicator_overlay", None)
        if overlay is not None:
            overlay.invalidate()

    @property
    def active_mode(self) -> Mode:
        """The mode currently handling input (top of the mode stack)."""
        return self.mode_stack[-1]

    def push_mode(self, mode: Mode) -> None:
        """Push a mode onto the stack and activate it.

        The pushed mode becomes the new active mode and receives all input.
        The mode below remains in the stack but is temporarily inactive.

        Raises:
            RuntimeError: If a mode of the same type is already in the stack.
        """
        if any(type(m) is type(mode) for m in self.mode_stack):
            raise RuntimeError(
                f"A mode of type {type(mode).__name__} is already in the stack"
            )
        # Append before enter() so nested push_mode calls work correctly
        # (e.g., CombatMode.enter() pushing PickerMode)
        self.mode_stack.append(mode)
        mode.enter()

    def pop_mode(self) -> None:
        """Pop the top mode and return to the mode below.

        ExploreMode (the base) cannot be popped - the stack always has at
        least one mode. Calling pop_mode when only ExploreMode remains is a no-op.
        """
        if len(self.mode_stack) > 1:
            old_mode = self.mode_stack.pop()
            old_mode._exit()

    def transition_to_mode(self, new_mode: Mode) -> None:
        """Replace the top mode with a new one (legacy compatibility).

        For stack-based modes (PickerMode, etc.), prefer push_mode/pop_mode.
        This method is kept for the ExploreMode <-> CombatMode transitions
        which conceptually replace rather than stack.

        If the new mode is already in the stack (e.g., transitioning back to
        ExploreMode), this will pop modes until that mode is on top.
        """
        if self.active_mode is new_mode:
            return  # Already in this mode

        # Pop the current top mode
        self.pop_mode()

        # Keep popping if the new mode is already in the stack below
        # (e.g., transitioning back to ExploreMode which is at the base)
        while self.active_mode is not new_mode and new_mode in self.mode_stack:
            self.pop_mode()

        # Only push if the new mode isn't already the active mode
        if self.active_mode is not new_mode:
            self.push_mode(new_mode)

    def enter_combat_mode(self) -> None:
        """Enter combat mode from current mode."""
        already_in_combat = self.is_combat_mode()
        self.transition_to_mode(self.combat_mode)
        if not already_in_combat:
            # NPCs may have wandered up with a full ambient energy bar. Zero it
            # so combat starts player-driven and nobody gets a free instant hit.
            self.turn_manager.clamp_npc_energy_for_combat()

    def exit_combat_mode(
        self, reason: CombatEndReason = CombatEndReason.MANUAL_EXIT
    ) -> None:
        """Exit combat mode back to explore mode.

        Warns the player if they voluntarily leave while hostiles are visible.

        Args:
            reason: Why combat ended. Used for ceremony hooks.
        """
        if reason != CombatEndReason.ALL_ENEMIES_DEAD and self.has_visible_hostiles():
            publish_event(
                MessageEvent("Standing down despite hostile presence.", colors.YELLOW)
            )
        publish_event(CombatEndedEvent(reason=reason))
        self.transition_to_mode(self.explore_mode)

    def is_combat_mode(self) -> bool:
        """Check if currently in combat mode (combat mode is in the stack).

        Note: When combat mode is entered, it pushes PickerMode on top for
        target selection. So we check if combat_mode is anywhere in the stack,
        not just if it's the active mode.
        """
        return self.combat_mode in self.mode_stack

    def invalidate_combat_tooltip(self) -> None:
        """Refresh the combat tooltip overlay if it is active."""
        if not self.is_combat_mode():
            return

        tooltip = self.frame_manager.combat_tooltip_overlay
        if tooltip.is_active:
            tooltip.invalidate()

    def _on_combat_initiated(self, event: CombatInitiatedEvent) -> None:
        """Auto-enter combat mode when combat is initiated.

        Triggered by:
        - NPC attacks player (hit or miss)
        - Player attacks non-hostile NPC
        - Player pushes non-hostile NPC
        - Player's noise alerts nearby NPCs
        """
        player = self.gw.player
        if player is None:
            return

        # Defensive guard: combat mode is a player-facing state, so events that
        # don't involve the player should never force mode transitions.
        if event.attacker is not player and event.defender is not player:
            return

        if not self.is_combat_mode():
            self.enter_combat_mode()

    def has_visible_hostiles(self) -> bool:
        """Check if any on-screen visible NPCs are hostile toward the player.

        Used to warn when exiting combat mode with enemies still in sight.
        Uses spatial index for efficient viewport-bounds query.
        """
        from brileta.game.actors.core import NPC

        player = self.gw.player
        if player is None:
            return False
        bounds = self.get_visible_bounds()
        if bounds is not None:
            nearby_actors = self.gw.actor_spatial_index.get_in_rect(bounds)
        else:
            # Fallback for headless/test contexts.
            nearby_actors = self.gw.actor_spatial_index.get_in_radius(
                player.x, player.y, radius=config.FOV_RADIUS
            )

        for actor in nearby_actors:
            if not isinstance(actor, NPC) or actor == player:
                continue
            if not actor.health.is_alive():
                continue
            if not self.gw.game_map.visible[actor.x, actor.y]:
                continue
            if actor.ai.is_hostile_toward(player):
                return True
        return False

    def select_target(self, target: Actor | None) -> None:
        """Set the selected target for the ActionPanel.

        Selected targets are sticky - they persist until explicitly deselected
        (by clicking empty ground or calling deselect_target). This enables
        RTS-style click-to-select where the panel locks to a target.

        Args:
            target: The actor to select, or None to deselect.
        """
        # Play selection sound when selecting a new target
        if target is not None and target is not self.selected_target:
            # Use player position so UI sound plays at full volume
            publish_event(
                SoundEvent(sound_id="ui_select", x=self.gw.player.x, y=self.gw.player.y)
            )
        self.selected_target = target

    def deselect_target(self) -> None:
        """Clear the selected target.

        Call this when the player clicks on empty ground or otherwise wants
        to clear the selection.
        """
        self.selected_target = None

    def start_consumable_targeting(self, item: Item) -> None:
        """Start consumable targeting via PickerMode.

        Clicking on self uses the item on the player. Clicking on another
        character uses it on them (pathfinding if needed).

        Args:
            item: The consumable Item to use.
        """
        from brileta import colors
        from brileta.events import MessageEvent, publish_event
        from brileta.game import ranges
        from brileta.game.actions.recovery import (
            UseConsumableIntent,
            UseConsumableOnTargetIntent,
        )
        from brileta.game.actors import Character
        from brileta.modes.picker import PickerResult

        player = self.gw.player

        def valid_filter(x: int, y: int) -> bool:
            """Check if a tile is a valid consumable target."""
            # Player's own tile is valid (self-target)
            if x == player.x and y == player.y:
                return True
            # Check for a living character at the tile
            actor = self.gw.get_actor_at_location(x, y)
            if actor is None:
                return False
            return (
                isinstance(actor, Character)
                and actor.health.is_alive()
                and self.gw.game_map.visible[x, y]
            )

        def on_select(result: PickerResult) -> None:
            """Handle target selection."""
            # Self-targeting: use on self
            if result.tile[0] == player.x and result.tile[1] == player.y:
                intent = UseConsumableIntent(self, player, item)
                self.queue_action(intent)
                return

            # Target is another character
            target = result.actor
            if target is not None and isinstance(target, Character):
                distance = ranges.calculate_distance(
                    player.x, player.y, target.x, target.y
                )
                if distance == 1:
                    # Adjacent target: use immediately
                    intent = UseConsumableOnTargetIntent(self, player, item, target)
                    self.queue_action(intent)
                else:
                    # Distant target: use ActionPlan to pathfind then use
                    from brileta.game.actions.recovery import UseConsumableOnTargetPlan

                    self.start_plan(
                        player,
                        UseConsumableOnTargetPlan,
                        target_actor=target,
                        target_position=(target.x, target.y),
                        item=item,
                    )

        # Show message about targeting
        publish_event(MessageEvent(f"Select target for {item.name}", colors.CYAN))

        self.picker_mode.start(
            on_select=on_select,
            on_cancel=lambda: None,
            valid_filter=valid_filter,
        )

    def create_resolver(self, **kwargs: object) -> ResolutionSystem:
        """Factory method for resolution systems.

        Currently returns a :class:`D20System` but allows future
        customization without changing action code.
        """
        from brileta.game.resolution.d20_system import D20System

        return D20System(**kwargs)  # ty: ignore[invalid-argument-type]

    def start_plan(
        self,
        actor: Character,
        plan: ActionPlan,
        target_actor: Character | Actor | None = None,
        target_position: WorldTilePos | None = None,
        weapon: Item | None = None,
        item: Item | None = None,
    ) -> bool:
        """Start an action plan for an actor.

        This is the unified method for starting any ActionPlan. It creates
        the PlanContext and ActivePlan, then assigns it to the actor.

        Args:
            actor: The character executing the plan.
            plan: The ActionPlan to execute.
            target_actor: Optional target actor for targeted actions.
            target_position: Optional target position. For actor targets,
                this is usually derived from (target_actor.x, target_actor.y).
            weapon: Optional weapon for combat actions.
            item: Optional item for consumable actions.

        Returns:
            True (always succeeds in creating the plan).
        """
        # Paused: refuse new plans so a shift-click/right-click target set while
        # paused doesn't auto-walk the moment we resume. See queue_action.
        # getattr tolerates lightweight test doubles that skip Controller.__init__.
        if getattr(self, "paused", False):
            return False

        from brileta.game.action_plan import ActivePlan, PlanContext

        context = PlanContext(
            actor=actor,
            controller=self,
            target_actor=target_actor,
            target_position=target_position,
            weapon=weapon,
            item=item,
        )
        actor.active_plan = ActivePlan(plan=plan, context=context)
        return True

    def stop_plan(self, actor: Character) -> None:
        """Cancel an actor's active plan."""
        actor.active_plan = None

    def _initialize_sound_system(self) -> None:
        """Initialize the sound system and audio backend."""
        # Sound system for managing audio playback
        self.sound_system = SoundSystem()

        # Initialize audio backend if enabled and not in test environment
        if config.AUDIO_ENABLED and not config.IS_TEST_ENVIRONMENT:
            try:
                from pathlib import Path

                from brileta.backends.miniaudio_audio import MiniaudioBackend

                audio_backend = MiniaudioBackend()
                audio_backend.initialize()
                self.sound_system.set_audio_backend(audio_backend)

                # Set assets path for loading sounds
                assets_path = Path(__file__).parent.parent / "assets"
                self.sound_system.set_assets_path(assets_path)
            except Exception as e:
                # Continue without audio; avoid user-visible warnings.
                logger.debug("Failed to initialize audio backend: %s", e)

    def cleanup(self) -> None:
        """Clean up resources when the controller is being shut down."""
        # Shutdown audio backend if it exists
        if self.sound_system.audio_backend:
            self.sound_system.audio_backend.shutdown()
