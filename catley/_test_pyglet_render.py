import time

import numpy as np
import pyglet
from pyglet.window import Window

from catley import colors, config
from catley.backends.pyglet.canvas import PygletCanvas
from catley.backends.pyglet.renderer import PygletRenderer
from catley.types import DeltaTime, Opacity
from catley.util.coordinates import Rect
from catley.view.render.effects.effects import EffectContext, EffectLibrary
from catley.view.render.effects.environmental import EnvironmentalEffectSystem
from catley.view.render.effects.particles import ParticleLayer, SubTileParticleSystem

# from catley.view.views.health_view import HealthView # Test a view

# Basic window setup
VSYNC = True
window = Window(
    width=config.SCREEN_WIDTH * 16,  # Start with some reasonable pixel size
    height=config.SCREEN_HEIGHT * 16,
    caption="Pyglet Renderer Test Harness",
    resizable=True,
    vsync=VSYNC,
)

# --- YOUR TEST SETUP GOES HERE ---
# 1. Instantiate your PygletRenderer
try:
    renderer = PygletRenderer(window)
    print("PygletRenderer created successfully.")
    print(f"Tile dimensions: {renderer.tile_dimensions}")
    print(
        f"Console size: {renderer.console_width_tiles}x{renderer.console_height_tiles}"
    )
except Exception as e:
    print(f"Failed to create PygletRenderer: {e}")
    exit()


# 2. Create a dummy map for testing
def create_test_map():
    """Create a simple test map with borders and some features."""
    width = min(config.SCREEN_WIDTH, 40)  # Smaller for easier testing
    height = min(config.SCREEN_HEIGHT, 30)

    # Create map filled with floor tiles
    test_map = np.full((height, width), ord("."), dtype=int)

    # Add borders
    for x in range(width):
        test_map[0, x] = ord("#")  # Top wall
        test_map[height - 1, x] = ord("#")  # Bottom wall
    for y in range(height):
        test_map[y, 0] = ord("#")  # Left wall
        test_map[y, width - 1] = ord("#")  # Right wall

    # Add some interesting features
    for i in range(5, width - 5, 7):
        for j in range(5, height - 5, 7):
            test_map[j, i] = ord("*")  # Pillars

    # Add a few doors in the walls
    test_map[height // 2, 0] = ord("+")  # Left door
    test_map[height // 2, width - 1] = ord("+")  # Right door
    test_map[0, width // 2] = ord("+")  # Top door
    test_map[height - 1, width // 2] = ord("+")  # Bottom door

    return test_map


test_map = create_test_map()
print(f"Created test map: {test_map.shape}")

# 3. Test actor for smooth rendering
test_actor_pos = [10.5, 8.7]  # Smooth float coordinates
test_actor_char = "@"
test_actor_color = colors.WHITE
test_actor_lighting = (1.0, 0.9, 0.8)  # Warm lighting

# 4. Animation state for testing movement
animation_time = 0.0

# 5. Test canvas
test_canvas = PygletCanvas(renderer)
print("PygletCanvas created successfully!")

# 6. Performance tracking
frame_count = 0
last_fps_time = time.time()
frame_times = []

# 7. Map rendering optimization - only render once
map_rendered = False

# 8. Test effects system
test_particle_system = SubTileParticleSystem(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
test_environmental_system = EnvironmentalEffectSystem()
effect_library = EffectLibrary()
viewport_bounds = Rect(0, 0, config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
view_offset = (0, 0)  # Simple tuple for view offset

# Create different effect contexts for testing
blood_context = EffectContext(
    particle_system=test_particle_system,
    environmental_system=test_environmental_system,
    x=15,
    y=10,
    intensity=1.0,
)

muzzle_context = EffectContext(
    particle_system=test_particle_system,
    environmental_system=test_environmental_system,
    x=20,
    y=8,
    intensity=0.8,
    direction_x=1.0,
    direction_y=0.0,
)

explosion_context = EffectContext(
    particle_system=test_particle_system,
    environmental_system=test_environmental_system,
    x=25,
    y=15,
    intensity=1.2,
)

smoke_context = EffectContext(
    particle_system=test_particle_system,
    environmental_system=test_environmental_system,
    x=10,
    y=20,
    intensity=0.7,
)

# Trigger effects every few seconds
effect_timer = 0.0
last_effect_time = 0.0


def render_test_map():
    """Render the test map using the sprite pool - only once."""
    global map_rendered

    if map_rendered:
        return  # Map is static, only render once

    height, width = test_map.shape

    for y in range(height):
        for x in range(width):
            # Get the character for this tile
            char_code = test_map[y, x]

            # Calculate sprite index (row-major order)
            sprite_index = y * config.SCREEN_WIDTH + x

            # Make sure we don't exceed our sprite pool
            if sprite_index < len(renderer.map_sprites):
                sprite = renderer.map_sprites[sprite_index]

                # Update sprite properties
                sprite.image = renderer.tile_atlas[char_code]
                screen_x, screen_y = renderer.console_to_screen_coords(x, y)
                sprite.x = screen_x
                sprite.y = screen_y
                sprite.visible = True

                # Color tiles based on type
                if char_code == ord("#"):
                    sprite.color = colors.GREY
                elif char_code == ord("+"):
                    sprite.color = colors.ORANGE
                elif char_code == ord("*"):
                    sprite.color = colors.YELLOW
                else:
                    sprite.color = colors.WHITE

    map_rendered = True


def animate_test_actor():
    """Update test actor position with simple circular movement."""
    global animation_time, test_actor_pos

    animation_time += 1.0 / 60.0  # Assume 60fps

    # Circular movement around center point
    center_x, center_y = 20.0, 15.0
    radius = 3.0

    test_actor_pos[0] = center_x + radius * np.cos(animation_time * 0.5)
    test_actor_pos[1] = center_y + radius * np.sin(animation_time * 0.5)

    # Draw the actor
    screen_x, screen_y = renderer.console_to_screen_coords(
        test_actor_pos[0], test_actor_pos[1]
    )

    renderer.draw_actor_smooth(
        test_actor_char, test_actor_color, screen_x, screen_y, test_actor_lighting
    )


# Create UI objects once and reuse them
ui_objects_created = False
ui_labels = []
ui_rects = []
ui_frame = None


def test_pyglet_canvas():
    """Test the PygletCanvas functionality - fully optimized."""
    global ui_objects_created, ui_labels, ui_rects, ui_frame

    # Only create UI objects once, don't update them every frame
    if not ui_objects_created:
        from pyglet.shapes import BorderedRectangle, Rectangle
        from pyglet.text import Label

        # Create labels once with the current batch
        ui_labels = [
            Label(
                "Hello Pyglet Canvas!",
                font_name=str(config.MESSAGE_LOG_FONT_PATH),
                font_size=16,
                x=50,
                y=renderer.window.height - 50,
                anchor_y="top",
                color=(*colors.GREEN, 255),
                batch=renderer.ui_batch,
            ),
            Label(
                "Text rendering test",
                font_name=str(config.MESSAGE_LOG_FONT_PATH),
                font_size=14,
                x=50,
                y=renderer.window.height - 80,
                anchor_y="top",
                color=(*colors.BLUE, 255),
                batch=renderer.ui_batch,
            ),
            Label(
                "Multiple lines work!",
                font_name=str(config.MESSAGE_LOG_FONT_PATH),
                font_size=12,
                x=50,
                y=renderer.window.height - 110,
                anchor_y="top",
                color=(*colors.RED, 255),
                batch=renderer.ui_batch,
            ),
        ]

        # Create rectangles once
        ui_rects = [
            Rectangle(
                x=300,
                y=renderer.window.height - 50 - 80,
                width=150,
                height=80,
                color=(*colors.YELLOW, 255),
                batch=renderer.ui_batch,
            ),
            Rectangle(
                x=320,
                y=renderer.window.height - 70 - 40,
                width=110,
                height=40,
                color=(*colors.CYAN, 255),
                batch=renderer.ui_batch,
            ),
        ]
        ui_rects[0].opacity = 128  # Unfilled rect

        # Create frame once
        tile_w, tile_h = renderer.tile_dimensions
        ui_frame = BorderedRectangle(
            x=2 * tile_w,
            y=renderer.window.height - (2 + 8) * tile_h,
            width=15 * tile_w,
            height=8 * tile_h,
            border=1,
            color=colors.BLACK,
            border_color=colors.WHITE,
            batch=renderer.ui_batch,
        )

        ui_objects_created = True

    # Since UI objects were created with the batch that gets recreated each frame,
    # they will automatically be included in the new batch when it's drawn.
    # No need to update batch references every frame.

    return  # No artifact needed


@window.event
def on_draw():
    global frame_count, last_fps_time, frame_times

    frame_start = time.time()

    # Use the renderer's proper methods instead of manual clearing
    prepare_start = time.time()
    renderer.prepare_to_present()
    prepare_time = time.time() - prepare_start

    # Canvas batch will be updated automatically

    # Test 1: Render the map
    map_start = time.time()
    render_test_map()
    map_time = time.time() - map_start

    # Test 2: Draw animated actor
    actor_start = time.time()
    animate_test_actor()
    actor_time = time.time() - actor_start

    # Test 3: Draw some tile highlights for testing
    highlight_start = time.time()
    renderer.draw_tile_highlight(5, 5, colors.YELLOW, Opacity(0.6))
    renderer.draw_tile_highlight(15, 10, colors.CYAN, Opacity(0.4))
    highlight_time = time.time() - highlight_start

    # Test 4: Update effects and render particles
    particles_start = time.time()
    global effect_timer, last_effect_time

    # Update timer
    dt = 1.0 / 60.0
    effect_timer += dt

    # Trigger different effects periodically
    if effect_timer - last_effect_time > 2.0:  # Every 2 seconds
        current_time = int(effect_timer) % 8  # Cycle through effects
        if current_time == 0:
            effect_library.trigger("blood_splatter", blood_context)
            print("Triggered blood splatter effect")
        elif current_time == 2:
            effect_library.trigger("muzzle_flash", muzzle_context)
            print("Triggered muzzle flash effect")
        elif current_time == 4:
            effect_library.trigger("explosion", explosion_context)
            print("Triggered explosion effect")
        elif current_time == 6:
            effect_library.trigger("smoke_cloud", smoke_context)
            print("Triggered smoke cloud effect")
        last_effect_time = effect_timer

    # Update particle and environmental systems
    test_particle_system.update(DeltaTime(dt))
    test_environmental_system.update(DeltaTime(dt))

    # Render particles and environmental effects
    renderer.render_particles(
        test_particle_system, ParticleLayer.OVER_ACTORS, viewport_bounds, view_offset
    )
    # Apply environmental effects (manually for testing)
    for effect in test_environmental_system.effects:
        renderer.apply_environmental_effect(
            position=effect.position,
            radius=effect.radius,
            tint_color=effect.tint_color,
            intensity=effect.intensity,
            blend_mode=effect.blend_mode,
        )
    particles_time = time.time() - particles_start

    # Test 5: Test the canvas
    canvas_start = time.time()
    test_pyglet_canvas()
    canvas_time = time.time() - canvas_start

    # Use the renderer's finalize method to draw all batches
    finalize_start = time.time()
    renderer.finalize_present()
    finalize_time = time.time() - finalize_start

    frame_end = time.time()
    total_frame_time = frame_end - frame_start
    frame_times.append(total_frame_time)

    frame_count += 1
    if frame_count % 60 == 0:  # Print stats every 60 frames
        current_time = time.time()
        fps = 60 / (current_time - last_fps_time)
        avg_frame_time = sum(frame_times[-60:]) / min(60, len(frame_times))

        print("\n=== PERFORMANCE STATS ===")
        print(f"FPS: {fps:.1f}")
        print(f"Avg frame time: {avg_frame_time * 1000:.2f}ms")
        print(f"  Prepare: {prepare_time * 1000:.2f}ms")
        print(f"  Map: {map_time * 1000:.2f}ms")
        print(f"  Actor: {actor_time * 1000:.2f}ms")
        print(f"  Highlights: {highlight_time * 1000:.2f}ms")
        print(f"  Particles: {particles_time * 1000:.2f}ms")
        print(f"  Canvas: {canvas_time * 1000:.2f}ms")
        print(f"  Finalize: {finalize_time * 1000:.2f}ms")
        other_time = (
            total_frame_time
            - prepare_time
            - map_time
            - actor_time
            - highlight_time
            - particles_time
            - canvas_time
            - finalize_time
        )
        print(f"  Other: {other_time * 1000:.2f}ms")

        last_fps_time = current_time


pyglet.app.run()
