import pyglet
from pyglet.window import Window

from catley import config
from catley.view.render.pyglet_renderer import PygletRenderer

# from catley.view.render.pyglet_canvas import PygletCanvas
# from catley.view.views.health_view import HealthView # Test a view

# Basic window setup
window = Window(
    width=config.SCREEN_WIDTH * 16,  # Start with some reasonable pixel size
    height=config.SCREEN_HEIGHT * 16,
    caption="Pyglet Renderer Test Harness",
    resizable=True,
)

# --- YOUR TEST SETUP GOES HERE ---
# 1. Instantiate your PygletRenderer
try:
    # This will fail until you create PygletRenderer, which is the point!
    renderer = PygletRenderer(window)
except NameError:
    print("PygletRenderer not yet created. Exiting.")
    exit()

# 2. Example: Test drawing a single sprite
# tile_atlas = renderer.get_tile_atlas() # A method you'll add to your renderer
# test_sprite = pyglet.sprite.Sprite(img=tile_atlas[ord('@')], batch=renderer.ui_batch)


# 3. Example: Test rendering a View
# health_view = HealthView(...) # Needs a dummy controller/gw
# health_view.canvas = PygletCanvas(renderer) # Manually assign the new canvas
# health_view.set_bounds(...)


@window.event
def on_draw():
    window.clear()

    # --- YOUR TEST DRAWING GOES HERE ---
    # Call the methods on your renderer to see if they work.

    # renderer.world_batch.draw()
    # renderer.ui_batch.draw()

    # health_view.draw_content() # Test the canvas
    # health_view.present() # Test texture presentation


pyglet.app.run()
