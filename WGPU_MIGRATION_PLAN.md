# WGPU Migration Plan for Catley

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py while maintaining Pyglet as the window manager.

## Overview

The migration strategy uses **Option 1: Keep Pyglet as Window Manager** by creating a custom `PygletWgpuCanvas` that bridges Pyglet's windowing system with wgpu's rendering pipeline.

## Phase 1: Pyglet-WGPU Bridge Implementation

### Step 1.1: Create PygletWgpuCanvas Base Implementation

**Location**: `catley/backends/wgpu/pyglet_canvas.py`

Create a custom canvas that implements the required `WgpuCanvasBase` interface:

```python
from wgpu.gui import WgpuCanvasBase
import wgpu
import sys

class PygletWgpuCanvas(WgpuCanvasBase):
    def __init__(self, pyglet_window):
        super().__init__()
        self._window = pyglet_window
        self._closed = False
        
    def get_pixel_ratio(self):
        """Return the ratio between logical and physical pixels."""
        # Pyglet doesn't expose this directly, but we can approximate
        # based on the window's framebuffer size vs window size
        return 1.0  # Start with 1.0, refine based on platform
        
    def get_logical_size(self):
        """Return the logical size in float pixels."""
        return (float(self._window.width), float(self._window.height))
        
    def get_physical_size(self):
        """Return the physical size in integer pixels."""
        # This should be the framebuffer size
        return (self._window.width, self._window.height)
        
    def set_logical_size(self, width, height):
        """Set the window size in logical pixels."""
        self._window.set_size(int(width), int(height))
        
    def set_title(self, title):
        """Set the window title."""
        self._window.set_caption(title)
        
    def is_closed(self):
        """Check if the window is closed."""
        return self._closed
        
    def _request_draw(self):
        """Schedule a draw - integrate with Pyglet's event loop."""
        # This will need to trigger a redraw in Pyglet's event system
        self._window.dispatch_event('on_draw')
```

### Step 1.2: Research Pyglet Native Window Access

**Task**: Investigate Pyglet's source code to find the correct way to access native window handles on each platform.

- Windows: Look for `_hwnd` or similar Win32 handle storage
- macOS: Look for `NSWindow` or Cocoa window references  
- Linux: Look for X11 window IDs and display connections

**Fallback Strategy**: If we cannot get Pyglet window handles working, we can switch to GLFW as wgpu already supports it. However, this would require reimplementing our entire `PygletApp` infrastructure including:
- Event handling and input management
- Window lifecycle management  
- Main loop integration
- All the custom event dispatch logic

Therefore, getting PygletWgpuCanvas working is strongly preferred to preserve our existing application infrastructure.

### Step 1.3: Implement Platform-Specific Surface Creation

Once we understand how to access Pyglet's native handles, implement `get_present_methods()`:

```python
def get_present_methods(self):
    """Return a dict with supported present methods for this canvas."""
    # Get the native window handle from Pyglet
    if sys.platform == "win32":
        # Windows: Get HWND
        from pyglet.libs.win32 import _user32
        hwnd = self._window._hwnd  # Pyglet stores this internally
        return {
            "screen": {
                "window": hwnd,
            }
        }
    elif sys.platform == "darwin":
        # macOS: Get NSWindow
        # This requires using Pyglet's Cocoa integration
        ns_window = self._window._nswindow  # Pyglet's internal reference
        return {
            "screen": {
                "window": ns_window,
            }
        }
    elif sys.platform.startswith("linux"):
        # Linux: Handle both X11 and Wayland
        # This is more complex as we need to detect the display server
        display = self._window.context.x_display  # For X11
        window = self._window._window  # X11 window ID
        return {
            "screen": {
                "window": window,
                "display": display,
            }
        }
```

### Step 1.4: Proof of Concept Testing

Before proceeding to Phase 2, create a minimal test to validate the Pyglet-WGPU bridge:

**Test Script**: `test_pyglet_wgpu_bridge.py`

```python
import pyglet
import wgpu
from catley.backends.wgpu.pyglet_canvas import PygletWgpuCanvas

# Create a simple Pyglet window
window = pyglet.window.Window(800, 600, "WGPU Bridge Test")

# Create our canvas bridge
canvas = PygletWgpuCanvas(window)

# Try to create a WGPU context
try:
    # Get adapter and device
    adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
    device = adapter.request_device()
    
    # Create canvas context
    context = canvas.create_canvas_context()
    
    # Configure the context
    context.configure(device=device, format=wgpu.TextureFormat.bgra8unorm)
    
    # Create a simple render pipeline that clears to red
    shader = device.create_shader_module(code='''
        @vertex
        fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
            var pos = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>( 3.0, -1.0),
                vec2<f32>(-1.0,  3.0)
            );
            return vec4<f32>(pos[vertex_index], 0.0, 1.0);
        }
        
        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);  // Red
        }
    ''')
    
    pipeline = device.create_render_pipeline(
        vertex={"module": shader, "entry_point": "vs_main"},
        fragment={"module": shader, "entry_point": "fs_main", "targets": [{"format": wgpu.TextureFormat.bgra8unorm}]},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    )
    
    # Render function
    def render():
        # Get the current texture to render to
        texture = context.get_current_texture()
        
        # Create command encoder
        command_encoder = device.create_command_encoder()
        
        # Create render pass
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": texture.create_view(),
                "load_op": wgpu.LoadOp.clear,
                "clear_value": (1, 0, 0, 1),  # Red
                "store_op": wgpu.StoreOp.store,
            }]
        )
        
        # Draw fullscreen triangle
        render_pass.set_pipeline(pipeline)
        render_pass.draw(3)
        render_pass.end()
        
        # Submit commands
        device.queue.submit([command_encoder.finish()])
        
        # Present the frame
        context.present()
    
    @window.event
    def on_draw():
        render()
    
    print("✅ SUCCESS: WGPU context created and configured!")
    print("✅ Window should display red background")
    print("✅ Pyglet-WGPU bridge is working!")
    
    pyglet.app.run()
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("Need to investigate Pyglet window handle access")
```

**Success Criteria**:
1. Window appears and shows red background
2. No crashes or errors
3. Can handle window resize/close events
4. Validates that native window handles are correctly accessed

**Failure Handling**:
- If this test fails, investigate the exact error
- Use debugger to inspect Pyglet window object attributes
- Check Pyglet version compatibility
- Consider GLFW fallback if absolutely necessary

## Phase 2: Backend Scaffolding

### Step 2.1: Create Directory Structure

```
catley/backends/wgpu/
├── __init__.py
├── graphics.py          # WGPUGraphicsContext
├── canvas.py           # WGPUCanvas (uses PygletWgpuCanvas)
├── pyglet_canvas.py    # PygletWgpuCanvas implementation
├── resource_manager.py # WGPU resource management
└── shader_utils.py     # WGSL shader utilities
```

### Step 2.2: Create WGPUGraphicsContext Skeleton

Implement a minimal `GraphicsContext` that can be instantiated:

```python
class WGPUGraphicsContext(GraphicsContext):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.canvas = PygletWgpuCanvas(window)
        self.device = None
        self.queue = None
        
    def initialize(self):
        """Initialize WGPU device and queue."""
        # Create canvas context
        self.context = self.canvas.create_canvas_context()
        
        # Get adapter and device
        adapter = wgpu.request_adapter(canvas=self.canvas, power_preference="high-performance")
        self.device = adapter.request_device()
        self.queue = self.device.queue
```

## Phase 3: Core Rendering Port

### Step 3.1: Performance Baseline

Before any porting work:
1. Create benchmark script measuring ModernGL performance
2. Test scenarios: 10, 50, 100, 200 lights with shadows
3. Measure: FPS, frame time, memory usage
4. Document baseline metrics

### Step 3.2: Shader Translation (GLSL → WGSL)

**Critical Shaders to Translate**:
1. `screen/main.vert` + `screen/main.frag` → `screen/main.wgsl`
2. `ui/texture.vert` + `ui/texture.frag` → `ui/texture.wgsl`
3. `glyph/render.vert` + `glyph/render.frag` → `glyph/render.wgsl`

**Translation Guidelines**:
- WGSL uses different syntax for uniforms (use `uniform` blocks)
- Vertex input/output uses `@location` attributes
- Fragment outputs use `@location(0)` for color
- Texture sampling syntax differs

### Step 3.3: Implement Basic Tile Rendering

Start with `ScreenRenderer` equivalent that can:
1. Render background tiles
2. Handle viewport transformations
3. Support basic texture atlasing

## Phase 4: GPU Lighting System Port

### Step 4.1: Translate Lighting Shader

**Special Attention Required**:
1. **Directional Shadow Algorithm**: The discrete tile-based stepping must be preserved exactly
2. **Shadow Direction Calculation**: Sign-based (not normalized) direction vectors
3. **Sky Exposure Sampling**: Texture coordinate mapping for indoor/outdoor detection

Critical sections to preserve:
```glsl
// GLSL (current)
float shadow_dx = u_sun_direction.x > 0.0 ? -1.0 : (u_sun_direction.x < 0.0 ? 1.0 : 0.0);
float shadow_dy = u_sun_direction.y > 0.0 ? -1.0 : (u_sun_direction.y < 0.0 ? 1.0 : 0.0);

// WGSL (translate to)
let shadow_dx = select(select(0.0, 1.0, u_sun_direction.x < 0.0), -1.0, u_sun_direction.x > 0.0);
let shadow_dy = select(select(0.0, 1.0, u_sun_direction.y < 0.0), -1.0, u_sun_direction.y > 0.0);
```

### Step 4.2: Port GPULightingSystem

Create `catley/backends/wgpu/gpu_lighting.py`:

Key methods to port:
- `_collect_light_data()` - Should be nearly identical
- `_set_lighting_uniforms()` - Adapt for WGPU uniform buffer updates
- `_set_directional_light_uniforms()` - Preserve all calculations
- `_update_sky_exposure_texture()` - Texture creation syntax differs
- `_compute_lightmap_gpu()` - Main rendering logic

### Step 4.3: Uniform Buffer Management

WGPU uses explicit uniform buffers instead of individual uniform setters:

```python
# Create uniform buffer for light data
light_buffer = self.device.create_buffer(
    size=light_data_size,
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

# Update uniform buffer
self.queue.write_buffer(light_buffer, 0, light_data_bytes)
```

## Phase 5: Integration and Validation

### Step 5.1: Modify Controller Integration

Update `Controller.__init__` to support WGPU backend selection:

```python
if config.WGPU_BACKEND_ENABLED and wgpu_available:
    from catley.backends.wgpu.graphics import WGPUGraphicsContext
    self.graphics = WGPUGraphicsContext(window)
else:
    from catley.backends.moderngl.graphics import ModernGLGraphicsContext  
    self.graphics = ModernGLGraphicsContext(window)
```

### Step 5.2: Visual Parity Testing

1. Implement screenshot capture for both backends
2. Create test scenes with known configurations:
   - Outdoor scene with directional shadows
   - Indoor scene with torch lighting
   - Transition areas with light spillover
3. Compare pixel-by-pixel for differences
4. Document any acceptable variations

### Step 5.3: Performance Validation

1. Run same benchmarks as Step 3.1
2. Compare WGPU vs ModernGL performance
3. Ensure WGPU meets or exceeds baseline
4. Profile any performance regressions

## Phase 6: Advanced Multi-Pass Architecture

**Reference**: LIGHTING_SYSTEM_PLAN.md Phase 2.3

Once basic parity is achieved, implement the advanced architecture:
1. Static light texture caching
2. Dynamic light per-frame rendering  
3. Intelligent cache invalidation
4. Compute shader optimization (if beneficial)

## Critical Success Factors

1. **Native Window Handle Access**: Must successfully get Pyglet window handles
2. **Shader Translation Accuracy**: Shadows must behave identically
3. **Performance Parity**: No regression from ModernGL
4. **Visual Parity**: Screenshots must match
5. **Pyglet Integration**: Event loop and window management must work seamlessly

## Risk Mitigation

1. **Window Handle Access Fails**: 
   - Fallback: Use offscreen rendering + Pyglet texture display
   - Fallback: Fork Pyglet to expose handles
   
2. **Shader Translation Issues**:
   - Use WGSL validator early and often
   - Test individual shader features in isolation
   
3. **Performance Regression**:
   - Profile early in the process
   - Consider keeping ModernGL as permanent fallback
   
4. **Integration Complexity**:
   - Build minimal proof-of-concept first
   - Test Pyglet+WGPU integration before full port

## Next Steps

1. Research Pyglet source for native window handles
2. Create minimal Pyglet+WGPU proof of concept
3. Validate window handle access works on target platforms
4. Begin shader translation with simple examples
5. Proceed with full implementation plan