# WGPU Migration Plan for Catley

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py.

## Phase 1: GLFW Migration (Prerequisite)

### Step 1.1: Pause WGPU Migration
**Rationale**: Pyglet is fundamentally incompatible with WGPU due to architectural differences:
- Pyglet is designed exclusively for OpenGL and assumes complete control of the rendering pipeline
- WGPU uses Metal on macOS, creating a conflict where both OpenGL (Pyglet) and Metal (WGPU) try to render to the same NSView
- The Pyglet-WGPU bridge approach is not viable

### Step 1.2: Add GLFW Dependency
Add `glfw` as a dependency to enable WGPU integration.

### Step 1.3: Implement GlfwApp
1. **Write new main loop** (`while not glfw.window_should_close()`)
2. **Write new input handlers** for keyboard, mouse, scroll, etc., translating their events into the tcod.event format your InputHandler expects. Re-implement the logic from your PygletApp
3. **Get current ModernGL backend working** on top of GlfwApp. This is a critical validation step

### Step 1.4: Validation
Once GlfwApp is working and stable with ModernGL, resume the WGPU migration.

## Phase 2: WGPU Implementation (After GLFW Migration)

### Step 2.1: GLFW-WGPU Bridge Implementation

Create a minimal test to validate the GLFW-WGPU bridge:

**Test Script**: `test_glfw_wgpu_bridge.py`

```python
import glfw
import wgpu
from wgpu.gui.glfw import GlfwWgpuCanvas

# Initialize GLFW
if not glfw.init():
    raise Exception("Failed to initialize GLFW")

# Create GLFW window
window = glfw.create_window(800, 600, "WGPU Bridge Test", None, None)
if not window:
    glfw.terminate()
    raise Exception("Failed to create GLFW window")

# Create WGPU canvas
canvas = GlfwWgpuCanvas(window)

# Get adapter and device
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()

# Create canvas context
context = canvas.get_context("wgpu")

# Get preferred format and configure the context
render_texture_format = context.get_preferred_format(adapter)
context.configure(device=device, format=render_texture_format)

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

pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={"module": shader, "entry_point": "vs_main"},
    fragment={"module": shader, "entry_point": "fs_main", "targets": [{"format": render_texture_format}]},
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

# Main loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    render()

print("✅ SUCCESS: WGPU context created and configured!")
print("✅ Window should display red background")
print("✅ GLFW-WGPU bridge is working!")

glfw.terminate()
```

**Success Criteria**:
1. Window appears and shows red background
2. No crashes or errors
3. Can handle window resize/close events
4. Validates that native window handles are correctly accessed

### Step 2.2: Backend Scaffolding

**Create directory structure**:
```
catley/backends/wgpu/
├── __init__.py
├── graphics.py          # WGPUGraphicsContext
├── canvas.py           # WGPUCanvas (uses GlfwWgpuCanvas)
├── glfw_canvas.py      # GlfwWgpuCanvas implementation
├── resource_manager.py # WGPU resource management
└── shader_utils.py     # WGSL shader utilities
```

**Create WGPUGraphicsContext skeleton** using correct WGPU API patterns:

```python
class WGPUGraphicsContext(GraphicsContext):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.canvas = GlfwWgpuCanvas(window)
        self.device = None
        self.queue = None

    def initialize(self):
        """Initialize WGPU device and queue."""
        # Correct WGPU API usage patterns:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = adapter.request_device_sync()
        self.queue = self.device.queue

        # Create and configure canvas context
        self.context = self.canvas.get_context("wgpu")
        render_texture_format = self.context.get_preferred_format(adapter)
        self.context.configure(device=self.device, format=render_texture_format)
```

**Key API Notes**:
- Use `wgpu.gpu.request_adapter_sync()` not `wgpu.request_adapter()`
- Use `adapter.request_device_sync()` not `adapter.request_device()`
- Use `canvas.get_context("wgpu")` not `canvas.create_canvas_context()`
- Use `context.get_preferred_format(adapter)` instead of hardcoded formats
- `create_render_pipeline()` requires `layout` parameter

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

**Critical sections to preserve exactly**:
1. **Directional Shadow Algorithm**: The discrete tile-based stepping must be preserved exactly
2. **Shadow Direction Calculation**: Sign-based (not normalized) direction vectors
3. **Sky Exposure Sampling**: Texture coordinate mapping for indoor/outdoor detection

**GLSL to WGSL translation example**:
```glsl
// GLSL (current)
float shadow_dx = u_sun_direction.x > 0.0 ? -1.0 : (u_sun_direction.x < 0.0 ? 1.0 : 0.0);
float shadow_dy = u_sun_direction.y > 0.0 ? -1.0 : (u_sun_direction.y < 0.0 ? 1.0 : 0.0);

// WGSL (translate to)
let shadow_dx = select(select(0.0, 1.0, u_sun_direction.x < 0.0), -1.0, u_sun_direction.x > 0.0);
let shadow_dy = select(select(0.0, 1.0, u_sun_direction.y < 0.0), -1.0, u_sun_direction.y > 0.0);
```

### Step 4.2: Port GPULightingSystem

**Create `catley/backends/wgpu/gpu_lighting.py`** with these key methods:
- `_collect_light_data()` - Should be nearly identical
- `_set_lighting_uniforms()` - Adapt for WGPU uniform buffer updates
- `_set_directional_light_uniforms()` - Preserve all calculations
- `_update_sky_exposure_texture()` - Texture creation syntax differs
- `_compute_lightmap_gpu()` - Main rendering logic

### Step 4.3: Uniform Buffer Management

**WGPU uses explicit uniform buffers** instead of individual uniform setters:

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

**Update `Controller.__init__`** to support WGPU backend selection:

```python
if config.WGPU_BACKEND_ENABLED and wgpu_available:
    from catley.backends.wgpu.graphics import WGPUGraphicsContext
    self.graphics = WGPUGraphicsContext(glfw_window)
else:
    from catley.backends.moderngl.graphics import ModernGLGraphicsContext
    self.graphics = ModernGLGraphicsContext(glfw_window)
```

### Step 5.2: Visual Parity Testing

1. **Implement screenshot capture** for both backends
2. **Create test scenes** with known configurations:
   - Outdoor scene with directional shadows
   - Indoor scene with torch lighting
   - Transition areas with light spillover
3. **Compare pixel-by-pixel** for differences
4. **Document any acceptable variations**

### Step 5.3: Performance Validation

1. **Run same benchmarks** as Step 3.1
2. **Compare WGPU vs ModernGL** performance
3. **Ensure WGPU meets or exceeds** baseline
4. **Profile any performance regressions**

## Phase 6: Advanced Multi-Pass Architecture

### Step 6.1: Advanced Architecture Implementation
**Reference**: LIGHTING_SYSTEM_PLAN.md Phase 2.3

Once basic parity is achieved, implement the advanced architecture:
1. **Static light texture caching**
2. **Dynamic light per-frame rendering**
3. **Intelligent cache invalidation**
4. **Compute shader optimization** (if beneficial)

## Critical Success Factors

1. **GLFW Migration**: Must successfully migrate from Pyglet to GLFW with ModernGL first
2. **Native Window Handle Access**: Must successfully get GLFW window handles for WGPU
3. **Shader Translation Accuracy**: Shadows must behave identically
4. **Performance Parity**: No regression from ModernGL
5. **Visual Parity**: Screenshots must match
6. **GLFW Integration**: Event loop and window management must work seamlessly

## Risk Mitigation

1. **GLFW Migration Complexity**:
   - Implement step-by-step with ModernGL first
   - Preserve existing input handling patterns
   - Test thoroughly before WGPU migration

2. **Window Handle Access Fails**:
   - GLFW has proven WGPU integration (standard pattern)
   - Use wgpu.gui.glfw.GlfwWgpuCanvas (well-tested)

3. **Shader Translation Issues**:
   - Use WGSL validator early and often
   - Test individual shader features in isolation

4. **Performance Regression**:
   - Profile early in the process
   - Consider keeping ModernGL as permanent fallback

5. **Integration Complexity**:
   - Build minimal proof-of-concept first
   - Test GLFW+WGPU integration before full port

## Next Steps

1. ✅ ~~Research Pyglet source for native window handles~~ **COMPLETE - Found incompatibility**
2. ✅ ~~Create minimal Pyglet+WGPU proof of concept~~ **COMPLETE - Confirmed incompatibility**
3. 🔄 **NEW PRIORITY**: Implement GLFW migration for ModernGL backend
4. Validate GLFW+ModernGL works with existing input/event systems
5. Create minimal GLFW+WGPU proof of concept
6. Begin shader translation with simple examples
7. Proceed with full WGPU implementation plan