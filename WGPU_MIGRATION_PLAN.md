# WGPU Migration Plan for Catley

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py.

## Phase 2: WGPU Implementation (After GLFW Migration)

### Step 2.2: Backend Scaffolding

**Create directory structure**:
```
catley/backends/wgpu/
├── __init__.py
├── graphics.py          # WGPUGraphicsContext (main entry point)
├── canvas.py           # WGPUCanvas (extends catley.view.render.canvas.Canvas)
├── resource_manager.py # WGPU resource management (buffers, textures, pipelines)
├── screen_renderer.py  # WGPU screen rendering (port of ModernGL version)
├── texture_renderer.py # WGPU texture rendering (port of ModernGL version)
├── shader_manager.py   # WGSL shader management and compilation
└── gpu_lighting.py     # WGPU lighting system (port of ModernGL version)
```

**Create WGPUGraphicsContext skeleton** using correct WGPU API patterns:

```python
class WGPUGraphicsContext(GraphicsContext):
    def __init__(self, window_size=(800, 600), title="Catley"):
        super().__init__()
        # Let GlfwWgpuCanvas create its own window - don't pass existing GLFW window
        self.canvas = GlfwWgpuCanvas(size=window_size, title=title)
        self.window = self.canvas._window  # Access GLFW window after creation
        self.device = None
        self.queue = None

    def initialize(self):
        """Initialize WGPU device and queue."""
        # Correct WGPU API usage patterns:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference=wgpu.PowerPreference.high_performance  # type: ignore
        )
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
- **CRITICAL**: Use `GlfwWgpuCanvas(size=..., title=...)` - don't pass existing GLFW window
- **CRITICAL**: Access GLFW window via `canvas._window` after creation
- **CRITICAL**: Add `# type: ignore` for PowerPreference enum to satisfy type checker

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
3. ✅ ~~Create minimal GLFW+WGPU proof of concept~~ **COMPLETE - Step 2.1 validated**
4. 🔄 **CURRENT PRIORITY**: Implement GLFW migration for ModernGL backend
5. Validate GLFW+ModernGL works with existing input/event systems
6. Begin Step 2.2: Backend Scaffolding (create `catley/backends/wgpu/` structure)
7. Begin shader translation with simple examples
8. Proceed with full WGPU implementation plan