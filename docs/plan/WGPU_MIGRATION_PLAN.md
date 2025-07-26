# WGPU Migration Plan for Catley

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py.

**Important**: Maintain implementation parity between backends to prevent subtle bugs. Unintentional divergences can cause hard-to-debug visual artifacts. For example, the introduction of dirty-tile optimization initially caused visual artifacts (ghosting of previously lit tiles). Root cause: the WGPU implementation had diverged from ModernGL by adding a transparent tile optimization that skipped generating vertices for fully transparent tiles. This caused stale vertex data to persist in the VBO. The fix was to remove this optimization and ensure WGPU matches ModernGL's behavior of always generating vertices for all tiles.

## Phase 3: WGPU Performance Optimization

**Current Performance Gap**: WGPU renders at ~180-240 FPS (~3.7-4.4ms render time) vs ModernGL at ~140-300 FPS (~1.7-2.7ms render time).

Analysis of the existing ModernGL and WGPU backends reveals that the performance gap is not due to a fundamental limitation of WGPU, but rather a specific, critical optimization present in the ModernGL backend that is currently missing from the WGPU implementation. The following plan prioritizes closing this gap first, followed by broader architectural improvements.

### **Step 3.3: Consolidate Command Encoder and Isolate Canvas Resources**

**Priority: HIGH** - Fixes a performance bottleneck and prevents a critical rendering bug.

1.  **Observation:** The main render loop in `WGPUGraphicsContext.finalize_present` correctly uses a single `command_encoder` for the final scene. However, the `WGPUTextureRenderer.render` method creates its own `command_encoder` and submits it to the GPU for *every single* render-to-texture operation (e.g., for each UI element).

2.  **Problem (Compound Issue):**
    *   **Primary Performance Issue:** Creating and submitting many command buffers per frame is a major performance anti-pattern in modern graphics APIs like WebGPU. It introduces significant driver overhead and prevents the GPU from efficiently scheduling work, leading to lower performance compared to the ModernGL backend where the driver handles batching implicitly.
    *   **Latent Correctness Bug:** A critical secondary problem exists that will be exposed by fixing the primary issue. The `WGPUTextureRenderer` uses a single, shared CPU-side vertex buffer. When command recording is deferred (as it will be with a single encoder), all UI views will write their vertex data to this shared buffer sequentially. By the time the GPU finally executes the commands at the end of the frame, the buffer will only contain the data from the *last* view that was drawn. This will cause all UI elements to render with the same incorrect content, creating severe visual artifacts.

3.  **Action (Combined Solution):** We must fix both problems at once. The solution is to refactor the rendering logic to use a single command buffer per frame **AND** simultaneously ensure that each UI canvas has its own isolated resources, preventing the data race. This will mirror the correct and performant architecture already present in the ModernGL backend.

    *   **Part A: Make the Renderer Stateless.** The `WGPUTextureRenderer` must be refactored to be a stateless service.
        *   **A.1:** In `catley/backends/wgpu/texture_renderer.py`, remove the `self.cpu_vertex_buffer`, `self.vertex_count`, and `self.vertex_buffer` instance variables from its `__init__` method. The renderer should no longer own any per-draw state.
        *   **A.2:** Modify the `render()` method signature in `WGPUTextureRenderer` to accept all necessary resources as arguments: a `command_encoder`, a `cpu_buffer_override`, and a `buffer_override` (the GPU buffer).
        *   **A.3:** The `render()` method's logic will now be to:
            1.  Populate the provided `cpu_buffer_override` with vertex data.
            2.  Record a `write_buffer` command into the provided `command_encoder` to copy data from the `cpu_buffer_override` to the `buffer_override`.
            3.  Record a render pass into the `command_encoder` that draws using the `buffer_override`.
            4.  It must **not** `finish()` or `submit()` the encoder.

    *   **Part B: Make the Canvas Stateful.** Each `WGPUCanvas` instance must own its own resources.
        *   **B.1:** In `catley/backends/wgpu/canvas.py`, ensure the `WGPUCanvas` class has its own instance variables: `self.cpu_vertex_buffer` (a NumPy array) and `self.vertex_buffer` (a `wgpu.GPUBuffer`).
        *   **B.2:** In the `configure_dimensions` method of `WGPUCanvas`, create or resize these two buffers. This ensures they are pre-allocated and reused, avoiding per-frame allocation overhead.
        *   **B.3:** Update `WGPUCanvas.create_texture` to pass its unique `self.cpu_vertex_buffer` and `self.vertex_buffer` to the graphics context's `render_glyph_buffer_to_texture` method.

    *   **Part C: Orchestrate the New Flow in the Graphics Context.**
        *   **C.1:** In `catley/backends/wgpu/graphics.py`, add a `self._frame_command_encoder` instance variable to `WGPUGraphicsContext`.
        *   **C.2:** In `prepare_to_present()`, create and store a single `GPUCommandEncoder` in `self._frame_command_encoder`.
        *   **C.3:** Update the `render_glyph_buffer_to_texture` method to accept the per-canvas CPU and GPU buffers (from Part B.3) and pass them, along with the shared `self._frame_command_encoder`, to the stateless renderer's `render` method (from Part A.2).
        *   **C.4:** In `finalize_present()`, use the single `self._frame_command_encoder` for the final on-screen render pass, then call `finish()` on it once, and `submit()` the resulting command buffer to the queue once. Clear `self._frame_command_encoder` to `None` at the end of the method.

-   **Combined Optimizations**: Should achieve or exceed ModernGL parity, targeting a **~2-3ms** render time.

### Success Metrics

1.  **Target Performance**: ≤3ms render time (matching ModernGL baseline).
2.  **Frame Rate**: ≥140 FPS average on the same hardware.
3.  **Memory Usage**: No significant increase from the current WGPU implementation.
4.  **Visual Parity**: No rendering artifacts or visual regressions compared to the ModernGL backend.

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

1. **Run WGPU benchmarks** using the realistic performance script:
   ```bash
   uv run python scripts/benchmark_realistic_performance.py --verbose --duration 10.0 --export-json benchmarks/wgpu_performance_results.json
   ```
2. **Compare against ModernGL baseline** found in `benchmarks/realistic_moderngl_baseline.json`
3. **Validate performance targets** from `benchmarks/REALISTIC_MODERNGL_BASELINE.md`:
   - **10-25 lights**: ≥100 FPS average, ≥60 FPS P95
   - **50 lights**: ≥200 FPS average, ≥80 FPS P95
   - **Frame drops**: ≤50 per 10-second test
   - **Memory**: ≤250MB total usage
4. **Profile any performance regressions** if WGPU fails to meet baseline targets

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