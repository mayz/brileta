# WGPU Migration Plan for Catley

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py.

**Important**: Maintain implementation parity between backends to prevent subtle bugs. Unintentional divergences can cause hard-to-debug visual artifacts. For example, the introduction of dirty-tile optimization initially caused visual artifacts (ghosting of previously lit tiles). Root cause: the WGPU implementation had diverged from ModernGL by adding a transparent tile optimization that skipped generating vertices for fully transparent tiles. This caused stale vertex data to persist in the VBO. The fix was to remove this optimization and ensure WGPU matches ModernGL's behavior of always generating vertices for all tiles.

## Phase 3: WGPU Performance Optimization

**Current Performance Gap**: WGPU renders at ~180-240 FPS (~3.7-4.4ms render time) vs ModernGL at ~140-300 FPS (~1.7-2.7ms render time).

Analysis of the existing ModernGL and WGPU backends reveals that the performance gap is not due to a fundamental limitation of WGPU, but rather a specific, critical optimization present in the ModernGL backend that is currently missing from the WGPU implementation. The following plan prioritizes closing this gap first, followed by broader architectural improvements.

### Step 3.2: Standardize All Rendering on a Batched Model (HIGH)

**Priority: HIGH** - Ensures architectural consistency and reduces draw call overhead.

1.  **Observation**: The WGPU backend has already made positive steps by batching background draws in `WGPUBackgroundRenderer`. However, the ModernGL backend's `apply_environmental_effect` uses an immediate-mode style draw (`_draw_environmental_effect_immediately`), a pattern that should be avoided in WGPU.
2.  **Problem**: Mixing batched and immediate-mode rendering complicates the render loop and can lead to inefficient use of the GPU.
3.  **Action**: Ensure that all rendering components follow a consistent batched pattern. When implementing `apply_environmental_effect` for WGPU, create a dedicated `WGPUEnvironmentalEffectRenderer` that queues effect quads, similar to how the `ScreenRenderer` and `UITextureRenderer` work. This will consolidate all drawing into a single, predictable render pass.

### Step 3.3: Consolidate Command Encoder Usage (MEDIUM)

**Priority: MEDIUM** - Reduces driver overhead.

1.  **Observation**: The main render loop in `WGPUGraphicsContext.finalize_present` correctly uses a single `command_encoder`. However, the `WGPUTextureRenderer.render` method creates its own `command_encoder` for every render-to-texture operation.
2.  **Problem**: Creating multiple command encoders per frame introduces unnecessary overhead, especially if several UI elements need to be redrawn simultaneously.
3.  **Action**: Refactor the rendering logic to allow multiple render-to-texture passes to be recorded into a single command buffer, which is then submitted once at the end of all UI drawing.

### Step 3.4: Integrate Performance Monitoring (MEDIUM)

**Priority: MEDIUM** - Long-term maintainability.

1.  **Action**: This step remains the same as the original plan.
    *   Integrate timing measurements around key rendering stages in the WGPU backend to identify regressions and bottlenecks.
    *   Establish automated performance testing in CI to ensure performance does not degrade over time.
    *   Use benchmarking tools to validate that performance targets are being met after optimizations.

### Expected Performance Improvements

-   **Dirty-Tile Optimization**: **~70-80% reduction** in total render time when UI elements are present and mostly static. This single change is expected to close most of the performance gap.
-   **Batched Rendering & Encoder Consolidation**: **~10-20% additional reduction** by minimizing state changes and reducing CPU-side driver overhead.
-   **Combined Optimizations**: Should achieve or exceed ModernGL parity, targeting a **~2-3ms** render time.

### Success Metrics

1.  **Target Performance**: ≤3ms render time (matching ModernGL baseline).
2.  **Frame Rate**: ≥140 FPS average on the same hardware.
3.  **Memory Usage**: No significant increase from the current WGPU implementation.
4.  **Visual Parity**: No rendering artifacts or visual regressions compared to the ModernGL backend.

### Implementation Order

1.  **Phase 3.1 (Critical)**: Implement dirty-tile optimization in `WGPUTextureRenderer`.
2.  **Phase 3.2 (High)**: Standardize all renderers on the batched model, starting with `apply_environmental_effect`.
3.  **Phase 3.3 (Medium)**: Consolidate command encoder usage.
4.  **Phase 3.4 (Medium)**: Integrate performance monitoring tools.

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