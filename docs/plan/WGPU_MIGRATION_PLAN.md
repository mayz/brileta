# WGPU Migration Plan for Catley

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py.

**Important**: Maintain implementation parity between backends to prevent subtle bugs. Unintentional divergences can cause hard-to-debug visual artifacts. For example, the introduction of dirty-tile optimization initially caused visual artifacts (ghosting of previously lit tiles). Root cause: the WGPU implementation had diverged from ModernGL by adding a transparent tile optimization that skipped generating vertices for fully transparent tiles. This caused stale vertex data to persist in the VBO. The fix was to remove this optimization and ensure WGPU matches ModernGL's behavior of always generating vertices for all tiles.

## Phase 3: WGPU Performance Optimization

**Current Performance Gap**: WGPU renders at ~180-240 FPS (~3.7-4.4ms render time) vs ModernGL at ~140-300 FPS (~1.7-2.7ms render time).

Analysis of the existing ModernGL and WGPU backends reveals that the performance gap is not due to a fundamental limitation of WGPU, but rather a specific, critical optimization present in the ModernGL backend that is currently missing from the WGPU implementation. The following plan prioritizes closing this gap first, followed by broader architectural improvements.

### Step 3.3: A Verifiable, Five-Step Refactor to a Single Command Encoder

Preamble: This plan is designed to be executed as five distinct, atomic steps. After each step, the application will be in a fully working, testable state. The ultimate goal is to fix the WGPU performance bottleneck by ensuring only one command encoder is used per frame. To achieve this safely, we must first refactor the rendering architecture to be compatible with this model. This plan explicitly accounts for the fact that WorldView manages its own GlyphBuffer directly, while other UI views use a Canvas object.

#### **Step 3.3.1: Isolate CPU-Side Resources in UI Canvases**
**This has been implemented and committed.**

#### **Step 3.3.2: Prepare the Rendering Pipeline for New Information**

*   **Goal:** To make the rendering pipeline aware of the newly isolated CPU buffers from Step 1, without actually changing the core rendering logic yet.

*   **Conceptual Action:** Update the method signatures along the data pipeline. When a `WGPUCanvas` creates its texture, it will now pass both its GPU buffer *and* its new CPU buffer to `WGPUGraphicsContext`. The graphics context will, in turn, update its own internal method to accept this new CPU buffer and pass it along to the `WGPUTextureRenderer`. For this step, the texture renderer will accept the new CPU buffer but will continue to ignore it and use its old, internal logic.

*   **Verification:** The game compiles and runs. There will be **no visual or performance changes**. The application remains in a stable, working state. We have successfully prepared the system for the next change without altering its behavior.

---

#### **Step 3.3.3: Activate CPU Buffer Isolation and Make the Renderer Stateless**

*   **Goal:** To make the `WGPUTextureRenderer` fully stateless, using the dedicated resources provided by each canvas and eliminating the last part of the CPU performance bug.

*   **Conceptual Action:** Modify the `WGPUTextureRenderer`. It will now be a pure service. Remove its internal, shared CPU buffer entirely. Its rendering method will now use the specific CPU buffer provided by each `WGPUCanvas` for its vertex calculations. It no longer owns any per-draw state.

*   **Verification:** The game compiles and runs. It is **visually identical** to the previous step. Performance should be the same as, or slightly better than, Step 1. The application is stable, and the resource management for all views is now correct and robust.

---

#### **Step 3.3.4: Implement and Test the High-Performance Render Path Safely**

*   **Goal:** To build and verify the new, high-performance, single-encoder rendering loop without breaking the existing, stable loop.

*   **Conceptual Action:** Create a "sidetrack" for testing. In `WGPUGraphicsContext`, create a *new, separate method* called `finalize_present_single_pass`. This new method will contain the ideal rendering logic: it creates one command encoder, orchestrates all `draw` and `present` calls for every view and overlay (recording all their commands into that single encoder), and then submits it once to the GPU. In the main application loop (`GlfwApp`), add a temporary boolean flag. Based on this flag, the app will call either the old, slow `finalize_present` or the new, fast `finalize_present_single_pass`.

*   **Verification:** You can now toggle the flag to switch between rendering paths. With the flag `False`, the game runs exactly as it did in Step 3. With the flag `True`, the game is **visually identical**, but the `dev.fps` counter is **significantly higher**. You have now verifiably achieved the performance goal in a safe, controlled, and directly comparable manner.

---

#### **Step 3.3.5: Decommission the Old Render Path and Finalize the Architecture**

*   **Goal:** To clean up the code by making the new, fast rendering path the only path, completing the refactor.

*   **Conceptual Action:** The high-speed path has been proven correct and performant. Now, we remove the old infrastructure. Delete the temporary boolean flag in `GlfwApp`. In `WGPUGraphicsContext`, delete the old `finalize_present` method and rename `finalize_present_single_pass` to `finalize_present`. The `FrameManager`'s rendering logic can now be simplified, as the graphics context is fully in control.

*   **Verification:** The game compiles and runs. It performs identically to the high-speed path in Step 4. The refactor is complete, and the code is now both correct and performant.

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