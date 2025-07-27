# WGPU Migration Plan for Catley

### Migration Status: Paused (July 2025)

**Decision:** The WGPU migration is being put on hold to prioritize development of the core game. The existing ModernGL backend is fully functional and exceeds performance baselines.

**Justification:**
- **Performance Regression:** The WGPU lighting system implementation currently performs at ~25 FPS, a significant regression from the 200+ FPS of its ModernGL counterpart. This could very well be a simple bug.
- **Development Overhead:** Resolving these low-level backend issues requires a disproportionate amount of time and effort compared to working on gameplay features.
- **Low Immediate Need:** While OpenGL is deprecated on macOS, the ModernGL backend remains fully supported and functional. There is no immediate risk that requires a mandatory migration.

**Next Steps:**
- The WGPU implementation will be kept in the `main` branch but will be disabled via a feature toggle.
- All new development will target the stable ModernGL backend.
- This plan and the existing WGPU code will serve as a starting point if/when the migration is resumed.

---

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py.

**Important**: Maintain implementation parity between backends to prevent subtle bugs. Unintentional divergences can cause hard-to-debug visual artifacts. For example, the introduction of dirty-tile optimization initially caused visual artifacts (ghosting of previously lit tiles). Root cause: the WGPU implementation had diverged from ModernGL by adding a transparent tile optimization that skipped generating vertices for fully transparent tiles. This caused stale vertex data to persist in the VBO. The fix was to remove this optimization and ensure WGPU matches ModernGL's behavior of always generating vertices for all tiles.

## Phase 4: GPU Lighting System Port

### Step 4.2: Port GPULightingSystem to WGPU

This is implemented, but there's currently a major performance regression in the WGPU version.

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

## Postponed WGPU Performance Optimization

**Current Performance Gap**: WGPU renders at ~180-240 FPS (~3.7-4.4ms render time) vs ModernGL at ~140-300 FPS (~1.7-2.7ms render time).

Single-Encoder Performance Optimization: An attempt was made to refactor the WGPU backend to use a single command encoder per frame. While theoretically correct for performance, this led to persistent and difficult-to-debug visual artifacts (flickering, incorrect composition). The issue appears to be a subtle race condition or architectural conflict between the FrameManager's two-pass draw/present loop and WGPU's explicit command submission model. Decision: Postponing this optimization to maintain development momentum. The current WGPU backend is functional but slower than ModernGL. This can be revisited after the rest of the WGPU migration (e.g., the lighting system) is complete.


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