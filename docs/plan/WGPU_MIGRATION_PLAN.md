# WGPU Migration Plan for Catley

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py.

## Phase 3: WGPU Performance Optimization

**Current Performance Gap**: WGPU renders at ~117-118 FPS (~8.5ms render time) vs ModernGL at ~140-300 FPS (~1.7-2.7ms render time).

**Step 3.1 Completed**: Command buffer multiplexing eliminated - background rendering converted to batched approach with bind group caching. Achieved ~7% performance improvement.

### Root Cause Analysis

**Critical Performance Bottlenecks Identified:**

#### 1. Excessive Command Buffer Creation (Major Bottleneck)
- **WGPU Current**: Creates **multiple command encoders per frame**
  - `finalize_present()`: Creates main command encoder in `graphics.py:460`
  - `background_renderer.render_immediately()`: Creates separate command encoder in `background_renderer.py:152` 
  - **Each background texture = separate command submission**
- **ModernGL**: Single draw call sequence, no command buffer overhead

#### 2. Inefficient Background Rendering (Major Bottleneck)
- **WGPU Current**: `draw_background()` calls `render_immediately()` which:
  - Creates fresh command encoder per background texture
  - Creates new bind group per texture (`background_renderer.py:189-205`)
  - Submits individual command buffer immediately
- **ModernGL**: Uses persistent VBO/VAO, single draw call with texture binding

#### 3. Bind Group Creation Per Frame
- **WGPU Current**: Background renderer creates new bind groups every frame
- **ModernGL**: Reuses shader programs and VAOs

#### 4. Multiple GPU Resource Uploads
- **WGPU Current**: Multiple `queue.write_buffer()` calls per frame across renderers
- **ModernGL**: Single VBO upload per renderer

### Step 3.2: Implement Additional Bind Group Caching

**Priority: HIGH** - Significant performance impact

1. **Extend bind group caching to other renderers** (background renderer already completed):
   - Cache atlas texture bind group in screen renderer
   - Cache bind groups in UI texture renderer by texture ID
   - Pre-create common bind groups during initialization

2. **Optimize bind group creation patterns**:
   - Cache uniform buffer bind groups where possible
   - Reduce bind group creation frequency in screen renderer

3. **Implement bind group cleanup**:
   - Remove cached bind groups when textures are destroyed
   - Periodic cleanup of unused bind groups

### Step 3.3: Optimize Buffer Management

**Priority: MEDIUM** - Incremental improvements

1. **Reduce buffer uploads per frame**:
   - Batch vertex data updates where possible
   - Use change detection to avoid unnecessary uploads
   - Implement dirty flagging for uniform buffers

2. **Use persistent staging buffers**:
   - Pre-allocate buffers for common sizes
   - Reuse buffers across similar operations

3. **Minimize `queue.write_buffer` calls**:
   - Combine multiple small uploads into single larger uploads
   - Use ring buffering for dynamic content

### Step 3.4: Architecture Alignment with ModernGL

**Priority: MEDIUM** - Long-term maintainability

1. **Standardize rendering patterns**:
   - Make all renderers follow batched approach
   - Consistent resource management across components
   - Unified error handling and cleanup

2. **Performance monitoring integration**:
   - Add timing measurements to identify regressions
   - Automated performance testing in CI
   - Benchmark comparison tools

### Expected Performance Improvements

- **Command buffer consolidation**: ~50-60% render time reduction
- **Bind group caching**: ~20-30% render time reduction  
- **Buffer optimization**: ~10-15% render time reduction
- **Combined optimizations**: Should achieve ModernGL parity (~2-3ms render time)

### Success Metrics

1. **Target Performance**: ≤3ms render time (matching ModernGL baseline)
2. **Frame Rate**: ≥140 FPS average on same hardware
3. **Memory Usage**: No significant increase from current WGPU implementation
4. **Visual Parity**: No rendering artifacts or visual regressions

### Implementation Order

1. **Phase 3.1** (Critical): ✅ **COMPLETED** - Fixed background rendering architecture
2. **Phase 3.2** (High): Implement additional bind group caching
3. **Phase 3.3** (Medium): Buffer management optimizations
4. **Phase 3.4** (Medium): Architecture cleanup and monitoring

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