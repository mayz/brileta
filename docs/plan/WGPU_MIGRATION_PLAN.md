# WGPU Migration Plan for Catley

This document outlines the specific technical steps for migrating Catley from ModernGL to wgpu-py.

## Current Status

**Active backend:** ModernGL (default in `config.py`). Recommended due to faster startup (~350ms vs ~1200ms for WGPU).

**WGPU backend:** Ready to go and future-proof (WebGPU/Vulkan/Metal). The ~1.2s startup cost comes from Metal shader compilation and adapter enumeration. Consider a loading screen when switching to WGPU as default.

**Upgrade strategy:** Keep wgpu-py reasonably current (within 2-3 versions) to avoid accumulating breaking changes. The canvas/GUI APIs have stabilized with the `rendercanvas` split.

**Important**: Maintain implementation parity between backends to prevent subtle bugs. Unintentional divergences can cause hard-to-debug visual artifacts. For example, the introduction of dirty-tile optimization initially caused visual artifacts (ghosting of previously lit tiles). Root cause: the WGPU implementation had diverged from ModernGL by adding a transparent tile optimization that skipped generating vertices for fully transparent tiles. This caused stale vertex data to persist in the VBO. The fix was to remove this optimization and ensure WGPU matches ModernGL's behavior of always generating vertices for all tiles.

## Phase 4: GPU Lighting System Port

### Step 4.2: Port GPULightingSystem to WGPU

This is still in progress and exhibiting a slight performance regression compared to the ModernGL lighting backend.

## Phase 5: Integration and Validation

### Step 5.1: Visual Parity Testing

1. **Implement screenshot capture** for both backends
2. **Create test scenes** with known configurations:
   - Outdoor scene with directional shadows
   - Indoor scene with torch lighting
   - Transition areas with light spillover
3. **Compare pixel-by-pixel** for differences
4. **Document any acceptable variations**

### Step 5.2: Performance Validation

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

## Deferred WGPU Command Encoder Performance Optimization

**Note:** This may no longer be necessary. The current WGPU backend is functional *and already appears faster than ModernGL* when used with the ModernGL lighting backend.

Single-Encoder Performance Optimization: An attempt was made to refactor the WGPU backend to use a single command encoder per frame. While theoretically correct for performance, this led to persistent and difficult-to-debug visual artifacts (flickering, incorrect composition). The issue appears to be a subtle race condition or architectural conflict between the FrameManager's two-pass draw/present loop and WGPU's explicit command submission model. Decision: Postponing this optimization to maintain development momentum.

## Deferred GPU Lightmap Retention Optimization

**Status:** Deferred - prioritizing game features over performance optimization.

**The bottleneck:** GPU→CPU lightmap transfer (`Texture.read_into` in ModernGL, `map_sync` in WGPU) consumes ~20% of frame time when profiled. This readback exists because visibility masking, animation effects, and tile appearance blending currently happen in Python on the CPU.

**The optimization:** Move visibility/animation/blending operations to GPU fragment shaders. The lightmap would stay on the GPU and only the final composited frame would be read back for display. This is standard practice in commercial games - they minimize CPU↔GPU transfers by doing as much work as possible on the GPU.

**Implementation scope:** This is architecturally significant. It requires:
1. Porting visibility masking logic to WGSL/GLSL shaders
2. Uploading FOV/explored state as GPU textures
3. Moving animation effect calculations (pulsation, etc.) to shaders
4. Compositing tile appearances with lighting in a final GPU pass

**See also:** `PERF:` comments in `gpu_lighting.py` for both backends mark the exact readback locations.

## Phase 6: Advanced Multi-Pass Architecture

### Step 6.1: Advanced Architecture Implementation
**Reference**: Resume `LIGHTING_SYSTEM_PLAN.md` Phase 2.3.

Once basic parity is achieved, implement the advanced architecture:
1. **Static light texture caching**
2. **Dynamic light per-frame rendering**
3. **Intelligent cache invalidation**
4. **Compute shader optimization** (if beneficial)

## Critical Success Factors

3. **Shader Translation Accuracy**: Shadows must behave identically
4. **Performance Parity**: No regression from ModernGL
5. **Visual Parity**: Screenshots must match

## Risk Mitigation

- Use WGSL validator early and often
- Test individual shader features in isolation
- Consider keeping ModernGL as permanent fallback
