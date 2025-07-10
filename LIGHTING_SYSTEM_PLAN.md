# GPU Lighting System Implementation Plan

**SHADER ASSETS LOCATION**: All shaders are organized in `assets/shaders/` with subdirectories:
- `glyph/` - Character rendering (render.vert, render.frag)
- `lighting/` - GPU lighting system (point_light.vert, point_light.frag)
- `screen/` - Main screen rendering (main.vert, main.frag)
- `ui/` - User interface rendering (texture.vert, texture.frag)

## Implementation Phases

### Phase 2: Feature Parity (High Priority)

################################################################
Do this next phase *before* the WebGPU migration.
################################################################

#### **Phase 2.1: GPU Directional Lighting & Shadow Parity (Immediate Priority)**

*   **Goal**: Achieve full visual and feature parity with the `CPULightingSystem` by implementing directional sunlight and its corresponding parallel shadows in the GPU pipeline. This makes the ModernGL implementation a feature-complete prototype.
*   **Priority**: 9/10 - This is the final step to stabilize the ModernGL renderer before focusing on gameplay and the eventual WebGPU migration.
*   **Implementation Approach**: A single-pass fragment shader enhancement that adds directional light calculations on top of the existing point-light logic.

**Key Deliverables**:

1.  **Python-Side Data Pipeline (`GPULightingSystem`)**
    *   Verify that `_set_directional_light_uniforms()` correctly finds the active `DirectionalLight` and passes its `direction`, `color`, and `intensity` to the shader.
    *   Verify that `_update_sky_exposure_texture()` correctly creates and binds a texture representing tile-by-tile sky exposure values. This texture is essential for the shader to know which areas are "outdoors".

2.  **Fragment Shader Uniforms (`assets/shaders/lighting/point_light.frag`)**
    *   Add the necessary uniforms to receive directional light data from the Python backend.
    ```glsl
    // Add to the uniform block
    uniform vec2 u_sun_direction;
    uniform vec3 u_sun_color;
    uniform float u_sun_intensity;

    uniform sampler2D u_sky_exposure_map; // For outdoor/indoor detection
    uniform float u_sky_exposure_power;   // To control light fall-off curve
    ```

3.  **Sky Exposure and Sunlight Logic (in `point_light.frag`)**
    *   After the point-light loop, sample the `u_sky_exposure_map` to determine if the current fragment (pixel) is outdoors.
    *   If `sky_exposure > 0.0`, calculate the sun's contribution. Use `pow(sky_exposure, u_sky_exposure_power)` to create a more natural, non-linear falloff at the edges of outdoor areas.
    *   Combine the sun's contribution with the point-light color using `max()`, preventing oversaturation and ensuring the brightest light source prevails.

4.  **Directional Shadow Logic (in `point_light.frag`)**
    *   This is a critical feature for visual parity with the CPU system.
    *   After the sunlight calculation, add a new loop that iterates through all `u_shadow_caster_positions`.
    *   For each shadow caster, determine if the current fragment lies in the "shadow path" cast by that caster *away from the sun's direction*. This is different from point-light shadows, which radiate from a central point.
    *   If the fragment is in a directional shadow path, reduce its final calculated light color by `u_shadow_intensity`.
    *   The length of the shadow can be controlled by `u_shadow_max_length`.

5.  **Final Blending and Output**
    *   The final fragment color will be the result of (Point Lights + Directional Light) * Directional Shadows.

6.  **Testing & Validation**
    *   Perform A/B testing by toggling `config.GPU_LIGHTING_ENABLED`.
    *   **Success Criteria**: Outdoor areas in GPU mode should be brightly lit by the sun, and both actors and shadow-casting tiles should cast long, parallel shadows away from the sun's direction, matching the visual output of the CPU implementation.

################################################################
Only do stuff here and below *after* the WebGPU migration.
################################################################

#### **Phase 2.3: Architectural Blueprint for High-Performance Lighting**

*   **Goal**: Define the architectural principles for an optimal, multi-pass lighting system that separates static and dynamic elements for maximum performance. This design is the target for the final WebGPU implementation.
*   **Priority**: 5/10 - The design is important, but its full implementation is deferred.
*   **Implementation Approach**: A multi-pass rendering pipeline that caches expensive, unchanging light and shadow data to a persistent GPU texture.

**Key Architectural Principles**:

1.  **Static Light Texture Caching**
    *   A persistent **GPU Texture** (the "static lightmap") will cache the combined contribution of all non-moving light sources.
    *   A dedicated **Framebuffer** will be used to render to this static lightmap.
    *   This static lightmap includes: ambient light, static point lights, and global directional lights (sun/moon).
    *   The cache will only be invalidated (re-rendered) when a crucial static element changes:
        *   Static light positions, colors, or radii.
        *   Global directional light properties (e.g., time of day).
        *   Static shadow caster positions (e.g., map structure changes).

2.  **Dynamic Light Rendering**
    *   A separate, per-frame **GPU Texture** (the "dynamic lightmap") will be used to render all moving and changing lights.
    *   This includes flickering torches, temporary spell effects, and lights attached to moving actors.
    *   Shadows from dynamic sources (like actors) will be calculated in this pass every frame.

3.  **Multi-Pass Shader Pipeline**
    *   **Static Pass Shader**: An optimized shader that calculates the contribution of all static and directional lights. It runs only when the static cache is invalidated.
    *   **Dynamic Pass Shader**: A shader that calculates the contribution of dynamic lights, including time-based effects like flickering. It runs every frame.
    *   **Composite Pass Shader**: A simple blending shader that combines the static and dynamic lightmaps (e.g., using `max()`) and applies final color corrections (like gamma) to produce the final lit scene.

4.  **Light and Shadow Collection**
    *   The system will maintain separate collections for `static_lights`, `dynamic_lights`, `static_shadow_casters`, and `dynamic_shadow_casters`.
    *   This allows data to be sent to the appropriate shader pass efficiently.

5.  **Intelligent Cache Invalidation**
    *   A `static_cache_valid` flag, tied to game state revisions (e.g., `game_map.structural_revision`), will control when the expensive static pass is re-run.

6.  **Backend-Agnostic Data Structures**
    *   Light and shadow data will be packed into uniform buffers or storage buffers for the GPU. The conceptual data structure is universal, even if the shader syntax changes.
    *   *(Note: The exact syntax will depend on the shader language, e.g., GLSL for OpenGL, WGSL for WebGPU, but the data layout remains the same.)*
    ```glsl
    // Example conceptual data layout
    struct Light {
        vec2 position;
        float radius;
        vec3 color;
        // ... other properties for flicker, type, etc.
    };
    ```

**Migration & Implementation Note:**

This plan describes the target architecture for a highly scalable lighting system. **The full implementation of this multi-pass system should be deferred until the WebGPU migration (Roadmap Step 3.5).**

For the current ModernGL prototype, a simplified single-pass approach that achieves visual parity is sufficient and more efficient from a development standpoint.

### Phase 3: Performance Benchmarking (High Priority)
- **Goal**: Create comprehensive benchmarking program to compare GPU vs CPU performance
- **Status**: NOT STARTED
- **Priority**: 8/10 - Critical for validating GPU performance gains
- **Prerequisites**: Phases 1.35, 1.4 and 2.x completion (full feature parity required)
- **Deliverables**:
  - Standalone benchmarking program comparing GPU vs CPU lighting
  - Multiple test scenarios (light counts, scene complexity, shadow density, directional lighting)
  - Performance metrics (FPS, frame time, memory usage)
  - Automated test suite for performance regression detection
  - Documentation of expected performance improvements
  - Integration validation and stress testing with high light counts
  - Before/after performance measurement for Phase 1.35 optimizations
- **Success Criteria**: Demonstrate 5-10x performance improvement with GPU system

### Phase 4: Advanced Features (Medium Priority)

#### 4.1 Continuous Lighting Option
- Goal: Sub-tile resolution lighting for smoother effects
- Deliverables:
  - Higher resolution lighting computation (2x, 4x, 8x per tile)
  - Smooth gradients across tile boundaries
  - Configuration option to choose tile-based vs continuous
  - Shadows that don't align to tile grid
- Priority: 8/10 - Major visual enhancement opportunity

### Phase 5: Performance & Polish (Lower Priority)

#### 5.1 Performance Optimizations
- Goal: Maximize GPU efficiency
- Deliverables:
  - Frustum culling for lights
  - Level-of-detail for distant lights
  - Batched light data updates
  - GPU memory usage optimization
  - Batch light processing with numpy operations for data formatting
  - More efficient viewport-based light culling algorithms
  - Performance monitoring hooks and memory usage validation
- Priority: 6/10 - Enables even more complex lighting scenarios

#### 5.2 Environmental Shadow Effects
- Goal: Dynamic environmental shadows for atmospheric outdoor scenes
- Deliverables:
  - Cloud shadow system with moving shadow patterns
  - Weather-based shadow effects (storm clouds, etc.)
  - Configurable shadow pattern textures
  - Integration with day/night cycle and directional lighting
- Priority: 7/10 - High visual impact for outdoor scenes, requires directional lighting + shadow system
- Dependencies: Directional lighting (2.1), Shadow casting (1.4), ideally Continuous lighting (4.1)

#### 5.3 Configuration & Quality Settings
- Goal: Flexible lighting system configuration
- Deliverables:
  - Runtime GPU/CPU selection
  - Quality presets (Performance/Balanced/Quality)
  - Debug visualization tools
  - Hardware capability detection
- Priority: 4/10 - User experience and debugging improvements

## Current Sunlight Implementation Issues

Based on extensive testing of the CPU sunlight system, several critical issues have been identified that need resolution before GPU implementation:

### High Priority Issues

#### Issue 1: Inconsistent Boulder/Wall Background Colors
- Problem: Boulders (walls) show blue backgrounds in some regions despite outdoor tile conversion
- Root Cause: Tile conversion only applies to test region, other walls still use indoor WALL tiles
- Symptoms: Blue backgrounds appear when player moves between regions or areas outside FOV
- Priority: 10/10 - Breaks outdoor scene immersion completely

#### Issue 2: Missing Directional Sun Shadows
- Problem: No directional shadows despite having sun angle/direction data
- Current State: Only torch shadows exist, creating unrealistic outdoor lighting
- Expected: All shadows should point away from sun direction (southeast per config)
- Priority: 9/10 - Core feature of directional lighting system

#### Issue 3: Torch Shadow Logic Reversed
- Problem: Torch light invisible outdoors but torch shadows still appear
- Root Cause: Daylight reduction affects light but not shadow calculation
- Result: Dark areas behind enemies with no visible light source
- Priority: 8/10 - Confusing and unrealistic lighting behavior

### Medium Priority Issues

#### Issue 4: Harsh Indoor/Outdoor Transitions
- Problem: Sharp cutoff between bright outdoor and dark indoor areas
- Missing: Light spillover from outdoor areas into indoor entrances
- Impact: Jarring visual discontinuity, breaks immersion
- Priority: 7/10 - Important for natural-feeling transitions

#### Issue 5: Torch "Appearing from Nowhere" Effect
- Problem: Torch suddenly becomes visible when moving indoors
- Cause: Abrupt change from 10% to 100% torch intensity
- Solution: Smoother intensity transitions between regions
- Priority: 6/10 - Player feedback and continuity issue

#### Issue 6: Flat Outdoor Visual Interest
- Problem: Outdoor scenes less visually engaging than indoor torch-lit areas
- Cause: Uniform sunlight lacks the gradient and variation of torch lighting
- Note: May be addressed by varied terrain rather than lighting changes
- Priority: 5/10 - Aesthetic enhancement opportunity

### Solutions Priority

#### Immediate Fixes (Before GPU Implementation)
1. Fix boulder background colors globally (10/10)
2. Implement directional sun shadows (9/10)
3. Fix torch shadow logic in daylight (8/10)
4. Improve torch visibility transition (6/10)

#### Future Enhancements (GPU Implementation Phase)
5. Light spillover system (7/10)
6. Varied outdoor terrain/materials (5/10)
7. Advanced shadow quality and soft shadows

### Implementation Notes

These issues reveal important design principles:
- Outdoor lighting needs fundamentally different behavior than indoor lighting
- Tile variants must be globally consistent, not region-specific
- Shadow systems need to consider primary light source (sun vs torch)
- Smooth transitions are crucial for believable lighting

The GPU implementation should address these core issues rather than simply porting the current CPU behavior.

### CPU System Status

#### ✅ Fixed Issues
1. **Boulder Background Colors** - Implemented region-aware tile appearance system that makes boulders inherit appropriate ground colors based on their region's sky exposure
2. **Directional Sun Shadows** - Added complete directional shadow system that casts realistic shadows from the sun based on its direction, affecting both actors and shadow-casting tiles in outdoor areas
3. **Torch Shadow Logic** - Fixed torch shadows to respect per-tile daylight reduction, preventing mysterious shadows from invisible torches in bright sunlight
4. **Per-Tile Lighting Logic** - Changed from player-region-based to per-tile-based torch effectiveness, preventing torch from affecting outdoor areas when player moves indoors
5. **Light Spillover Basic Implementation** - Working spillover system that streams light from outdoor areas into indoor areas with gradual falloff
6. **FOV Integration** - Fixed field-of-view darkening to work properly with spillover effects (areas outside FOV are dimmed but spillover remains visible)

#### 🔄 Partially Working: Light Spillover
**Status: FUNCTIONAL BUT NEEDS REFINEMENT**
- Basic spillover is working - light streams from outdoor areas into indoor hallways
- Spillover respects actual outdoor light intensities (not fixed values)
- Spillover has distance-based falloff and doesn't exceed source brightness
- Current implementation in `_apply_final_spillover()` uses region-based detection

**Working Features:**
- Light streams through doorways into dark indoor areas
- Gradual falloff over 2-3 tiles from outdoor sources
- Spillover visible even outside player's field of view
- Physics-correct: spillover never brighter than source

#### ❌ Outstanding Issues
1. **Torch Lighting Inconsistency** - Dark bands or inconsistent illumination patterns from torch light on walls
2. **Door Background Colors** - Doors not matching surrounding wall colors consistently
3. **Wall Lighting Uniformity** - Some walls show banding or inconsistent lighting treatment

**Current State**: The spillover system works for the basic use case of outdoor→indoor light streaming, but there are rendering inconsistencies with torch lighting and tile appearance that create visual artifacts.

#### 📋 GPU Implementation Notes

**General Light Diffusion System (High Priority for GPU)**
The current CPU system uses targeted spillover for doorways, but the GPU system should implement a **general light diffusion system** that handles natural light bleeding between adjacent tiles based on intensity differences. This would support:
- Outdoor → Indoor spillover
- Bright room → Dark room transitions
- Torch → Shadow area diffusion
- Any lighting scenario with intensity gradients

This general system would replace the current targeted spillover and provide more realistic, physically-based lighting behavior across all scenarios.

## Technical Architecture

### Core Components

#### GPULightingSystem
- Location: `catley/backends/moderngl/gpu_lighting.py`
- Purpose: Main class implementing LightingSystem interface
- Key methods:
  - `compute_lightmap()` - Main API, returns numpy array
  - `update()` - Handle time-based effects
  - `on_light_*()` - Cache management

#### Compute Shaders
- Point Light Shader: Distance-based falloff, color blending
- Directional Light Shader: Sky exposure, directional illumination
- Shadow Shader: Raycast shadows from light sources
- Composite Shader: Combine lights with ambient, apply gamma

#### GPU Memory Layout
```glsl
struct LightData {
    vec2 position;      // World position
    float radius;       // Light radius
    float intensity;    // Current intensity (includes flicker)
    vec3 color;         // RGB color (0.0-1.0)
    uint lightType;     // 0=static, 1=dynamic, 2=directional
    vec2 direction;     // For directional lights
    float padding;      // Memory alignment
};
```

### Integration Points

#### ModernGLGraphicsContext Integration
- Resource Management: Use existing FBO caching system
- Texture Pipeline: Output compatible with existing renderers
- Shader Management: Integrate with existing shader infrastructure

#### Fallback Strategy
- Hardware Detection: Check compute shader support
- Graceful Degradation: Fall back to CPU system if needed
- Error Handling: Robust error recovery and logging

## Expected Performance Gains

### Current CPU Bottlenecks
- Nested loops in per-pixel calculations
- Serial light processing
- Expensive shadow ray casting
- CPU-bound distance calculations

### GPU Advantages
- Parallel light computation (100+ lights simultaneously)
- Hardware-accelerated math operations
- Efficient memory bandwidth utilization
- Native texture sampling for smooth operations

### Projected Improvements
- 5-10x performance for scenes with many lights
- Consistent frame times regardless of light count
- Enables complex effects previously too expensive
- Scalable architecture for future enhancements

## Risk Mitigation

### Technical Risks
- Hardware Compatibility: Fallback to CPU system for older GPUs
- Visual Consistency: Extensive A/B testing vs CPU system
- Performance Regression: Careful profiling and optimization

### Development Risks
- Scope Creep: Focus on core functionality first
- Integration Complexity: Incremental testing and validation
- Resource Management: Proper GPU memory handling

## Testing Strategy

### Visual Validation
- Side-by-side comparison with CPU system
- Automated screenshot diffing for regression detection
- Manual validation of lighting effects

### Performance Testing
- Benchmarking suite for various lighting scenarios
- Memory usage monitoring
- Frame time analysis

### Hardware Testing
- Multiple GPU vendors (NVIDIA, AMD, Intel)
- Various performance levels (integrated to high-end)
- Fallback mechanism validation

## Future Enhancement Opportunities

### Advanced Shadow Techniques
- Soft shadows with penumbra
- Volumetric lighting effects
- Multiple shadow cascades

### Physically-Based Lighting
- Realistic light falloff
- Color temperature simulation
- HDR lighting pipeline

### Dynamic Effects
- Animated light patterns
- Weather-based lighting effects (storm darkening, lightning flashes)
- Particle-light interactions
- Time-of-day lighting transitions
- Seasonal lighting variations

### Environmental Systems
- Cloud shadow patterns with realistic movement
- Fog and atmospheric scattering effects
- Underground/cave lighting transitions
- Water reflection and refraction effects
- Fire/explosion dynamic lighting

## Success Metrics

### Performance Metrics
- Frame rate improvement in lighting-heavy scenes
- Memory usage comparison
- GPU utilization efficiency

### Quality Metrics
- Visual parity with CPU system
- Enhanced effect quality with continuous lighting
- Stability across hardware configurations

### Development Metrics
- Implementation timeline adherence
- Code quality and maintainability
- Test coverage of new functionality

## Completed Phases

## Conclusion

This plan provides a clear path to significantly improved lighting performance while maintaining the existing game architecture. The phased approach allows for incremental implementation, testing, and validation at each stage. The dual rendering capability (tile-based and continuous) ensures both visual consistency and future enhancement opportunities.

The focus on high-impact phases ensures maximum return on implementation effort, while the comprehensive risk mitigation strategies minimize the chance of integration issues or performance regressions.

#### 🎯 **ARCHITECTURAL DECISION - Compute vs Fragment Shaders**:

**Problem Identified**: macOS OpenGL 4.1 vs Compute Shader Requirements (OpenGL 4.3+)
- **Root Cause**: Compute shaders require OpenGL 4.3+, but macOS caps OpenGL at 4.1
- **Solution**: Complete rewrite to use fragment shaders instead of compute shaders
- **Benefits**:
  - Works on OpenGL 3.3+ (including macOS OpenGL 4.1)
  - Single clean implementation (no dual compute/fragment maintenance burden)
  - Same parallel GPU performance characteristics
  - Simpler architecture and debugging

#

### Performance Benchmarks

**Location**: Performance benchmarks are integrated into the comprehensive test suite at:
- `/Users/mayz/catley/tests/rendering/effects/test_gpu_lighting_system.py`
- Test classes: `TestGPULightingPerformance` and `TestGPULightingVisualRegression`

**Benchmark Coverage**:
- Light count scaling (0-300+ lights)
- Memory usage optimization validation
- Buffer size calculations for 12-float format
- Visual regression testing vs CPU system
- Flicker determinism verification

**Current benchmark status**: Only unit test mocks exist. Real performance benchmarks require working GPU implementation.

**To run current tests**:
```bash
uv run pytest tests/rendering/effects/test_gpu_lighting_system.py::TestGPULightingPerformance -v
```

**TODO**: Create standalone benchmark script comparing CPU vs GPU performance with real lighting scenarios.

### How to Switch Lighting Systems

**Enable GPU Lighting** (default):
```python
# In catley/config.py
GPU_LIGHTING_ENABLED = True
```

**Disable GPU Lighting** (force CPU):
```python
# In catley/config.py
GPU_LIGHTING_ENABLED = False
```

The system automatically:
1. Tries GPU lighting when enabled
2. Falls back to CPU if GPU unavailable
3. Provides identical visual output regardless of backend
4. Maintains all existing lighting features (flicker, shadows, sun/moon)
