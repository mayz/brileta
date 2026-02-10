# GPU Lighting System Implementation Plan

**SHADER ASSETS LOCATION**: All shaders are organized in `assets/shaders/wgsl/` with subdirectories:
- `glyph/` - Character rendering (render.wgsl)
- `lighting/` - GPU lighting system (point_light.wgsl)
- `screen/` - Main screen rendering (main.wgsl)
- `effects/` - Atmospheric effects (cloud_layer.wgsl)

## Implementation Phases

### Phase 2: Feature Parity (High Priority)

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

**Implementation Note:**

This plan describes the target architecture for a highly scalable lighting system. The current WGPU implementation uses a simplified single-pass approach that achieves visual parity. The multi-pass system described above is a future optimization.

### Phase 3: Performance Benchmarking
- **Goal**: Standalone benchmarking for the GPU lighting system
- **Status**: NOT STARTED
- **Deliverables**:
  - Multiple test scenarios (light counts, scene complexity, shadow density, directional lighting)
  - Performance metrics (FPS, frame time, memory usage)
  - Automated performance regression detection
  - Stress testing with high light counts

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
- **Status**: Core cloud shadows implemented (`AtmosphericLayerSystem`, `cloud_layer.wgsl`). Moving cloud shadow patterns, ground mist, noise-based generation, and sky exposure masking are all working.
- Remaining work:
  - Weather-based shadow effects (storm clouds, etc.)
  - Configurable shadow pattern textures

#### 5.3 Configuration & Quality Settings
- Goal: Flexible lighting system configuration
- Deliverables:
  - Quality presets (Performance/Balanced/Quality)
  - Debug visualization tools
  - Hardware capability detection
- Priority: 4/10 - User experience and debugging improvements

## Technical Architecture

### Core Components

#### GPULightingSystem
- Location: `brileta/backends/wgpu/gpu_lighting.py`
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
```wgsl
struct LightData {
    position: vec2f,     // World position
    radius: f32,         // Light radius
    intensity: f32,      // Current intensity (includes flicker)
    color: vec3f,        // RGB color (0.0-1.0)
    light_type: u32,     // 0=static, 1=dynamic, 2=directional
    direction: vec2f,    // For directional lights
    padding: f32,        // Memory alignment
}
```

### Integration Points

#### WGPUGraphicsContext Integration
- Resource Management: Use existing texture/buffer caching system
- Texture Pipeline: Output compatible with existing renderers
- Shader Management: Integrate with existing shader infrastructure

#### Error Handling
- Robust error recovery and logging

## Future Enhancement Ideas

- Soft shadows with penumbra
- Volumetric lighting effects
- Color temperature simulation
- Weather-based lighting effects (storm darkening, lightning flashes)
- Fog and atmospheric scattering
- Water reflection and refraction effects

## Completed

The core GPU lighting system is implemented and is the only lighting backend. There is no CPU fallback. Key completed work:

- WGPU fragment shader pipeline (chose fragment shaders over compute shaders for macOS compatibility)
- Point lights with distance falloff, flicker, and shadow casting
- Directional sun with ray-marched terrain shadows
- Sky exposure for indoor/outdoor transitions
- Tile emission
- Atmospheric effects (cloud shadows, ground mist)
- GPU compositing with fog-of-war integration
- Performance caching (revision tracking, sky exposure texture caching, uniform buffer hashing)
