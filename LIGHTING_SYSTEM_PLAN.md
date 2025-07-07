# GPU Lighting System Implementation Plan

## Overview

This plan outlines the implementation of a GPU-based lighting system to replace the current CPU implementation, delivering significant performance improvements while maintaining visual consistency and enabling future enhancements.

## Key Design Decisions

### 1. Implementation Sequence
- Merge wip-sunlight branch first - Integrate DirectionalLight support before starting GPU implementation
- Implement GPU system with full feature set - Support both point lights AND directional lights from the start
- Avoid technical debt - Single implementation effort rather than retrofitting

### 2. Visual Rendering Strategy
- Phase 1: Tile-based lighting - Maintain current discrete/pixelated aesthetic (drop-in replacement)
- Phase 2+: Continuous lighting option - Add sub-tile resolution for smoother effects
- Dual capability - Support both rendering modes as configuration options

### 3. Architecture Approach
- Interface compatibility - Implement existing `LightingSystem` interface for seamless replacement
- ModernGL integration - Leverage existing shader infrastructure and resource management
- Fallback strategy - Graceful degradation to CPU system for older hardware

## Implementation Phases

### Phase 1: Foundation & Integration (High Priority)

#### 1.3 Tile-based Point Light Rendering
- Goal: GPU-accelerated point lights with identical visual output to CPU system
- Deliverables:
  - Point light compute shader with distance falloff
  - Color blending matching CPU behavior
  - Tile-aligned output (same resolution as current system)
- Priority: 10/10 - Immediate performance gains with zero visual regression

### Phase 2: Feature Parity (High Priority)

#### 2.1 Static vs Dynamic Light Separation
- Goal: Implement caching system for optimal performance
- Deliverables:
  - Static light pre-computation and caching
  - Dynamic light per-frame updates
  - Cache invalidation on light changes
- Priority: 8/10 - Major performance optimization for scenes with many static lights

#### 2.2 Flicker Effects
- Goal: GPU-based dynamic lighting effects
- Deliverables:
  - GPU noise generation for flicker
  - Time-based intensity modulation
  - Configurable flicker parameters matching CPU system
- Priority: 6/10 - Maintains visual parity, enables future enhancements

#### 2.3 Directional Lighting (Sunlight)
- Goal: GPU implementation of directional lights from wip-sunlight
- Deliverables:
  - Directional light compute shader
  - Sky exposure calculations
  - Day/night cycle support
- Priority: 8/10 - Enables rich environmental lighting

### Phase 3: Advanced Features (Medium Priority)

#### 3.1 Shadow Casting
- Goal: GPU-accelerated shadow computation
- Deliverables:
  - Actor shadow casting matching CPU behavior
  - Terrain shadow support from tile data
  - Configurable shadow quality settings
- Priority: 6/10 - Significant performance improvement for shadow-heavy scenes

#### 3.2 Continuous Lighting Option
- Goal: Sub-tile resolution lighting for smoother effects
- Deliverables:
  - Higher resolution lighting computation (2x, 4x, 8x per tile)
  - Smooth gradients across tile boundaries
  - Configuration option to choose tile-based vs continuous
  - Shadows that don't align to tile grid
- Priority: 8/10 - Major visual enhancement opportunity

### Phase 4: Performance & Polish (Lower Priority)

#### 4.1 Performance Optimizations
- Goal: Maximize GPU efficiency
- Deliverables:
  - Frustum culling for lights
  - Level-of-detail for distant lights
  - Batched light data updates
  - GPU memory usage optimization
- Priority: 6/10 - Enables even more complex lighting scenarios

#### 4.2 Environmental Shadow Effects
- Goal: Dynamic environmental shadows for atmospheric outdoor scenes
- Deliverables:
  - Cloud shadow system with moving shadow patterns
  - Weather-based shadow effects (storm clouds, etc.)
  - Configurable shadow pattern textures
  - Integration with day/night cycle and directional lighting
- Priority: 7/10 - High visual impact for outdoor scenes, requires directional lighting + shadow system
- Dependencies: Directional lighting (2.3), Shadow casting (3.1), ideally Continuous lighting (3.2)

#### 4.3 Configuration & Quality Settings
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

### CPU System Status (July 2024)

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
- Location: `catley/view/render/lighting/gpu.py`
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

### ✅ Phase 1.0: CPU Code Cleanup & Stabilization (COMPLETED - Dec 2024)
- **Goal**: Remove complex spillover code and stabilize for merge
- **Status**: COMPLETED ✅
- **Achievements**:
  - Removed 6 complex spillover methods from cpu.py (341 line reduction, exceeded ~300 line target)
  - Simplified spillover system to basic outdoor→indoor light streaming only
  - Eliminated crashes and game-breaking bugs
  - All 444 tests pass, all quality checks pass
  - Core functionality preserved: torch lighting, directional sunlight, shadows, FOV integration
- **Files Modified**:
  - `/catley/view/render/lighting/cpu.py` - Removed complex spillover methods
  - `/catley/game/game_world.py` - Removed forced starting room outdoor setup

### ✅ Room Generation Enhancement (COMPLETED - Dec 2024)
- **Goal**: Randomize outdoor room distribution instead of fixed starting room
- **Status**: COMPLETED ✅
- **Changes**:
  - Each room now has 20% probability of being outdoor (instead of just starting room)
  - Uses `MapRegion.create_outdoor_region()` and `create_indoor_region()` factory methods
  - Outdoor rooms get proper `sky_exposure=1.0` for sunlight system
- **Files Modified**:
  - `/catley/environment/generators.py` - Added random outdoor room generation
  - `/catley/game/game_world.py` - Removed starting room outdoor forcing

### ✅ Visual Bug Fix: Outdoor Tile Colors (COMPLETED - Dec 2024)
- **Goal**: Fix outdoor floor tiles appearing blue outside FOV
- **Status**: COMPLETED ✅
- **Problem Solved**: Outdoor room floor tiles were using indoor `DARK_GROUND` color instead of `OUTDOOR_DARK_GROUND`
- **Solution**: Extended region-aware appearance system to include floor tiles (previously only boulders)
- **Files Modified**:
  - `/catley/environment/map.py` - Added floor tiles to region-aware color system

### ✅ Phase 1.0a: Investigation Results (COMPLETED - 2024)
- **Goal**: Test and refine outdoor sunlight system
- **Status**: COMPLETED - Revealed fundamental architectural issues
- **Key Findings**:
  - CPU approach fights against natural light physics
  - 12+ spillover methods create unmaintainable complexity
  - Visual bugs persist despite extensive debugging
  - **Strategic Decision**: Simplify CPU, fast-track GPU physics implementation

## Conclusion

This plan provides a clear path to significantly improved lighting performance while maintaining the existing game architecture. The phased approach allows for incremental implementation, testing, and validation at each stage. The dual rendering capability (tile-based and continuous) ensures both visual consistency and future enhancement opportunities.

The focus on high-impact phases ensures maximum return on implementation effort, while the comprehensive risk mitigation strategies minimize the chance of integration issues or performance regressions.

### ✅ Phase 1.1: Merge wip-sunlight Branch (COMPLETED)
- **Goal**: Integrate cleaned-up DirectionalLight support into main branch
- **Priority**: 9/10 - Unblocks GPU development
- **Status**: COMPLETED ✅
- **Deliverables**:
  - DirectionalLight class available in lighting system
  - Sun/moonlight capability in CPU system
  - Stable, simplified lighting code in main branch

### ✅ Phase 1.2: GPU Infrastructure Setup (COMPLETED - Jan 2025)
- **Goal**: Create core GPU lighting architecture
- **Status**: COMPLETED ✅
- **Deliverables**:
  - `catley/view/render/lighting/gpu.py` - GPULightingSystem class ✅
  - Basic compute shader programs ✅
  - GPU buffer management for light data ✅
  - Integration with ModernGLGraphicsContext ✅
  - Comprehensive unit test suite (21 test cases) ✅
- **Priority**: 10/10 - Enables all future GPU lighting work
- **Key Features**:
  - Hardware detection and graceful fallback to CPU system
  - GLSL 4.3 compute shaders with 8x8 work groups
  - Resource management with proper cleanup
  - Interface-compatible drop-in replacement for CPU system
  - Handles up to 256 lights with overflow protection

**Current Status**: Phase 1.0, 1.1, and 1.2 complete. Ready to proceed with Phase 1.3 (tile-based point light rendering).